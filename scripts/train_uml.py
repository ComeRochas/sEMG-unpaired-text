"""Train the dual-branch UML model.

Two scheduling modes (``--epoch-mode``):

* ``alternate`` (default) — each optimizer step processes 1 EMG batch + 1
  audio batch (both backward, then ``optim.step()``). The epoch ends after
  one full EMG pass; audio is cycled, so per epoch only ~``n_emg_batches``
  audio batches are seen out of the full audio dataset.

* ``both`` — each optimizer step processes 1 batch from a SINGLE modality.
  The epoch's schedule is the union of all EMG and all audio batches
  (length = ``n_emg_batches + n_audio_batches``), shuffled. Every batch
  from both datasets is processed exactly once per epoch.

The Transformer is shared (same Python object) between the two branches;
the AudioFrontend (wav2vec2-base) is always frozen.

EMG branch matches ``scripts/train_baseline.py``:
  - ``CachedRawEMGDataset`` + ``build_batches(max_batch_len)``
  - ``combine_fixed_length(raw_emg, fixed_raw_len)`` then per-sample CTC
    via ``decollate_tensor`` + ``pad_sequence``.

Usage
-----
    python scripts/train_uml.py --config configs/train_uml.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from semg_jepa.architecture import GaddyRawEMGEncoder, CTCHead, factor_to_strides
from semg_jepa.augmentations import RawEMGAugment
from semg_jepa.cached_dataset import CachedRawEMGDataset, build_batches
from semg_jepa.config_utils import parse_with_config, setup_stdout_logging
from semg_jepa.ctc_utils import evaluate
from semg_jepa.data_utils import combine_fixed_length, decollate_tensor
from semg_jepa.tokenizers import build_tokenizer
from semg_jepa.wandb_utils import finish_wandb, init_wandb, wandb_log

from uml.audio_dataset import LibriSpeechCharDataset
from uml.text_dataset import TextCorpusDataset
from uml.model import UMLModel, ctc_loss_from_logits


class _EMGInferenceWrapper(nn.Module):
    """Adapts ``UMLModel`` for ``ctc_utils.evaluate`` (which calls ``model(raw)``
    and runs log_softmax itself). The wrapper returns raw EMG-branch logits.
    """

    def __init__(self, uml: UMLModel):
        super().__init__()
        self.uml = uml

    def forward(self, raw_emg: torch.Tensor) -> torch.Tensor:
        return self.uml.forward_emg(raw_emg)


def _load_unit_targets(units_dir: str, split: str) -> dict:
    """Load a precomputed HuBERT-unit cache: {sample_id -> list[int]}."""
    path = os.path.join(units_dir, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HuBERT unit cache not found: {path}. Run scripts/precompute_hubert_units.py first."
        )
    payload = torch.load(path, map_location="cpu")
    return payload["units"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to YAML config; CLI flags override its values.")
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data")
    p.add_argument("--librispeech-cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data/libri_cache")
    p.add_argument("--librispeech-split", default="train-clean-100")
    p.add_argument("--output-directory", default="/scratch/cr4206/sEMG-unpaired-text/runs/uml")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-batch-len", type=int, default=88000)
    p.add_argument("--fixed-raw-len", type=int, default=1600)
    p.add_argument("--audio-batch-size", type=int, default=8)
    p.add_argument("--audio-num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--learning-rate-warmup", type=int, default=1000)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Number of training steps to accumulate before optim.step. "
                        "In 'alternate' mode a step = (EMG+audio) pair; in 'both' "
                        "mode a step = a single batch from one modality.")
    p.add_argument("--lr-decay-milestones", type=int, nargs="+", default=[125, 150, 175])
    p.add_argument("--lr-decay-gamma", type=float, default=0.5)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--lambda-uml", type=float, default=1.0,
                   help="Weight on the second-branch (audio/text) CTC loss.")
    p.add_argument("--share-ctc-head", action="store_true",
                   help="If set, EMG and the second branch share the same CTCHead.")
    # Phase-2 §C: the second branch can be unpaired TEXT instead of audio.
    p.add_argument("--second-branch", choices=["audio", "text"], default="audio",
                   help="Auxiliary modality sharing the transformer (default audio).")
    # Frontend-symmetry knob: how much of wav2vec2 runs before the SHARED transformer.
    # full = whole wav2vec2 incl. its 12 self-attention layers (last_hidden_state); audio
    # reaches the shared transformer already contextualized while EMG (conv-only) has not.
    # conv = wav2vec2 conv feature extractor ONLY (no attention), mirroring the conv-only EMG
    # frontend so the shared transformer contextualizes both modalities from local features.
    p.add_argument("--audio-frontend", choices=["full", "conv"], default="full",
                   help="Audio frontend depth before the shared transformer (audio branch).")
    # Multi-auxiliary UML (paper Thm 1: Fisher info compounds across modalities). When set to
    # more than one, ALL listed auxiliaries share the transformer, each with its own CTC head;
    # per step we do 1 EMG + 1 batch of EACH aux (weighted by --lambda-{audio,text}). If unset,
    # falls back to the single --second-branch (weighted by --lambda-uml).
    p.add_argument("--aux-branches", nargs="+", choices=["audio", "text"], default=None,
                   help="Auxiliary modalities to combine (e.g. 'audio text'). Overrides --second-branch.")
    p.add_argument("--lambda-audio", type=float, default=None,
                   help="Audio-branch loss weight in multi-aux mode (default: --lambda-uml).")
    p.add_argument("--lambda-text", type=float, default=None,
                   help="Text-branch loss weight in multi-aux mode (default: --lambda-uml).")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed torch/np/random for paired multi-aux-vs-baseline comparisons.")
    p.add_argument("--text-source", choices=["libri", "gaddy"], default="libri",
                   help="Unpaired-text corpus when --second-branch text.")
    p.add_argument("--text-frontend", choices=["embed", "frozen"], default="embed",
                   help="embed: trainable char embedding (corruption+upsample in collate). "
                        "frozen: frozen CANINE char encoder + trainable Linear (mirrors the "
                        "audio frontend); features are upsampled in the frontend, so the "
                        "collate does NOT pre-upsample.")
    p.add_argument("--text-frozen-arch", choices=["canine", "byt5"], default="canine",
                   help="Frozen text-encoder arch for --text-frontend frozen. canine (char, ~120M) "
                        "or byt5 (byte-level, ~300M+; paper lesson: a stronger frozen text encoder "
                        "gives bigger UML gains). Both are tokenization-free (1 char = 1 position).")
    p.add_argument("--text-frozen-model", default=None,
                   help="HF id of the frozen text encoder (default: per-arch — canine-s / byt5-small).")
    p.add_argument("--text-cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data/text_cache")
    p.add_argument("--text-batch-size", type=int, default=8)
    p.add_argument("--text-num-workers", type=int, default=2)
    p.add_argument("--text-p-mask", type=float, default=0.15)
    p.add_argument("--text-p-sub", type=float, default=0.10)
    p.add_argument("--text-p-del", type=float, default=0.05)
    p.add_argument("--text-mask-span-mean", type=float, default=3.0)
    p.add_argument("--text-upsample", type=int, default=3)
    p.add_argument("--text-jitter", type=int, default=1)
    # EMG token unit + temporal resolution (matches train_baseline.py; char @ 16x is the
    # decided phase-2 config). --downsample-factor takes precedence over --conv-strides.
    p.add_argument("--unit", choices=["char", "subword", "phoneme", "hubert"], default="char")
    p.add_argument("--subword-model", default=None)
    p.add_argument("--phoneme-dict", default=None)
    # HuBERT-unit target (audio-derived, precomputed via scripts/precompute_hubert_units.py).
    p.add_argument("--hubert-k", type=int, default=100, help="k-means K for --unit hubert.")
    p.add_argument("--hubert-units-dir", default="/scratch/cr4206/sEMG-unpaired-text/data/hubert_units",
                   help="Dir with {train,dev}.pt unit caches for --unit hubert.")
    p.add_argument("--audio-units-dir", default=None,
                   help="When --unit hubert and --second-branch audio: dir with "
                        "<librispeech_split>.pt audio-unit targets (precompute_audio_hubert_units.py). "
                        "Makes the audio branch predict the SAME km100 units as the EMG branch "
                        "(the proper UML setup); without it the audio branch would predict chars.")
    p.add_argument("--voiced-eval-ids", default=None,
                   help="Path to a newline-separated list of VOICED sample_ids to HOLD OUT of "
                        "training and evaluate as a second dev set ('vdev'). Separates genuine "
                        "silent-speech difficulty from (ex-)target degeneracy: vdev is voiced "
                        "EMG -> its own (clean) units, the main dev is silent EMG. hubert only.")
    p.add_argument("--conv-strides", type=int, nargs="+", default=[2, 2, 2])
    p.add_argument("--downsample-factor", type=int, default=None)
    # Option 1 (frontend symmetry): EMG-private pre-transformer depth. >0 gives the EMG branch
    # its own attention stack BEFORE the shared transformer, so EMG reaches it already
    # contextualized (like audio via wav2vec2's full 12-layer transformer). The aux branches
    # bypass these layers. MUST match --num-private-layers at finetune time.
    p.add_argument("--num-private-layers", type=int, default=0,
                   help="EMG-private transformer layers before the shared transformer (Option 1).")
    p.add_argument("--no-private-gate", action="store_true",
                   help="Disable the zero-init LayerScale gate around the private pre-transformer "
                        "(default: gated). num_private_layers>=6 WITHOUT the gate collapses to the "
                        "CTC all-blank attractor within ~10 epochs (confirmed 2026-07, independent "
                        "of clip_grad_norm) — only disable for that specific ablation.")
    p.add_argument("--epoch-mode", choices=["alternate", "both"], default="alternate",
                   help="alternate (default): each step does 1 EMG batch + 1 audio "
                        "batch (paired backward, one optim.step). Epoch ends after "
                        "one EMG pass; audio sees a fraction of its dataset per epoch. "
                        "both: each step does 1 batch from ONE modality (single "
                        "backward). The schedule is the union of all EMG and all "
                        "audio batches, shuffled — every batch from both datasets is "
                        "processed exactly once per epoch (audio dominates because it "
                        "has more batches; tune lambda_uml to rebalance).")
    p.add_argument("--model-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--start-training-from", default=None)
    p.add_argument("--eval-method", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="UMLforVideoLab")
    p.add_argument("--wandb-project", default="JEPAforsEMG")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--emg-channel-dropout", type=float, default=0.0,
                   help="Per-channel zero-out probability applied to EMG batches.")
    p.add_argument("--emg-time-mask-prob", type=float, default=0.0,
                   help="Per-sample probability of applying a single time mask.")
    p.add_argument("--emg-time-mask-max", type=int, default=0,
                   help="Maximum width (in raw samples) of the EMG time mask.")
    p.add_argument("--emg-noise-std", type=float, default=0.0,
                   help="Std of additive Gaussian noise on EMG batches (strong-aug for low-label).")
    p.add_argument("--emg-amp-scale", type=float, default=0.0,
                   help="Per-sample amplitude jitter magnitude on EMG batches (0=off).")
    # ---- Low-label / few-shot UML sweep ----
    # Subsample the LABELLED EMG train set to a fraction (dev + audio stay full). The subset is a
    # deterministic random draw keyed by --label-subset-seed, so control (lam=0) and UML (lam>0)
    # at the same seed see the SAME labelled utterances.
    p.add_argument("--label-fraction", type=float, default=1.0,
                   help="Fraction of labelled EMG train utts to keep (1.0 = all).")
    p.add_argument("--label-subset-seed", type=int, default=0,
                   help="Seed selecting WHICH utts are kept when --label-fraction < 1.")
    # Step-based training (few-shot): when set, train for a FIXED number of optim steps regardless
    # of label fraction (the small EMG set cycles while audio progresses over LibriSpeech), with
    # frequent dev eval + early stopping. Absent -> the epoch-based loop (unchanged).
    p.add_argument("--max-steps", type=int, default=None,
                   help="Total optim steps (step-based few-shot mode). None -> epoch loop.")
    p.add_argument("--eval-every", type=int, default=500,
                   help="Step-based mode: eval dev + checkpoint every N steps.")
    p.add_argument("--patience", type=int, default=25,
                   help="Step-based mode: stop after this many evals with no dev-WER improvement.")
    return parse_with_config(p)


def _save_emg_branch(model: UMLModel, path: str) -> None:
    """Save just the EMG-branch weights (encoder + EMG CTC head) so they map
    1-to-1 onto a baseline ``BaselineCTCModel`` for fine-tuning.
    """
    state = {}
    for k, v in model.emg_encoder.state_dict().items():
        state[f"encoder.{k}"] = v
    for k, v in model.emg_ctc_head.state_dict().items():
        state[f"ctc_head.{k}"] = v
    torch.save(state, path)


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def train(args):
    run = init_wandb(args, default_name_prefix="uml")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        logging.info("seed=%d (torch/np/random)", args.seed)

    # EMG token resolution: --downsample-factor takes precedence over --conv-strides.
    if getattr(args, "downsample_factor", None):
        conv_strides = factor_to_strides(args.downsample_factor)
    else:
        conv_strides = tuple(args.conv_strides)
    factor = math.prod(conv_strides)
    assert args.fixed_raw_len % factor == 0, (
        f"fixed_raw_len={args.fixed_raw_len} must be divisible by downsample factor {factor}")

    tokenizer = build_tokenizer(
        args.unit, subword_model=args.subword_model, phoneme_dict=args.phoneme_dict,
        hubert_k=args.hubert_k,
    )
    n_chars = tokenizer.vocab_size
    # HuBERT units are audio-derived, so CTC targets are read from a precomputed cache
    # (keyed by sample_id) instead of being re-encoded from the transcript.
    if tokenizer.targets_from_audio:
        train_units = _load_unit_targets(args.hubert_units_dir, "train")
        dev_units = _load_unit_targets(args.hubert_units_dir, "dev")
        logging.info("loaded HuBERT units: train=%d dev=%d (dev_wer below is a UNIT error rate)",
                     len(train_units), len(dev_units))
    else:
        train_units = dev_units = None

    # Optional held-out VOICED eval set (§3 diagnostic): hold these voiced sample_ids out of
    # train and evaluate them separately ("vdev"), to tell silent difficulty from target noise.
    voiced_eval_ids = None
    if args.voiced_eval_ids:
        with open(args.voiced_eval_ids) as fh:
            voiced_eval_ids = {ln.strip() for ln in fh if ln.strip()}
        logging.info("voiced-eval: holding out %d voiced sample_ids from train", len(voiced_eval_ids))

    trainset = CachedRawEMGDataset(args.cache_dir, "train", tokenizer=tokenizer,
                                   downsample_factor=factor, unit_targets=train_units,
                                   exclude_ids=voiced_eval_ids)
    # Low-label sweep: keep only a (seeded) random fraction of the labelled EMG train set.
    if args.label_fraction < 1.0:
        n_full = len(trainset)
        trainset = trainset.subset(args.label_fraction, seed=args.label_subset_seed)
        logging.info("low-label: kept %d/%d train utts (fraction=%.3f, subset_seed=%d)",
                     len(trainset), n_full, args.label_fraction, args.label_subset_seed)
    devset = CachedRawEMGDataset(args.cache_dir, "dev", tokenizer=tokenizer,
                                 downsample_factor=factor, unit_targets=dev_units)
    vdevset = None
    if voiced_eval_ids:
        vdevset = CachedRawEMGDataset(args.cache_dir, "train", tokenizer=tokenizer,
                                      downsample_factor=factor, unit_targets=train_units,
                                      include_ids=voiced_eval_ids)
        logging.info("voiced-eval: vdev=%d samples (voiced EMG -> own units)", len(vdevset))

    # Units without a word LM (phoneme/hubert) must score greedy: beam glues the labels
    # with no spaces and mis-renders them (cf. train_baseline.py).
    eval_method = args.eval_method if tokenizer.supports_word_lm else "greedy"
    if eval_method != args.eval_method:
        logging.info("eval_method forced to 'greedy' (unit=%s has no word LM)", args.unit)

    # Auxiliary modalities (multi-aux = paper Thm 1: Fisher info compounds). --aux-branches
    # overrides --second-branch; each aux keeps its OWN dataset + loss weight and shares the
    # transformer. A pure baseline is any subset with its lambda=0 (branch present but inert).
    aux_names = []
    for b in (args.aux_branches or [args.second_branch]):
        if b not in aux_names:
            aux_names.append(b)
    lam_of = {
        "audio": args.lambda_audio if args.lambda_audio is not None else args.lambda_uml,
        "text": args.lambda_text if args.lambda_text is not None else args.lambda_uml,
    }

    def _build_aux(name):
        if name == "text":
            # frozen frontend upsamples features itself -> the collate must NOT pre-upsample.
            ds_upsample = 1 if args.text_frontend == "frozen" else args.text_upsample
            ds_jitter = 0 if args.text_frontend == "frozen" else args.text_jitter
            ds = TextCorpusDataset(
                args.text_cache_dir, args.text_source,
                vocab_size=tokenizer.vocab_size,
                p_mask=args.text_p_mask, p_sub=args.text_p_sub, p_del=args.text_p_del,
                mask_span_mean=args.text_mask_span_mean,
                upsample=ds_upsample, jitter=ds_jitter,
            )
            return ds, f"text:{args.text_source}:{args.text_frontend}"
        audio_units_path = None
        if args.audio_units_dir:
            audio_units_path = os.path.join(args.audio_units_dir, f"{args.librispeech_split}.pt")
            if not os.path.exists(audio_units_path):
                raise FileNotFoundError(
                    f"audio unit targets not found: {audio_units_path} "
                    f"(run scripts/precompute_audio_hubert_units.py for split '{args.librispeech_split}')")
        ds = LibriSpeechCharDataset(
            args.librispeech_cache_dir, args.librispeech_split,
            unit_targets_path=audio_units_path,
        )
        return ds, (f"audio:{args.librispeech_split}:{'units' if audio_units_path else 'char'}"
                    f":frontend={args.audio_frontend}")

    aux_specs = []
    for name in aux_names:
        ds, desc = _build_aux(name)
        aux_specs.append({"name": name, "dataset": ds, "desc": desc, "lam": lam_of[name]})
    logging.info(
        "unit=%s vocab=%d conv_strides=%s factor=%d private_layers=%d | emg_train=%d emg_dev=%d | aux=[%s]",
        args.unit, tokenizer.vocab_size, conv_strides, factor, args.num_private_layers,
        len(trainset), len(devset),
        ", ".join(f"{a['name']}({a['desc']},n={len(a['dataset'])},lam={a['lam']})" for a in aux_specs),
    )

    model = UMLModel(
        vocab_size=n_chars,
        model_size=args.model_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        share_ctc_head=bool(args.share_ctc_head),
        conv_strides=conv_strides,
        num_private_layers=args.num_private_layers,
        private_gate=not args.no_private_gate,
        aux_branches=tuple(aux_names),
        audio_frontend_mode=args.audio_frontend,
        text_frontend=args.text_frontend,
        text_frozen_model=args.text_frozen_model,
        text_frozen_arch=args.text_frozen_arch,
        text_upsample=args.text_upsample,
    ).to(device)

    if args.start_training_from:
        sd = torch.load(args.start_training_from, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logging.info("loaded init from %s (missing=%d unexpected=%d)",
                     args.start_training_from, len(missing), len(unexpected))

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("model parameters: total=%d trainable=%d", n_total, n_train)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.l2)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=args.lr_decay_milestones, gamma=args.lr_decay_gamma,
    )

    def set_lr(new_lr):
        for pg in optim.param_groups:
            pg["lr"] = new_lr

    def schedule_lr(iteration):
        iteration += 1
        if iteration <= args.learning_rate_warmup:
            set_lr(iteration * args.learning_rate / args.learning_rate_warmup)

    for a in aux_specs:
        if a["name"] == "text":
            a["loader"] = DataLoader(
                a["dataset"], batch_size=args.text_batch_size, shuffle=True,
                num_workers=args.text_num_workers, pin_memory=(device == "cuda"),
                persistent_workers=(args.text_num_workers > 0),
                collate_fn=a["dataset"].collate_fn, drop_last=True,
            )
        else:
            a["loader"] = DataLoader(
                a["dataset"], batch_size=args.audio_batch_size, shuffle=True,
                num_workers=args.audio_num_workers, pin_memory=(device == "cuda"),
                persistent_workers=(args.audio_num_workers > 0),
                collate_fn=LibriSpeechCharDataset.collate_fn, drop_last=True,
            )

    os.makedirs(args.output_directory, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M")
    best_wer = float("inf")
    global_step = 0
    optim.zero_grad()

    eval_wrapper = _EMGInferenceWrapper(model)

    emg_augment = RawEMGAugment(
        channel_dropout=args.emg_channel_dropout,
        time_mask_prob=args.emg_time_mask_prob,
        time_mask_max=args.emg_time_mask_max,
        noise_std=args.emg_noise_std,
        amp_scale=args.emg_amp_scale,
    )
    emg_augment_enabled = (
        args.emg_channel_dropout > 0
        or (args.emg_time_mask_prob > 0 and args.emg_time_mask_max > 0)
        or args.emg_noise_std > 0
        or args.emg_amp_scale > 0
    )
    if emg_augment_enabled:
        logging.info(
            "EMG augment: channel_dropout=%.3f time_mask_prob=%.3f time_mask_max=%d "
            "noise_std=%.3f amp_scale=%.3f",
            args.emg_channel_dropout, args.emg_time_mask_prob, args.emg_time_mask_max,
            args.emg_noise_std, args.emg_amp_scale,
        )

    def _make_emg_loader():
        batches_local = build_batches(trainset, args.max_batch_len)
        loader_local = DataLoader(
            trainset,
            pin_memory=(device == "cuda"),
            num_workers=0,
            collate_fn=CachedRawEMGDataset.collate_raw,
            batch_sampler=batches_local,
        )
        return loader_local, len(batches_local)

    def _emg_step(example, t):
        t1 = time.perf_counter()
        raw = combine_fixed_length(example["raw_emg"], args.fixed_raw_len).to(device)
        if emg_augment_enabled and model.training:
            raw = emg_augment(raw)
        emg_logits = model.forward_emg(raw)                              # (n_blocks, T_block, V+1)
        emg_logp = F.log_softmax(emg_logits.float(), dim=-1)
        emg_logp = nn.utils.rnn.pad_sequence(
            decollate_tensor(emg_logp, example["lengths"]), batch_first=False,
        )                                                                 # (T_max, B, V+1)
        targets = nn.utils.rnn.pad_sequence(example["text_int"], batch_first=True).to(device)
        loss_emg = F.ctc_loss(
            emg_logp, targets, example["lengths"], example["text_int_lengths"],
            blank=n_chars, zero_infinity=True,
        )
        _sync(device)
        t["fwd_emg"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        loss_emg.backward()
        _sync(device)
        t["bwd_emg"] += time.perf_counter() - t2
        return loss_emg.item()

    def _audio_step(audio_batch, t, lam):
        t1 = time.perf_counter()
        wav = audio_batch["audio_features"].to(device)
        audio_lengths = audio_batch["audio_lengths"].to(device)
        a_targets = audio_batch["text_int"].to(device)
        a_target_lengths = audio_batch["text_int_lengths"].to(device)
        audio_logits, audio_input_lengths = model.forward_audio(wav, audio_lengths)
        audio_input_lengths = audio_input_lengths.to(device)
        loss_audio = ctc_loss_from_logits(
            audio_logits, a_targets, audio_input_lengths, a_target_lengths,
            blank=model.blank_id,
        )
        _sync(device)
        t["fwd_audio"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        (lam * loss_audio).backward()
        _sync(device)
        t["bwd_audio"] += time.perf_counter() - t2
        return loss_audio.item()

    def _text_step(text_batch, t, lam):
        t1 = time.perf_counter()
        tok_ids = text_batch["text_input"].to(device)
        tok_lens = text_batch["text_input_lengths"].to(device)
        t_targets = text_batch["text_int"].to(device)
        t_target_lengths = text_batch["text_int_lengths"].to(device)
        text_logits, text_input_lengths = model.forward_text(tok_ids, tok_lens)
        loss_text = ctc_loss_from_logits(
            text_logits, t_targets, text_input_lengths.to(device), t_target_lengths,
            blank=model.blank_id,
        )
        _sync(device)
        t["fwd_audio"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        (lam * loss_text).backward()
        _sync(device)
        t["bwd_audio"] += time.perf_counter() - t2
        return loss_text.item()

    step_of = {"audio": _audio_step, "text": _text_step}

    # ------------------------------------------------------------------
    # Step-based few-shot training (low-label sweep): a FIXED number of optim
    # steps regardless of label fraction. The (small) EMG set cycles — reshuffled
    # each pass — while the aux loader progresses over LibriSpeech, so the auxiliary
    # is equally trained at every fraction. Frequent dev eval + early stopping;
    # a lambda=0 control skips the aux branch entirely (EMG-only fast path).
    # ------------------------------------------------------------------
    if args.max_steps:
        model.train()
        active_aux = [a for a in aux_specs if a["lam"] > 0]
        if not active_aux:
            logging.info("step-based: all aux lambda=0 -> CONTROL (aux branch skipped, EMG-only)")
        milestones_steps = [int(f * args.max_steps) for f in (0.625, 0.75, 0.875)]
        logging.info("step-based: max_steps=%d eval_every=%d patience=%d lr_milestones(steps)=%s",
                     args.max_steps, args.eval_every, args.patience, milestones_steps)

        def step_lr(step):
            if step < args.learning_rate_warmup:
                set_lr((step + 1) * args.learning_rate / max(1, args.learning_rate_warmup))
            else:
                n_dec = sum(1 for m in milestones_steps if step >= m)
                set_lr(args.learning_rate * (args.lr_decay_gamma ** n_dec))

        def emg_batch_gen():
            while True:                                   # reshuffles each pass over the subset
                loader_local, _ = _make_emg_loader()
                for b in loader_local:
                    yield b
        emg_gen = emg_batch_gen()
        aux_iters = {a["name"]: iter(a["loader"]) for a in active_aux}

        def next_aux(a):
            try:
                return next(aux_iters[a["name"]])
            except StopIteration:
                aux_iters[a["name"]] = iter(a["loader"])  # linear pass over LibriSpeech, then reshuffle
                return next(aux_iters[a["name"]])

        t = {k: 0.0 for k in ("data_emg", "data_audio", "fwd_emg", "bwd_emg",
                              "fwd_audio", "bwd_audio", "opt")}
        emg_losses, aux_losses = [], {a["name"]: [] for a in active_aux}
        evals_no_improve = 0
        t_train0 = time.perf_counter()
        for step in range(args.max_steps):
            step_lr(step)
            emg_losses.append(_emg_step(next(emg_gen), t))
            for a in active_aux:
                aux_losses[a["name"]].append(step_of[a["name"]](next_aux(a), t, a["lam"]))
            if (step + 1) % args.grad_accum_steps == 0:
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optim.step()
                optim.zero_grad()

            if (step + 1) % args.eval_every == 0 or (step + 1) == args.max_steps:
                model.eval()
                wer, cer = evaluate(eval_wrapper, devset, device, method=eval_method)
                model.train()
                train_emg = float(np.mean(emg_losses)) if emg_losses else 0.0
                train_aux = {n: (float(np.mean(v)) if v else 0.0) for n, v in aux_losses.items()}
                cur_lr = optim.param_groups[0]["lr"]
                improved = wer < best_wer
                logging.info(
                    "step=%d/%d lr=%.2e emg_loss=%.4f %s dev_wer=%.3f dev_cer=%.3f best=%.3f "
                    "no_improve=%d t=%.0fs",
                    step + 1, args.max_steps, cur_lr, train_emg,
                    " ".join(f"{a['name']}_loss={train_aux[a['name']]:.4f}(lam{a['lam']:.2f})"
                             for a in active_aux) or "(control)",
                    wer, cer, min(best_wer, wer), evals_no_improve, time.perf_counter() - t_train0,
                )
                wandb_log(run, {
                    "eval/wer": wer, "eval/cer": cer, "train/emg_loss": train_emg,
                    **{f"train/{a['name']}_loss": train_aux[a["name"]] for a in active_aux},
                    "train/lr": cur_lr, "step": step + 1,
                })
                torch.save(model.state_dict(), os.path.join(args.output_directory, "last.pt"))
                _save_emg_branch(model, os.path.join(args.output_directory, "last_emg_branch.pt"))
                if improved:
                    best_wer = wer
                    evals_no_improve = 0
                    torch.save(model.state_dict(), os.path.join(args.output_directory, "best.pt"))
                    _save_emg_branch(model, os.path.join(args.output_directory, "best_emg_branch.pt"))
                else:
                    evals_no_improve += 1
                    if evals_no_improve >= args.patience:
                        logging.info("early stop @ step %d (no dev improvement for %d evals; best=%.3f)",
                                     step + 1, args.patience, best_wer)
                        break
                emg_losses, aux_losses = [], {a["name"]: [] for a in active_aux}

        torch.save(model.emg_encoder.state_dict(),
                   os.path.join(args.output_directory, "pretrained_encoder.pt"))
        finish_wandb(run)
        return

    for epoch in range(args.epochs):
        model.train()

        emg_loader, n_emg_batches = _make_emg_loader()
        emg_iter = iter(emg_loader)
        aux_iters = {a["name"]: iter(a["loader"]) for a in aux_specs}

        emg_losses = []
        aux_losses = {a["name"]: [] for a in aux_specs}
        t = {"data_emg": 0.0, "data_audio": 0.0,
             "fwd_emg": 0.0, "bwd_emg": 0.0,
             "fwd_audio": 0.0, "bwd_audio": 0.0,
             "opt": 0.0}
        epoch_start = time.perf_counter()
        n_steps = 0

        def _run_aux(a):
            td0 = time.perf_counter()
            try:
                batch = next(aux_iters[a["name"]])
            except StopIteration:
                aux_iters[a["name"]] = iter(a["loader"])  # reshuffles
                batch = next(aux_iters[a["name"]])
            t["data_audio"] += time.perf_counter() - td0
            aux_losses[a["name"]].append(step_of[a["name"]](batch, t, a["lam"]))

        def _maybe_step():
            t_opt0 = time.perf_counter()
            if (global_step + 1) % args.grad_accum_steps == 0:
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optim.step()
                optim.zero_grad()
                _sync(device)
            t["opt"] += time.perf_counter() - t_opt0

        if args.epoch_mode == "alternate":
            # 1 EMG + 1 batch of EACH aux per step, all backward, single optim.step.
            t0 = time.perf_counter()
            for _ in range(n_emg_batches):
                schedule_lr(global_step)
                emg_batch = next(emg_iter)
                t["data_emg"] += time.perf_counter() - t0
                emg_losses.append(_emg_step(emg_batch, t))
                for a in aux_specs:
                    _run_aux(a)
                _maybe_step()
                global_step += 1
                n_steps += 1
                t0 = time.perf_counter()
        else:  # "both": one modality per step, all batches seen once (single aux only).
            if len(aux_specs) != 1:
                raise ValueError("epoch_mode 'both' supports exactly one aux branch")
            a0 = aux_specs[0]
            sched = ["emg"] * n_emg_batches + ["aux"] * len(a0["loader"])
            random.shuffle(sched)
            t0 = time.perf_counter()
            for modality in sched:
                schedule_lr(global_step)
                if modality == "emg":
                    emg_batch = next(emg_iter)
                    t["data_emg"] += time.perf_counter() - t0
                    emg_losses.append(_emg_step(emg_batch, t))
                else:
                    _run_aux(a0)
                _maybe_step()
                global_step += 1
                n_steps += 1
                t0 = time.perf_counter()

        train_emg = float(np.mean(emg_losses)) if emg_losses else 0.0
        train_aux = {n: (float(np.mean(v)) if v else 0.0) for n, v in aux_losses.items()}

        eval_start = time.perf_counter()
        wer, cer = evaluate(eval_wrapper, devset, device, method=eval_method)
        vdev_wer = vdev_cer = None
        if vdevset is not None:
            vdev_wer, vdev_cer = evaluate(eval_wrapper, vdevset, device, method=eval_method)
        t_eval = time.perf_counter() - eval_start

        lr_sched.step()
        t_epoch = time.perf_counter() - epoch_start
        cur_lr = optim.param_groups[0]["lr"]

        vdev_str = (f" vdev_wer={vdev_wer:.3f} vdev_cer={vdev_cer:.3f}"
                    if vdev_wer is not None else "")
        aux_str = " ".join(f"{a['name']}_loss={train_aux[a['name']]:.4f}(lam{a['lam']:.2f})"
                           for a in aux_specs)
        logging.info(
            "epoch=%d/%d steps=%d lr=%.2e emg_loss=%.4f %s "
            "dev_wer=%.3f dev_cer=%.3f%s t_data_emg=%.1fs t_data_aux=%.1fs "
            "t_fwd_emg=%.1fs t_bwd_emg=%.1fs t_fwd_aux=%.1fs t_bwd_aux=%.1fs "
            "t_opt=%.1fs t_eval=%.1fs t_epoch=%.1fs",
            epoch + 1, args.epochs, n_steps, cur_lr,
            train_emg, aux_str, wer, cer, vdev_str,
            t["data_emg"], t["data_audio"],
            t["fwd_emg"], t["bwd_emg"], t["fwd_audio"], t["bwd_audio"],
            t["opt"], t_eval, t_epoch,
        )
        wandb_log(run, {
            "eval/wer": wer, "eval/cer": cer,
            **({"eval/vdev_wer": vdev_wer, "eval/vdev_cer": vdev_cer} if vdev_wer is not None else {}),
            "train/emg_loss": train_emg,
            **{f"train/{a['name']}_loss": train_aux[a["name"]] for a in aux_specs},
            "train/total_loss": train_emg + sum(a["lam"] * train_aux[a["name"]] for a in aux_specs),
            "train/lr": cur_lr,
            **{f"uml/lambda_{a['name']}": a["lam"] for a in aux_specs},
            "time/data_emg": t["data_emg"], "time/data_aux": t["data_audio"],
            "time/fwd_emg": t["fwd_emg"], "time/bwd_emg": t["bwd_emg"],
            "time/fwd_aux": t["fwd_audio"], "time/bwd_aux": t["bwd_audio"],
            "time/opt": t["opt"],
            "time/eval": t_eval, "time/epoch": t_epoch,
            "epoch": epoch + 1,
        })

        # Full UML state (resume + second branch retained)
        torch.save(model.state_dict(), os.path.join(args.output_directory, "last.pt"))
        torch.save(model.state_dict(), os.path.join(args.output_directory, f"last_{run_ts}.pt"))
        # EMG-only weights, ready for finetune_from_jepa-style loading.
        _save_emg_branch(model, os.path.join(args.output_directory, "last_emg_branch.pt"))

        if wer < best_wer:
            best_wer = wer
            torch.save(model.state_dict(), os.path.join(args.output_directory, "best.pt"))
            torch.save(model.state_dict(), os.path.join(args.output_directory, f"best_{run_ts}.pt"))
            _save_emg_branch(model, os.path.join(args.output_directory, "best_emg_branch.pt"))

    # Final convenience copy: encoder-only weights for finetune_from_jepa.py.
    torch.save(
        model.emg_encoder.state_dict(),
        os.path.join(args.output_directory, "pretrained_encoder.pt"),
    )

    finish_wandb(run)


if __name__ == "__main__":
    setup_stdout_logging()
    train(parse_args())
