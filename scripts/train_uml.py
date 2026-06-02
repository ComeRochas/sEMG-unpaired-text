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
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from semg_jepa.architecture import GaddyRawEMGEncoder, CTCHead
from semg_jepa.augmentations import RawEMGAugment
from semg_jepa.cached_dataset import CachedRawEMGDataset, build_batches
from semg_jepa.config_utils import parse_with_config, setup_stdout_logging
from semg_jepa.ctc_utils import evaluate
from semg_jepa.data_utils import combine_fixed_length, decollate_tensor
from semg_jepa.wandb_utils import finish_wandb, init_wandb, wandb_log

from uml.audio_dataset import LibriSpeechCharDataset
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
                   help="Weight on the audio CTC loss.")
    p.add_argument("--share-ctc-head", action="store_true",
                   help="If set, EMG and audio branches share the same CTCHead.")
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
    trainset = CachedRawEMGDataset(args.cache_dir, "train")
    devset = CachedRawEMGDataset(args.cache_dir, "dev")
    n_chars = len(devset.text_transform.chars)

    audio_train = LibriSpeechCharDataset(args.librispeech_cache_dir, args.librispeech_split)
    logging.info(
        "data emg_train=%d emg_dev=%d audio_train=%d (split=%s)",
        len(trainset), len(devset), len(audio_train), args.librispeech_split,
    )

    model = UMLModel(
        vocab_size=n_chars,
        model_size=args.model_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        share_ctc_head=bool(args.share_ctc_head),
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

    audio_loader = DataLoader(
        audio_train,
        batch_size=args.audio_batch_size,
        shuffle=True,
        num_workers=args.audio_num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.audio_num_workers > 0),
        collate_fn=LibriSpeechCharDataset.collate_fn,
        drop_last=True,
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
    )
    emg_augment_enabled = (
        args.emg_channel_dropout > 0
        or (args.emg_time_mask_prob > 0 and args.emg_time_mask_max > 0)
    )
    if emg_augment_enabled:
        logging.info(
            "EMG augment: channel_dropout=%.3f time_mask_prob=%.3f time_mask_max=%d",
            args.emg_channel_dropout, args.emg_time_mask_prob, args.emg_time_mask_max,
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

    def _audio_step(audio_batch, t):
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
        (args.lambda_uml * loss_audio).backward()
        _sync(device)
        t["bwd_audio"] += time.perf_counter() - t2
        return loss_audio.item()

    for epoch in range(args.epochs):
        model.train()

        emg_loader, n_emg_batches = _make_emg_loader()
        n_audio_batches = len(audio_loader)
        emg_iter = iter(emg_loader)
        audio_iter = iter(audio_loader)

        emg_losses, audio_losses = [], []
        t = {"data_emg": 0.0, "data_audio": 0.0,
             "fwd_emg": 0.0, "bwd_emg": 0.0,
             "fwd_audio": 0.0, "bwd_audio": 0.0,
             "opt": 0.0}
        epoch_start = time.perf_counter()
        n_steps = 0

        if args.epoch_mode == "alternate":
            # 1 EMG + 1 audio per step, paired backward, single optim.step.
            t0 = time.perf_counter()
            for _ in range(n_emg_batches):
                schedule_lr(global_step)

                emg_batch = next(emg_iter)
                t["data_emg"] += time.perf_counter() - t0
                emg_losses.append(_emg_step(emg_batch, t))

                t_audio_data0 = time.perf_counter()
                try:
                    audio_batch = next(audio_iter)
                except StopIteration:
                    audio_iter = iter(audio_loader)  # reshuffles
                    audio_batch = next(audio_iter)
                t["data_audio"] += time.perf_counter() - t_audio_data0
                audio_losses.append(_audio_step(audio_batch, t))

                t_opt0 = time.perf_counter()
                if (global_step + 1) % args.grad_accum_steps == 0:
                    if args.clip_grad_norm and args.clip_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optim.step()
                    optim.zero_grad()
                    _sync(device)
                t["opt"] += time.perf_counter() - t_opt0

                global_step += 1
                n_steps += 1
                t0 = time.perf_counter()
        else:  # "both": each step is one modality, all batches seen exactly once.
            sched = ["emg"] * n_emg_batches + ["audio"] * n_audio_batches
            random.shuffle(sched)
            t0 = time.perf_counter()
            for modality in sched:
                schedule_lr(global_step)
                if modality == "emg":
                    emg_batch = next(emg_iter)
                    t["data_emg"] += time.perf_counter() - t0
                    emg_losses.append(_emg_step(emg_batch, t))
                else:
                    audio_batch = next(audio_iter)
                    t["data_audio"] += time.perf_counter() - t0
                    audio_losses.append(_audio_step(audio_batch, t))

                t_opt0 = time.perf_counter()
                if (global_step + 1) % args.grad_accum_steps == 0:
                    if args.clip_grad_norm and args.clip_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optim.step()
                    optim.zero_grad()
                    _sync(device)
                t["opt"] += time.perf_counter() - t_opt0

                global_step += 1
                n_steps += 1
                t0 = time.perf_counter()

        train_emg = float(np.mean(emg_losses)) if emg_losses else 0.0
        train_audio = float(np.mean(audio_losses)) if audio_losses else 0.0

        eval_start = time.perf_counter()
        wer, cer = evaluate(eval_wrapper, devset, device, method=args.eval_method)
        t_eval = time.perf_counter() - eval_start

        lr_sched.step()
        t_epoch = time.perf_counter() - epoch_start
        cur_lr = optim.param_groups[0]["lr"]

        logging.info(
            "epoch=%d/%d steps=%d lr=%.2e emg_loss=%.4f audio_loss=%.4f lambda=%.3f "
            "dev_wer=%.3f dev_cer=%.3f t_data_emg=%.1fs t_data_audio=%.1fs "
            "t_fwd_emg=%.1fs t_bwd_emg=%.1fs t_fwd_audio=%.1fs t_bwd_audio=%.1fs "
            "t_opt=%.1fs t_eval=%.1fs t_epoch=%.1fs",
            epoch + 1, args.epochs, n_steps, cur_lr,
            train_emg, train_audio, args.lambda_uml, wer, cer,
            t["data_emg"], t["data_audio"],
            t["fwd_emg"], t["bwd_emg"], t["fwd_audio"], t["bwd_audio"],
            t["opt"], t_eval, t_epoch,
        )
        wandb_log(run, {
            "eval/wer": wer, "eval/cer": cer,
            "train/emg_loss": train_emg,
            "train/audio_loss": train_audio,
            "train/total_loss": train_emg + args.lambda_uml * train_audio,
            "train/lr": cur_lr,
            "uml/lambda": args.lambda_uml,
            "time/data_emg": t["data_emg"], "time/data_audio": t["data_audio"],
            "time/fwd_emg": t["fwd_emg"], "time/bwd_emg": t["bwd_emg"],
            "time/fwd_audio": t["fwd_audio"], "time/bwd_audio": t["bwd_audio"],
            "time/opt": t["opt"],
            "time/eval": t_eval, "time/epoch": t_epoch,
            "epoch": epoch + 1,
        })

        # Full UML state (resume + audio branch retained)
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
