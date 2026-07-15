"""Fine-tune the EMG branch of a UML model on EMG-only CTC.

Differs from ``finetune_from_jepa.py`` in one essential way: this script
loads BOTH the EMG encoder weights AND the (already-trained) EMG CTC head
from a UML training checkpoint, instead of starting the head from random.

Expected input checkpoint
-------------------------
Either:

* ``--emg-branch <path>`` to a ``*_emg_branch.pt`` file saved by
  ``scripts/train_uml.py`` (keys ``encoder.*`` + ``ctc_head.*``), OR
* ``--uml-full-ckpt <path>`` to a full ``last.pt`` / ``best.pt`` UMLModel
  checkpoint — the script strips ``emg_encoder.*`` and ``emg_ctc_head.*``
  in memory.

If both are given, ``--emg-branch`` wins.

All other knobs (optim, schedule, eval) mirror the supervised baseline so
the fine-tune dynamics match the baseline's.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from semg_jepa.architecture import BaselineCTCModel, factor_to_strides
from semg_jepa.augmentations import RawEMGAugment
from semg_jepa.cached_dataset import CachedRawEMGDataset, build_batches
from semg_jepa.config_utils import parse_with_config, setup_stdout_logging
from semg_jepa.ctc_utils import evaluate
from semg_jepa.data_utils import combine_fixed_length, decollate_tensor
from semg_jepa.tokenizers import build_tokenizer
from semg_jepa.wandb_utils import finish_wandb, init_wandb, wandb_log


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _load_emg_branch_state(path: str) -> dict:
    """Return a state dict with keys ``encoder.*`` and ``ctc_head.*``."""
    sd = torch.load(path, map_location="cpu")
    keys = list(sd.keys())
    if any(k.startswith("encoder.") for k in keys) and any(k.startswith("ctc_head.") for k in keys):
        return sd
    # full UMLModel checkpoint: rebuild
    remapped: dict = {}
    for k, v in sd.items():
        if k.startswith("emg_encoder."):
            remapped[f"encoder.{k[len('emg_encoder.'):]}"] = v
        elif k.startswith("emg_ctc_head."):
            remapped[f"ctc_head.{k[len('emg_ctc_head.'):]}"] = v
    if not any(k.startswith("encoder.") for k in remapped):
        raise ValueError(
            f"checkpoint {path} contains neither 'encoder.*' nor 'emg_encoder.*' keys"
        )
    if not any(k.startswith("ctc_head.") for k in remapped):
        raise ValueError(
            f"checkpoint {path} has no EMG CTC head (no 'ctc_head.*' / 'emg_ctc_head.*')"
        )
    return remapped


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="Path to YAML config; CLI flags override its values.")
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data")
    p.add_argument(
        "--emg-branch", default=None,
        help="Path to a *_emg_branch.pt file (encoder.* + ctc_head.* keys). "
             "Either --emg-branch or --uml-full-ckpt must be given.",
    )
    p.add_argument(
        "--uml-full-ckpt", default=None,
        help="Path to a full UMLModel state dict; the EMG encoder + CTC head "
             "are extracted in memory. Ignored if --emg-branch is set.",
    )
    p.add_argument(
        "--output-directory",
        default="/scratch/cr4206/sEMG-unpaired-text/runs/finetune_from_uml",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-batch-len", type=int, default=88000)
    p.add_argument("--fixed-raw-len", type=int, default=1600)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--learning-rate-warmup", type=int, default=500)
    p.add_argument("--lr-decay-milestones", type=int, nargs="+", default=[125, 150, 175])
    p.add_argument("--lr-decay-gamma", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--clip-grad-norm", type=float, default=0.0)
    p.add_argument("--model-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    # EMG token unit + temporal resolution — MUST match the UML checkpoint being
    # finetuned (phase-2 default: char @ 16x). --downsample-factor beats --conv-strides.
    p.add_argument("--unit", choices=["char", "subword", "phoneme"], default="char")
    p.add_argument("--subword-model", default=None)
    p.add_argument("--phoneme-dict", default=None)
    p.add_argument("--conv-strides", type=int, nargs="+", default=[2, 2, 2])
    p.add_argument("--downsample-factor", type=int, default=None)
    # MUST match the UML checkpoint's --num-private-layers (Option 1 EMG pre-transformer),
    # else the encoder state dict won't align. --no-private-gate must also match (the gate adds
    # a private_gate_scale parameter to the state dict).
    p.add_argument("--num-private-layers", type=int, default=0)
    p.add_argument("--no-private-gate", action="store_true")
    # Low-label sweep: finetune on the SAME seeded EMG subset as the UML ckpt being resumed.
    p.add_argument("--label-fraction", type=float, default=1.0)
    p.add_argument("--label-subset-seed", type=int, default=0)
    # EMG augmentation (strong-aug regularizer for the few-shot finetune).
    p.add_argument("--emg-channel-dropout", type=float, default=0.0)
    p.add_argument("--emg-time-mask-prob", type=float, default=0.0)
    p.add_argument("--emg-time-mask-max", type=int, default=0)
    p.add_argument("--emg-noise-std", type=float, default=0.0)
    p.add_argument("--emg-amp-scale", type=float, default=0.0)
    p.add_argument("--eval-method", choices=["greedy", "beam"], default="beam")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="UMLforVideoLab")
    p.add_argument("--wandb-project", default="JEPAforsEMG")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--cpu", action="store_true")
    args = parse_with_config(p)
    if not args.emg_branch and not args.uml_full_ckpt:
        p.error("either --emg-branch or --uml-full-ckpt must be given (CLI or YAML)")
    return args


def train(args):
    logging.info("args: %s", vars(args))
    run = init_wandb(args, default_name_prefix="finetune_uml")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    if getattr(args, "downsample_factor", None):
        conv_strides = factor_to_strides(args.downsample_factor)
    else:
        conv_strides = tuple(args.conv_strides)
    factor = math.prod(conv_strides)

    tokenizer = build_tokenizer(
        args.unit, subword_model=args.subword_model, phoneme_dict=args.phoneme_dict,
    )
    n_chars = tokenizer.vocab_size
    trainset = CachedRawEMGDataset(args.cache_dir, "train", tokenizer=tokenizer, downsample_factor=factor)
    if args.label_fraction < 1.0:
        n_full = len(trainset)
        trainset = trainset.subset(args.label_fraction, seed=args.label_subset_seed)
        logging.info("low-label: kept %d/%d train utts (fraction=%.3f, subset_seed=%d)",
                     len(trainset), n_full, args.label_fraction, args.label_subset_seed)
    devset = CachedRawEMGDataset(args.cache_dir, "dev", tokenizer=tokenizer, downsample_factor=factor)
    logging.info("unit=%s vocab=%d conv_strides=%s factor=%d", args.unit, n_chars, conv_strides, factor)

    model = BaselineCTCModel(
        model_size=args.model_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        vocab_size=n_chars,
        conv_strides=conv_strides,
        num_private_layers=args.num_private_layers,
        private_gate=not args.no_private_gate,
    ).to(device)

    ckpt_path = args.emg_branch or args.uml_full_ckpt
    state = _load_emg_branch_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info(
        "loaded encoder+head from %s (missing=%d unexpected=%d)",
        ckpt_path, len(missing), len(unexpected),
    )
    if missing:
        logging.info("  missing keys (first 10): %s", missing[:10])
    if unexpected:
        logging.info("  unexpected keys (first 10): %s", unexpected[:10])

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
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
        logging.info("EMG augment: channel_dropout=%.3f time_mask_prob=%.3f time_mask_max=%d "
                     "noise_std=%.3f amp_scale=%.3f", args.emg_channel_dropout,
                     args.emg_time_mask_prob, args.emg_time_mask_max, args.emg_noise_std,
                     args.emg_amp_scale)

    os.makedirs(args.output_directory, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M")
    best_wer = float("inf")
    global_step = 0
    optim.zero_grad()

    for epoch in range(args.epochs):
        batches = build_batches(trainset, args.max_batch_len)
        dataloader = torch.utils.data.DataLoader(
            trainset,
            pin_memory=(device == "cuda"),
            num_workers=0,
            collate_fn=CachedRawEMGDataset.collate_raw,
            batch_sampler=batches,
        )
        losses = []
        t = {"data": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}
        epoch_start = time.perf_counter()
        n_steps = 0
        t0 = time.perf_counter()
        for example in dataloader:
            t["data"] += time.perf_counter() - t0

            schedule_lr(global_step)

            t1 = time.perf_counter()
            raw = combine_fixed_length(example["raw_emg"], args.fixed_raw_len).to(device)
            if emg_augment_enabled and model.training:
                raw = emg_augment(raw)
            pred = F.log_softmax(model(raw), dim=-1)
            pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, example["lengths"]), batch_first=False)
            targets = nn.utils.rnn.pad_sequence(example["text_int"], batch_first=True).to(device)
            loss = F.ctc_loss(pred, targets, example["lengths"], example["text_int_lengths"], blank=n_chars)
            _sync(device)
            t["fwd"] += time.perf_counter() - t1

            t2 = time.perf_counter()
            loss.backward()
            _sync(device)
            t["bwd"] += time.perf_counter() - t2

            losses.append(loss.item())

            t3 = time.perf_counter()
            if (global_step + 1) % args.grad_accum_steps == 0:
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optim.step()
                optim.zero_grad()
                _sync(device)
            t["opt"] += time.perf_counter() - t3

            global_step += 1
            n_steps += 1
            t0 = time.perf_counter()

        train_loss = float(np.mean(losses)) if losses else 0.0
        eval_start = time.perf_counter()
        wer, cer = evaluate(model, devset, device, method=args.eval_method)
        t_eval = time.perf_counter() - eval_start

        lr_sched.step()
        t_epoch = time.perf_counter() - epoch_start
        cur_lr = optim.param_groups[0]["lr"]

        logging.info(
            "epoch=%d/%d steps=%d lr=%.2e train_loss=%.4f dev_wer=%.3f dev_cer=%.3f "
            "t_data=%.1fs t_fwd=%.1fs t_bwd=%.1fs t_opt=%.1fs t_eval=%.1fs t_epoch=%.1fs",
            epoch + 1, args.epochs, n_steps, cur_lr, train_loss, wer, cer,
            t["data"], t["fwd"], t["bwd"], t["opt"], t_eval, t_epoch,
        )
        wandb_log(run, {
            "eval/wer": wer, "eval/cer": cer,
            "train/loss": train_loss, "train/lr": cur_lr,
            "time/data": t["data"], "time/fwd": t["fwd"],
            "time/bwd": t["bwd"], "time/opt": t["opt"],
            "time/eval": t_eval, "time/epoch": t_epoch,
            "epoch": epoch + 1,
        })

        torch.save(model.state_dict(), os.path.join(args.output_directory, "last.pt"))
        torch.save(model.state_dict(), os.path.join(args.output_directory, f"last_{run_ts}.pt"))
        if wer < best_wer:
            best_wer = wer
            torch.save(model.state_dict(), os.path.join(args.output_directory, "best.pt"))
            torch.save(model.state_dict(), os.path.join(args.output_directory, f"best_{run_ts}.pt"))

    finish_wandb(run)


if __name__ == "__main__":
    setup_stdout_logging()
    train(parse_args())
