import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import math

from semg_jepa.architecture import BaselineCTCModel, factor_to_strides
from semg_jepa.cached_dataset import CachedRawEMGDataset, build_batches
from semg_jepa.config_utils import parse_with_config, setup_stdout_logging
from semg_jepa.ctc_utils import evaluate
from semg_jepa.data_utils import combine_fixed_length, decollate_tensor
from semg_jepa.tokenizers import build_tokenizer
from semg_jepa.wandb_utils import finish_wandb, init_wandb, wandb_log


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def train(args):
    run = init_wandb(args, default_name_prefix="baseline")

    # --downsample-factor (e.g. 25) takes precedence over --conv-strides (e.g. 2 2 2).
    if getattr(args, "downsample_factor", None):
        conv_strides = factor_to_strides(args.downsample_factor)
    else:
        conv_strides = tuple(args.conv_strides)
    factor = math.prod(conv_strides)
    assert args.fixed_raw_len % factor == 0, (
        f"fixed_raw_len={args.fixed_raw_len} must be divisible by downsample factor {factor} "
        f"(conv_strides={conv_strides})")

    tokenizer = build_tokenizer(
        args.unit, subword_model=args.subword_model, phoneme_dict=args.phoneme_dict,
    )
    logging.info("unit=%s vocab_size=%d blank=%d conv_strides=%s downsample_factor=%d",
                 args.unit, tokenizer.vocab_size, tokenizer.blank_index, conv_strides, factor)

    trainset = CachedRawEMGDataset(args.cache_dir, "train", tokenizer=tokenizer, downsample_factor=factor)
    devset = CachedRawEMGDataset(args.cache_dir, "dev", tokenizer=tokenizer, downsample_factor=factor)
    blank_id = tokenizer.blank_index

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = BaselineCTCModel(
        model_size=args.model_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        vocab_size=tokenizer.vocab_size,
        conv_strides=conv_strides,
    ).to(device)

    if args.start_training_from:
        model.load_state_dict(torch.load(args.start_training_from, map_location=device), strict=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=args.lr_decay_milestones, gamma=args.lr_decay_gamma,
    )

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group["lr"] = new_lr

    def schedule_lr(iteration):
        iteration += 1
        if iteration <= args.learning_rate_warmup:
            set_lr(iteration * args.learning_rate / args.learning_rate_warmup)

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
            pred = F.log_softmax(model(raw), dim=-1)
            pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, example["lengths"]), batch_first=False)
            targets = nn.utils.rnn.pad_sequence(example["text_int"], batch_first=True).to(device)
            loss = F.ctc_loss(pred, targets, example["lengths"], example["text_int_lengths"], blank=blank_id)
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
            "time/data": t["data"], "time/fwd": t["fwd"], "time/bwd": t["bwd"], "time/opt": t["opt"],
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to YAML config; CLI flags override its values.")
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data")
    p.add_argument("--output-directory", default="/scratch/cr4206/sEMG-unpaired-text/runs/baseline")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-batch-len", type=int, default=128000)
    p.add_argument("--fixed-raw-len", type=int, default=1600)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--learning-rate-warmup", type=int, default=1000)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--lr-decay-milestones", type=int, nargs="+", default=[125, 150, 175])
    p.add_argument("--lr-decay-gamma", type=float, default=0.5)
    p.add_argument("--clip-grad-norm", type=float, default=0.0,
                   help="Max grad norm before optim.step (0 disables).")
    p.add_argument("--model-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    # Phase-2 knobs: target unit and EMG token temporal resolution.
    p.add_argument("--unit", choices=["char", "subword", "phoneme"], default="char",
                   help="CTC target unit. Retrain a baseline per unit.")
    p.add_argument("--subword-model", default=None,
                   help="Path to a SentencePiece .model (required for --unit subword).")
    p.add_argument("--phoneme-dict", default=None,
                   help="CMUdict-style file for --unit phoneme (else uses g2p_en if installed).")
    p.add_argument("--conv-strides", type=int, nargs="+", default=[2, 2, 2],
                   help="Per-ResBlock strides; downsample factor = product "
                        "(default 8x ~86Hz; e.g. 2 2 = 4x, 2 2 2 2 = 16x).")
    p.add_argument("--downsample-factor", type=int, default=None,
                   help="EMG token downsample factor (alternative to --conv-strides; "
                        "must divide fixed_raw_len=1600, i.e. be 2^a*5^b). "
                        "E.g. 25 -> strides (5,5), 20 -> (2,2,5). Takes precedence.")
    p.add_argument("--start-training-from", default=None)
    p.add_argument("--eval-method", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="UMLforVideoLab")
    p.add_argument("--wandb-project", default="JEPAforsEMG")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--cpu", action="store_true")
    return parse_with_config(p)


if __name__ == "__main__":
    setup_stdout_logging()
    train(parse_args())
