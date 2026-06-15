"""Phoneme CTC diagnostics: greedy PER + phone-level reconstructions per checkpoint.

For each phoneme checkpoint we forward the test set, greedy-collapse the CTC output,
and report:
  - greedy **PER** (phone error rate; word boundaries `|` kept as tokens), and
  - K example  ref / hyp  phone strings, to eyeball the error structure.

This is the "feel the models" tool — it is *not* word decoding (see
`scripts/phoneme_to_words_lattice.py` for phones -> words). Reuses the standard eval
forward pass so the numbers match `train_baseline`/`evaluate` greedy PER.

Usage (GPU):
    PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text \
    python scripts/phoneme_diag.py            # the 3 default checkpoints, last.pt
    python scripts/phoneme_diag.py --runs runs/baseline_phoneme_16x:16 --k 12
"""
from __future__ import annotations

import argparse
import os

import torch

from semg_jepa.architecture import BaselineCTCModel, factor_to_strides
from semg_jepa.cached_dataset import CachedRawEMGDataset
from semg_jepa.ctc_utils import _greedy_collapse, compute_log_probs
from semg_jepa.metrics import compute_wer
from semg_jepa.tokenizers import build_tokenizer

# (run_dir, downsample_factor) defaults — the three phoneme baselines.
DEFAULT_RUNS = [
    ("runs/baseline_phoneme_8x", 8),
    ("runs/baseline_phoneme_10x", 10),
    ("runs/baseline_phoneme_16x", 16),
]


def eval_checkpoint(run_dir, factor, ckpt_name, cache_dir, split, phoneme_dict, device, k):
    tokenizer = build_tokenizer("phoneme", phoneme_dict=phoneme_dict)
    conv_strides = factor_to_strides(factor)
    model = BaselineCTCModel(vocab_size=tokenizer.vocab_size, conv_strides=conv_strides).to(device)
    ckpt = os.path.join(run_dir, ckpt_name)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

    ds = CachedRawEMGDataset(cache_dir, split, tokenizer=tokenizer, downsample_factor=factor)
    log_probs_list, references = compute_log_probs(model, ds, device)

    blank_id = tokenizer.blank_index
    hyps = [
        tokenizer.int_to_text(_greedy_collapse(lp.argmax(-1).tolist(), blank_id))
        for lp in log_probs_list
    ]
    per = compute_wer(references, hyps)

    tag = os.path.basename(run_dir.rstrip("/"))
    print(f"\n===== {tag} ({factor}x, {ckpt_name}) [{split}] | N={len(references)} =====")
    print(f"greedy PER : {per:.4f}")
    for r, h in list(zip(references, hyps))[:k]:
        print(f"  ref: {r}")
        print(f"  hyp: {h or '<empty>'}")
    return tag, factor, per


def parse_runs(items):
    out = []
    for it in items:
        path, _, fac = it.partition(":")
        out.append((path, int(fac)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=None,
                   help="run_dir:factor pairs (default: the 3 phoneme baselines)")
    p.add_argument("--ckpt-name", default="last.pt")
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--phoneme-dict", default="data/tokenizers/phoneme_g2p.dict")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    runs = parse_runs(args.runs) if args.runs else DEFAULT_RUNS
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[phoneme_diag] device={device} | {len(runs)} checkpoint(s) | {args.ckpt_name}", flush=True)

    summary = []
    for run_dir, factor in runs:
        summary.append(eval_checkpoint(run_dir, factor, args.ckpt_name, args.cache_dir,
                                       args.split, args.phoneme_dict, device, args.k))

    print(f"\n===== summary (greedy PER, {args.split}) =====")
    print(f"{'run':32s} {'factor':>7s} {'PER':>8s}")
    for tag, factor, per in summary:
        print(f"{tag:32s} {factor:>6d}x {per:>8.4f}")


if __name__ == "__main__":
    main()
