"""Measure EMG token resolution vs. target-unit duration, to pick `conv_strides`.

For every cached utterance we know the raw EMG length (689.06 Hz samples) and can
re-encode the transcript to any unit (char/subword/phoneme). This script reports,
per unit:

  - utterance duration stats (s);
  - number of target tokens per utterance;
  - **per-unit duration** = duration / n_tokens  (mean / median / p5 / min) — the
    physical time one character/subword/phoneme occupies on average;
  - for each candidate downsample factor F (= prod(conv_strides)): the CTC frame
    period (F / 689.06 s), and the CTC feasibility of that F across the dataset
    (fraction of utterances with `raw_len // F >= n_tokens`, and the frames/tokens
    ratio = how many EMG frames the model gets per target token).

Rule of thumb: CTC needs output_frames >= target_tokens for *every* training
utterance (hard floor, set by the *densest* utterance = min per-unit duration),
and works best with a few frames per token (room for blanks / repeats). So pick the
largest F whose frames/tokens ratio stays comfortably > 1 on the p5/min utterance.

Usage:
    PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text \
    python scripts/analyze_unit_durations.py --unit char
    ... --unit subword --subword-model data/tokenizers/subword_500.model
    ... --unit phoneme --phoneme-dict /path/to/cmudict
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from semg_jepa.tokenizers import build_tokenizer

SAMPLE_RATE = 689.06  # Hz, the stored raw_emg rate (read_emg.subsample target)


def load_split(cache_dir: str, split: str):
    payload = torch.load(os.path.join(cache_dir, f"{split}.pt"), map_location="cpu")
    return payload["samples"]


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--unit", choices=["char", "subword", "phoneme"], default="char")
    p.add_argument("--subword-model", default=None)
    p.add_argument("--phoneme-dict", default=None)
    p.add_argument("--factors", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    args = p.parse_args()

    tok = build_tokenizer(args.unit, subword_model=args.subword_model,
                          phoneme_dict=args.phoneme_dict)

    raw_lens, n_tokens = [], []
    for split in args.splits:
        for s in load_split(args.cache_dir, split):
            raw = s["raw_emg"]
            n = raw.shape[0] if hasattr(raw, "shape") else len(raw)
            raw_lens.append(int(n))
            n_tokens.append(len(tok.text_to_int(s["text"])))

    raw_lens = np.asarray(raw_lens, dtype=np.float64)
    n_tokens = np.asarray(n_tokens, dtype=np.float64)
    n_empty = int((n_tokens == 0).sum())               # transcripts with no target tokens
    if n_empty:
        keep = n_tokens > 0
        raw_lens, n_tokens = raw_lens[keep], n_tokens[keep]
    dur = raw_lens / SAMPLE_RATE                       # seconds
    per_unit_ms = 1000.0 * dur / n_tokens              # ms per target token
    units_per_s = n_tokens / dur

    print(f"\n=== unit={args.unit}  splits={args.splits}  N={len(raw_lens)} "
          f"(+{n_empty} empty dropped)  vocab={tok.vocab_size} ===")
    print(f"utterance duration (s): mean={dur.mean():.2f} median={np.median(dur):.2f} "
          f"min={dur.min():.2f} max={dur.max():.2f}")
    print(f"tokens/utt           : mean={n_tokens.mean():.1f} median={np.median(n_tokens):.0f} "
          f"min={n_tokens.min():.0f} max={n_tokens.max():.0f}")
    print(f"units/sec            : mean={units_per_s.mean():.1f} median={np.median(units_per_s):.1f}")
    print(f"per-unit duration(ms): mean={per_unit_ms.mean():.1f} median={np.median(per_unit_ms):.1f} "
          f"p5={pct(per_unit_ms,5):.1f} min={per_unit_ms.min():.1f}  "
          f"(min = densest utterance = the CTC hard floor)")

    print(f"\n  F   strides        frame_period   frames/token(median)  feasible(all frames>=tokens)")
    for f in args.factors:
        frames = np.floor(raw_lens / f)
        ratio = frames / n_tokens
        feasible = float((frames >= n_tokens).mean())
        period_ms = 1000.0 * f / SAMPLE_RATE
        strides = "x".join(["2"] * int(round(np.log2(f)))) if (f & (f - 1)) == 0 else str(f)
        flag = "  <-- char-8x default" if f == 8 else ""
        print(f"  {f:<3d} ({strides:<9}) {period_ms:6.1f} ms     "
              f"median={np.median(ratio):5.2f}  min={ratio.min():4.2f}   "
              f"{100*feasible:5.1f}%{flag}")
    print()


if __name__ == "__main__":
    main()
