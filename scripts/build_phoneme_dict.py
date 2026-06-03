"""Precompute a COMPLETE phoneme pronunciation dict for the EMG transcripts.

Why: CMUdict is a fixed lookup table, so any word not in it (proper nouns like
"Weybridge", rare inflections/compounds like "unscrewing", numbers) is
**out-of-vocabulary (OOV)** -> no pronunciation -> the word is silently dropped
from the phoneme CTC target. To get lossless coverage we run `g2p_en` (a neural
grapheme->phoneme model that predicts a pronunciation for ANY spelling) ONCE over
every unique word in the dataset and write a CMUdict-format file. Training then
uses fast dict lookups (`--phoneme-dict`) with no per-epoch G2P and no OOV gaps.

Output (default): data/tokenizers/phoneme_g2p.dict  — lines "WORD  P H O N E S"
(lowercase phones, stress stripped), parsed by PhonemeTokenizer._load_dict.

Usage:
    PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text \
    python scripts/build_phoneme_dict.py            # train+dev+test vocab
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from semg_jepa.tokenizers import ARPABET, clean_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--out", default="data/tokenizers/phoneme_g2p.dict")
    args = p.parse_args()

    words = set()
    for split in args.splits:
        payload = torch.load(os.path.join(args.cache_dir, f"{split}.pt"), map_location="cpu")
        for s in payload["samples"]:
            words.update(clean_text(s["text"]).split())
    words = sorted(w for w in words if w)
    print(f"[build_phoneme_dict] {len(words)} unique words across {args.splits}", flush=True)

    from g2p_en import G2p  # neural G2P; handles OOV words

    g2p = G2p()
    arp = set(ARPABET)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    n_written = 0
    with open(args.out, "w") as f:
        for i, w in enumerate(words):
            phones = []
            for ph in g2p(w):
                ph2 = ph.strip().rstrip("012").lower()
                if ph2 in arp:
                    phones.append(ph2)
            if not phones:
                continue
            f.write(w.upper() + "  " + " ".join(phones) + "\n")
            n_written += 1
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(words)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[build_phoneme_dict] wrote {n_written} entries -> {args.out} "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
