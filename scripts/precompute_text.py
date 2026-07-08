#!/usr/bin/env python
"""Precompute the unpaired-text corpus cache for the UML **text branch** (phase-2 §C).

Encodes a corpus to clean char-level ``text_int`` (same ``CharTokenizer`` vocab as
the EMG/audio branches) and writes ``<out-dir>/<source>.pt``:

    text_int : list[LongTensor (L_i,)]  — clean char ids
    version  : 1

Two sources (``--text-source``), mirroring phase-1's audio choice:

* ``libri``  — LibriSpeech ``train-clean-100`` transcripts (the same text the audio
  branch used). **Deduped** against the gold books (War of the Worlds + Sherlock
  Holmes, recovered from the EMG cache ``text`` fields) so a "text gain" cannot be
  test/dev leakage. Dedup drops any sentence whose normalized form exactly matches a
  gold sentence OR shares an ``--shingle-n`` (default 8) word contiguous shingle.

* ``gaddy``  — the Gaddy **train**-split transcripts (in-distribution). Not deduped:
  these are the exact labels the EMG branch is already supervised on, so the text
  branch adds no information beyond the paired labels.

Usage
-----
    python scripts/precompute_text.py --text-source libri
    python scripts/precompute_text.py --text-source gaddy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from semg_jepa.tokenizers import CharTokenizer, clean_text
from scripts.precompute_audio import build_manifest


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _libri_sentences(librispeech_dir: str, split: str) -> list[str]:
    tasks = build_manifest(librispeech_dir, split)
    return [text for _flac, text in tasks]


def _gaddy_sentences(cache_dir: str, split: str = "train") -> list[str]:
    payload = torch.load(os.path.join(cache_dir, f"{split}.pt"), map_location="cpu")
    return [s["text"] for s in payload["samples"]]


def _log_book_composition(cache_dir: str) -> None:
    for split in ("train", "dev", "test"):
        path = os.path.join(cache_dir, f"{split}.pt")
        if not os.path.isfile(path):
            continue
        payload = torch.load(path, map_location="cpu")
        books = Counter(
            str(s.get("book_location", ("", -1))[0]) for s in payload["samples"]
        )
        print(f"[precompute_text] {split}.pt book composition: {dict(books)}", flush=True)


# ---------------------------------------------------------------------------
# Dedup (libri only): gold = WotW + Sherlock, recovered from the EMG cache text.
# ---------------------------------------------------------------------------

def _word_shingles(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def build_gold_filter(cache_dir: str, shingle_n: int) -> tuple[set, set]:
    """Exact-sentence set + n-gram-shingle set over all train/dev/test gold text."""
    exact: set[str] = set()
    shingles: set[tuple[str, ...]] = set()
    for split in ("train", "dev", "test"):
        path = os.path.join(cache_dir, f"{split}.pt")
        if not os.path.isfile(path):
            continue
        payload = torch.load(path, map_location="cpu")
        for s in payload["samples"]:
            norm = clean_text(s["text"])
            if not norm:
                continue
            exact.add(norm)
            words = norm.split()
            if len(words) >= shingle_n:
                shingles |= _word_shingles(words, shingle_n)
    return exact, shingles


def dedup_sentences(
    sentences: list[str], exact: set, shingles: set, shingle_n: int
) -> tuple[list[str], int]:
    """Drop any sentence that exactly matches gold or shares a gold shingle."""
    kept, dropped = [], 0
    for raw in sentences:
        norm = clean_text(raw)
        if not norm:
            dropped += 1
            continue
        if norm in exact:
            dropped += 1
            continue
        words = norm.split()
        if len(words) >= shingle_n and (_word_shingles(words, shingle_n) & shingles):
            dropped += 1
            continue
        kept.append(raw)
    return kept, dropped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text-source", choices=["libri", "gaddy"], default="libri")
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data",
                   help="EMG cache dir (for the gaddy source + the gold dedup books).")
    p.add_argument("--librispeech-dir", default="/scratch/cr4206/data/librispeech")
    p.add_argument("--librispeech-split", default="train-clean-100")
    p.add_argument("--out-dir", default="/scratch/cr4206/sEMG-unpaired-text/data/text_cache")
    p.add_argument("--shingle-n", type=int, default=8,
                   help="Word-shingle length for libri dedup against the gold books.")
    p.add_argument("--no-dedup", action="store_true",
                   help="Skip dedup (libri only; gaddy is never deduped).")
    p.add_argument("--min-chars", type=int, default=1,
                   help="Drop sentences whose encoded length is below this.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = CharTokenizer()
    _log_book_composition(args.cache_dir)

    if args.text_source == "libri":
        print(f"[precompute_text] reading LibriSpeech transcripts ({args.librispeech_split}) ...",
              flush=True)
        sentences = _libri_sentences(args.librispeech_dir, args.librispeech_split)
        print(f"[precompute_text]   {len(sentences)} raw sentences", flush=True)
        if not args.no_dedup:
            exact, shingles = build_gold_filter(args.cache_dir, args.shingle_n)
            print(f"[precompute_text] gold filter: {len(exact)} sentences, "
                  f"{len(shingles)} {args.shingle_n}-word shingles", flush=True)
            sentences, dropped = dedup_sentences(sentences, exact, shingles, args.shingle_n)
            pct = 100.0 * dropped / max(1, dropped + len(sentences))
            print(f"[precompute_text] dedup dropped {dropped} sentences "
                  f"({pct:.2f}%); {len(sentences)} survive", flush=True)
            # Hard gate: nothing surviving may overlap gold (it shouldn't, post-filter).
            assert not any(clean_text(s) in exact for s in sentences), \
                "dedup audit failed: a gold sentence survived"
    else:  # gaddy
        print("[precompute_text] reading Gaddy train transcripts ...", flush=True)
        sentences = _gaddy_sentences(args.cache_dir, "train")
        print(f"[precompute_text]   {len(sentences)} train transcripts (no dedup; "
              f"these are the supervised labels)", flush=True)

    text_int_list: list[torch.Tensor] = []
    n_skipped = 0
    for raw in sentences:
        try:
            ids = tok.text_to_int(raw)
        except ValueError:
            n_skipped += 1
            continue
        if len(ids) < args.min_chars:
            n_skipped += 1
            continue
        text_int_list.append(torch.tensor(ids, dtype=torch.long))

    out_path = os.path.join(args.out_dir, f"{args.text_source}.pt")
    torch.save({"text_int": text_int_list, "version": 1}, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[precompute_text] wrote {len(text_int_list)} sentences "
          f"(skipped {n_skipped}) → {out_path} ({size_mb:.1f} MB) "
          f"in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
