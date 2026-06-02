"""Train a SentencePiece subword tokenizer for the `--unit subword` baseline.

Trains on the (cleaned) EMG training-split transcripts by default, so the subword
vocabulary matches the in-domain text. Use ``--text-file`` to train on an external
corpus instead (e.g. for the phase-2 unpaired-text branch).

Example:
    python scripts/train_subword.py --vocab-size 500
    # -> data/tokenizers/subword_500.model  (+ .vocab)
"""
from __future__ import annotations

import argparse
import os
import tempfile

import torch

from semg_jepa.tokenizers import clean_text


def collect_cache_text(cache_dir: str, split: str) -> list[str]:
    payload = torch.load(os.path.join(cache_dir, f"{split}.pt"), map_location="cpu")
    return [clean_text(s["text"]) for s in payload["samples"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--split", default="train")
    p.add_argument("--text-file", default=None,
                   help="Train on this text file (one sentence per line) instead of the EMG cache.")
    p.add_argument("--vocab-size", type=int, default=500)
    p.add_argument("--model-type", choices=["bpe", "unigram"], default="unigram")
    p.add_argument("--output-dir", default="data/tokenizers")
    p.add_argument("--model-prefix", default=None,
                   help="Defaults to <output-dir>/subword_<vocab-size>.")
    args = p.parse_args()

    import sentencepiece as spm

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.model_prefix or os.path.join(args.output_dir, f"subword_{args.vocab_size}")

    if args.text_file:
        text_path = args.text_file
        cleanup = False
    else:
        lines = collect_cache_text(args.cache_dir, args.split)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        tmp.write("\n".join(lines))
        tmp.close()
        text_path, cleanup = tmp.name, True
        print(f"[train_subword] {len(lines)} {args.split} transcripts -> {text_path}")

    try:
        spm.SentencePieceTrainer.train(
            input=text_path,
            model_prefix=prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=1.0,
            # Match the char vocab's alphabet so EMG transcripts encode losslessly.
            user_defined_symbols=[],
            bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1,
        )
    finally:
        if cleanup:
            os.unlink(text_path)

    print(f"[train_subword] wrote {prefix}.model (+ .vocab)")


if __name__ == "__main__":
    main()
