"""Apples-to-apples decode: run char AND phoneme through the SAME lexicon-constrained
flashlight (torchaudio) decoder + the SAME KenLM, so the only thing that differs is the
acoustic unit. Isolates "is the phoneme a better unit?" from "is a lexicon-constrained
WFST decoder stronger than pyctcdecode?".

For every unit we use the SAME open-vocab word vocabulary (LibriSpeech `unigrams.txt`,
157k words), spelled in that unit's tokens:
  - char    : word -> its letters,  e.g. "hello h e l l o |"   (sil='|' == the space class)
  - phoneme : word -> its phones    (lexicon_librispeech.txt, built earlier)
Token strings are just labels; only index-alignment to the CTC logits matters. For char the
space class (index 36) is RELABELED '|' so flashlight can use it as the word separator.

Subword is intentionally excluded: SentencePiece encodes word boundaries in the '▁' piece
prefix, not a separate emitted separator token, so it has no clean flashlight lexicon mapping
(its pyctcdecode+KenLM number stays the reference for that unit).

Usage (GPU): python scripts/decode_lexicon_flashlight.py            # char {16,20,25}x + phoneme {10,16}x
"""
from __future__ import annotations

import argparse
import os

import torch

from semg_jepa.architecture import factor_to_strides  # noqa: F401 (used via load_model)
from semg_jepa.cached_dataset import CachedRawEMGDataset
from semg_jepa.ctc_utils import _greedy_collapse
from semg_jepa.metrics import compute_wer
from semg_jepa.tokenizers import build_tokenizer, clean_text

from scripts.phoneme_to_words_lattice import (
    BLANK_TOKEN, SIL_TOKEN, make_decoder, decode_words, forward_logits,
    load_model, load_tokens,
)

# Most-promising runs per unit (TODO.md §B): char best 20x 0.292 / 16x 0.296 / 25x;
# phoneme best 16x (PER .161) / 10x (.169). Subword-250@16x (0.290) is reference-only.
DEFAULT_RUNS = [
    ("char", "runs/baseline_char_16x", 16),
    ("char", "runs/baseline_char_20x", 20),
    ("char", "runs/baseline_char_25x", 25),
    ("phoneme", "runs/baseline_phoneme_10x", 10),
    ("phoneme", "runs/baseline_phoneme_16x", 16),
]


def build_char_lexicon(unigrams_path, out_lexicon, out_tokens):
    """char lexicon (word -> letters + sep) + tokens index-aligned to char logits."""
    from semg_jepa.tokenizers import CharTokenizer
    chars = CharTokenizer().chars            # 'a..z0..9 ' ; space at index 36
    vocab = set(chars[:-1])                   # letters/digits (exclude the space class)
    tokens = list(chars[:-1]) + [SIL_TOKEN] + [BLANK_TOKEN]  # space(36)->'|', blank(37)
    with open(out_tokens, "w") as f:
        f.write("\n".join(tokens) + "\n")
    n = 0
    with open(unigrams_path) as fi, open(out_lexicon, "w") as fo:
        for line in fi:
            w = clean_text(line.strip())
            w = w.split()[0] if w.split() else ""
            if not w or any(c not in vocab for c in w):
                continue
            fo.write(w + " " + " ".join(w) + " " + SIL_TOKEN + "\n")
            n += 1
    print(f"[flashlight] char lexicon {n} words -> {out_lexicon}", flush=True)


def unit_files(unit, args):
    if unit == "char":
        lex = "data/tokenizers/lexicon_char.txt"
        tok = "data/tokenizers/char_tokens.txt"
        if not (os.path.exists(lex) and os.path.exists(tok)):
            build_char_lexicon(args.unigrams, lex, tok)
        return lex, tok, build_tokenizer("char")
    if unit == "phoneme":
        return (args.phoneme_lexicon, args.phoneme_tokens,
                build_tokenizer("phoneme", phoneme_dict=args.phoneme_dict))
    raise ValueError(unit)


def greedy_word_wer(log_probs_list, tokenizer, refs):
    """Native greedy word string (CTC collapse + int_to_text) vs word refs."""
    blank = tokenizer.blank_index
    hyps = [tokenizer.int_to_text(_greedy_collapse(lp.argmax(-1).tolist(), blank))
            for lp in log_probs_list]
    return compute_wer(refs, hyps)


def tune_dev(dlp, drefs, lexicon, tokens, args):
    best = None
    for lm_w in args.lm_weights:
        for ws in args.word_scores:
            dec = make_decoder(lexicon, tokens, args.lm_path, lm_w, ws, args.beam_size)
            wer = compute_wer(drefs, decode_words(dec, dlp))
            print(f"    dev lm_w={lm_w:<4} ws={ws:<5} -> WER {wer:.4f}", flush=True)
            if best is None or wer < best[2]:
                best = (lm_w, ws, wer)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=None, help="unit:run_dir:factor (default: chart's best)")
    p.add_argument("--ckpt-name", default="last.pt")
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--dev-split", default="dev")
    p.add_argument("--unigrams", default="data/unigrams.txt")
    p.add_argument("--phoneme-lexicon", default="data/tokenizers/lexicon_librispeech.txt")
    p.add_argument("--phoneme-tokens", default="data/tokenizers/phoneme_tokens.txt")
    p.add_argument("--phoneme-dict", default="data/tokenizers/phoneme_g2p.dict")
    p.add_argument("--lm-path", default="data/lm.binary")
    p.add_argument("--beam-size", type=int, default=100)
    p.add_argument("--lm-weights", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--word-scores", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.runs:
        runs = []
        for it in args.runs:
            u, rd, fac = it.split(":")
            runs.append((u, rd, int(fac)))
    else:
        runs = DEFAULT_RUNS

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[flashlight] device={device} | identical lexicon+KenLM decoder across units", flush=True)

    summary = []
    for unit, run_dir, factor in runs:
        lexicon, tokens_path, tokenizer = unit_files(unit, args)
        tokens = load_tokens(tokens_path)
        tag = os.path.basename(run_dir.rstrip("/"))
        print(f"\n##### {unit} | {tag} ({factor}x) | tokens={len(tokens)} #####", flush=True)
        model = load_model(run_dir, factor, args.ckpt_name, tokenizer.vocab_size, device)

        dev_ds = CachedRawEMGDataset(args.cache_dir, args.dev_split, tokenizer=tokenizer,
                                     downsample_factor=factor)
        dlp, dtex = forward_logits(model, dev_ds, device)
        drefs = [clean_text(t) for t in dtex]
        lm_w, ws, dwer = tune_dev(dlp, drefs, lexicon, tokens, args)
        print(f"  best dev: lm_w={lm_w} ws={ws} WER {dwer:.4f}", flush=True)

        test_ds = CachedRawEMGDataset(args.cache_dir, args.split, tokenizer=tokenizer,
                                      downsample_factor=factor)
        tlp, ttex = forward_logits(model, test_ds, device)
        trefs = [clean_text(t) for t in ttex]
        greedy = greedy_word_wer(tlp, tokenizer, trefs)
        dec = make_decoder(lexicon, tokens, args.lm_path, lm_w, ws, args.beam_size)
        hyp = decode_words(dec, tlp)
        flash = compute_wer(trefs, hyp)
        print(f"  [{args.split}] greedy(native) WER {greedy:.4f} | flashlight lex+KenLM WER {flash:.4f}"
              f"  (lm_w={lm_w}, ws={ws})", flush=True)
        for r, h in list(zip(trefs, hyp))[: args.k]:
            print(f"    ref: {r}\n    hyp: {h or '<empty>'}")
        summary.append((unit, tag, factor, greedy, flash))

    print(f"\n===== apples-to-apples summary (word WER, {args.split}) =====")
    print(f"{'unit':8s} {'run':26s} {'factor':>6s} {'greedy':>8s} {'flashlight+KenLM':>17s}")
    for unit, tag, factor, g, fl in summary:
        print(f"{unit:8s} {tag:26s} {factor:>5d}x {g:>8.4f} {fl:>17.4f}")


if __name__ == "__main__":
    main()
