"""Phonemes -> words via lexicon + KenLM beam over the FULL phone CTC lattice.

Method A (recommended) from the plan: `torchaudio.models.decoder.ctc_decoder`
(flashlight LexiconDecoder) decodes the phone CTC emissions directly into words, using
  - an OPEN-VOCAB lexicon  (g2p'd LibriSpeech unigrams; build with
    `scripts/build_lexicon_from_unigrams.py`), independent of the dataset/test words, and
  - the SAME KenLM word LM (`data/lm.binary`) that backs the char/subword beams,
so the resulting word WER is directly comparable to char-8x 0.315 / char-20x 0.292.

Unlike `scripts/phoneme_to_words.py` (greedy 1-best phones -> nearest-pron map, method B),
this searches the whole lattice and lets the word LM resolve phone errors / homophones.

The CTC logits layout is [ARPABET(0..46), '|'(47), <blank>(48)]; the `tokens` file from
the lexicon builder is index-aligned to that, with sil_token='|' and blank_token='<blank>'.

Self-test (no GPU/model) — verifies token/lexicon/blank alignment:
    python scripts/phoneme_to_words_lattice.py --selftest
Full eval (GPU): tunes (lm_weight, word_score) on dev, reports test word WER for the 3 ckpts:
    python scripts/phoneme_to_words_lattice.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from torchaudio.models.decoder import ctc_decoder

from semg_jepa.architecture import BaselineCTCModel, factor_to_strides
from semg_jepa.cached_dataset import CachedRawEMGDataset
from semg_jepa.ctc_utils import _greedy_collapse, _make_eval_collate
from semg_jepa.metrics import compute_wer
from semg_jepa.tokenizers import build_tokenizer, clean_text

BLANK_TOKEN = "<blank>"
SIL_TOKEN = "|"

DEFAULT_RUNS = [
    ("runs/baseline_phoneme_8x", 8),
    ("runs/baseline_phoneme_10x", 10),
    ("runs/baseline_phoneme_16x", 16),
]


# ---- decoder ------------------------------------------------------------------

def make_decoder(lexicon, tokens, lm_path, lm_weight, word_score, beam_size):
    return ctc_decoder(
        lexicon=lexicon, tokens=tokens, lm=lm_path,
        nbest=1, beam_size=beam_size, lm_weight=lm_weight, word_score=word_score,
        blank_token=BLANK_TOKEN, sil_token=SIL_TOKEN,
    )


def decode_words(decoder, log_probs_list):
    """Decode each utterance's [T,N] log-prob lattice to a word string."""
    preds = []
    for lp in log_probs_list:
        emis = torch.from_numpy(np.ascontiguousarray(lp, dtype=np.float32)).unsqueeze(0)
        res = decoder(emis)
        words = res[0][0].words if res and res[0] else []
        preds.append(" ".join(words))
    return preds


def load_tokens(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


# ---- forward ------------------------------------------------------------------

def forward_logits(model, dataset, device):
    """Return (log_probs_list [T,N], raw_texts) for the split."""
    model.eval()
    collate = _make_eval_collate(dataset.downsample_factor)
    bs = 16 if str(device).startswith("cuda") else 1
    dl = torch.utils.data.DataLoader(dataset, batch_size=bs, collate_fn=collate)
    lps, texts_all = [], []
    with torch.no_grad():
        for raw, seq_lens, texts in dl:
            lp = F.log_softmax(model(raw.to(device)), -1).cpu()
            for i, T in enumerate(seq_lens.tolist()):
                lps.append(lp[i, :T].numpy().astype(np.float32))
            texts_all.extend(texts)
    return lps, texts_all


def load_model(run_dir, factor, ckpt_name, vocab_size, device):
    model = BaselineCTCModel(vocab_size=vocab_size, conv_strides=factor_to_strides(factor)).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, ckpt_name), map_location=device),
                          strict=False)
    return model


# ---- comparison row: greedy + exact open-vocab lexicon (no LM) -----------------

def build_exact_lexicon(lexicon_path):
    """{(phones tuple): word} from the torchaudio lexicon file (first word per pron)."""
    pron2word = {}
    with open(lexicon_path) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 2:
                continue
            word, phones = parts[0], parts[1:]
            if phones and phones[-1] == SIL_TOKEN:
                phones = phones[:-1]
            pron2word.setdefault(tuple(phones), word)
    return pron2word


def greedy_exact_words(log_probs_list, blank_id, symbols, pron2word):
    preds = []
    for lp in log_probs_list:
        ids = _greedy_collapse(lp.argmax(-1).tolist(), blank_id)
        syms = [symbols[j] for j in ids]
        words, cur = [], []
        for s in syms + [SIL_TOKEN]:
            if s == SIL_TOKEN:
                w = pron2word.get(tuple(cur))
                if w:
                    words.append(w)
                cur = []
            else:
                cur.append(s)
        preds.append(" ".join(words))
    return preds


# ---- self-test ----------------------------------------------------------------

def run_selftest(args):
    tokens = load_tokens(args.tokens)
    tok2idx = {t: i for i, t in enumerate(tokens)}
    blank_id = tok2idx[BLANK_TOKEN]
    n = len(tokens)
    pron2word = build_exact_lexicon(args.lexicon)
    # pick a couple of words whose pron uses only known tokens
    word2pron = {}
    with open(args.lexicon) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 3 and p[-1] == SIL_TOKEN:
                word2pron.setdefault(p[0], p[1:-1])
    picks = [w for w in ("the", "dog", "house", "water") if w in word2pron][:2]
    if len(picks) < 2:
        picks = list(word2pron)[:2]
    spelling = []
    for w in picks:
        spelling.extend(word2pron[w] + [SIL_TOKEN])
    # build a near-one-hot emission: 2 frames per token, leading/trailing blank
    rows = [blank_id] + [t for s in spelling for t in (tok2idx[s], tok2idx[s])] + [blank_id]
    emis = np.full((len(rows), n), -20.0, dtype=np.float32)
    for r, idx in enumerate(rows):
        emis[r, idx] = 0.0
    decoder = make_decoder(args.lexicon, tokens, args.lm_path,
                           args.lm_weight, args.word_score, args.beam_size)
    out = decode_words(decoder, [emis])[0]
    print(f"[selftest] tokens={n} blank_id={blank_id} sil_id={tok2idx[SIL_TOKEN]}")
    print(f"[selftest] target words : {' '.join(picks)}")
    print(f"[selftest] decoded words: {out}")
    ok = out.split() == picks
    print(f"[selftest] {'PASS' if ok else 'FAIL'} — token/lexicon/blank alignment")
    if not ok:
        sys.exit(1)


# ---- main ---------------------------------------------------------------------

def parse_runs(items):
    out = []
    for it in items:
        path, _, fac = it.partition(":")
        out.append((path, int(fac)))
    return out


def tune_on_dev(log_probs_list, refs, tokens, args):
    """Grid (lm_weight, word_score) on dev; return best (lm_w, word_s, wer)."""
    best = None
    for lm_w in args.lm_weights:
        for ws in args.word_scores:
            dec = make_decoder(args.lexicon, tokens, args.lm_path, lm_w, ws, args.beam_size)
            wer = compute_wer(refs, decode_words(dec, log_probs_list))
            print(f"    dev: lm_w={lm_w:<4} word_score={ws:<5} -> WER {wer:.4f}", flush=True)
            if best is None or wer < best[2]:
                best = (lm_w, ws, wer)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=None, help="run_dir:factor (default: 3 baselines)")
    p.add_argument("--ckpt-name", default="last.pt")
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--dev-split", default="dev")
    p.add_argument("--phoneme-dict", default="data/tokenizers/phoneme_g2p.dict",
                   help="for the tokenizer vocab only (decode uses --lexicon)")
    p.add_argument("--lexicon", default="data/tokenizers/lexicon_librispeech.txt")
    p.add_argument("--tokens", default="data/tokenizers/phoneme_tokens.txt")
    p.add_argument("--lm-path", default="data/lm.binary")
    p.add_argument("--lm-weight", type=float, default=2.0)
    p.add_argument("--word-score", type=float, default=0.0)
    p.add_argument("--beam-size", type=int, default=100)
    p.add_argument("--lm-weights", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--word-scores", type=float, nargs="+", default=[-1.0, 0.0, 1.0])
    p.add_argument("--no-tune", action="store_true", help="skip dev tuning, use --lm-weight/--word-score")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.selftest:
        run_selftest(args)
        return

    tokens = load_tokens(args.tokens)
    pron2word = build_exact_lexicon(args.lexicon)
    print(f"[p2w-lattice] tokens={len(tokens)} lexicon prons={len(pron2word)}", flush=True)

    runs = parse_runs(args.runs) if args.runs else DEFAULT_RUNS
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tokenizer = build_tokenizer("phoneme", phoneme_dict=args.phoneme_dict)
    vocab_size = tokenizer.vocab_size
    blank_id = tokenizer.blank_index
    symbols = tokenizer.symbols

    summary = []
    for run_dir, factor in runs:
        tag = os.path.basename(run_dir.rstrip("/"))
        print(f"\n##### {tag} ({factor}x, {args.ckpt_name}) #####", flush=True)
        model = load_model(run_dir, factor, args.ckpt_name, vocab_size, device)

        if args.no_tune:
            lm_w, ws = args.lm_weight, args.word_score
        else:
            dev_ds = CachedRawEMGDataset(args.cache_dir, args.dev_split, tokenizer=tokenizer,
                                         downsample_factor=factor)
            dlp, dtex = forward_logits(model, dev_ds, device)
            drefs = [clean_text(t) for t in dtex]
            lm_w, ws, dwer = tune_on_dev(dlp, drefs, tokens, args)
            print(f"  best dev: lm_w={lm_w} word_score={ws} WER {dwer:.4f}", flush=True)

        test_ds = CachedRawEMGDataset(args.cache_dir, args.split, tokenizer=tokenizer,
                                      downsample_factor=factor)
        tlp, ttex = forward_logits(model, test_ds, device)
        trefs = [clean_text(t) for t in ttex]

        dec = make_decoder(args.lexicon, tokens, args.lm_path, lm_w, ws, args.beam_size)
        hyp_lattice = decode_words(dec, tlp)
        wer_lattice = compute_wer(trefs, hyp_lattice)
        hyp_greedy = greedy_exact_words(tlp, blank_id, symbols, pron2word)
        wer_greedy = compute_wer(trefs, hyp_greedy)

        print(f"\n  [{args.split}] N={len(trefs)}")
        print(f"  word WER  greedy+exact-lexicon (no LM) : {wer_greedy:.4f}")
        print(f"  word WER  lattice+lexicon+KenLM (A)    : {wer_lattice:.4f}  "
              f"(lm_w={lm_w}, word_score={ws}, beam={args.beam_size})")
        for r, h in list(zip(trefs, hyp_lattice))[: args.k]:
            print(f"    ref: {r}")
            print(f"    hyp: {h or '<empty>'}")
        summary.append((tag, factor, wer_greedy, wer_lattice))

    print(f"\n===== summary (word WER, {args.split}) =====")
    print(f"{'run':28s} {'factor':>7s} {'greedy+lex':>12s} {'lattice+LM':>12s}")
    for tag, factor, wg, wl in summary:
        print(f"{tag:28s} {factor:>6d}x {wg:>12.4f} {wl:>12.4f}")


if __name__ == "__main__":
    main()
