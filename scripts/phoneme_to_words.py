"""Decode a phoneme CTC model to WORDS and report a real word WER.

`pyctcdecode` can't do this: it scores a *word* LM over glued phone text. The
standard route is **lexicon + word-LM decoding** — map each predicted phone group
(between `|` separators) back to words via the pronunciation dictionary, and use the
KenLM word LM to resolve homophones (to/too/two) and pick a fluent sentence.

Pipeline (per utterance):
  1. greedy-collapse the phone CTC output -> phones split on `|` into word groups;
  2. for each group, candidate words = exact lexicon matches, plus the K nearest
     pronunciations by phone edit distance (handles phone errors);
  3. left-to-right beam search scoring  lm_weight*KenLM(word|history)
     - dist_weight*(phone_edit_distance / pron_len);
  4. WER of the decoded word string vs the (cleaned) reference words.

Reports greedy PER, word WER without LM (nearest-pron only), and word WER with LM.

Self-test (no model/GPU):  python scripts/phoneme_to_words.py --selftest
Full eval (GPU):           python scripts/phoneme_to_words.py \
    --checkpoint runs/baseline_phoneme_10x/last.pt --downsample-factor 10
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from semg_jepa.architecture import BaselineCTCModel, factor_to_strides
from semg_jepa.cached_dataset import CachedRawEMGDataset
from semg_jepa.ctc_utils import _greedy_collapse, _make_eval_collate
from semg_jepa.metrics import compute_wer
from semg_jepa.tokenizers import build_tokenizer, clean_text

import kenlm


# ---- lexicon -------------------------------------------------------------------

def build_lexicon(tokenizer):
    """From the phoneme tokenizer's dict (word -> phones), build lookup structures."""
    word2pron = tokenizer._dict  # {word: [phones]} loaded from --phoneme-dict
    if not word2pron:
        raise SystemExit("phoneme tokenizer has no dict; pass --phoneme-dict")
    exact = {}
    by_len = {}
    for word, phones in word2pron.items():
        pron = tuple(phones)
        if not pron:
            continue
        exact.setdefault(pron, [])
        if word not in exact[pron]:
            exact[pron].append(word)
    for pron, words in exact.items():
        by_len.setdefault(len(pron), []).append((pron, words))
    return exact, by_len


def edit_distance(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[lb]


def candidates(group, exact, by_len, max_cands, max_dist):
    """(word, pron_len, phone_dist) candidates for one predicted phone group."""
    out = [(w, len(group), 0) for w in exact.get(group, [])]
    if len(out) < max_cands:
        scored = []
        for L in range(max(1, len(group) - 2), len(group) + 3):
            for pron, words in by_len.get(L, []):
                d = edit_distance(pron, group)
                if d <= max_dist:
                    for w in words:
                        scored.append((d, w, len(pron)))
        scored.sort(key=lambda x: x[0])
        seen = {w for w, _, _ in out}
        for d, w, plen in scored:
            if w not in seen:
                out.append((w, plen, d))
                seen.add(w)
            if len(out) >= max_cands:
                break
    return out[:max_cands]


def decode_utterance(groups, exact, by_len, lm, lm_weight, dist_weight,
                     beam_width, max_cands, max_dist):
    """Lexicon + KenLM beam search over predicted phone groups -> word list."""
    beams = [(0.0, [], None)]  # (score, words, kenlm_state)
    start = kenlm.State()
    lm.BeginSentenceWrite(start)
    beams = [(0.0, [], start)]
    for group in groups:
        if not group:
            continue
        cands = candidates(group, exact, by_len, max_cands, max_dist)
        if not cands:
            continue
        nxt = []
        for score, words, state in beams:
            for word, plen, dist in cands:
                out_state = kenlm.State()
                lm_lp = lm.BaseScore(state, word, out_state) if lm is not None else 0.0
                ns = score + lm_weight * lm_lp - dist_weight * (dist / max(1, plen))
                nxt.append((ns, words + [word], out_state))
        nxt.sort(key=lambda x: -x[0])
        beams = nxt[:beam_width]
    return beams[0][1] if beams else []


# ---- model forward -------------------------------------------------------------

def forward_phones(model, dataset, device, blank_id, symbols):
    """Return (predicted_phone_groups, reference_word_strings) for the split."""
    model.eval()
    collate = _make_eval_collate(dataset.downsample_factor)
    bs = 16 if str(device).startswith("cuda") else 1
    dl = torch.utils.data.DataLoader(dataset, batch_size=bs, collate_fn=collate)
    groups_all, refs = [], []
    with torch.no_grad():
        for raw, seq_lens, texts in dl:
            lp = F.log_softmax(model(raw.to(device)), -1).cpu()
            for i, T in enumerate(seq_lens.tolist()):
                ids = _greedy_collapse(lp[i, :T].numpy().argmax(-1).tolist(), blank_id)
                syms = [symbols[j] for j in ids]
                groups, cur = [], []
                for s in syms:
                    if s == "|":
                        groups.append(tuple(cur)); cur = []
                    else:
                        cur.append(s)
                groups.append(tuple(cur))
                groups_all.append([g for g in groups if g])
                refs.append(clean_text(texts[i]))
    return groups_all, refs


def per_from_groups(groups, tokenizer, refs_text):
    """Phone error rate: predicted phone tokens vs g2p(reference)."""
    ref_ph = [tokenizer.reference_text(t) for t in refs_text]
    hyp_ph = []
    for gs in groups:
        toks = []
        for k, g in enumerate(gs):
            if k > 0:
                toks.append("|")
            toks.extend(g)
        hyp_ph.append(" ".join(toks))
    return compute_wer(ref_ph, hyp_ph)


# ---- main / selftest -----------------------------------------------------------

def run_selftest(args, tokenizer, exact, by_len, lm):
    """No model: g2p reference texts, optionally corrupt phones, check recovery."""
    payload = torch.load(os.path.join(args.cache_dir, f"{args.split}.pt"), map_location="cpu")
    texts = [s["text"] for s in payload["samples"] if clean_text(s["text"]).strip()][: args.n_selftest]
    rng = random.Random(0)
    phones = list(tokenizer.symbols[:-1])
    refs, groups = [], []
    for t in texts:
        syms = tokenizer._encode_symbols(t)  # phones with '|'
        gs, cur = [], []
        for s in syms:
            if s == "|":
                gs.append(tuple(cur)); cur = []
            else:
                cur.append(s)
        gs.append(tuple(cur))
        # corrupt ~15% of phones to mimic the model's PER
        gs2 = []
        for g in gs:
            g = list(g)
            for k in range(len(g)):
                if rng.random() < args.corrupt:
                    g[k] = rng.choice(phones)
            gs2.append(tuple(g))
        groups.append([g for g in gs2 if g])
        refs.append(clean_text(t))
    report(args, groups, refs, tokenizer, exact, by_len, lm, tag=f"SELFTEST (corrupt={args.corrupt})")


def report(args, groups, refs, tokenizer, exact, by_len, lm, tag):
    per = per_from_groups(groups, tokenizer, refs)
    hyp_nolm = [" ".join(decode_utterance(gs, exact, by_len, lm, 0.0, args.dist_weight,
                                           args.beam_width, args.max_cands, args.max_dist))
                for gs in groups]
    hyp_lm = [" ".join(decode_utterance(gs, exact, by_len, lm, args.lm_weight, args.dist_weight,
                                        args.beam_width, args.max_cands, args.max_dist))
              for gs in groups]
    print(f"\n===== {tag} | N={len(refs)} =====")
    print(f"greedy PER                 : {per:.3f}")
    print(f"word WER (lexicon, no LM)   : {compute_wer(refs, hyp_nolm):.3f}")
    print(f"word WER (lexicon + KenLM)  : {compute_wer(refs, hyp_lm):.3f}  "
          f"(lm_w={args.lm_weight}, dist_w={args.dist_weight})")
    for r, h in list(zip(refs, hyp_lm))[: args.k]:
        print(f"  ref: {r}")
        print(f"  hyp: {h}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/baseline_phoneme_10x/last.pt")
    p.add_argument("--downsample-factor", type=int, default=10)
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--phoneme-dict", default="data/tokenizers/phoneme_g2p.dict")
    p.add_argument("--lm-path", default="data/lm.binary")
    p.add_argument("--lm-weight", type=float, default=0.5)
    p.add_argument("--dist-weight", type=float, default=8.0)
    p.add_argument("--beam-width", type=int, default=16)
    p.add_argument("--max-cands", type=int, default=6)
    p.add_argument("--max-dist", type=int, default=3)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--corrupt", type=float, default=0.15)
    p.add_argument("--n-selftest", type=int, default=40)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    tokenizer = build_tokenizer("phoneme", phoneme_dict=args.phoneme_dict)
    exact, by_len = build_lexicon(tokenizer)
    lm = kenlm.Model(args.lm_path)
    print(f"[p2w] lexicon: {len(exact)} unique prons; LM loaded", flush=True)

    if args.selftest:
        run_selftest(args, tokenizer, exact, by_len, lm)
        return

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    conv_strides = factor_to_strides(args.downsample_factor)
    model = BaselineCTCModel(vocab_size=tokenizer.vocab_size, conv_strides=conv_strides).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    ds = CachedRawEMGDataset(args.cache_dir, args.split, tokenizer=tokenizer,
                             downsample_factor=args.downsample_factor)
    print(f"[p2w] forwarding {len(ds)} {args.split} utts on {device} ...", flush=True)
    groups, refs = forward_phones(model, ds, device, tokenizer.blank_index, tokenizer.symbols)
    report(args, groups, refs, tokenizer, exact, by_len, lm,
           tag=f"{os.path.basename(os.path.dirname(args.checkpoint))} [{args.split}]")


if __name__ == "__main__":
    main()
