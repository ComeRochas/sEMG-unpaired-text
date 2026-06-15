"""Summarise the token-resolution sweep for a team presentation.

Discovers `runs/baseline_<tag>_<F>x` checkpoints, evaluates each on the test set with
the unit-appropriate decoder, and prints (and saves) a markdown report:

  - a table of TEST scores per run (frame length, frames/unit, dev-from-log, test WER/CER);
  - example reconstructions (reference vs hypothesis) for the best run of each unit;
  - a headline pick.

Decoder per unit (matches how each model is meant to be read):
  - char / subword : best.pt + beam search + KenLM word LM  -> word WER/CER
  - phoneme        : last.pt + greedy                       -> phone error rate (PER)
    (phoneme best.pt was selected by a broken beam metric; beam glues phones with no
     spaces, so we score phonemes with greedy, which renders via int_to_text.)

Usage (GPU recommended; --cpu works but slower):
    PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text \
    python scripts/analyze_sweep_results.py --split test --k 6 --out runs/sweep_summary.md
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import OrderedDict

import numpy as np
import torch

from semg_jepa.architecture import BaselineCTCModel, factor_to_strides
from semg_jepa.cached_dataset import CachedRawEMGDataset
from semg_jepa.ctc_utils import (_decode_beam, _greedy_collapse, build_decoder,
                                  compute_log_probs)
from semg_jepa.metrics import compute_cer, compute_wer
from semg_jepa.tokenizers import build_tokenizer

SAMPLE_RATE = 689.06
RUN_RE = re.compile(r"^baseline_(?P<tag>.+)_(?P<f>\d+)x$")
CHAR8X_REF_TEST = 0.315  # phase-1 reference (char, 8x), beam+KenLM


def tag_to_unit(tag, subword_dir, phoneme_dict):
    """Map a run tag (char / phoneme / subword_<N>) to build_tokenizer kwargs."""
    if tag == "char":
        return "char", {}
    if tag == "phoneme":
        return "phoneme", {"phoneme_dict": phoneme_dict}
    m = re.match(r"subword_(\d+)", tag)
    if m:
        return "subword", {"subword_model": os.path.join(subword_dir, f"subword_{m.group(1)}.model")}
    raise ValueError(f"unknown run tag: {tag}")


def parse_sweep_logs(logs_dir):
    """Return {(tag, factor): {'dev_wer','dev_cer','done'}} from sweep_*.out logs."""
    info = {}
    for f in sorted(glob.glob(os.path.join(logs_dir, "sweep_*.out"))):
        tag = None
        cur = None
        for line in open(f, errors="ignore"):
            m = re.search(r"unit=\S+ tag=(\S+) factors=", line)
            if m:
                tag = m.group(1)
            m = re.search(r"===== factor=(\d+)x", line)
            if m:
                cur = (tag, int(m.group(1)))
                info.setdefault(cur, {"dev_wer": math.inf, "dev_cer": math.inf, "done": False})
            m = re.search(r"factor=(\d+)x done", line)
            if m and tag is not None:
                info.setdefault((tag, int(m.group(1))), {"dev_wer": math.inf, "dev_cer": math.inf, "done": False})["done"] = True
            m = re.search(r"epoch=\d+/\d+ .*dev_wer=([\d.]+) dev_cer=([\d.]+)", line)
            if m and cur is not None:
                info[cur]["dev_wer"] = min(info[cur]["dev_wer"], float(m.group(1)))
                info[cur]["dev_cer"] = min(info[cur]["dev_cer"], float(m.group(2)))
    return info


_ntok_cache = {}


def frames_per_unit(tokenizer, factor, raw_lens, texts, key):
    """Median (#CTC frames / #tokens) over train utterances at this factor."""
    if key not in _ntok_cache:
        _ntok_cache[key] = np.array([len(tokenizer.text_to_int(t)) for t in texts], dtype=np.float64)
    n = _ntok_cache[key]
    mask = n > 0
    frames = np.floor(raw_lens[mask] / factor)
    return float(np.median(frames / n[mask]))


def refs_and_preds(model, dataset, device, method, beam_kw):
    """One forward pass -> (references, predictions) for the whole split."""
    lps, refs = compute_log_probs(model, dataset, device)
    if method == "greedy":
        blank = dataset.tokenizer.blank_index
        preds = [dataset.tokenizer.int_to_text(_greedy_collapse(lp.argmax(-1).tolist(), blank))
                 for lp in lps]
    else:
        bw = beam_kw["beam_width"]
        dec_kw = {k: v for k, v in beam_kw.items() if k != "beam_width"}  # build_decoder has no beam_width
        decoder = build_decoder(dataset.tokenizer, **dec_kw)
        try:
            preds = _decode_beam(lps, decoder, bw)
        except ValueError:
            # pyctcdecode returns no beams on near-uniform logits (e.g. the failed/nan
            # char-25x checkpoint) -> "max() arg is an empty sequence". Fall back to greedy
            # so one bad run doesn't kill the whole table.
            blank = dataset.tokenizer.blank_index
            preds = [dataset.tokenizer.int_to_text(_greedy_collapse(lp.argmax(-1).tolist(), blank))
                     for lp in lps]
    return refs, preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--subword-dir", default="data/tokenizers")
    p.add_argument("--phoneme-dict", default="data/tokenizers/phoneme_g2p.dict")
    p.add_argument("--lm-path", default="data/lm.binary")
    p.add_argument("--unigrams-path", default="data/unigrams.txt")
    p.add_argument("--beam-width", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.90)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--k", type=int, default=6, help="reconstruction examples per shown model")
    p.add_argument("--only", default=None, help="substring filter on run dir (smoke test)")
    p.add_argument("--out", default="runs/sweep_summary.md")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    beam_kw = dict(lm_path=args.lm_path, unigrams_path=args.unigrams_path,
                   alpha=args.alpha, beta=args.beta, beam_width=args.beam_width)
    log_info = parse_sweep_logs(args.logs_dir)

    # Train durations for the frames/unit column (raw_lens are unit-independent).
    train = torch.load(os.path.join(args.cache_dir, "train.pt"), map_location="cpu")["samples"]
    raw_lens = np.array([s["raw_emg"].shape[0] for s in train], dtype=np.float64)
    texts = [s["text"] for s in train]

    runs = []
    for d in sorted(glob.glob(os.path.join(args.runs_dir, "baseline_*"))):
        m = RUN_RE.match(os.path.basename(d))
        if not m:
            continue
        if args.only and args.only not in d:
            continue
        tag, factor = m.group("tag"), int(m.group("f"))
        unit, tok_kw = tag_to_unit(tag, args.subword_dir, args.phoneme_dict)
        # phoneme: best.pt was mis-selected by the broken beam metric -> use last.pt + greedy.
        if unit == "phoneme":
            ckpt_name, method = "last.pt", "greedy"
        else:
            ckpt_name, method = "best.pt", "beam"
        ckpt = os.path.join(d, ckpt_name)
        if not os.path.exists(ckpt):
            alt = os.path.join(d, "last.pt" if ckpt_name == "best.pt" else "best.pt")
            if not os.path.exists(alt):
                continue
            ckpt, ckpt_name = alt, os.path.basename(alt)
        runs.append(dict(dir=d, tag=tag, unit=unit, factor=factor, tok_kw=tok_kw,
                         ckpt=ckpt, ckpt_name=ckpt_name, method=method))

    print(f"[analyze] {len(runs)} runs on '{args.split}' (device={device})", flush=True)

    results = []
    examples = {}
    for r in runs:
        tok = build_tokenizer(r["unit"], **r["tok_kw"])
        conv_strides = factor_to_strides(r["factor"])
        model = BaselineCTCModel(vocab_size=tok.vocab_size, conv_strides=conv_strides).to(device)
        model.load_state_dict(torch.load(r["ckpt"], map_location=device), strict=False)
        ds = CachedRawEMGDataset(args.cache_dir, args.split, tokenizer=tok, downsample_factor=r["factor"])
        print(f"[analyze] {os.path.basename(r['dir'])}: {r['ckpt_name']} / {r['method']} ...", flush=True)
        refs, preds = refs_and_preds(model, ds, device, r["method"], beam_kw)
        wer, cer = compute_wer(refs, preds), compute_cer(refs, preds)

        key = r["tag"]
        fpu = frames_per_unit(tok, r["factor"], raw_lens, texts, key)
        li = log_info.get((r["tag"], r["factor"]), {})
        r.update(dict(
            test_wer=wer, test_cer=cer,
            frame_ms=1000.0 * r["factor"] / SAMPLE_RATE, fpu=fpu,
            dev_wer=li.get("dev_wer", math.inf), done=li.get("done", None),
            metric="PER(greedy)" if r["unit"] == "phoneme" else "WER(beam+LM)",
        ))
        results.append(r)
        examples[r["dir"]] = list(zip(refs, preds))[: args.k]

    render(results, examples, args)


def render(results, examples, args):
    lines = []
    w = lines.append
    w(f"# Token-resolution sweep — {args.split} results\n")
    w("Decoder per unit: **char/subword** = word WER via beam + KenLM word LM; "
      "**phoneme** = phone error rate (PER) via greedy (no LM). "
      "**WER and PER are not directly comparable.** "
      f"Phase-1 reference: char-8x test WER = {CHAR8X_REF_TEST:.3f}.\n")

    w("| unit | F | frame (ms) | frames/unit | dev (log) | **test** | metric | ckpt | done |")
    w("|---|---|---|---|---|---|---|---|---|")
    order = {"char": 0, "phoneme": 1, "subword_250": 2, "subword_500": 3, "subword_1000": 4}
    for r in sorted(results, key=lambda x: (order.get(x["tag"], 9), x["factor"])):
        dev = "—" if (r["unit"] == "phoneme" or not math.isfinite(r["dev_wer"])) else f"{r['dev_wer']:.3f}"
        beats = " ⭐" if (r["metric"].startswith("WER") and r["test_wer"] < CHAR8X_REF_TEST) else ""
        ref = " (ref)" if (r["tag"] == "char" and r["factor"] == 8) else ""
        done = {True: "✓", False: "partial", None: "?"}[r["done"]]
        w(f"| {r['tag']}{ref} | {r['factor']}x | {r['frame_ms']:.1f} | {r['fpu']:.1f} | {dev} | "
          f"**{r['test_wer']:.3f}**{beats} / {r['test_cer']:.3f} | {r['metric']} | {r['ckpt_name']} | {done} |")
    w("")

    # Best per unit + headline (word units only for the WER headline).
    by_unit = {}
    for r in results:
        by_unit.setdefault(r["unit"], []).append(r)
    w("## Best per unit\n")
    word_best = None
    for unit, rs in by_unit.items():
        b = min(rs, key=lambda x: x["test_wer"])
        metric = "PER" if unit == "phoneme" else "WER"
        w(f"- **{unit}**: best {metric} = {b['test_wer']:.3f} at {b['factor']}x "
          f"({b['frame_ms']:.0f} ms/frame, {b['fpu']:.1f} frames/unit)")
        if unit != "phoneme" and (word_best is None or b["test_wer"] < word_best["test_wer"]):
            word_best = b
    if word_best:
        delta = CHAR8X_REF_TEST - word_best["test_wer"]
        w(f"\n**Headline (word WER):** best decodable unit = **{word_best['tag']} at "
          f"{word_best['factor']}x** ({word_best['frame_ms']:.0f} ms/frame) — test WER "
          f"{word_best['test_wer']:.3f} vs char-8x {CHAR8X_REF_TEST:.3f} "
          f"({'+' if delta < 0 else '−'}{abs(delta):.3f}).")
    w("")

    # Reconstructions for the best run of each unit (+ char-8x reference).
    w("## Example reconstructions (reference → hypothesis)\n")
    shown = {min(rs, key=lambda x: x["test_wer"])["dir"] for rs in by_unit.values()}
    shown |= {r["dir"] for r in results if r["tag"] == "char" and r["factor"] == 8}
    for r in sorted(results, key=lambda x: (order.get(x["tag"], 9), x["factor"])):
        if r["dir"] not in shown:
            continue
        w(f"### {r['tag']} {r['factor']}x  ({r['metric']}, test={r['test_wer']:.3f})")
        for ref, hyp in examples[r["dir"]]:
            w(f"- ref: `{ref}`")
            w(f"  hyp: `{hyp}`")
        w("")

    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[analyze] wrote {args.out}")


if __name__ == "__main__":
    main()
