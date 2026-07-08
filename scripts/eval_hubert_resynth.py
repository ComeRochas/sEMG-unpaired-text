"""Real word-WER for an EMG->HuBERT-units model, via re-synthesis + ASR.

HuBERT units are not text, so a word-WER needs the units turned back into speech and
transcribed. Pipeline (3 stages, 2 envs + the HuBERTVoc container — orchestrated by
``slurm/eval_hubert_resynth.slurm``):

  A. ``--stage predict`` (project env): run the EMG branch over a split, greedy-collapse
     the CTC output -> predicted unit sequence per utterance. Writes ``pred.units`` and
     ``gold.units`` (HuBERTVoc ``uttid|u u u`` format) + ``refs.json`` (uttid -> clean text),
     and reports the **UER** (units, no vocoder needed).
  B. vocode (HuBERTVoc container): ``pred.units``/``gold.units`` -> wavs, via the repo's own
     ``scripts/hubert_pipeline.py unit2speech``. No new vocoder code.
  C. ``--stage asr`` (project env): Whisper-transcribe the wavs -> text -> word-WER (jiwer)
     vs the gold transcript. Reports EMG->units WER and the **topline** (gold units -> wav ->
     ASR), which bounds what's achievable through this vocoder+ASR — the honest anchor next
     to the char-16x baseline (test WER 0.296).

Stage A loads the EMG-branch checkpoint (``best_emg_branch.pt`` from ``train_uml.py``, keys
``encoder.*``/``ctc_head.*``) into a plain ``BaselineCTCModel`` — no UML second-branch config
needed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def _fmt_units_line(uttid: str, units: list[int]) -> str:
    # HuBERTVoc's loader splits on "|" and parses space-separated ints; a non-empty unit
    # list is required for the vocoder, so fall back to a single unit for empty predictions.
    if not units:
        units = [0]
    return f"{uttid}|{' '.join(str(int(u)) for u in units)}"


# ----------------------------------------------------------------------------------------
# Stage A: EMG -> predicted units
# ----------------------------------------------------------------------------------------
def stage_predict(args):
    from semg_jepa.architecture import BaselineCTCModel, factor_to_strides
    from semg_jepa.cached_dataset import CachedRawEMGDataset
    from semg_jepa.ctc_utils import _greedy_collapse, _make_eval_collate
    from semg_jepa.metrics import compute_wer
    from semg_jepa.tokenizers import CharTokenizer, HubertUnitTokenizer

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    factor = args.downsample_factor
    tokenizer = HubertUnitTokenizer(k=args.hubert_k)

    gold_units = torch.load(Path(args.hubert_units_dir) / f"{args.split}.pt",
                            map_location="cpu")["units"]
    dataset = CachedRawEMGDataset(args.cache_dir, args.split, tokenizer=tokenizer,
                                  downsample_factor=factor, unit_targets=gold_units)
    sample_ids = [s["sample_id"] for s in dataset.samples]
    clean = CharTokenizer().clean_text
    gold_text = [clean(s["text"]) for s in dataset.samples]

    model = BaselineCTCModel(
        vocab_size=args.hubert_k, conv_strides=factor_to_strides(factor),
        model_size=args.model_size, num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    ckpt = os.path.join(args.run_dir, args.ckpt)
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    print(f"[predict] loaded {ckpt} (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    model.eval()

    # Sequential forward; predictions align 1:1 with dataset order.
    collate = _make_eval_collate(factor)
    bs = 16 if device == "cuda" else 1
    dl = torch.utils.data.DataLoader(dataset, batch_size=bs, collate_fn=collate)
    pred_units: list[list[int]] = []
    with torch.no_grad():
        for raw, seq_lens, _texts in dl:
            lp = F.log_softmax(model(raw.to(device)), -1).cpu()
            for i, T in enumerate(seq_lens.tolist()):
                ids = _greedy_collapse(lp[i, :T].numpy().argmax(-1).tolist(), args.hubert_k)
                pred_units.append(ids)

    assert len(pred_units) == len(sample_ids), (len(pred_units), len(sample_ids))

    # UER (no vocoder): WER over space-joined unit ids.
    refs = [tokenizer.int_to_text(gold_units[sid]) for sid in sample_ids]
    hyps = [tokenizer.int_to_text(u) for u in pred_units]
    uer = compute_wer(refs, hyps)
    n_empty = sum(1 for u in pred_units if not u)

    out = Path(args.out_dir) / args.split
    out.mkdir(parents=True, exist_ok=True)
    uttids = [f"{i:05d}" for i in range(len(sample_ids))]
    with open(out / "pred.units", "w") as f:
        for uid, u in zip(uttids, pred_units):
            f.write(_fmt_units_line(uid, u) + "\n")
    with open(out / "gold.units", "w") as f:
        for uid, sid in zip(uttids, sample_ids):
            f.write(_fmt_units_line(uid, gold_units[sid]) + "\n")
    with open(out / "refs.json", "w") as f:
        json.dump({uid: txt for uid, txt in zip(uttids, gold_text)}, f)
    with open(out / "metrics_predict.json", "w") as f:
        json.dump({"split": args.split, "uer": uer, "n": len(sample_ids),
                   "n_empty_pred": n_empty,
                   "mean_pred_units": sum(len(u) for u in pred_units) / len(pred_units),
                   "mean_gold_units": sum(len(gold_units[s]) for s in sample_ids) / len(sample_ids)},
                  f, indent=2)
    print(f"[predict] split={args.split} n={len(sample_ids)} UER={uer:.4f} "
          f"empty_preds={n_empty} -> {out}", flush=True)


# ----------------------------------------------------------------------------------------
# Stage C: wavs -> Whisper -> word-WER
# ----------------------------------------------------------------------------------------
def _transcribe_dir(wav_dir: Path, refs: dict, asr, clean) -> tuple[list[str], list[str]]:
    import soundfile as sf
    import torchaudio.functional as AF

    ref_list, hyp_list = [], []
    for uid in sorted(refs.keys()):
        wav_path = wav_dir / f"{uid}.wav"
        if not wav_path.exists():
            print(f"[asr] WARN missing wav {wav_path}", flush=True)
            continue
        wav, sr = sf.read(str(wav_path))
        if wav.ndim == 2:
            wav = wav.mean(-1)
        wav = torch.as_tensor(wav, dtype=torch.float32)
        if sr != 16000:
            wav = AF.resample(wav, sr, 16000)
        text = asr({"raw": wav.numpy(), "sampling_rate": 16000})["text"]
        ref_list.append(refs[uid])
        hyp_list.append(clean(text))
    return ref_list, hyp_list


def stage_asr(args):
    from transformers import pipeline

    from semg_jepa.metrics import compute_wer
    from semg_jepa.tokenizers import CharTokenizer

    device = 0 if torch.cuda.is_available() and not args.cpu else -1
    clean = CharTokenizer().clean_text
    out = Path(args.out_dir) / args.split
    refs = json.load(open(out / "refs.json"))

    asr = pipeline("automatic-speech-recognition", model=args.whisper_model, device=device)
    print(f"[asr] model={args.whisper_model} device={device} n={len(refs)}", flush=True)

    results = {}
    for which, sub in (("pred", "pred_wavs"), ("gold", "gold_wavs")):
        wav_dir = out / sub
        if not wav_dir.exists():
            print(f"[asr] SKIP {which}: {wav_dir} not found (run stage B vocoding first)", flush=True)
            continue
        ref_list, hyp_list = _transcribe_dir(wav_dir, refs, asr, clean)
        wer = compute_wer(ref_list, hyp_list)
        results[which] = {"wer": wer, "n": len(ref_list)}
        print(f"[asr] {which:4s} WER={wer:.4f} (n={len(ref_list)})", flush=True)

    pm = out / "metrics_predict.json"
    uer = json.load(open(pm))["uer"] if pm.exists() else None
    with open(out / "metrics_resynth.json", "w") as f:
        json.dump({"split": args.split, "uer": uer, "results": results,
                   "char16x_baseline_wer": 0.296}, f, indent=2)
    print("\n==== HuBERT-unit re-synthesis WER ====")
    print(f"split={args.split}")
    if uer is not None:
        print(f"  UER (units)            : {uer:.4f}")
    if "pred" in results:
        print(f"  WER EMG->units->ASR    : {results['pred']['wer']:.4f}")
    if "gold" in results:
        print(f"  WER topline (gold units): {results['gold']['wer']:.4f}  <- ceiling (vocoder+ASR)")
    print(f"  char-16x baseline (ref) : 0.296  (direct text-WER, not directly comparable)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["predict", "asr"], required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", default="/scratch/cr4206/sEMG-unpaired-text/runs/hubert_resynth")
    p.add_argument("--cpu", action="store_true")
    # stage predict
    p.add_argument("--run-dir", default=None, help="Dir with the EMG-branch checkpoint.")
    p.add_argument("--ckpt", default="best_emg_branch.pt")
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data")
    p.add_argument("--hubert-units-dir", default="/scratch/cr4206/sEMG-unpaired-text/data/hubert_units")
    p.add_argument("--hubert-k", type=int, default=100)
    p.add_argument("--downsample-factor", type=int, default=8)
    p.add_argument("--model-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    # stage asr
    p.add_argument("--whisper-model", default="openai/whisper-small")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "predict":
        if not args.run_dir:
            raise SystemExit("--run-dir is required for --stage predict")
        stage_predict(args)
    else:
        stage_asr(args)


if __name__ == "__main__":
    main()
