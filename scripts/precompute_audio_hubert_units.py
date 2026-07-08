"""Precompute HuBERT km100 units for the UML AUDIO branch, aligned to an existing audio cache.

For a real UML run with HuBERT-unit targets, the audio second branch must predict the SAME
target space as the EMG branch (km100 units), not characters. The audio cache
(``scripts/precompute_audio.py`` / ``precompute_audio_gaddy.py``) stores normalized 16 kHz
waveforms in a fixed order; this script extracts km100 units per waveform with the same
reader+kmeans+dedup as the EMG-side ``precompute_hubert_units.py``, so the audio branch's CTC
target is unit ids 1:1-aligned (by list index) with the cached waveforms.

Key detail — no normalization mismatch: HuBERT's ``get_feats`` applies ``F.layer_norm(x, x.shape)``
(full-sequence zero-mean/unit-variance) when ``cfg.normalize``, which is exactly the
normalization the audio cache already applied. So feeding the cached waveform straight into the
model reproduces the file-based pipeline (the extra layer_norm is idempotent on normalized input).

Output ``{out_dir}/{split}.pt`` = ``{"version", "metadata", "units": list[list[int]]}`` aligned to
the cache's ``audio`` list. Runs INSIDE the HuBERTVoc Singularity overlay (fairseq) — see
``slurm/precompute_audio_hubert_units.slurm``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Reuse the checkpoint loader + run-length dedup from the EMG-side precompute.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from precompute_hubert_units import load_speech2unit, dedup


def feats_from_wav(reader, wav, max_chunk: int = 1600000):
    """HuBERT layer-`reader.layer` features for an in-memory waveform — mirrors
    ``HubertFeatureReader.get_feats`` but skips the file read (we already have the waveform)."""
    x = torch.as_tensor(wav).float()
    if reader.use_cuda:
        x = x.cuda()
    if reader.task.cfg.normalize:
        x = F.layer_norm(x, x.shape)           # idempotent on already-normalized cache audio
    x = x.view(1, -1)
    feat = []
    with torch.no_grad():
        for start in range(0, x.size(1), max_chunk):
            x_chunk = x[:, start:start + max_chunk]
            feat_chunk, _ = reader.model.extract_features(
                source=x_chunk, padding_mask=None, mask=False, output_layer=reader.layer,
            )
            feat.append(feat_chunk)
    return torch.cat(feat, 1).squeeze(0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-cache-dir", required=True,
                   help="Dir holding the audio cache <split>.pt (audio + text_int lists).")
    p.add_argument("--split", required=True, help="Cache split name, e.g. train-clean-100 or gaddy_internal.")
    p.add_argument("--out-dir", default="/scratch/cr4206/sEMG-unpaired-text/data/audio_hubert_units")
    p.add_argument("--hubertvoc-root", default="/scratch/th3482/HuBERTVoc")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--no-dedup", action="store_true", help="Keep raw 50 Hz units (no collapse).")
    return p.parse_args()


def main():
    args = parse_args()
    do_dedup = not args.no_dedup
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] HuBERT reader + km100 from {args.hubertvoc_root} (layer {args.layer})", flush=True)
    reader, kmeans = load_speech2unit(args.hubertvoc_root, args.layer)

    cache_path = Path(args.audio_cache_dir) / f"{args.split}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"audio cache not found: {cache_path}")
    print(f"[load] audio cache {cache_path}", flush=True)
    audio_list = torch.load(cache_path, map_location="cpu")["audio"]
    n = len(audio_list)
    print(f"[{args.split}] {n} waveforms", flush=True)

    units_list: list[list[int]] = []
    lengths, n_infeasible = [], 0
    t0 = time.perf_counter()
    for i, wav in enumerate(audio_list):
        feat = feats_from_wav(reader, wav.numpy()).cpu().numpy()      # (T_frames, 768) @ 50 Hz
        u = [int(x) for x in kmeans.predict(feat)]
        if do_dedup:
            u = dedup(u)
        units_list.append(u)
        lengths.append(len(u))
        # wav2vec2-base frontend (the audio branch encoder) emits ~ (len-400)//320+1 frames.
        audio_frames = max(1, (len(wav) - 400) // 320 + 1)
        n_infeasible += int(len(u) > audio_frames)
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"[{args.split}] {i+1}/{n} | {rate:.1f} utt/s", flush=True)

    meta = {
        "model": "hubert_base_ls960", "layer": args.layer, "k": 100, "dedup": do_dedup,
        "source_cache": str(cache_path), "n_samples": n,
    }
    out_path = Path(args.out_dir) / f"{args.split}.pt"
    torch.save({"version": 1, "metadata": meta, "units": units_list}, out_path)
    mean_len = sum(lengths) / len(lengths)
    print(
        f"[{args.split}] wrote {n} -> {out_path} | mean_units={mean_len:.1f} "
        f"min={min(lengths)} max={max(lengths)} | "
        f"CTC-infeasible (vs wav2vec2 frames): {n_infeasible}/{n} ({100*n_infeasible/n:.1f}%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
