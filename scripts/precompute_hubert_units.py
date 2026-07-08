"""Precompute HuBERT km100 discrete units for every EMG utterance.

A HuBERT unit target is a function of the *audio*, not the transcript, so it can't be
re-encoded from the cached ``text`` like char/subword/phoneme. This script extracts, for
each EMG cache sample, the discrete units of its co-located parallel voiced recording
(``{source_path}/{index}_audio_clean.flac``, present for both silent and voiced sessions)
and stores them keyed by ``sample_id`` — the dataset reads them via ``unit_targets``.

Units = HuBERT ``hubert_base_ls960`` layer-6 features -> km100 cluster ids, then
**run-length deduplicated** (consecutive repeats merged). Dedup (a) shortens the target so
CTC stays feasible and (b) matches the GSLM vocoder, which collapses repeats anyway; CTC's
own greedy collapse also yields a deduplicated sequence, so train and synth are consistent.

This reuses the HuBERTVoc package (k-means + fairseq HuBERT reader), which lives in a
Singularity overlay env. Run it inside that container — see
``slurm/precompute_hubert_units.slurm``. The script imports only torch + the HuBERTVoc
``gslm`` package (no ``semg_jepa`` deps), so it runs cleanly in the torch2 overlay env.

Output: ``{out_dir}/{split}.pt`` = ``{"version", "metadata", "units": {sample_id: [int,...]}}``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# --- HuBERTVoc reuse -------------------------------------------------------------------
HUBERTVOC_ROOT_DEFAULT = "/scratch/th3482/HuBERTVoc"


def load_speech2unit(hubertvoc_root: str, layer: int):
    """Build the (fairseq HuBERT reader, k-means) pair from the HuBERTVoc checkpoints."""
    root = Path(hubertvoc_root)
    sys.path.insert(0, str(root))                      # so `import gslm...` resolves
    sys.path.insert(0, str(root / "gslm" / "unit2speech"))

    import logging
    for name in ("fairseq", "fairseq.tasks.hubert_pretraining", "fairseq.models.hubert.hubert"):
        logging.getLogger(name).setLevel(logging.ERROR)

    import joblib
    from gslm.speech2unit.pretrained.utils import get_feature_reader

    s2u_ckpt = root / "pretrained_gslm" / "speech2units" / "hubert_base_ls960.pt"
    km_path = root / "pretrained_gslm" / "speech2units" / "kmeans" / "hubert_km_100.bin"
    for p in (s2u_ckpt, km_path):
        if not p.exists():
            raise FileNotFoundError(p)

    reader = get_feature_reader("hubert")(checkpoint_path=str(s2u_ckpt), layer=layer)
    kmeans = joblib.load(open(km_path, "rb"))
    kmeans.verbose = False
    return reader, kmeans


def dedup(units: list[int]) -> list[int]:
    """Run-length collapse: drop consecutive repeats (matches GSLM collapse_code)."""
    out: list[int] = []
    prev = None
    for u in units:
        if u != prev:
            out.append(u)
            prev = u
    return out


def extract_units(reader, kmeans, audio_path: str, do_dedup: bool) -> list[int]:
    feats = reader.get_feats(audio_path).cpu().numpy()        # (T_frames, 768) @ 50 Hz
    units = [int(x) for x in kmeans.predict(feats)]
    return dedup(units) if do_dedup else units


def build_voiced_index(voiced_root: str, audio_suffix: str) -> dict:
    """Map (book, sentence_index) -> voiced `_audio_clean.flac` path, scanning the raw
    voiced_parallel_data folder. The EMG cache stores book_location=(book, sentence_index);
    we resolve a silent sample's spoken-content audio through that key. Single-speaker data,
    so any voiced take of the sentence is a valid target; we keep the first (sorted) for
    determinism and skip empty/uncatalogued takes (book='' / sentence_index=-1).
    """
    import glob
    import json
    index: dict = {}
    for info in sorted(glob.glob(os.path.join(voiced_root, "*", "*_info.json"))):
        try:
            meta = json.load(open(info))
        except Exception:
            continue
        book, sidx = meta.get("book", ""), meta.get("sentence_index", -1)
        if not book or sidx is None or sidx < 0:
            continue
        key = (book, sidx)
        if key in index:
            continue  # keep first occurrence
        d = os.path.dirname(info)
        i = os.path.basename(info).split("_")[0]
        ap = os.path.join(d, f"{i}{audio_suffix}")
        if os.path.exists(ap):
            index[key] = ap
    return index


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data",
                   help="Dir with the EMG cache {split}.pt files.")
    p.add_argument("--out-dir", default="/scratch/cr4206/sEMG-unpaired-text/data/hubert_units")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--hubertvoc-root", default=HUBERTVOC_ROOT_DEFAULT)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--audio-suffix", default="_audio_clean.flac",
                   help="Audio file = {source_path}/{index}{audio_suffix}.")
    p.add_argument("--silent-target", choices=["self", "voiced"], default="self",
                   help="Source audio for SILENT samples. 'self' (default, legacy): the "
                        "silent recording's own near-silent mic audio -> units are noise. "
                        "'voiced': the parallel VOICED take of the same sentence (matched by "
                        "book_location) -> units encode the spoken content (the fix). Voiced "
                        "and nonparallel samples always use their own audio.")
    p.add_argument("--voiced-root",
                   default="/scratch/cr4206/data/emg_data/emg_data/voiced_parallel_data",
                   help="Dir scanned for voiced parallels when --silent-target voiced.")
    p.add_argument("--no-dedup", action="store_true", help="Keep raw 50 Hz units (no collapse).")
    p.add_argument("--allow-missing", action="store_true",
                   help="Skip (warn) samples whose audio file is missing instead of erroring.")
    p.add_argument("--check-factor", type=int, default=8,
                   help="Report CTC-feasibility: fraction of utts with len(units) > frames@factor.")
    return p.parse_args()


def main():
    args = parse_args()
    do_dedup = not args.no_dedup
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] HuBERT reader + km100 from {args.hubertvoc_root} (layer {args.layer})", flush=True)
    reader, kmeans = load_speech2unit(args.hubertvoc_root, args.layer)

    voiced_index = None
    if args.silent_target == "voiced":
        voiced_index = build_voiced_index(args.voiced_root, args.audio_suffix)
        print(f"[voiced] indexed {len(voiced_index)} voiced parallels from {args.voiced_root}", flush=True)

    for split in args.splits:
        cache_path = Path(args.cache_dir) / f"{split}.pt"
        if not cache_path.exists():
            print(f"[{split}] SKIP (no cache at {cache_path})", flush=True)
            continue
        samples = torch.load(cache_path, map_location="cpu")["samples"]

        units_by_id: dict[str, list[int]] = {}
        lengths, frames_at_factor, n_infeasible, missing = [], [], 0, 0
        n_rerouted = 0
        t0 = time.perf_counter()
        for i, s in enumerate(samples):
            # SILENT samples optionally target their parallel VOICED take (spoken content)
            # instead of their own near-silent mic audio (whose units are noise).
            if voiced_index is not None and bool(s.get("silent", False)):
                key = tuple(s["book_location"])
                audio_path = voiced_index.get(key)
                if audio_path is None:
                    missing += 1
                    if args.allow_missing:
                        print(f"[{split}] WARN no voiced parallel for {key}", flush=True)
                        continue
                    raise FileNotFoundError(f"[{split}] no voiced parallel for book_location={key}")
                n_rerouted += 1
            else:
                audio_path = os.path.join(s["source_path"], f"{s['index']}{args.audio_suffix}")
                if not os.path.exists(audio_path):
                    missing += 1
                    if args.allow_missing:
                        print(f"[{split}] WARN missing audio: {audio_path}", flush=True)
                        continue
                    raise FileNotFoundError(f"[{split}] audio not found: {audio_path}")

            u = extract_units(reader, kmeans, audio_path, do_dedup)
            units_by_id[s["sample_id"]] = u

            lengths.append(len(u))
            frames = (8 * int(s["ctc_length"])) // args.check_factor   # raw_len // factor
            frames_at_factor.append(frames)
            n_infeasible += int(len(u) > frames)

            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.perf_counter() - t0)
                print(f"[{split}] {i+1}/{len(samples)} | {rate:.1f} utt/s", flush=True)

        meta = {
            "model": "hubert_base_ls960", "layer": args.layer, "k": 100,
            "dedup": do_dedup, "audio_suffix": args.audio_suffix,
            "n_samples": len(units_by_id), "n_missing": missing,
            "silent_target": args.silent_target, "n_silent_rerouted": n_rerouted,
        }
        out_path = Path(args.out_dir) / f"{split}.pt"
        torch.save({"version": 1, "metadata": meta, "units": units_by_id}, out_path)

        if lengths:
            mean_len = sum(lengths) / len(lengths)
            print(
                f"[{split}] wrote {len(units_by_id)} -> {out_path} | "
                f"silent_target={args.silent_target} rerouted={n_rerouted} | "
                f"mean_units={mean_len:.1f} min={min(lengths)} max={max(lengths)} | "
                f"CTC-infeasible @{args.check_factor}x: {n_infeasible}/{len(lengths)} "
                f"({100*n_infeasible/len(lengths):.1f}%) | missing={missing}",
                flush=True,
            )


if __name__ == "__main__":
    main()
