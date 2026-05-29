#!/usr/bin/env python
"""Precompute Gaddy-internal audio (the voiced recordings packaged with the
EMG dataset) into the same cache format as ``scripts/precompute_audio.py``.

Walks the *voiced* session directories
(``voiced_parallel_data`` and ``nonparallel_data``), reads each
``{idx}_audio_clean.flac`` + ``{idx}_info.json``, and writes a single
``<out_dir>/<split>.pt`` payload identical in schema to the LibriSpeech cache:

    audio    : list[Tensor (T_i,)]  — fp16 zero-mean / unit-variance waveforms
    text_int : list[Tensor (L_i,)]  — int64 char-level token ids
    version  : 1

Audio in this dataset is already 16 kHz mono. Silent sessions are skipped on
purpose: their ``audio_clean.flac`` is a separate voiced re-recording but
overlaps with files in ``voiced_parallel_data``, and pulling it in would
introduce duplicate utterances. Utterances with empty / non-letter text
(boundary-of-silence clips with ``sentence_index = -1``) are dropped.

Usage
-----
    python scripts/precompute_audio_gaddy.py \\
        --emg-data-root /scratch/cr4206/data/emg_data/emg_data \\
        --out-dir       /scratch/cr4206/sEMGencoderJEPA/data/gaddy_audio_cache \\
        --split-name    gaddy_internal \\
        --num-workers   16
"""
from __future__ import annotations

import argparse
import json
import os
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Keep BLAS single-threaded so num_workers processes don't oversubscribe cores.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import soundfile as sf
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from semg_jepa.data_utils import TextTransform


TARGET_SR = 16_000
EPS = 1e-7

# Voiced directories only — these contain unique voiced utterances. Silent
# session dirs also ship audio_clean.flac files but they are re-recordings
# of the same parallel-set sentences already present in voiced_parallel_data.
_VOICED_SUBDIRS = ("voiced_parallel_data", "nonparallel_data")


_TT: TextTransform | None = None


def _init_worker() -> None:
    global _TT
    _TT = TextTransform()


def _zero_mean_unit_var(x: np.ndarray) -> np.ndarray:
    """Match ``Wav2Vec2FeatureExtractor._zero_mean_unit_var_norm`` (per-sample)."""
    return (x - x.mean()) / np.sqrt(x.var() + EPS)


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio
    import torchaudio
    t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, TARGET_SR)
    return t.squeeze(0).numpy()


def _process_sample(task):
    audio_path, text = task
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = _resample(audio, sr)
        audio = _zero_mean_unit_var(audio).astype(np.float16)
        try:
            text_int = np.asarray(_TT.text_to_int(text), dtype=np.int64)
        except ValueError:
            return None
        if text_int.shape[0] == 0 or audio.shape[0] < 16:
            return None
        return {"audio": audio, "text_int": text_int}
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{audio_path}: {exc!r}"}


def _has_letter(text: str) -> bool:
    return any(c in string.ascii_letters for c in text)


def build_manifest(emg_data_root: str) -> list[tuple[str, str]]:
    tasks: list[tuple[str, str]] = []
    for sub in _VOICED_SUBDIRS:
        sub_path = os.path.join(emg_data_root, sub)
        if not os.path.isdir(sub_path):
            print(f"[precompute_audio_gaddy] WARN missing dir: {sub_path}", flush=True)
            continue
        for session_dir_name in sorted(os.listdir(sub_path)):
            session_dir = os.path.join(sub_path, session_dir_name)
            if not os.path.isdir(session_dir):
                continue
            for fname in sorted(os.listdir(session_dir)):
                if not fname.endswith("_info.json"):
                    continue
                idx_str = fname[: -len("_info.json")]
                if not idx_str.isdigit():
                    continue
                info_path = os.path.join(session_dir, fname)
                audio_path = os.path.join(session_dir, f"{idx_str}_audio_clean.flac")
                if not os.path.isfile(audio_path):
                    continue
                try:
                    with open(info_path) as fh:
                        info = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue
                # Skip boundary-of-silence clips and empty transcripts.
                if int(info.get("sentence_index", -1)) < 0:
                    continue
                text = info.get("text") or ""
                if not _has_letter(text):
                    continue
                tasks.append((audio_path, text))
    return tasks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--emg-data-root",
        default="/scratch/cr4206/data/emg_data/emg_data",
        help="Root containing voiced_parallel_data/ and nonparallel_data/",
    )
    p.add_argument(
        "--out-dir",
        default="/scratch/cr4206/sEMGencoderJEPA/data/gaddy_audio_cache",
        help="Output directory; writes <split-name>.pt there.",
    )
    p.add_argument(
        "--split-name",
        default="gaddy_internal",
        help="Filename stem for the output cache (default: gaddy_internal).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel CPU workers (default: #cores - 1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print(f"[precompute_audio_gaddy] emg_data_root = {args.emg_data_root}", flush=True)
    print(f"[precompute_audio_gaddy] out_dir       = {args.out_dir}", flush=True)
    print(f"[precompute_audio_gaddy] split_name    = {args.split_name}", flush=True)
    print(f"[precompute_audio_gaddy] num_workers   = {args.num_workers}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[precompute_audio_gaddy] scanning sessions ...", flush=True)
    tasks = build_manifest(args.emg_data_root)
    print(f"[precompute_audio_gaddy] manifest = {len(tasks)} samples", flush=True)
    if not tasks:
        sys.exit("no samples found — check --emg-data-root")

    split_t0 = time.time()
    audio_list: list[torch.Tensor] = []
    text_int_list: list[torch.Tensor] = []
    n_errors = 0
    log_every = max(1, len(tasks) // 40)

    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker) as ex:
        for i, result in enumerate(ex.map(_process_sample, tasks, chunksize=16), start=1):
            if result is None:
                pass
            elif "_error" in result:
                n_errors += 1
                if n_errors <= 5:
                    print(f"  [err] {result['_error']}", flush=True)
            else:
                audio_list.append(torch.from_numpy(result["audio"]))
                text_int_list.append(torch.from_numpy(result["text_int"]))

            if i % log_every == 0 or i == len(tasks):
                elapsed = time.time() - split_t0
                rate = i / max(1e-3, elapsed)
                eta = (len(tasks) - i) / max(rate, 1e-3)
                print(
                    f"  [gaddy] {i}/{len(tasks)} ({100*i/len(tasks):5.1f}%) "
                    f"rate={rate:6.1f} samp/s elapsed={elapsed:6.1f}s eta={eta:6.0f}s",
                    flush=True,
                )

    out_path = os.path.join(args.out_dir, f"{args.split_name}.pt")
    payload = {"audio": audio_list, "text_int": text_int_list, "version": 1}
    print(
        f"[precompute_audio_gaddy] saving {len(audio_list)} samples → {out_path}",
        flush=True,
    )
    torch.save(payload, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    total_seconds = sum(int(a.shape[0]) for a in audio_list) / TARGET_SR
    print(
        f"[precompute_audio_gaddy] wrote {len(audio_list)} samples "
        f"({size_mb:.1f} MB, {total_seconds/3600:.2f} h) "
        f"in {time.time() - split_t0:.1f}s (errors={n_errors})",
        flush=True,
    )
    print(f"[precompute_audio_gaddy] all done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
