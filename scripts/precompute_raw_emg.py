#!/usr/bin/env python
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(var, "1")


def _build_config(args):
    if args.data_config:
        with open(args.data_config) as f:
            return json.load(f)

    if not args.emg_data_dir:
        raise ValueError("Provide either --data-config or --emg-data-dir")

    base = Path(args.emg_data_dir)
    if not args.testset_file:
        raise ValueError("When using --emg-data-dir you must also provide --testset-file")

    return {
        "testset_file": args.testset_file,
        "silent_data_directories": [str(base / "silent_parallel_data")],
        "voiced_data_directories": [str(base / "voiced_parallel_data"), str(base / "nonparallel_data")],
        "remove_channels": args.remove_channels,
    }


def _task_from_example(example):
    directory_info, idx = example
    return {
        "directory": directory_info.directory,
        "idx": int(idx),
        "session_index": int(directory_info.session_index),
        "silent": bool(directory_info.silent),
    }


def _worker(task, remove_channels):
    import torch

    from semg_jepa.data_utils import TextTransform
    from semg_jepa.read_emg import load_utterance

    text_transform = TextTransform()
    utt = load_utterance(task["directory"], task["idx"], remove_channels=remove_channels)

    raw = torch.from_numpy(utt["raw_emg"]).to(torch.float16)
    text_int = torch.tensor(text_transform.text_to_int(utt["text"]), dtype=torch.long)

    ctc_length = int(utt["ctc_length"])
    assert raw.shape[0] == 8 * ctc_length

    return {
        "raw_emg": raw,
        "text": utt["text"],
        "text_int": text_int,
        "ctc_length": ctc_length,
        "session_index": task["session_index"],
        "book_location": utt["book_location"],
        "silent": task["silent"],
        "sample_id": f"{task['directory']}::{task['idx']}",
        "source_path": task["directory"],
        "index": task["idx"],
        "version": 1,
    }


def _precompute_split(config, split, cache_dir, num_workers):
    import torch

    from semg_jepa.read_emg import EMGDataset

    is_dev = split == "dev"
    is_test = split == "test"
    dataset = EMGDataset(config, dev=is_dev, test=is_test)
    tasks = [_task_from_example(ex) for ex in dataset.example_indices]

    out_path = Path(cache_dir) / f"{split}.pt"
    print(f"[{split}] samples={len(tasks)} | out={out_path}", flush=True)

    start = time.time()
    samples = [None] * len(tasks)
    errors = 0

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_worker, task, config.get("remove_channels", [])): i for i, task in enumerate(tasks)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            done += 1
            try:
                samples[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[{split}] ERROR at idx={idx}: {exc}", flush=True)

            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                elapsed = time.time() - start
                sps = done / max(elapsed, 1e-6)
                eta = (len(tasks) - done) / max(sps, 1e-6)
                print(f"[{split}] {done}/{len(tasks)} | {sps:.2f} samples/s | ETA {eta/60:.2f} min | errors {errors}", flush=True)

    samples = [s for s in samples if s is not None]
    payload = {
        "version": 1,
        "split": split,
        "metadata": {
            "preprocessing": {
                "notch_harmonics_hz": [60, 120, 180, 240, 300, 360, 420],
                "highpass_hz": 2,
                "subsample_hz": 689.06,
                "crop_rule": "raw = emg_orig[8:8+8*T], T=floor((len(emg_orig)-8)/8)",
                "normalization": "raw=raw/20; raw=50*tanh(raw/50)",
                "dtype_on_disk": "float16",
            },
            "remove_channels": config.get("remove_channels", []),
            "num_errors": errors,
        },
        "samples": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    elapsed = time.time() - start
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[{split}] done in {elapsed/60:.2f} min | saved={out_path} | size={size_mb:.2f} MB | errors={errors}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Precompute and cache raw-EMG tensors for train/dev/test splits.")
    p.add_argument("--data-config", default=None)
    p.add_argument("--emg-data-dir", default=None)
    p.add_argument("--testset-file", default=None)
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMGencoderJEPA/data",
                   help="Directory where precomputed .pt files will be saved")
    p.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--remove-channels", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    config = _build_config(args)
    print(f"[precompute] cache_dir={args.cache_dir} num_workers={args.num_workers} data_source={args.data_config or args.emg_data_dir}", flush=True)
    for split in ["train", "dev", "test"]:
        _precompute_split(config, split, args.cache_dir, args.num_workers)


if __name__ == "__main__":
    main()
