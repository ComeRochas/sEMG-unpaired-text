from __future__ import annotations

import random
from copy import copy
from pathlib import Path

import torch

from .tokenizers import CharTokenizer


def build_batches(dataset: "CachedRawEMGDataset", max_len: int) -> list[list[int]]:
    """Build size-aware batches from a CachedRawEMGDataset.

    Groups examples so total raw_emg samples per batch <= max_len.
    Returns a shuffled list of index lists (one list per batch).
    Budgets by raw EMG length (``8 * cached ctc_length``), which is independent
    of the encoder's downsample factor.
    """
    lengths = [8 * s["ctc_length"] for s in dataset.samples]
    indices = list(range(len(lengths)))
    random.shuffle(indices)
    batches: list[list[int]] = []
    batch: list[int] = []
    batch_len = 0
    for i in indices:
        if lengths[i] > max_len:
            continue
        if batch_len + lengths[i] > max_len and batch:
            batches.append(batch)
            batch, batch_len = [], 0
        batch.append(i)
        batch_len += lengths[i]
    if batch:
        batches.append(batch)
    return batches


class CachedRawEMGDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str, split: str, tokenizer=None, downsample_factor: int = 8):
        cache_path = Path(cache_dir) / f"{split}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache split file not found: {cache_path}")

        payload = torch.load(cache_path, map_location="cpu")
        self.version = payload.get("version", 0)
        self.metadata = payload.get("metadata", {})
        self.samples = payload["samples"]
        # Target unit is decoupled from the cache: the raw `text` string is stored,
        # so any tokenizer re-encodes it on the fly (no recache per unit).
        self.tokenizer = tokenizer if tokenizer is not None else CharTokenizer()
        self.text_transform = self.tokenizer  # back-compat alias
        self.downsample_factor = int(downsample_factor)

    def __len__(self):
        return len(self.samples)

    def subset(self, fraction: float):
        result = copy(self)
        result.samples = self.samples[: int(fraction * len(self.samples))]
        return result

    def __getitem__(self, i):
        sample = self.samples[i]
        raw = sample["raw_emg"]
        if not isinstance(raw, torch.Tensor):
            raw = torch.tensor(raw)
        raw = raw.to(torch.float32)

        assert raw.ndim == 2 and raw.size(1) == 8, f"Expected [F*T,8], got {tuple(raw.shape)}"
        f = self.downsample_factor
        # Right-crop so the raw length is a multiple of the downsample factor (drops
        # < f trailing samples, i.e. < f/689 s). Lets any token resolution be used.
        t = raw.size(0) // f  # encoder output frames at this token resolution
        if raw.size(0) != t * f:
            raw = raw[: t * f]

        # Re-encode the transcript to the configured unit (char/subword/phoneme).
        text_int = torch.tensor(self.tokenizer.text_to_int(sample["text"]), dtype=torch.long)

        session_index = int(sample.get("session_index", sample.get("session_id", 0)))
        session_ids = torch.full((t,), session_index, dtype=torch.long)

        return {
            "raw_emg": raw,
            "text": sample["text"],
            "text_int": text_int,
            "session_ids": session_ids,
            "silent": bool(sample.get("silent", False)),
            "book_location": tuple(sample.get("book_location", ("", -1))),
            "length": t,
        }

    @staticmethod
    def collate_raw(batch):
        return {
            "raw_emg": [ex["raw_emg"] for ex in batch],
            "lengths": [ex["length"] for ex in batch],
            "session_ids": [ex["session_ids"] for ex in batch],
            "silent": [ex["silent"] for ex in batch],
            "text_int": [ex["text_int"] for ex in batch],
            "text_int_lengths": [ex["text_int"].shape[0] for ex in batch],
            "text": [ex["text"] for ex in batch],
            "book_location": [ex["book_location"] for ex in batch],
        }
