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
    def __init__(self, cache_dir: str, split: str, tokenizer=None, downsample_factor: int = 8,
                 unit_targets: dict | None = None,
                 include_ids: set | None = None, exclude_ids: set | None = None):
        cache_path = Path(cache_dir) / f"{split}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache split file not found: {cache_path}")

        payload = torch.load(cache_path, map_location="cpu")
        self.version = payload.get("version", 0)
        self.metadata = payload.get("metadata", {})
        self.samples = payload["samples"]
        # Optional sample_id filtering: carve a held-out subset out of a split (e.g. a
        # voiced eval set held out of the train split). include wins precedence; both keep
        # the original cache order. Used by train_uml.py's --voiced-eval-ids diagnostic.
        if include_ids is not None:
            include_ids = set(include_ids)
            self.samples = [s for s in self.samples if s["sample_id"] in include_ids]
        if exclude_ids is not None:
            exclude_ids = set(exclude_ids)
            self.samples = [s for s in self.samples if s["sample_id"] not in exclude_ids]
        # Target unit is decoupled from the cache: the raw `text` string is stored,
        # so any tokenizer re-encodes it on the fly (no recache per unit).
        self.tokenizer = tokenizer if tokenizer is not None else CharTokenizer()
        self.text_transform = self.tokenizer  # back-compat alias
        self.downsample_factor = int(downsample_factor)
        # Audio-derived targets (e.g. HuBERT units): {sample_id -> list[int]}. When set,
        # CTC targets are read from here instead of `tokenizer.text_to_int(text)` — the
        # transcript can't produce them (cf. tokenizer.targets_from_audio).
        self.unit_targets = unit_targets
        if unit_targets is not None:
            missing = [s["sample_id"] for s in self.samples if s["sample_id"] not in unit_targets]
            if missing:
                raise KeyError(
                    f"unit_targets missing {len(missing)} of {len(self.samples)} sample_ids "
                    f"(e.g. {missing[0]}). Re-run scripts/precompute_hubert_units.py for split '{split}'."
                )

    def __len__(self):
        return len(self.samples)

    def subset(self, fraction: float, seed: int | None = None):
        """Return a copy keeping a ``fraction`` of the samples.

        ``seed=None`` keeps the original head-slice (deterministic, cache order).
        With an int ``seed`` the kept subset is a deterministic *random* draw (shuffle
        by seed, then take the first ``fraction``) — used by the low-label sweep so a
        given (fraction, seed) always selects the SAME labelled EMG utterances across
        the control and UML arms.
        """
        result = copy(self)
        n_keep = int(fraction * len(self.samples))
        if seed is None:
            result.samples = self.samples[:n_keep]
        else:
            order = list(range(len(self.samples)))
            random.Random(seed).shuffle(order)
            keep = sorted(order[:n_keep])          # keep cache order among the drawn subset
            result.samples = [self.samples[i] for i in keep]
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

        # Targets: either audio-derived units (precomputed, keyed by sample_id) or the
        # transcript re-encoded to the configured unit (char/subword/phoneme) on the fly.
        if self.unit_targets is not None:
            text_int = torch.tensor(self.unit_targets[sample["sample_id"]], dtype=torch.long)
        else:
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
            "sample_id": sample.get("sample_id", ""),
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
