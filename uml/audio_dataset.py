"""LibriSpeech cache reader for the UML audio branch.

Reads the precomputed cache produced by ``scripts/precompute_audio.py``.

Per-sample dict:
    audio_features   : FloatTensor (T_audio,)  — normalized fp32 waveform
    text_int         : LongTensor  (L,)
    text_int_lengths : int

The cache file at ``<cache_dir>/<split>.pt`` is a dict with keys:
    audio    : list[Tensor (T_i,)]  — fp16 zero-mean / unit-variance waveforms
    text_int : list[Tensor (L_i,)]  — int64 char-level token ids (TextTransform)
    version  : 1
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class LibriSpeechCharDataset(Dataset):
    def __init__(self, cache_dir: str, split: str = "train-clean-100",
                 unit_targets_path: str | None = None):
        cache_path = Path(cache_dir) / f"{split}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"LibriSpeech cache not found: {cache_path}\n"
                f"Run scripts/precompute_audio.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        self.audio_list: list[torch.Tensor] = payload["audio"]
        self.text_int_list: list[torch.Tensor] = payload["text_int"]
        assert len(self.audio_list) == len(self.text_int_list)
        self.version = int(payload.get("version", 1))
        # UML with HuBERT-unit targets: the audio branch predicts km100 units (same target
        # space as the EMG branch), not chars. Units are precomputed from THESE waveforms
        # (scripts/precompute_audio_hubert_units.py) so they align 1:1 by list index.
        if unit_targets_path is not None:
            up = torch.load(unit_targets_path, map_location="cpu")["units"]
            if len(up) != len(self.audio_list):
                raise ValueError(
                    f"audio unit targets ({len(up)}) misaligned with cache audio "
                    f"({len(self.audio_list)}) — re-run precompute_audio_hubert_units.py for {cache_path}"
                )
            self.text_int_list = [torch.as_tensor(u, dtype=torch.long) for u in up]

    def __len__(self) -> int:
        return len(self.audio_list)

    def __getitem__(self, idx: int) -> dict:
        audio = self.audio_list[idx]
        text_int = self.text_int_list[idx]
        return {
            "audio_features": audio.float(),
            "text_int": text_int.long(),
            "text_int_lengths": int(text_int.shape[0]),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        audio_list = [b["audio_features"] for b in batch]
        text_list = [b["text_int"] for b in batch]
        audio_lengths = torch.tensor([a.shape[0] for a in audio_list], dtype=torch.long)
        text_lengths = torch.tensor([b["text_int_lengths"] for b in batch], dtype=torch.long)
        audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
        text_padded = pad_sequence(text_list, batch_first=True, padding_value=0)
        return {
            "audio_features": audio_padded,    # (B, T_audio_max)
            "audio_lengths": audio_lengths,    # (B,)
            "text_int": text_padded,           # (B, L_max)
            "text_int_lengths": text_lengths,  # (B,)
        }
