"""Unpaired-text corpus reader for the UML **text branch** (phase-2 §C).

Mirrors ``uml/audio_dataset.py``: reads a precomputed cache produced by
``scripts/precompute_text.py`` and yields CTC training batches. Where the audio
branch maps a waveform → clean text, the text branch maps a **corrupted,
upsampled** character stream → clean text (a denoising-CTC objective), so the
shared Transformer learns English character-sequence structure without a trivial
identity map.

Cache file ``<cache_dir>/<source>.pt`` is a dict:
    text_int : list[LongTensor (L_i,)]  — clean char-level token ids (CharTokenizer)
    version  : 1

The denoising corruption + jittered upsample happen in ``collate_fn`` (fresh
noise every epoch). Per batch it returns padded tensors + lengths, matching the
key layout the training loop's ``_text_step`` expects:
    text_input        : LongTensor (B, T_max)   — corrupted + upsampled input ids
    text_input_lengths: LongTensor (B,)         — true per-sample input frames
    text_int          : LongTensor (B, L_max)   — clean CTC target ids
    text_int_lengths  : LongTensor (B,)

Input ids are in ``[0, vocab_size]``: ``vocab_size`` is the input-side ``MASK``
token (embedded by :class:`uml.model.TextFrontend`), distinct from the CTC blank
(also ``vocab_size``) which only ever appears in output logit space.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TextCorpusDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        source: str = "libri",
        *,
        vocab_size: int = 37,
        p_mask: float = 0.15,
        p_sub: float = 0.10,
        p_del: float = 0.05,
        mask_span_mean: float = 3.0,
        upsample: int = 3,
        jitter: int = 1,
    ):
        cache_path = Path(cache_dir) / f"{source}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Text cache not found: {cache_path}\n"
                f"Run scripts/precompute_text.py --text-source {source} first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        self.text_int_list: list[torch.Tensor] = payload["text_int"]
        self.version = int(payload.get("version", 1))

        self.vocab_size = int(vocab_size)
        self.mask_id = int(vocab_size)           # input-side MASK (embedding has vocab+1 rows)
        self.p_mask = float(p_mask)
        self.p_sub = float(p_sub)
        self.p_del = float(p_del)
        self.mask_span_mean = float(mask_span_mean)
        self.upsample = int(upsample)
        self.jitter = int(jitter)

    def __len__(self) -> int:
        return len(self.text_int_list)

    def __getitem__(self, idx: int) -> dict:
        return {"clean": self.text_int_list[idx].long()}

    # ------------------------------------------------------------------
    # Denoising corruption + jittered upsample (the text-branch "augment")
    # ------------------------------------------------------------------

    def _corrupt_and_upsample(self, clean: torch.Tensor) -> torch.Tensor:
        c = clean.tolist()
        L = len(c)
        V = self.vocab_size

        # 1) Span-mask / substitute / delete on the clean id list.
        ids: list[int] = []
        i = 0
        p_span = 1.0 / max(self.mask_span_mean, 1.0)
        while i < L:
            r = random.random()
            if r < self.p_mask:
                # geometric span length (mean ~ mask_span_mean), all -> MASK
                span = 1
                while random.random() > p_span:
                    span += 1
                for _ in range(span):
                    if i < L:
                        ids.append(self.mask_id)
                        i += 1
                continue
            if r < self.p_mask + self.p_del:
                i += 1                                   # delete (target keeps it)
                continue
            if r < self.p_mask + self.p_del + self.p_sub and V > 1:
                sub = random.randrange(V - 1)            # substitute a *different* char
                if sub >= c[i]:
                    sub += 1
                ids.append(sub)
                i += 1
                continue
            ids.append(int(c[i]))                        # keep
            i += 1

        if not ids:                                      # safety: never empty
            ids = [self.mask_id]

        # 2) Upsample each surviving token with per-token jitter so length and
        #    boundaries resemble the EMG frontend's ~3-4 frames/char.
        up: list[int] = []
        for tok in ids:
            rep = max(1, self.upsample + random.randint(-self.jitter, self.jitter))
            up.extend([tok] * rep)

        # 3) Guarantee the CTC constraint input_len >= target_len (deletions could
        #    otherwise push T below L for a short/heavily-deleted sample).
        if len(up) < L:
            up.extend([self.mask_id] * (L - len(up)))

        return torch.tensor(up, dtype=torch.long)

    def collate_fn(self, batch: list[dict]) -> dict:
        clean_list = [b["clean"] for b in batch]
        input_list, input_lengths = [], []
        for clean in clean_list:
            ids = self._corrupt_and_upsample(clean)
            assert ids.shape[0] >= clean.shape[0], "CTC input_len < target_len"
            input_list.append(ids)
            input_lengths.append(ids.shape[0])

        input_padded = pad_sequence(input_list, batch_first=True, padding_value=self.mask_id)
        text_padded = pad_sequence(clean_list, batch_first=True, padding_value=0)
        return {
            "text_input": input_padded,                                          # (B, T_max)
            "text_input_lengths": torch.tensor(input_lengths, dtype=torch.long), # (B,)
            "text_int": text_padded,                                             # (B, L_max)
            "text_int_lengths": torch.tensor(                                    # (B,)
                [c.shape[0] for c in clean_list], dtype=torch.long),
        }
