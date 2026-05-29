from __future__ import annotations

from typing import Iterable

import jiwer


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def compute_wer(references: Iterable[str], predictions: Iterable[str]) -> float:
    return jiwer.wer(list(references), list(predictions))


def compute_cer(references: Iterable[str], predictions: Iterable[str]) -> float:
    refs = list(references)
    preds = list(predictions)
    if hasattr(jiwer, "cer"):
        return jiwer.cer(refs, preds)

    total_dist = 0
    total_chars = 0
    for ref, pred in zip(refs, preds):
        total_dist += _levenshtein_distance(ref, pred)
        total_chars += len(ref)
    return total_dist / max(1, total_chars)


def compute_text_metrics(references: Iterable[str], predictions: Iterable[str]) -> dict[str, float]:
    refs = list(references)
    preds = list(predictions)
    return {
        "wer": compute_wer(refs, preds),
        "cer": compute_cer(refs, preds),
    }
