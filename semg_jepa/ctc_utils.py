import multiprocessing
import os
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

# Import kenlm BEFORE pyctcdecode and inject it into pyctcdecode's namespace.
# pyctcdecode does `try: import kenlm except: ...` at module load — on compute
# nodes that import sometimes fails silently (different system libs from the
# login node), leaving `kenlm` undefined inside pyctcdecode. We force-inject it
# here so `kenlm.Model(...)` works downstream.
try:
    import kenlm
except ImportError:
    kenlm = None

import pyctcdecode.decoder as _pyctc_decoder
from pyctcdecode import build_ctcdecoder

if kenlm is not None and not hasattr(_pyctc_decoder, "kenlm"):
    _pyctc_decoder.kenlm = kenlm

from semg_jepa.metrics import compute_cer, compute_wer
from semg_jepa.unigrams import build_unigrams


def load_unigrams(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


_lm_available = None
_cached_unigrams = {}


def _check_lm_available(lm_path):
    """Check if KenLM can be loaded (cached result).

    Probes via `kenlm.Model(...)` directly rather than building a dummy
    pyctcdecode decoder — the latter emits spurious "no unigrams provided" /
    "space token missing" warnings unrelated to the actual decode.
    """
    global _lm_available
    if _lm_available is not None:
        return _lm_available

    if kenlm is None or not Path(lm_path).exists():
        _lm_available = False
        return False

    try:
        kenlm.Model(str(lm_path))
        _lm_available = True
        print("[ctc_utils] KenLM loaded successfully", flush=True)
        return True
    except Exception as e:
        _lm_available = False
        print(f"[ctc_utils] KenLM unavailable ({type(e).__name__}: {e}); using beam search without LM", flush=True)
        return False


def _ensure_unigrams(lm_path, unigrams_path):
    """Return cached unigram list, building from LibriSpeech if the file is missing."""
    if "unigrams" in _cached_unigrams:
        return _cached_unigrams["unigrams"]

    unigrams_path = Path(unigrams_path)
    if not unigrams_path.exists():
        if not Path(lm_path).exists():
            return None
        print(f"[ctc_utils] unigrams missing at {unigrams_path}; building from LibriSpeech vocab", flush=True)
        build_unigrams(lm_path, unigrams_path)

    _cached_unigrams["unigrams"] = load_unigrams(unigrams_path)
    return _cached_unigrams["unigrams"]


def build_decoder(tokenizer, lm_path="data/lm.binary", unigrams_path="data/unigrams.txt",
                  alpha=1.5, beta=1.85):
    labels = list(tokenizer.labels) + [""]   # blank last, aligned with logits

    kwargs = {}
    # The KenLM word LM + unigrams only make sense for word-decodable units
    # (char, subword). Phoneme/other units beam-search without an LM.
    if tokenizer.supports_word_lm and _check_lm_available(lm_path):
        kwargs["kenlm_model_path"] = lm_path
        kwargs["alpha"] = alpha
        kwargs["beta"] = beta
        unigrams = _ensure_unigrams(lm_path, unigrams_path)
        if unigrams is not None:
            kwargs["unigrams"] = unigrams

    return build_ctcdecoder(labels, **kwargs)


def _make_eval_collate(downsample_factor):
    def _collate_eval(batch):
        raw_list = [ex["raw_emg"] for ex in batch]
        seq_lens = torch.tensor([r.shape[0] // downsample_factor for r in raw_list])
        raw_padded = torch.nn.utils.rnn.pad_sequence(raw_list, batch_first=True)
        texts = [ex["text"] for ex in batch]
        return raw_padded, seq_lens, texts
    return _collate_eval


def _greedy_collapse(int_seq, blank_id):
    out = []
    prev = -1
    for p in int_seq:
        if p != prev and p != blank_id:
            out.append(p)
        prev = p
    return out


def compute_log_probs(model, dataset, device, batch_size=None):
    """Single batched forward pass. Returns (log_probs_list, references)."""
    model.eval()
    on_gpu = str(device).startswith("cuda")
    bs = batch_size if batch_size is not None else (16 if on_gpu else 1)
    collate = _make_eval_collate(getattr(dataset, "downsample_factor", 8))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, collate_fn=collate)

    log_probs_list, references = [], []
    with torch.no_grad():
        for raw_padded, seq_lens, texts in tqdm.tqdm(dataloader, "Forward", disable=None):
            raw_padded = raw_padded.to(device)
            lp = F.log_softmax(model(raw_padded), -1).cpu()
            for i, T in enumerate(seq_lens.tolist()):
                log_probs_list.append(lp[i, :T].numpy().astype(np.float32))
                references.append(dataset.tokenizer.reference_text(texts[i]))
    model.train()
    return log_probs_list, references


def _decode_beam(log_probs_list, decoder, beam_width, num_workers=None):
    n_workers = min(num_workers or os.cpu_count() or 4, 16)
    with multiprocessing.pool.Pool(n_workers) as pool:
        return decoder.decode_batch(pool, log_probs_list, beam_width=beam_width)


def evaluate(model, dataset, device, method="greedy", batch_size=None,
             beam_width=100, alpha=1.5, beta=1.85,
             lm_path="data/lm.binary", unigrams_path="data/unigrams.txt",
             num_workers=None):
    """Return (wer, cer).

    method="greedy": batched GPU argmax + CTC collapse. Fast (~1s).
    method="beam":   batched GPU forward + parallel pyctcdecode beam search.
                     num_workers controls pool size (default: all CPUs, max 16).
    """
    if method not in ("greedy", "beam"):
        raise ValueError(f"unknown eval method: {method}")

    log_probs_list, references = compute_log_probs(model, dataset, device, batch_size)

    if method == "greedy":
        blank_id = dataset.tokenizer.blank_index
        predictions = [
            dataset.tokenizer.int_to_text(_greedy_collapse(lp.argmax(-1).tolist(), blank_id))
            for lp in log_probs_list
        ]
    else:
        decoder = build_decoder(dataset.tokenizer,
                                lm_path=lm_path, unigrams_path=unigrams_path,
                                alpha=alpha, beta=beta)
        try:
            predictions = _decode_beam(log_probs_list, decoder, beam_width, num_workers)
        except ValueError:
            # pyctcdecode raises "max() arg is an empty sequence" when its pruning leaves
            # no beams — happens on near-uniform logits (untrained model early in training,
            # especially at high downsample factors with few frames). Fall back to greedy
            # for this eval so training doesn't crash.
            blank_id = dataset.tokenizer.blank_index
            predictions = [
                dataset.tokenizer.int_to_text(_greedy_collapse(lp.argmax(-1).tolist(), blank_id))
                for lp in log_probs_list
            ]

    return compute_wer(references, predictions), compute_cer(references, predictions)


def grid_search(model, dataset, device, beam_widths, alphas, betas,
                lm_path="data/lm.binary", unigrams_path="data/unigrams.txt",
                batch_size=None, num_workers=None):
    """Run beam search over a grid of (beam_width, alpha, beta).

    Forward pass runs once; only decoders are rebuilt per combo. Returns a list
    of (beam_width, alpha, beta, wer, cer) sorted by ascending WER.
    """
    log_probs_list, references = compute_log_probs(model, dataset, device, batch_size)

    results = []
    combos = list(product(alphas, betas, beam_widths))
    for alpha, beta, bw in tqdm.tqdm(combos, "GridSearch", disable=None):
        decoder = build_decoder(dataset.tokenizer,
                                lm_path=lm_path, unigrams_path=unigrams_path,
                                alpha=alpha, beta=beta)
        preds = _decode_beam(log_probs_list, decoder, bw, num_workers)
        wer = compute_wer(references, preds)
        cer = compute_cer(references, preds)
        results.append((bw, alpha, beta, wer, cer))
    results.sort(key=lambda r: r[3])
    return results
