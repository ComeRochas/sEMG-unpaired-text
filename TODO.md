# Phase 2 roadmap — unpaired text + token resolution

> Phase 1 (baseline / JEPA / UML-audio / SSL) is complete and archived in `sEMGencoderJEPA`
> at tag `v1-uml-ssl-archive`. Best phase-1 result: **test WER 0.287** (UML, Gaddy-internal
> audio, λ=0.3, + EMG-only finetune) vs 0.325 supervised baseline. JEPA and unsupervised SSL
> gave no transfer at full labels. This repo carries over the reusable core only.

## Hypotheses

- The shared-transformer UML recipe wins because the second branch injects **external
  linguistic structure**. Phase-1 used audio for this; **text** is a cheaper, more abundant
  source of the same structure. Does a text branch match or beat the audio branch?
- The character vocabulary may be suboptimal for CTC on EMG. **Subwords / phonemes** change
  the target temporal resolution; the right resolution may improve WER and/or the
  unpaired-text transfer.

## Workstreams

### A. Tokenization + resolution infrastructure — **DONE**
- [x] Pluggable target-unit tokenizer (`semg_jepa/tokenizers.py`): `char` (port of the
      phase-1 `TextTransform`), `subword` (SentencePiece, train with
      `scripts/train_subword.py`), `phoneme` (ARPAbet + `|`; needs a G2P backend). Selected
      by `--unit`; vocab + blank index + decode/reference rendering all derive from it.
- [x] Cache stays unit-agnostic: `CachedRawEMGDataset` re-encodes `text_int` from the stored
      raw `text` per tokenizer — **no recache per unit**.
- [x] Configurable EMG token resolution: `GaddyRawEMGEncoder(conv_strides=...)` →
      downsample factor `prod(conv_strides)`; dataset right-crops raw to a multiple of it.
      Threaded through `train_baseline.py` + `evaluate.py` (`--unit`, `--conv-strides`).
- [x] Beam decode is unit-aware: KenLM word LM applies for char/subword (SentencePiece `▁`
      handled by `pyctcdecode`); phoneme beams without an LM (reported metric = phone error
      rate). Greedy works for all units.
- [x] Token-resolution analysis (`scripts/analyze_unit_durations.py`): per-unit duration
      (char/subword/phoneme) vs EMG token period for each downsample factor, with CTC
      feasibility — the tool used to pick the matched resolution per unit.
- [x] Non-power-of-2 resolutions: `factor_to_strides()` decomposes any factor dividing
      `fixed_raw_len=1600` (2^a·5^b: 8,10,16,20,25,32,40,50) into conv strides (stride-5
      ResBlocks widen the kernel); `--downsample-factor` flag + `DOWNSAMPLE_FACTOR` env.
- [x] Phoneme G2P: `PhonemeTokenizer` CMUdict loader handles nltk's variant-index format;
      `scripts/build_phoneme_dict.py` precomputes a lossless OOV-covering dict via `g2p_en`
      (`data/tokenizers/phoneme_g2p.dict`) so training uses fast lookups (no per-epoch G2P).
- [x] Sweep launchers: `slurm/sweep_resolution.sh` (one parallel job per factor) and
      `slurm/sweep_unit.slurm` (all factors sequentially in one allocation).

### B. Supervised baselines per (unit × resolution) — **IN PROGRESS**

Matched-resolution method (`analyze_unit_durations.py`, train split): set the EMG token
period (`factor / 689.06 s`) a few × below the per-unit duration. The *densest* utterance is
the hard CTC floor (frames ≥ labels); the *median* sets the target (~7–8 frames/unit, char's
proven sweet spot). Per-unit medians and the matched factor band:

| unit | median ms/unit | matched factor(s) (~7–8 frames/unit) |
|---|---|---|
| char | 87 | **8×** (done); 10× being checked |
| subword-250 | 197 | 16–20× |
| subword-500 | 241 | 20–25× |
| subword-1000 | 289 | 25–32× |
| phoneme | 105 | 8–10× |

- [x] **char @ 8× reference** (`runs/baseline_char_8x`): **test WER 0.315 / CER 0.145**
      (beam+KenLM), dev WER 0.326 (best @ epoch 167/200, ~4 h). Reproduces *and beats* the
      phase-1 0.325 supervised baseline. (Greedy was test 0.434 — the KenLM beam does heavy
      lifting on EMG, so the headline is always the beam number.)
- [ ] Resolution sweeps running (one `train_baseline` per factor, 200 ep, schedule
      `[125,150,175]`; → `runs/baseline_<tag>_<F>x`), as of 2026-06-03, matched factor first:
      - char `[10 16 20 25]` — job 10156693 (h100, running)
      - subword_250 `[16 20 25 32 40]` — job 10156694 (h100, running)
      - subword_500 `[20 25 32]` — job 10162648 (a100, queued)
      - subword_1000 `[25 32 20]` — job 10162649 (h200, queued)
      - phoneme `[10 8 16]` (g2p dict) — job 10162650 (a100, queued)
      (subword/phoneme moved off the saturated h100 queue to a100/h200 for overnight turnaround.)
- [ ] Collect test WER/CER (PER for phoneme) per (unit, factor) with `evaluate.py` once
      checkpoints land → build the (unit × resolution) table. This is the reference each
      unpaired-text experiment (C) is measured against.
- Open question: does a coarser unit + matched-coarser EMG resolution beat char-8× (0.315)?
  (fewer CTC frames → less peaky alignment, but a larger softmax and sparser supervision.)

#### Potential follow-up: phoneme → words (only if PER is low enough)
The phoneme baseline's reported metric is a **phone error rate (PER)** — there is no phoneme
LM, so its beam runs LM-free while char/subword beams get the KenLM word LM. If the phoneme
PER comes out competitive, convert phone hypotheses back to **words** to get a comparable (and
possibly better) WER: beam-decode phones against a **pronunciation lexicon + word LM** (a
phoneme→word FST, e.g. inverting `phoneme_g2p.dict`), or a small phoneme→grapheme model. That
would let the denser-supervision phoneme unit borrow the same linguistic prior the char/subword
beams already use. Go/no-go from the PER numbers; skip if PER is not competitive with char.

### C. Text as the unpaired modality
- [ ] Build a `TextFrontend` (text → embedding sequence feeding the shared transformer) and
      a text dataset reader, mirroring `uml/model.py::AudioFrontend` and
      `uml/audio_dataset.py`. Reuse the shared-transformer plumbing in `UMLModel`.
- [ ] Two corpora, compared head-to-head:
      - large generic English text corpus;
      - the transcripts shipped with the Gaddy audio (in-distribution text).
- [ ] Train EMG+text UML per chosen resolution, then EMG-only finetune
      (`finetune_from_uml.py`). Compare to the per-resolution supervised baseline (B) and to
      the phase-1 audio-UML 0.287.
- Carry over phase-1 lessons: `clip_grad_norm` disabled by default; `share_ctc_head` off;
  λ≈0.3 (down-weighted second branch) was best for both audio sources after finetune —
  start there for text.

### D. Unsupervised setting
- [ ] Unsupervised objective consuming un-transcribed text/EMG on the shared transformer
      (phase-1 SSL on audio gave no gain at ~100 h; revisit only if a text signal looks
      promising). Decide go/no-go by an honest CTC finetune, not a linear probe (the phase-1
      probe was misleading).

## Notes carried from phase 1
- Eval defaults: split=test, method=beam, KenLM `data/lm.binary` + leakage-free unigrams.
- UML inference uses the EMG branch only; the second branch is a training-time auxiliary.
- Validation during training is EMG-only on `dev`.
