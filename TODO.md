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

### B. Supervised baselines per (unit × resolution) — **NEXT (run these)**
- [ ] Train the char baseline at 8× as the reference (`runs/baseline_char_8x`), confirm it
      reproduces phase-1 (~0.31–0.33 test WER).
- [ ] Subword: train `scripts/train_subword.py` (sweep `--vocab-size`, e.g. 250/500/1000),
      then baselines sweeping `conv_strides` (8× vs 16×) to find the resolution that matches
      the coarser unit.
- [ ] Phoneme: install a G2P (`pip install g2p_en`) or drop in a CMUdict, then baseline at
      8× (and finer 4× if phones are dense).
- [ ] Report test WER/CER (PER for phoneme) per (unit, resolution). This is the reference
      each unpaired-text experiment is measured against.
- Open question: does a coarser unit + matched-coarser EMG resolution beat char-8×? (fewer
  CTC frames → less peaky alignment, but a larger softmax and sparser supervision.)

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
