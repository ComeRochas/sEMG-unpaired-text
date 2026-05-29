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

### A. Tokenization infrastructure (prerequisite)
- [ ] Add subword tokenizer (e.g. SentencePiece/BPE) and phoneme tokenizer (e.g. g2p_en /
      CMUdict) alongside the existing char `TextTransform`. One config knob selects
      resolution; CTC vocab + blank index derived from it.
- [ ] Make `text_int` in the EMG cache resolution-aware (re-encode from `text`, no signal
      recompute needed). Keep KenLM/beam decoding working per resolution (subword/phoneme LMs
      or decode-then-detokenize).

### B. Supervised baselines per resolution
- [ ] Re-train the supervised CTC baseline at char / subword / phoneme resolution.
- [ ] Pick the adapted **token resolution** per case (frames-per-token ratio vs EMG frame
      rate ~86 Hz). Report test WER/CER per resolution. This is the reference each text-UML
      experiment is measured against.

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
