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

### B. Supervised baselines per (unit × resolution) — **DONE → chosen: characters @ 16×**

Matched-resolution method (`analyze_unit_durations.py`, train split): set the EMG token
period (`factor / 689.06 s`) a few × below the per-unit duration. The *densest* utterance is
the hard CTC floor (frames ≥ labels); the *median* sets the target (~7–8 frames/unit, char's
proven sweet spot). Per-unit medians and the matched factor band:

| unit | median ms/unit | matched factor(s) (~7–8 frames/unit) |
|---|---|---|
| char | 87 | optimum 16–20× (see results below) |
| subword-250 | 197 | 16–20× |
| subword-500 | 241 | 20–25× |
| subword-1000 | 289 | 25–32× |
| phoneme | 105 | 8–10× |

- [x] **char @ 8× reference** (`runs/baseline_char_8x`): **test WER 0.315 / CER 0.145**
      (beam+KenLM), dev WER 0.326 (best @ epoch 167/200, ~4 h). Reproduces *and beats* the
      phase-1 0.325 supervised baseline. (Greedy was test 0.434 — the KenLM beam does heavy
      lifting on EMG, so the headline is always the beam number.)
- [x] **Resolution sweeps run** (single `train_baseline` per factor via `sweep_unit.slurm`,
      200 ep, schedule `[125,150,175]`; → `runs/baseline_<tag>_<F>x`). Regenerate the table any
      time with `sbatch slurm/analyze_sweep.slurm` → `runs/sweep_summary_test.md`. **Test WER**
      (pyctcdecode beam + KenLM; char/subword directly comparable; phoneme = greedy PER):

      | unit | best test WER (factor) | curve |
      |---|---|---|
      | **char** | **0.292 (20×)** / 0.296 (16×) | 8× .315 → 10× .300 → 16× .296 → 20× .292 |
      | subword-60 | ~0.31 (16×, dev) | dev 10× .331 / 16× .311 / 20× .357 — ≈ char-10×, ≤ char |
      | subword-100 | ~0.33 (10×, dev) | dev 10× .330 / 16× .333 / 20× .357 — ≤ char |
      | subword-250 | 0.283 (25×) | 16× .290 / 20× .294 / 25× .283 |
      | subword-500 | 0.298 (16×) | 16× .298 / 20× .319 / 25× .357 / 32× .328 |
      | subword-1000 | 0.333 (16×) | 16× .333 / 20× .335 |
      | phoneme | dropped | greedy PER 0.161 (16×); 8× collapses (all-blank, too fine) |

- **Finding — token resolution matters, and coarser-than-phase-1 wins.** char improves
  monotonically 8×(0.315)→20×(0.292): the phase-1 8× default was *too fine*. The optimum sits
  at ~16–20× (≈23–29 ms/frame, ~3–4 frames/char), squarely in the speech-ASR 20–40 ms band →
  the best EMG frame rate is set by the **signal's information rate, not the unit length**
  ("H-signal"). True for every unit (even char, the densest, peaks ≥16×).
- **DECISION → predict CHARACTERS at 16×** (`--unit char --downsample-factor 16`;
  `conv_strides=[2,2,2,2]`, ~43 Hz, ~23 ms/frame). char is best-or-tied on the *fair*
  (identical-decoder) comparison and simplest (no tokenizer/lexicon, graceful greedy). No
  subword vocab (60/100/250/500/1000) robustly beats it — subword-250-25× (0.283) is one noisy
  point; on dev every subword ≥ char.
- **Phonemes dropped.** Apples-to-apples (`flash_cmp` log — char *and* phoneme through the SAME
  flashlight lexicon+KenLM decoder, `scripts/decode_lexicon_flashlight.py`): char-16× **0.248**
  < phoneme-16× **0.275**. Phonemes only looked competitive when handed a stronger decoder than
  char; equalize the decoder → char wins. They also can't decode without a lexicon (greedy =
  garbage), lean harder on the leaky LM, and add OOV complexity. Phoneme→word exploration kept
  for reference: `phoneme_to_words[_lattice].py`, `build_lexicon_from_unigrams.py`,
  `phoneme_diag.py`, `decode_lexicon_flashlight.py`.

> **⚠️ LM-leakage caveat (applies to ALL beam/LM numbers above):** the test set is *War of the
> Worlds*, whose text is in the LibriSpeech KenLM (`data/lm.binary`) → absolute WERs are
> optimistically low (the *relative* unit/resolution ordering is trustworthy; absolutes aren't).
> Decoder also matters: char-16× = 0.296 (pyctcdecode) vs 0.248 (flashlight lexicon).
> **Freeze ONE decoder** for all phase-2 comparisons.

### C. Text as the unpaired modality — **NEXT**
EMG branch is fixed: **char @ 16×** (`--unit char --downsample-factor 16`).
- [ ] Build a `TextFrontend` (text → embedding sequence into the shared transformer) and a text
      dataset reader, mirroring `uml/model.py::AudioFrontend` and `uml/audio_dataset.py`; reuse
      the shared-transformer plumbing in `UMLModel` (inference = EMG branch only).
- [ ] Two corpora head-to-head: (a) a large generic English corpus; (b) the Gaddy transcripts
      (in-distribution). **🔴 Dedup any generic corpus against the test/dev transcripts (War of
      the Worlds + the Gaddy books) before training — else an unpaired-text "gain" is just text
      leakage.**
- [ ] Train EMG+text UML at char-16×, then EMG-only finetune (`finetune_from_uml.py`). Compare
      to the char-16× supervised baseline (≈0.296 pyctcdecode / 0.248 flashlight) and the
      phase-1 audio-UML **0.287**.
- Carry phase-1 UML lessons: `clip_grad_norm` off; `share_ctc_head` off; **λ≈0.3** (down-weighted
      second branch) was best after finetune — start there.
- Metric hygiene: freeze one decoder across conditions; report a leakage-controlled signal
      (greedy/CER deltas, or a KenLM rebuilt without the test book) so a "win" reflects EMG
      learning, not the LM.

### D. Unsupervised setting
- [ ] Unsupervised objective consuming un-transcribed text/EMG on the shared transformer
      (phase-1 SSL on audio gave no gain at ~100 h; revisit only if a text signal looks
      promising). Decide go/no-go by an honest CTC finetune, not a linear probe (the phase-1
      probe was misleading).

## Notes carried from phase 1
- Eval defaults: split=test, method=beam, KenLM `data/lm.binary` + leakage-free unigrams.
- UML inference uses the EMG branch only; the second branch is a training-time auxiliary.
- Validation during training is EMG-only on `dev`.
