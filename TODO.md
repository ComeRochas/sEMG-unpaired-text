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

### C. Unpaired modality (text / audio / multi-aux) — **DONE → NO GAIN over a pure-EMG control**

EMG fixed at **char @ 16×**. Ran the full UML program: single-aux text (2026-06), single-aux audio,
and — this round (2026-07) — **multi-auxiliary EMG + audio + text** (paper arXiv:2510.08492 Thm 1:
Fisher info compounds across modalities), each with seed-matched λ=0 controls + an EMG-only finetune.

**Verdict: no unpaired auxiliary — audio, text, or BOTH; CANINE or the stronger ByT5 text encoder —
beats a well-trained pure-EMG (λ=0) control at matched effective batch (grad_accum=2).**

Multi-aux numbers (char-16×, dev=200 utts, beam+KenLM WER; raw UML EMG-branch → after EMG-finetune):
| arm | raw dev WER | finetuned dev WER (best) |
|---|---|---|
| **λ=0 control** (seeds 0,1) | 0.306 / 0.319 | **0.280 / 0.313**  (best 0.280) |
| multi-aux audio+text, CANINE (0,1) | 0.334 / 0.343 | 0.308 / 0.324 |
| multi-aux audio+text, **ByT5** (0) | 0.336 | 0.308 |

Reads:
- The λ=0 control's best seed (**0.280**) is the best of all runs; multi-aux never beats it
  (best 0.308). Pre-finetune the control leads by ~0.02–0.03; the finetune shrinks but doesn't
  close the gap.
- Auxiliaries **trained correctly** (audio→char CTC → 0.137, text-denoise → 1.13, ByT5 → 1.16) —
  a genuine negative result, not a wiring bug. Compounding audio+text did NOT reduce variance
  (identical train `emg_loss` ~0.046 but worse dev) — it slightly pulled the shared transformer
  off EMG-optimal.
- **Stronger frozen text encoder gave nothing**: ByT5-small (~300M, byte-level) == CANINE-s (~120M),
  0.308 = 0.308. (For a char-CTC task the right "stronger" encoder is char/byte-level, not a subword
  LLM — ByT5 was the correct choice, and it still didn't move the needle.)
- **Seed variance dominates the effect**: control spread 0.033 > the control-vs-multiaux gap 0.019.
  Defensible claim = "no gain, weak evidence of slight harm, within ±0.02–0.03 noise". Any future
  claim needs ≥3–5 seeds + a leakage-free **test** metric (these are dev).
- Consistent with the single-aux finding: audio/text singles also failed to beat the control
  (their best 0.276–0.284 ≈ control ~0.28; see [[phase2-text-branch]] memory).

Why the paper doesn't transfer here: its gains add Fisher info to a *variance-limited* shared net
with *frozen, semantically-aligned* encoders (few-shot image classification, CLIP alignment). Here
the EMG encoder trains from scratch, the shared transformer must serve EMG *recognition*, and
audio/text share only the char **output** structure (not the input representation) → co-training
competes rather than complements. And EMG→char is capped by the **information/label ceiling** (data
quantity + silent-speech difficulty), which no unpaired auxiliary can lift. **UML is the wrong tool
for the full-label regime here.**

Implementation shipped (reusable): `train_uml.py --aux-branches audio text --lambda-audio/-text
--seed`; `UMLModel(aux_branches=...)`; `TextFrontend(frozen_arch=canine|byt5)` /
`--text-frozen-arch`; `configs/train_uml_multiaux{,_byt5}.yaml`; `slurm/train_uml_multiaux.slurm`.
Metric hygiene still stands: one frozen decoder across conditions; the test book (WotW) leaks into
the KenLM, so trust CER/greedy deltas and cross-seed spread, not absolute beam WER.

### C.2 Frontend symmetry (tutor critique) — conv-only audio frontend — **NO GAIN; HARMFUL on Gaddy**

Tutor observation (correct): the audio branch reaches the shared transformer already contextualized
by wav2vec2's **12 self-attention layers**, while the EMG branch reaches it **conv-only** (4
ResBlocks @ char-16×, no attention). Tested the "equalize down" fix (Option 2):
`AudioFrontend(mode=conv)` = wav2vec2 **feature-extractor only** (no transformer), so audio also
arrives at the shared transformer from raw conv features. Knob: `--audio-frontend {full,conv}`; no
recache. char-16×, seed 0, finetune-from-UML best dev (beam+KenLM):

| frontend | audio source | λ | dev WER | dev CER |
|---|---|---|---|---|
| **FULL** wav2vec2 (12 attn) | Gaddy | 0.3 | **0.276** | 0.132 |
| conv-only | Libri | 1.0 | 0.288 | 0.139 |
| conv-only | Libri | 0.3 | 0.292 | 0.139 |
| conv-only | Gaddy | 1.0 | 0.336 | 0.167 |
| conv-only | Gaddy | 0.3 | OOM @ep126 (UML dev 0.425) | — |
| λ=0 pure-EMG control (seed 0) | — | 0 | 0.280 | 0.139 |

**Verdict: Option 2 fails.** Matched-source Gaddy: full 0.276 → conv **0.336** (+0.060 WER, +0.035
CER) — a large regression. On Libri, conv (0.288 / CER 0.139) is **identical in CER to the λ=0
control** (0.280 / 0.139) → the conv-frontend audio branch is inert. Interpretation: wav2vec2's
transformer (linguistic contextualization) **is** what makes the audio auxiliary useful; stripping
it removes the transferable signal, most damagingly on the small, speaker-matched Gaddy corpus where
the full-frontend win lived. **The asymmetry is a feature, not a bug.** (Gaddy λ=0.3 conv OOM'd at
ep126 on l40s-44GB — long-utterance O(L²) rel-pos attention; rerun on h200-80GB for the exact
number, but conv-Gaddy is decisively worse regardless of λ.)

Does **not** rescue UML: the champion full-frontend 0.276 still only ties the control best 0.280
(within the 0.033 seed spread; and conv is single-seed so Libri 0.288-vs-0.280 is within noise —
only the Gaddy regression exceeds it). If frontend symmetry is pursued, the correct direction is
**Option 1 — raise EMG's contextualization UP** (an EMG-private pre-transformer, wav2vec2 kept full),
not equalize down; expected small since UML is a wash at full labels. Shipped:
`--audio-frontend conv`, `AudioFrontend(mode=...)`, `UMLModel(audio_frontend_mode=...)`.

### D. Where next — UML concluded (no gain); redirect

UML (§C) is closed as a full-label EMG→char lever. Directions, most-promising first:

1. **Low-label / few-shot UML (the one honest UML-salvage).** UML is a *variance reducer*; the
   paper's gains are largest **few-shot**. At full labels the EMG model is bias/ceiling-limited, so
   UML can't help — but with the labelled EMG set artificially cut to 10–25%, the shared net becomes
   variance-limited and audio/text may finally transfer. Re-run the multi-aux vs λ=0 control at
   {10, 25, 50, 100}% labels (all infra exists; just subsample `train.pt`). This also = the phase-3
   "supervised → unsupervised" roadmap item. **Best next experiment if UML is pursued at all.**
2. **EMG-side gains (the real bottleneck).** The ceiling (~0.28 dev) is data + silent-speech
   difficulty, not the auxiliary:
   - More/better labelled EMG (biggest lever; out of our control but worth flagging).
   - The deferred **Conformer** EMG encoder; or EMG self-sup pretraining (JEPA gave nothing at full
     labels — revisit only jointly with #1's low-label regime).
   - **LLM rescoring** of the CTC beam (cf. Gaddy 2024 cross-modal + LLM-enhanced recognition) — a
     decode-time lever orthogonal to the encoder, likely larger than any UML delta.
3. **Paired cross-modal alignment (CLIP-style), not unpaired UML.** The paper notes *aligned*
   encoders give bigger gains; we have PARALLEL voiced EMG+audio. Align EMG↔audio representations on
   the parallel data first, then probe — a different method than unpaired UML, closer to the paper's
   strongest setting.
4. **Metric rigor before any claim.** All deltas are inside ±0.02–0.03 (dev 200 / test 99 utts);
   need ≥3–5 seeds and a leakage-free test metric (KenLM rebuilt without WotW, or flashlight lexicon)
   to distinguish signal from seed noise. Decide go/no-go by an honest CTC finetune, not a linear
   probe (the phase-1 probe was misleading).

## Notes carried from phase 1
- Eval defaults: split=test, method=beam, KenLM `data/lm.binary` + leakage-free unigrams.
- UML inference uses the EMG branch only; the second branch is a training-time auxiliary.
- Validation during training is EMG-only on `dev`.
