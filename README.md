# sEMG-unpaired-text

EMG-to-text silent speech framework — **phase 2**. Studies **text as an unpaired modality**
and the effect of **token resolution** (characters / subwords / phonemes) on silent-speech
CTC, in both supervised and unsupervised settings.

Seeded from the reusable core of [`sEMGencoderJEPA`](../sEMGencoderJEPA) (archived at tag
`v1-uml-ssl-archive`). Built on Gaddy & Klein (2021) and the unpaired-modality method of
[Gupta et al. 2025 ("Better Together")](https://arxiv.org/abs/2510.08492).

## Background (phase 1, in the archive)

A single Transformer shared between an EMG branch and an *audio* branch (each with its own
CTC head on its own unpaired labelled data) reached **test WER 0.287** vs a 0.325 supervised
CTC baseline. JEPA and unsupervised SSL gave no transfer at full labels. Phase 2 swaps the
audio modality for **text** and adds the token-resolution study.

## Phase 2 plan

See [TODO.md](TODO.md). Priority order:
1. **Target unit × token resolution** (current focus, implemented): predict char / subword /
   phoneme and tune the EMG token temporal resolution to match; re-train a supervised
   baseline per (unit, resolution).
2. **Text as the unpaired modality** (implemented + evaluated — see results below).
3. **Supervised then unsupervised** settings.

### Target unit & token resolution (implemented)

The CTC **unit** (`char`/`subword`/`phoneme`) and the EMG **token resolution**
(`conv_strides` → downsample factor) are config knobs threaded through `train_baseline.py`
and `evaluate.py`. The EMG cache stores raw `text`, so units re-encode on the fly (no
recache). See [CLAUDE.md](CLAUDE.md) for the design.

**Result (test):** token resolution matters and *coarser than the phase-1 8× wins* — char
improves 8×(0.315)→16×(0.296)→20×(0.292), optimum ~16–20× (≈23 ms/frame). **Decision: predict
characters at 16×** (`--unit char --downsample-factor 16`). Subwords (60–1000) don't robustly
beat char; **phonemes dropped** — under the *identical* flashlight lexicon+KenLM decoder char-16×
0.249 < phoneme 0.281, and phonemes have no usable greedy (decoder-free) output at all (see the
[same-decoder comparison](#phonemes-vs-characters--same-decoder-apples-to-apples-char-wins) below).
⚠️ beam WERs are LM-leakage-inflated (the test text is in the LibriSpeech LM). See [TODO.md](TODO.md).

```bash
# Sweep a unit's token resolution: one training run per factor, in one slurm allocation.
# Factors must divide 1600 (2^a*5^b): 8, 10, 16, 20, 25, 32, 40, 50.
UNIT=char FACTORS="8 10 16 20 25" sbatch slurm/sweep_unit.slurm
python scripts/train_subword.py --vocab-size 500     # -> data/tokenizers/subword_500.model
UNIT=subword SUBWORD_MODEL=data/tokenizers/subword_500.model FACTORS="16 20 25 32" sbatch slurm/sweep_unit.slurm
UNIT=phoneme PHONEME_DICT=data/tokenizers/phoneme_g2p.dict FACTORS="10 16" sbatch slurm/sweep_unit.slurm
# single run: UNIT=... DOWNSAMPLE_FACTOR=25 OUTPUT_DIR=runs/... sbatch slurm/train_baseline.slurm

# Summarise every run on test (WER for char/subword, PER for phoneme) + reconstructions
sbatch slurm/analyze_sweep.slurm                     # -> runs/sweep_summary_test.md

# Phoneme -> words (pronunciation lexicon + KenLM) for a comparable word WER
CKPT=runs/baseline_phoneme_10x/last.pt FACTOR=10 sbatch slurm/phoneme_to_words.slurm
```

> Token resolution is also pickable directly as `--downsample-factor N` (alt to
> `--conv-strides`); pick it per unit with `scripts/analyze_unit_durations.py`.

> Requires `sentencepiece` (installed in the env) for the subword unit.

### Text as the unpaired modality (implemented + evaluated)

A **text branch** mirrors the audio branch into the shared Transformer (`--second-branch text`).
Since text is the *target* modality, "predict text from text" is trivial, so the branch trains a
**denoising-CTC** task: corrupt the clean characters (span-mask 15% → `MASK`, substitute 10%,
delete 5%), feed them through a frontend → shared Transformer → text CTC head, and reconstruct the
**full clean** sequence (CTC). Two frontends: `embed` (trainable char embedding; corruption +
×3 jittered upsample in the collate) and `frozen` (**frozen CANINE** char encoder + trainable
Linear — the architectural mirror of frozen wav2vec2; features upsampled ×3 after the encoder).
Corpora via `--text-source {libri,gaddy}` (LibriSpeech transcripts, deduped vs the test books;
or the in-distribution Gaddy transcripts). EMG branch fixed at char-16×, `epoch_mode=alternate`,
λ=0.3, then EMG-only finetune.

**⚠️ Methodology — compare to the λ=0 control, not the supervised baseline.** Every UML arm is a
**2-stage** procedure (`train_uml` → `finetune_from_uml`), so the correct reference is a **λ=0
control** (same 2 stages, second-branch loss weighted to 0), not the single-stage
`train_baseline`. A real bug surfaced here: the UML configs had `grad_accum_steps: 1` vs the
baseline's `2`, so the EMG stage trained at **half the effective batch** and the control was
~0.027 WER under-trained. **Fixed to 2**; re-run with the fix below.

**Results (test, char-16×, open-vocab beam+KenLM; 99 utterances → ±~0.01 WER noise):**

| condition (grad_accum=2) | WER | CER |
|---|---|---|
| λ=0 control (no 2nd branch) | **0.284** | **0.132** |
| gaddy in-domain text (λ=0.3) | 0.286 | 0.134 |
| audio (λ=0.3) | 0.293 | 0.142 |
| supervised baseline (1-stage) | 0.298 | 0.136 |

**Conclusion: at matched, properly-tuned training, no UML second branch — audio *or* text —
reproducibly beats a well-trained EMG-only control.** All conditions land at ~0.28–0.30 WER,
within the small-test-set + LM-leakage noise. The phase-1 "audio-UML 0.287 vs 0.325" gain does
**not** reproduce at 16×: that comparison was confounded three ways — (a) it was measured against
the *1-stage supervised baseline*, never a λ=0 control (phase 1 had none), so it conflated the
audio branch with the 2-stage train+finetune procedure; (b) the **identical `grad_accum=1` bug
was present in phase 1 too** (the archived `train_uml.yaml` uses `grad_accum_steps: 1` while its
`train_baseline.yaml` uses `2`); and (c) 8× is a *weak resolution* (the sweep later showed 8× is
too fine). At 16× the clean λ=0 control matches the audio arm → the apparent gain was the 2-stage
procedure, not the second branch. **This was then confirmed airtight at 8×** (the resolution where
the gain was first seen) — see the dedicated re-examination below.

Two **real** (large, reproducible) effects, both about *avoiding harm*, not adding gain:
- **Out-of-distribution text hurts** at λ=0.3 with a trainable embedding (LibriSpeech text → 0.39 WER),
- …and is **rescued** to ~0.29 by either the **frozen CANINE** frontend or a low **λ=0.1** — i.e. the
  frozen frontend is *defensive* (don't let foreign text corrupt the shared Transformer), not additive.

Differences below the seed/test-noise floor → a defensible "helps / doesn't help" claim needs
**multiple seeds** + a leakage-controlled metric. (Phonemes were also revisited with an audio
second branch — same null result; see the
[same-decoder comparison](#phonemes-vs-characters--same-decoder-apples-to-apples-char-wins) below.)

### Re-examining the phase-1 "audio-UML helps" result (settled: it was the 2-stage, not the audio)

Phase 1 reported **audio-UML (Gaddy, λ=0.3) → finetune = 0.287** vs a supervised baseline, and
read it as "the unpaired audio modality improves the EMG model." Re-examined with the control it
lacked, that attribution does **not** hold. The phase-1 comparison entangled the audio branch with
**six** other differences; the headline pair `runs/baseline/last.pt` (0.307) vs
`runs/finetune_uml_gaddy_lambda0.3/best.pt` (0.287) differs in: **(1) 1-stage vs 2-stage** (pretrain
→ finetune = two full LR warmup+decay cycles), **(2) audio branch present vs absent** (the variable
under test, *entangled* with #1), (3) `grad_accum` 2 vs 1, (4) `clip_grad_norm` 0 vs 1.0, (5)
`alternate` mode, (6) **`last` vs `best-dev` checkpoint** (≈0.018 alone — that run's `best.pt`=0.325
> `last.pt`=0.307), (7) 400-ep single cycle vs 200+200 two-cycle schedule.

The clean test isolates the audio branch by holding the 2-stage procedure fixed and varying **only
λ**. Done at both resolutions under one identical grid-searched beam+KenLM decoder:

| condition (grad_accum=2, 2-stage unless noted) | 8× WER / CER | 16× WER / CER |
|---|---|---|
| supervised baseline (1-stage) | 0.3125 / 0.1440 | 0.298 / 0.136 |
| **λ=0 control (2-stage, no 2nd branch)** | **0.3015 / 0.1326** | **0.284 / 0.132** |
| audio λ=0.3 (2-stage) | 0.3058 / 0.1387 | 0.293 / 0.142 |

**At both resolutions the λ=0 control beats the audio arm and the 1-stage baseline.** So the ~0.01–0.015
gain over the baseline is the **2-stage pretrain→finetune procedure** (a second LR cycle / warm
restart), **not** the unpaired audio. Phase-1's own grid corroborates: the UML EMG branch *before*
finetune was 0.321 (≈ baseline 0.325); the **finetune** is what reached 0.287. **What holds from phase
1:** the model works and 2-stage training genuinely helps. **What doesn't:** crediting the gain to the
audio modality — a missing-control artifact, re-attributed, not a fabrication.

```bash
# Text-branch UML (denoising-CTC), char-16x, alternate, lambda=0.3
TEXT_SOURCE=gaddy sbatch slurm/train_uml_text.slurm                       # embed frontend
TEXT_SOURCE=libri EXTRA_FLAGS="--text-frontend frozen" sbatch slurm/train_uml_text.slurm   # frozen CANINE
python scripts/precompute_text.py --text-source libri                    # build/dedup a text cache
# lambda=0 control (the correct reference): EXTRA_FLAGS="--lambda-uml 0"
```

### Phonemes vs characters — same-decoder, apples-to-apples (char wins)

A tutor-suggested follow-up revisited phonemes *with* the audio second branch. Two phoneme UML
arms were trained at phoneme-16× (its best PER resolution) and EMG-only finetuned — `audio`
(Gaddy-internal, λ=0.3) and a `λ=0 control` — then decoded **through the exact same** flashlight
lexicon+KenLM decoder as the two best char-16× models (`scripts/decode_lexicon_flashlight.py`,
decoder weights `lm_weight`/`word_score` dev-tuned, then test-reported). Same decoder, same LM,
same lexicon for every row — so the *only* thing that varies is the acoustic unit.

| unit | model | greedy word WER | flashlight lex+KenLM WER |
|---|---|---|---|
| char | UML λ=0 control, ft | 0.379 | **0.249** |
| char | supervised baseline | 0.379 | 0.257 |
| phoneme | UML λ=0 control, ft | 4.43 | 0.281 |
| phoneme | UML + audio (λ=0.3), ft | 4.46 | 0.290 |

**Conclusion: decoding to phonemes is strictly worse than characters, on both metrics.**
(1) **Greedy / decoder-free: char ≈ 0.38 is a real word WER; phoneme ≈ 4.4 is unusable.**
(2) **Same decoder: char 0.249–0.257 < phoneme 0.281–0.290** — a ~0.03–0.04 WER gap in char's
favour, with the lexicon+KenLM held identical. And, mirroring char/text, the **audio branch did
not help phonemes** (λ=0 control 0.281 *better than* audio 0.290).

**Why phonemes needed a different decoder, and what that implies.** A char model emits letters +
a space symbol, so the open-vocab `pyctcdecode`+KenLM decoder (the one behind every char WER
here) reads its greedy output directly as words — greedy alone already gives a real 0.38 WER. A
phoneme model emits ARPAbet symbols with no orthography and no word boundary the word-LM decoder
recognizes: `pyctcdecode` glues the phones into one blob (WER≈1.0), which is exactly the 4.4
"greedy word WER" above (a phone string scored as words = massive spurious insertions). **A
phoneme model therefore produces no words at all without a pronunciation lexicon** — a closed
dictionary mapping phone-sequences→words, decoded with the torchaudio/flashlight WFST. That
dependency is the catch, and it cuts against phonemes twice: (a) the lexicon is **closed-
vocabulary**, so any word absent from it can never be output — the metric is capped by lexicon
coverage, not by the acoustic model; and (b) our lexicon is built from LibriSpeech unigrams,
which **contain the test book (War of the Worlds) → leakage** that *deflates* the phoneme number.
So the only decoder that gives phonemes a "good" WER is simultaneously constrained and leaky —
and *even handed that favourable decoder, applied identically to char*, phonemes still lose. The
char metric needs no lexicon to be usable (greedy 0.38) and improves further with an open-vocab
LM. This is why the project decodes to **characters**, not phonemes.

## Setup

```bash
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text
PYTHON=/scratch/cr4206/envs/silent_speech/bin/python
```

`data/` is a symlink to the shared cache directory (`sEMGencoderJEPA/data`) — the EMG cache,
KenLM `lm.binary`, unigrams, and audio caches are reused as-is.

## Current (carried) workflow

```bash
# Supervised CTC baseline (character-level)
sbatch slurm/train_baseline.slurm

# Dual-branch UML (audio branch — template for the upcoming text branch)
sbatch slurm/train_uml.slurm                 # LibriSpeech audio
sbatch slurm/train_uml_gaddy_audio.slurm     # Gaddy-internal audio

# Finetune EMG-only from a UML EMG-branch checkpoint
EMG_BRANCH=runs/uml/best_emg_branch.pt sbatch slurm/finetune_from_uml.slurm

# Evaluate (defaults: split=test, method=beam)
sbatch slurm/evaluate.slurm
CHECKPOINTS="runs/baseline/best.pt" GRID_SEARCH=1 sbatch slurm/evaluate.slurm
```

## Scripts

| File | Purpose |
|------|---------|
| `scripts/train_baseline.py` | Supervised CTC baseline (`--unit char/subword/phoneme`, `--conv-strides`/`--downsample-factor`) |
| `scripts/train_subword.py` | Train a SentencePiece subword tokenizer (prerequisite for `--unit subword`) |
| `scripts/analyze_unit_durations.py` | Per-unit duration vs EMG token period per factor (CTC feasibility) — pick the resolution |
| `scripts/build_phoneme_dict.py` | Precompute a complete g2p pronunciation dict (OOV-covering) for `--unit phoneme` |
| `scripts/analyze_sweep_results.py` | Test-set WER/CER (PER) table + reconstructions across `runs/baseline_*` |
| `scripts/phoneme_to_words.py` | Decode a phoneme checkpoint → words (lexicon + KenLM) → real word WER |
| `scripts/train_uml.py` | Dual-branch shared-Transformer UML (`--second-branch audio/text`, `--text-frontend embed/frozen`, `--text-source libri/gaddy`) |
| `scripts/precompute_text.py` | Build/dedup a text-branch corpus cache (`--text-source libri/gaddy`; dedups libri vs the test books) |
| `scripts/finetune_from_uml.py` | CTC finetune from UML EMG-branch (encoder + EMG head) |
| `scripts/finetune_from_jepa.py` | CTC finetune from any encoder-only pretrain (head reset) |
| `scripts/evaluate.py` | WER + CER (greedy/beam, optional dev grid search) |
| `scripts/precompute_raw_emg.py` | Precompute raw EMG cache |
| `scripts/precompute_audio.py` / `_gaddy.py` | Audio cache builders (UML) |

## Architecture (carried)

`GaddyRawEMGEncoder` (3× ResBlock conv → Linear → relative-pos Transformer) +
`CTCHead`. `UMLModel` shares `model.emg_encoder.transformer` between the EMG branch and a
second-modality branch — phase 2 replaces the audio frontend with a text frontend, keeping
the shared-transformer plumbing. See [CLAUDE.md](CLAUDE.md) for details.
