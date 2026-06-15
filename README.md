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
2. **Text as the unpaired modality** (next) — large generic corpus vs Gaddy transcripts.
3. **Supervised then unsupervised** settings.

### Target unit & token resolution (implemented)

The CTC **unit** (`char`/`subword`/`phoneme`) and the EMG **token resolution**
(`conv_strides` → downsample factor) are config knobs threaded through `train_baseline.py`
and `evaluate.py`. The EMG cache stores raw `text`, so units re-encode on the fly (no
recache). See [CLAUDE.md](CLAUDE.md) for the design.

**Result (test):** token resolution matters and *coarser than the phase-1 8× wins* — char
improves 8×(0.315)→16×(0.296)→20×(0.292), optimum ~16–20× (≈23 ms/frame). **Decision: predict
characters at 16×** (`--unit char --downsample-factor 16`). Subwords (60–1000) don't robustly
beat char; **phonemes dropped** (char-16× 0.248 < phoneme 0.275 under the same lexicon decoder).
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
| `scripts/train_uml.py` | Dual-branch shared-Transformer UML (audio = template for text branch) |
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
