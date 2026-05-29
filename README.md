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

See [TODO.md](TODO.md). Three axes:
1. **Text as the unpaired modality** — large generic corpus vs Gaddy transcripts (try both).
2. **Token resolution** — char / subword / phoneme; pick the adapted temporal resolution and
   re-train a supervised baseline per resolution.
3. **Supervised and unsupervised** settings.

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
| `scripts/train_baseline.py` | Supervised CTC baseline |
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
