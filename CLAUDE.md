# sEMG-unpaired-text — Project Context

## What this project is

EMG-to-text silent speech framework. Phase 2 of a research line that began in
`sEMGencoderJEPA` (archived at tag `v1-uml-ssl-archive`). This repo is **seeded from the
reusable core** of that one: the EMG encoder, shared-transformer UML model, CTC/eval
utilities, and data pipeline carry over unchanged; the JEPA and SSL experiment scripts do
**not** (they stayed in the archive — they produced no downstream gain at full labels).

### Phase 1 result we build on

Supervised **UML won**: a single Transformer shared between an EMG branch and an *audio*
branch (each with its own CTC head, trained on its own unpaired labelled data) reached
**test WER 0.287** (Gaddy-internal audio, λ=0.3, + EMG-only finetune) vs a 0.325 supervised
CTC baseline. Method: "Better Together: Leveraging Unpaired Multimodal Data for Stronger
Unimodal Models", Gupta et al. 2025, arXiv:2510.08492. JEPA and unsupervised SSL gave no
transfer at full labels. (Full phase-1 results: see the archived repo's `TODO.md`.)

### Phase 2 direction (this repo)

Three axes — see [TODO.md](TODO.md):
1. **Text as the unpaired modality** (replacing / complementing the audio branch). Two text
   corpora to compare: a large generic English corpus vs the transcripts shipped with the
   Gaddy audio.
2. **Token resolution**: characters / subwords / phonemes. Study the adapted temporal token
   resolution in each case and **re-train a supervised baseline per resolution**.
3. **Supervised and unsupervised** settings.

## Cluster & environment

- **Login node:** `torch-login-a-2` (NYU HPC)
- **Python env:** `/scratch/cr4206/envs/silent_speech/bin/python` (torch 2.0.1)
- **Package not pip-installed** — always set `PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text`
- **Slurm account:** `torch_pr_39_tandon_advanced`
- **GPU partitions:** `h100_tandon`, `h200_public`, `a100_tandon`, `l40s_public` — check idle with `sinfo` before submitting
- **CPU partition:** `cpu_short` — used for precomputation (no GPU)
- **Do not over-poll squeue/sacct** — account can get rate-limited

## Data

- **Raw EMG:** `/scratch/cr4206/data/emg_data/emg_data/{silent_parallel_data,voiced_parallel_data,nonparallel_data}`
- **Config:** `configs/data_config.json` — absolute paths to raw EMG, testset JSON, normalizers
- **Precomputed cache:** `data/` is a **symlink** to `/scratch/cr4206/sEMGencoderJEPA/data`
  (the EMG cache, `lm.binary`, `unigrams.txt`, and audio caches are identical and shared —
  not copied). `data/{train,dev,test}.pt`: 8055/200/99 samples; each is
  `raw_emg` fp16 [8T,8], `text` str, `text_int` long, `ctc_length` T.

Both **voiced and silent** utterances are in the cache and used for training; the CTC loss
treats them identically.

## Signal processing (`load_utterance` in `semg_jepa/read_emg.py`)

1000 Hz, 8-channel EMG → notch (60 Hz harmonics) + 2 Hz high-pass → subsample to 689.06 Hz
→ tanh normalize (`50*tanh(x/20/50)`). T = (len−8)//8 ≈ 86 frames/s. Returns
`{raw_emg [8T,8], text, ctc_length T}`. No MFCC / EMG features / phonemes computed here
(the parent codebase had them; removed). **Phase 2 note:** phoneme/subword targets are a
*label-side* change (a new `TextTransform` / tokenizer + CTC vocab), not a signal-side one.

## Ground truth labels (current = characters)

`text` → `TextTransform.clean_text()`: `unidecode()` → strip punctuation → lowercase → map
to char indices in `"abcdefghijklmnopqrstuvwxyz0123456789 "` (37 chars). CTC blank = 37,
vocab = 38. **Phase 2** will add subword and phoneme tokenizers alongside this char one and
re-train baselines per resolution.

## Model architecture (carried from phase 1)

`GaddyRawEMGEncoder` (`semg_jepa/architecture.py`): `[B, 8T, 8] → [B, T, D]` — 3× ResBlock
Conv1d (stride 2 each = 8× downsample) → Linear → N-layer Transformer with relative
positional embeddings (window 100). Defaults: model_size=768, num_layers=6, nhead=8,
dim_feedforward=3072. Random temporal shift augmentation during training (clones input).

`BaselineCTCModel` = encoder + `CTCHead`.

`UMLModel` (`uml/model.py`) = dual-branch shared-Transformer model. The other branch's
frontend (`AudioFrontend` = frozen `facebook/wav2vec2-base` + trainable `Linear(768,
model_size)`) feeds the **same** `nn.Module` transformer (`model.emg_encoder.transformer`)
as the EMG branch. Inference uses the EMG branch only. **This is the template for the text
branch** in phase 2: swap `AudioFrontend` for a `TextFrontend` and `uml/audio_dataset.py`
for a text dataset reader; the shared-transformer plumbing stays.

## Training & evaluation pipeline (carried)

Cache-only datasets via `CachedRawEMGDataset`. `build_batches(dataset, max_len)` makes
size-aware batches per epoch. `combine_fixed_length(list, 1600)` reshapes for the encoder
during training (not eval). Eval: `evaluate(...)` in `semg_jepa/ctc_utils.py` — `greedy`
(GPU argmax + CTC collapse) or `beam` (pyctcdecode + KenLM `data/lm.binary` + leakage-free
unigrams). `grid_search(...)` tunes `(beam_width, alpha, beta)` on dev.

## Scripts (carried core)

| Python | Slurm | Purpose |
|---|---|---|
| `scripts/precompute_raw_emg.py` | `slurm/precompute_raw_emg.slurm` | EMG cache builder (cache already shared via symlink) |
| `scripts/train_baseline.py` | `slurm/train_baseline.slurm` | Supervised CTC baseline |
| `scripts/train_uml.py` | `slurm/train_uml.slurm`, `slurm/train_uml_gaddy_audio.slurm` | Dual-branch shared-transformer UML (audio branch — **template** for the text branch) |
| `scripts/finetune_from_uml.py` | `slurm/finetune_from_uml.slurm` | CTC finetune from a UML EMG-branch ckpt (encoder + EMG head loaded) |
| `scripts/finetune_from_jepa.py` | `slurm/finetune_from_jepa.slurm` | CTC finetune from any encoder-only pretrain (head reset) |
| `scripts/evaluate.py` | `slurm/evaluate.slurm` | WER + CER (greedy/beam, optional dev grid search) |
| `scripts/precompute_audio.py` | `slurm/precompute_audio.slurm` | LibriSpeech audio cache (UML) |
| `scripts/precompute_audio_gaddy.py` | `slurm/precompute_audio_gaddy.slurm` | Gaddy-internal audio cache (UML) |

New phase-2 scripts (text frontend, tokenizers, per-resolution baselines) will be added here.

## W&B logging

`--wandb` on all training scripts (offline by default; `WANDB_MODE=online` to stream).
Entity `UMLforVideoLab`, project `JEPAforsEMG`.

## Key design decisions (inherited)

- **Cache-only** preprocessing (no scipy on every epoch).
- **No MFCC/EMG features** — removed from the parent codebase.
- **build_batches per epoch** for reshuffled size-aware batches.
- **Batched evaluation** (~16× faster than batch_size=1).
- **In-place mutation fix** in `GaddyRawEMGEncoder` (clones before temporal shift).
