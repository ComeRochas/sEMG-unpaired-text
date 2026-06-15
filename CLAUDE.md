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

Priority order (see [TODO.md](TODO.md)):
1. **Target unit × token resolution** (DONE): swept char/subword/phoneme × resolution.
   **Result: predict CHARACTERS at 16×** (`--unit char --downsample-factor 16`). Coarser than
   the phase-1 8× wins; subwords don't robustly beat char; phonemes dropped. See [TODO.md](TODO.md) §B.
2. **Text as the unpaired modality** (CURRENT NEXT): add a text branch to the shared transformer
   (mirroring the kept audio branch), at char-16×, comparing a large generic corpus vs the Gaddy
   transcripts. See [TODO.md](TODO.md) §C.
3. **Supervised then unsupervised** settings.

(A Conformer encoder was considered and **deferred** — not in scope now.)

## Cluster & environment

- **Login node:** `torch-login-a-2` (NYU HPC)
- **Python env:** `/scratch/cr4206/envs/silent_speech/bin/python` (torch 2.0.1; `sentencepiece`
  installed for the subword unit; `g2p_en` installed for phonemes, and a complete OOV-covering
  pronunciation dict is precomputed at `data/tokenizers/phoneme_g2p.dict`)
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

## Target units — pluggable tokenizer (`semg_jepa/tokenizers.py`)

The CTC target unit is a config knob (`unit: char|subword|phoneme`). A tokenizer is the
single source of truth for the vocab, blank index, decode rendering, and the WER/CER
reference rendering, so the rest of the pipeline stays unit-agnostic. The EMG cache stores
the raw `text` string, so any unit re-encodes on the fly — **no recache per unit**.

- **char** (`CharTokenizer`): `a-z 0-9 space` (37 symbols), blank 37. Phase-1 default.
  `clean_text` = `unidecode` → strip punctuation → lowercase.
- **subword** (`SubwordTokenizer`): SentencePiece model trained with
  `scripts/train_subword.py` (→ `data/tokenizers/subword_<N>.model`). `▁` word-start marker
  is understood by `pyctcdecode`, so the KenLM word LM still applies for beam decode.
- **phoneme** (`PhonemeTokenizer`): ARPAbet inventory + `|` word separator. Needs a G2P
  backend — `g2p_en` (pip) or a CMUdict file via `--phoneme-dict` (use the precomputed
  `data/tokenizers/phoneme_g2p.dict`). No phoneme LM, so the reported metric is a phone error
  rate (PER). **Score phonemes with greedy, not beam** — pyctcdecode glues phones with no
  spaces and mis-renders them (WER≈1.0); `train_baseline.py` auto-forces greedy when the unit
  has no word LM, so phoneme `best.pt` is selected correctly (pre-fix runs: use `last.pt`).
  To get a comparable *word* WER, `scripts/phoneme_to_words.py` decodes phones→words via the
  inverted lexicon + KenLM.

Build via `build_tokenizer(unit, subword_model=..., phoneme_dict=...)`.

## Token temporal resolution (EMG side)

`GaddyRawEMGEncoder(conv_strides=...)` sets the downsample factor = `prod(conv_strides)`,
i.e. the encoder output frame rate that the CTC head emits at. This is the knob to **match
the EMG resolution to the target unit**: characters (many tokens) want the fine 8× rate
(`[2,2,2]` ≈ 86 Hz, default); coarser subwords/phonemes (fewer tokens) tolerate 16×
(`[2,2,2,2]`) or want 4× (`[2,2]`). The factor must divide `fixed_raw_len` (1600) and stay
≤ each utterance's target length; the dataset right-crops raw EMG to a multiple of the
factor. `train_baseline.py` and `evaluate.py` both take `--unit` and `--conv-strides`, plus
`--downsample-factor N` (alt to `--conv-strides`; `factor_to_strides()` decomposes any divisor
of `fixed_raw_len=1600 = 2^a·5^b` — 8,10,16,20,25,32,40,50 — into strides, widening the kernel
for stride>3 so non-power-of-2 factors don't skip samples).

**Finding (test): coarser than phase-1 8× is better → DECISION: predict CHARACTERS at 16×**
(`--unit char --downsample-factor 16`). char improves 8×(0.315)→16×(0.296)→20×(0.292); the EMG
optimum ~16–20× tracks the **signal's information rate, not the unit length** (speech-ASR's
20–40 ms band). Subwords (60–1000) don't robustly beat char; **phonemes dropped** (apples-to-
apples, same flashlight lexicon decoder: char-16× 0.248 < phoneme 0.275). ⚠️ all beam WERs are
LM-leakage-inflated (test = War-of-the-Worlds ∈ LibriSpeech LM). See [TODO.md](TODO.md) §B.

## Model architecture (carried from phase 1)

`GaddyRawEMGEncoder` (`semg_jepa/architecture.py`): `[B, F*T, 8] → [B, T, D]` — one ResBlock
Conv1d per entry of `conv_strides` (default `(2,2,2)` = 8× downsample, F=8) → Linear →
N-layer Transformer with relative positional embeddings (window 100). Defaults:
model_size=768, num_layers=6, nhead=8, dim_feedforward=3072. Random temporal shift
augmentation (≤ F samples) during training. `conv_strides` is the **token-resolution knob**
(see above); `encoder.downsample_factor` exposes F.

`BaselineCTCModel(vocab_size, conv_strides)` = encoder + `CTCHead(vocab_size)`.

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
| `scripts/train_subword.py` | — | Train a SentencePiece subword tokenizer (`--unit subword` prerequisite) |
| `scripts/train_baseline.py` | `slurm/train_baseline.slurm`, `slurm/sweep_unit.slurm` | Supervised CTC baseline (`--unit`, `--conv-strides`/`--downsample-factor`); sweep = one job per factor |
| `scripts/analyze_unit_durations.py` | — | Per-unit duration vs EMG token period per factor (CTC feasibility) — pick the resolution |
| `scripts/build_phoneme_dict.py` | — | Precompute a complete g2p pronunciation dict (OOV-covering) → `data/tokenizers/phoneme_g2p.dict` |
| `scripts/analyze_sweep_results.py` | `slurm/analyze_sweep.slurm` | Test-set WER/CER table + reconstructions across `runs/baseline_*` |
| `scripts/phoneme_to_words.py` | `slurm/phoneme_to_words.slurm` | Decode a phoneme ckpt → words (lexicon + KenLM) → real word WER |
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
- never watch or launch loops to monitor slurm job states (squeue, sacct, ...) otherwise my account will be banned from HPC clusters

## Phase-2 gotchas (learned this round)

- **LR schedule + epochs live in `configs/train_baseline.yaml`** (200 ep, milestones
  `[125,150,175]`, γ=0.5); the slurm no longer hard-overrides them.
- **Phoneme metric:** beam mis-renders phonemes (pyctcdecode glues phones → WER≈1.0) → always
  score phonemes **greedy** (auto for no-word-LM units in `train_baseline.py`); pre-fix phoneme
  runs have a mis-selected `best.pt` → use `last.pt`.
- **Phoneme 8× collapses** to all-blank (too fine, ~8.8 frames/phoneme → blank-dominated
  alignment); a CTC optimization instability, *not* a bug. 10×/16× train fine (PER ~0.17).
- **Beam dev-eval** in `evaluate()` falls back to greedy if pyctcdecode returns no beams
  (near-uniform logits on an untrained model at a high factor, e.g. char-25× epoch 1).
- **Phoneme→words** (`scripts/phoneme_to_words.py`) is the active next thread — see [TODO.md](TODO.md) §B.