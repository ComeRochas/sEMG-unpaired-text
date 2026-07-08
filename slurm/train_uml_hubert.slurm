#!/bin/bash
#SBATCH --job-name=train_uml_hubert
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_hubert_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_hubert_%j.err

# Baseline = UML with lambda_uml=0: the EMG branch predicts HuBERT km100 units; the audio
# second branch is built but contributes nothing (loss * 0). Config: configs/train_uml_hubert.yaml.

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CONFIG=${CONFIG:-configs/train_uml_hubert.yaml}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data}
UNITS_DIR=${UNITS_DIR:-/scratch/cr4206/sEMG-unpaired-text/data/hubert_units}
LIBRI_CACHE_DIR=${LIBRI_CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data/libri_cache}
LIBRI_SPLIT=${LIBRI_SPLIT:-train-clean-100}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMG-unpaired-text/runs/uml_hubert}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=${HF_HOME:-/scratch/cr4206/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/cr4206/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/scratch/cr4206/.cache/torch}
mkdir -p "$HF_HOME" "$TORCH_HOME" /scratch/cr4206/sEMG-unpaired-text/logs "$OUTPUT_DIR"

# Pre-flight: EMG cache + HuBERT-unit cache + LibriSpeech cache (audio second branch).
for f in "${CACHE_DIR}/train.pt" "${CACHE_DIR}/dev.pt"; do
    [ -f "$f" ] || { echo "[error] EMG cache missing: $f (run precompute_raw_emg)"; exit 1; }
done
for f in "${UNITS_DIR}/train.pt" "${UNITS_DIR}/dev.pt"; do
    [ -f "$f" ] || { echo "[error] HuBERT unit cache missing: $f (run precompute_hubert_units)"; exit 1; }
done
[ -f "${LIBRI_CACHE_DIR}/${LIBRI_SPLIT}.pt" ] || { echo "[error] LibriSpeech cache missing"; exit 1; }

echo "[train_uml_hubert] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[train_uml_hubert] config=$CONFIG output_dir=$OUTPUT_DIR units_dir=$UNITS_DIR"

"$PYTHON" -u scripts/train_uml.py \
  --config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --hubert-units-dir "$UNITS_DIR" \
  --librispeech-cache-dir "$LIBRI_CACHE_DIR" \
  --librispeech-split "$LIBRI_SPLIT" \
  --output-directory "$OUTPUT_DIR" \
  $EXTRA_FLAGS \
  --wandb

echo "[train_uml_hubert] finished at=$(date -Is)"
