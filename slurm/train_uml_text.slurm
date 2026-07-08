#!/bin/bash
#SBATCH --job-name=train_uml_text
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_text_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_text_%j.err

set -euo pipefail

# Phase-2 §C: dual-branch UML with the unpaired-TEXT branch (denoising-CTC), EMG @ char-16x.
# Select the corpus arm with TEXT_SOURCE (libri | gaddy); output dir defaults per arm.
#   TEXT_SOURCE=libri sbatch slurm/train_uml_text.slurm
#   TEXT_SOURCE=gaddy sbatch slurm/train_uml_text.slurm
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data}
TEXT_SOURCE=${TEXT_SOURCE:-libri}
TEXT_CACHE_DIR=${TEXT_CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data/text_cache}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMG-unpaired-text/runs/uml_text_${TEXT_SOURCE}}
# EXTRA_FLAGS="..." -> arbitrary extra flags appended verbatim (e.g. --lambda-uml 0.5)
EXTRA_FLAGS=${EXTRA_FLAGS:-}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
# Frozen text frontend (--text-frontend frozen) loads CANINE from the HF cache.
export HF_HOME=${HF_HOME:-/scratch/cr4206/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/cr4206/.cache/huggingface}
mkdir -p "$HF_HOME" /scratch/cr4206/sEMG-unpaired-text/logs "$OUTPUT_DIR"

# Pre-flight: EMG cache + the selected text cache.
if [ ! -f "${CACHE_DIR}/train.pt" ] || [ ! -f "${CACHE_DIR}/dev.pt" ]; then
    echo "[error] EMG cache not found under ${CACHE_DIR}"
    exit 1
fi
if [ ! -f "${TEXT_CACHE_DIR}/${TEXT_SOURCE}.pt" ]; then
    echo "[error] Text cache not found at ${TEXT_CACHE_DIR}/${TEXT_SOURCE}.pt"
    echo "        Run: python scripts/precompute_text.py --text-source ${TEXT_SOURCE}"
    exit 1
fi

echo "[train_uml_text] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[train_uml_text] text_source=$TEXT_SOURCE text_cache_dir=$TEXT_CACHE_DIR output_dir=$OUTPUT_DIR"

"$PYTHON" -u scripts/train_uml.py \
  --config configs/train_uml_text.yaml \
  --cache-dir "$CACHE_DIR" \
  --text-cache-dir "$TEXT_CACHE_DIR" \
  --text-source "$TEXT_SOURCE" \
  --output-directory "$OUTPUT_DIR" \
  $EXTRA_FLAGS \
  --wandb

echo "[train_uml_text] finished at=$(date -Is)"
