#!/bin/bash
#SBATCH --job-name=train_uml_multiaux
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_multiaux_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_multiaux_%j.err

set -euo pipefail

# Multi-auxiliary UML: EMG + AUDIO (Gaddy-internal) + TEXT (LibriSpeech, frozen CANINE), all
# sharing the transformer, EMG @ char-16x. Set SEED and (for the baseline) the lambdas via
# EXTRA_FLAGS. Examples:
#   SEED=0 OUTPUT_DIR=.../runs/uml_multiaux_s0 sbatch slurm/train_uml_multiaux.slurm
#   SEED=0 OUTPUT_DIR=.../runs/uml_baseline_s0 \
#     EXTRA_FLAGS="--lambda-audio 0 --lambda-text 0" sbatch slurm/train_uml_multiaux.slurm
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CONFIG=${CONFIG:-configs/train_uml_multiaux.yaml}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data}
AUDIO_CACHE_DIR=${AUDIO_CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data/gaddy_audio_cache}
AUDIO_SPLIT=${AUDIO_SPLIT:-gaddy_internal}
TEXT_CACHE_DIR=${TEXT_CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data/text_cache}
TEXT_SOURCE=${TEXT_SOURCE:-libri}
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMG-unpaired-text/runs/uml_multiaux_s${SEED}}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=${HF_HOME:-/scratch/cr4206/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/cr4206/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/scratch/cr4206/.cache/torch}
mkdir -p "$HF_HOME" "$TORCH_HOME" /scratch/cr4206/sEMG-unpaired-text/logs "$OUTPUT_DIR"

# Pre-flight: EMG cache + audio cache (wav2vec2 branch) + text cache (CANINE branch).
if [ ! -f "${CACHE_DIR}/train.pt" ] || [ ! -f "${CACHE_DIR}/dev.pt" ]; then
    echo "[error] EMG cache not found under ${CACHE_DIR}"; exit 1
fi
if [ ! -f "${AUDIO_CACHE_DIR}/${AUDIO_SPLIT}.pt" ]; then
    echo "[error] audio cache not found at ${AUDIO_CACHE_DIR}/${AUDIO_SPLIT}.pt"; exit 1
fi
if [ ! -f "${TEXT_CACHE_DIR}/${TEXT_SOURCE}.pt" ]; then
    echo "[error] text cache not found at ${TEXT_CACHE_DIR}/${TEXT_SOURCE}.pt"; exit 1
fi

echo "[train_uml_multiaux] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[train_uml_multiaux] config=$CONFIG seed=$SEED output_dir=$OUTPUT_DIR"
echo "[train_uml_multiaux] audio=$AUDIO_CACHE_DIR/$AUDIO_SPLIT text=$TEXT_CACHE_DIR/$TEXT_SOURCE extra='$EXTRA_FLAGS'"

"$PYTHON" -u scripts/train_uml.py \
  --config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --librispeech-cache-dir "$AUDIO_CACHE_DIR" \
  --librispeech-split "$AUDIO_SPLIT" \
  --text-cache-dir "$TEXT_CACHE_DIR" \
  --text-source "$TEXT_SOURCE" \
  --output-directory "$OUTPUT_DIR" \
  --seed "$SEED" \
  $EXTRA_FLAGS \
  --wandb

echo "[train_uml_multiaux] finished at=$(date -Is)"
