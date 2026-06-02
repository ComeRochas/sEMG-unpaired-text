#!/bin/bash
#SBATCH --job-name=train_uml_gaddy_audio
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_gaddy_audio_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_gaddy_audio_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data}
AUDIO_CACHE_DIR=${AUDIO_CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data/gaddy_audio_cache}
AUDIO_SPLIT=${AUDIO_SPLIT:-gaddy_internal}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMG-unpaired-text/runs/uml_gaddy_audio}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}

export HF_HOME=${HF_HOME:-/scratch/cr4206/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/cr4206/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/scratch/cr4206/.cache/torch}
mkdir -p "$HF_HOME" "$TORCH_HOME" /scratch/cr4206/sEMG-unpaired-text/logs "$OUTPUT_DIR"

if [ ! -f "${CACHE_DIR}/train.pt" ] || [ ! -f "${CACHE_DIR}/dev.pt" ]; then
    echo "[error] EMG cache not found under ${CACHE_DIR}"
    exit 1
fi
if [ ! -f "${AUDIO_CACHE_DIR}/${AUDIO_SPLIT}.pt" ]; then
    echo "[error] Gaddy-audio cache not found at ${AUDIO_CACHE_DIR}/${AUDIO_SPLIT}.pt"
    echo "        Run sbatch slurm/precompute_audio_gaddy.slurm first."
    exit 1
fi

echo "[train_uml_gaddy_audio] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[train_uml_gaddy_audio] cache_dir=$CACHE_DIR audio_cache=$AUDIO_CACHE_DIR/$AUDIO_SPLIT output=$OUTPUT_DIR"

"$PYTHON" -u scripts/train_uml.py \
  --config configs/train_uml.yaml \
  --cache-dir "$CACHE_DIR" \
  --librispeech-cache-dir "$AUDIO_CACHE_DIR" \
  --librispeech-split "$AUDIO_SPLIT" \
  --output-directory "$OUTPUT_DIR" \
  --clip-grad-norm 0.0 \
  $EXTRA_FLAGS \
  --wandb

echo "[train_uml_gaddy_audio] finished at=$(date -Is)"
