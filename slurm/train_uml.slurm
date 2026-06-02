#!/bin/bash
#SBATCH --job-name=train_uml
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/train_uml_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data}
LIBRI_CACHE_DIR=${LIBRI_CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data/libri_cache}
LIBRI_SPLIT=${LIBRI_SPLIT:-train-clean-100}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMG-unpaired-text/runs/uml}
# Override config knobs from sbatch CLI:
#   SHARE_CTC_HEAD=1  -> add --share-ctc-head (else use config default)
#   EPOCH_MODE=both   -> override --epoch-mode (default: config)
#   EXTRA_FLAGS="..." -> arbitrary extra flags appended verbatim
SHARE_CTC_HEAD=${SHARE_CTC_HEAD:-}
EPOCH_MODE=${EPOCH_MODE:-}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}

# Redirect HuggingFace caches to scratch (wav2vec2-base download).
export HF_HOME=${HF_HOME:-/scratch/cr4206/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/cr4206/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/scratch/cr4206/.cache/torch}
mkdir -p "$HF_HOME" "$TORCH_HOME" /scratch/cr4206/sEMG-unpaired-text/logs "$OUTPUT_DIR"

# Pre-flight: EMG cache + LibriSpeech cache
if [ ! -f "${CACHE_DIR}/train.pt" ] || [ ! -f "${CACHE_DIR}/dev.pt" ]; then
    echo "[error] EMG cache not found under ${CACHE_DIR}"
    echo "        Run sbatch slurm/precompute_raw_emg.slurm first."
    exit 1
fi
if [ ! -f "${LIBRI_CACHE_DIR}/${LIBRI_SPLIT}.pt" ]; then
    echo "[error] LibriSpeech cache not found at ${LIBRI_CACHE_DIR}/${LIBRI_SPLIT}.pt"
    echo "        Run sbatch slurm/precompute_audio.slurm first."
    exit 1
fi

echo "[train_uml] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[train_uml] cache_dir=$CACHE_DIR libri_cache_dir=$LIBRI_CACHE_DIR libri_split=$LIBRI_SPLIT"
echo "[train_uml] output_dir=$OUTPUT_DIR"

SHARE_FLAG=""
[[ "$SHARE_CTC_HEAD" == "1" ]] && SHARE_FLAG="--share-ctc-head"
EPOCH_MODE_FLAG=""
[[ -n "$EPOCH_MODE" ]] && EPOCH_MODE_FLAG="--epoch-mode $EPOCH_MODE"

"$PYTHON" -u scripts/train_uml.py \
  --config configs/train_uml.yaml \
  --cache-dir "$CACHE_DIR" \
  --librispeech-cache-dir "$LIBRI_CACHE_DIR" \
  --librispeech-split "$LIBRI_SPLIT" \
  --output-directory "$OUTPUT_DIR" \
  $SHARE_FLAG $EPOCH_MODE_FLAG $EXTRA_FLAGS \
  --wandb

echo "[train_uml] finished at=$(date -Is)"
