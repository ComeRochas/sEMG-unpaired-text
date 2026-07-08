#!/bin/bash
#SBATCH --job-name=precompute_audio_hubert_units
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/precompute_audio_hubert_units_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/precompute_audio_hubert_units_%j.err

# Extract HuBERT km100 units for the UML AUDIO branch from a precomputed audio cache's
# waveforms (aligned 1:1 by index). Runs INSIDE the HuBERTVoc Singularity overlay (fairseq).
# Set AUDIO_CACHE_DIR + SPLIT for each audio source (LibriSpeech / Gaddy-internal).

set -euo pipefail

REPO=${REPO:-/scratch/cr4206/sEMG-unpaired-text}
AUDIO_CACHE_DIR=${AUDIO_CACHE_DIR:-${REPO}/data/libri_cache}
SPLIT=${SPLIT:-train-clean-100}
OUT_DIR=${OUT_DIR:-${REPO}/data/audio_hubert_units}
LAYER=${LAYER:-6}
HUBERTVOC_ROOT=${HUBERTVOC_ROOT:-/scratch/th3482/HuBERTVoc}
SIF=${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
OVERLAY=${OVERLAY:-${HUBERTVOC_ROOT}/singularity/overlay-50G-10M.ext3}
SINGULARITY_BIN=${SINGULARITY_BIN:-singularity}

mkdir -p "${REPO}/logs" "$OUT_DIR"

echo "[precompute_audio_hubert_units] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[precompute_audio_hubert_units] audio_cache=$AUDIO_CACHE_DIR split=$SPLIT out=$OUT_DIR layer=$LAYER"

"$SINGULARITY_BIN" exec --nv \
  --overlay "${OVERLAY}:ro" \
  --bind /scratch:/scratch \
  "$SIF" \
  bash -lc "
    source /ext3/env.sh
    conda activate /ext3/miniforge3/envs/torch2
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    export NUMBA_CACHE_DIR=/tmp/hubertvoc_numba_\${SLURM_JOB_ID:-0}; mkdir -p \"\$NUMBA_CACHE_DIR\"
    python -u ${REPO}/scripts/precompute_audio_hubert_units.py \
      --audio-cache-dir ${AUDIO_CACHE_DIR} \
      --split ${SPLIT} \
      --out-dir ${OUT_DIR} \
      --layer ${LAYER} \
      --hubertvoc-root ${HUBERTVOC_ROOT}
  "

echo "[precompute_audio_hubert_units] finished at=$(date -Is)"
