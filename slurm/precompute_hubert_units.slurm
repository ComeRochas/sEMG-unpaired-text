#!/bin/bash
#SBATCH --job-name=precompute_hubert_units
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/precompute_hubert_units_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/precompute_hubert_units_%j.err

# Extract HuBERT km100 units for every EMG utterance, from its parallel voiced
# `_audio_clean.flac`. Runs INSIDE the HuBERTVoc Singularity overlay (it has fairseq +
# the HuBERT/k-means checkpoints); the project env lacks fairseq. The shared CUDA image
# and th3482's overlay are world-readable.

set -euo pipefail

REPO=${REPO:-/scratch/cr4206/sEMG-unpaired-text}
CACHE_DIR=${CACHE_DIR:-${REPO}/data}
# Default = the FIX: silent samples target their parallel voiced take (spoken content),
# written to a separate dir so the old (degenerate, silent=self) cache is preserved.
SILENT_TARGET=${SILENT_TARGET:-voiced}
OUT_DIR=${OUT_DIR:-${REPO}/data/hubert_units_voiced}
VOICED_ROOT=${VOICED_ROOT:-/scratch/cr4206/data/emg_data/emg_data/voiced_parallel_data}
SPLITS=${SPLITS:-"train dev test"}
LAYER=${LAYER:-6}
HUBERTVOC_ROOT=${HUBERTVOC_ROOT:-/scratch/th3482/HuBERTVoc}
SIF=${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
OVERLAY=${OVERLAY:-${HUBERTVOC_ROOT}/singularity/overlay-50G-10M.ext3}
SINGULARITY_BIN=${SINGULARITY_BIN:-singularity}

mkdir -p "${REPO}/logs" "$OUT_DIR"

echo "[precompute_hubert_units] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[precompute_hubert_units] cache=$CACHE_DIR out=$OUT_DIR splits='$SPLITS' layer=$LAYER"
echo "[precompute_hubert_units] hubertvoc=$HUBERTVOC_ROOT sif=$SIF"

# Run the precompute in the torch2 overlay env. Outer double quotes let the host shell
# expand ${REPO} etc.; \$VARS are escaped to expand inside the container.
"$SINGULARITY_BIN" exec --nv \
  --overlay "${OVERLAY}:ro" \
  --bind /scratch:/scratch \
  "$SIF" \
  bash -lc "
    source /ext3/env.sh
    conda activate /ext3/miniforge3/envs/torch2
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    export NUMBA_CACHE_DIR=/tmp/hubertvoc_numba_\${SLURM_JOB_ID:-0}; mkdir -p \"\$NUMBA_CACHE_DIR\"
    python -u ${REPO}/scripts/precompute_hubert_units.py \
      --cache-dir ${CACHE_DIR} \
      --out-dir ${OUT_DIR} \
      --splits ${SPLITS} \
      --layer ${LAYER} \
      --silent-target ${SILENT_TARGET} \
      --voiced-root ${VOICED_ROOT} \
      --hubertvoc-root ${HUBERTVOC_ROOT}
  "

echo "[precompute_hubert_units] finished at=$(date -Is)"
