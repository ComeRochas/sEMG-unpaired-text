#!/bin/bash
#SBATCH --job-name=precompute_audio_gaddy
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/precompute_audio_gaddy_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/precompute_audio_gaddy_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
EMG_DATA_ROOT=${EMG_DATA_ROOT:-/scratch/cr4206/data/emg_data/emg_data}
OUT_DIR=${OUT_DIR:-/scratch/cr4206/sEMG-unpaired-text/data/gaddy_audio_cache}
SPLIT_NAME=${SPLIT_NAME:-gaddy_internal}
NUM_WORKERS=${NUM_WORKERS:-16}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p /scratch/cr4206/sEMG-unpaired-text/logs "$OUT_DIR"

echo "[precompute_audio_gaddy] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[precompute_audio_gaddy] emg_data_root=$EMG_DATA_ROOT out_dir=$OUT_DIR split_name=$SPLIT_NAME"

"$PYTHON" -u scripts/precompute_audio_gaddy.py \
  --emg-data-root "$EMG_DATA_ROOT" \
  --out-dir       "$OUT_DIR" \
  --split-name    "$SPLIT_NAME" \
  --num-workers   "$NUM_WORKERS"

echo "[precompute_audio_gaddy] finished at=$(date -Is)"
