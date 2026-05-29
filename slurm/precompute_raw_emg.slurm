#!/bin/bash
#SBATCH --job-name=precompute_raw_emg
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cr4206/sEMGencoderJEPA/logs/precompute_raw_emg_%j.out
#SBATCH --error=/scratch/cr4206/sEMGencoderJEPA/logs/precompute_raw_emg_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
DATA_CONFIG=${DATA_CONFIG:-/scratch/cr4206/sEMGencoderJEPA/configs/data_config.json}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMGencoderJEPA/data}
NUM_WORKERS=${NUM_WORKERS:-16}

cd /scratch/cr4206/sEMGencoderJEPA
export PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p /scratch/cr4206/sEMGencoderJEPA/logs "$CACHE_DIR"

echo "[precompute_raw_emg] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[precompute_raw_emg] data_config=$DATA_CONFIG cache_dir=$CACHE_DIR num_workers=$NUM_WORKERS"

"$PYTHON" -u scripts/precompute_raw_emg.py \
  --data-config "$DATA_CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --num-workers "$NUM_WORKERS"

echo "[precompute_raw_emg] finished at=$(date -Is)"
