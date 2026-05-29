#!/bin/bash
#SBATCH --job-name=precompute_audio
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cr4206/sEMGencoderJEPA/logs/precompute_audio_%j.out
#SBATCH --error=/scratch/cr4206/sEMGencoderJEPA/logs/precompute_audio_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
LIBRISPEECH_DIR=${LIBRISPEECH_DIR:-/scratch/cr4206/data/librispeech}
OUT_DIR=${OUT_DIR:-/scratch/cr4206/sEMGencoderJEPA/data/libri_cache}
SPLITS=${SPLITS:-train-clean-100}
NUM_WORKERS=${NUM_WORKERS:-16}

cd /scratch/cr4206/sEMGencoderJEPA
export PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p /scratch/cr4206/sEMGencoderJEPA/logs "$OUT_DIR"

echo "[precompute_audio] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[precompute_audio] librispeech_dir=$LIBRISPEECH_DIR out_dir=$OUT_DIR splits=$SPLITS"

"$PYTHON" -u scripts/precompute_audio.py \
  --librispeech-dir "$LIBRISPEECH_DIR" \
  --out-dir "$OUT_DIR" \
  --splits $SPLITS \
  --num-workers "$NUM_WORKERS"

echo "[precompute_audio] finished at=$(date -Is)"
