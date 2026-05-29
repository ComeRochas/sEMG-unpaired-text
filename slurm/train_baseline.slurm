#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMGencoderJEPA/logs/train_baseline_%j.out
#SBATCH --error=/scratch/cr4206/sEMGencoderJEPA/logs/train_baseline_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMGencoderJEPA/data}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMGencoderJEPA/runs/baseline}

cd /scratch/cr4206/sEMGencoderJEPA
export PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA${PYTHONPATH:+:$PYTHONPATH}

mkdir -p /scratch/cr4206/sEMGencoderJEPA/logs "$OUTPUT_DIR"

echo "[train_baseline] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[train_baseline] cache_dir=$CACHE_DIR output_dir=$OUTPUT_DIR"

"$PYTHON" -u scripts/train_baseline.py \
  --config configs/train_baseline.yaml \
  --cache-dir "$CACHE_DIR" \
  --output-directory "$OUTPUT_DIR" \
  --wandb \
  --epochs 400 --lr-decay-milestones 250 300 350

echo "[train_baseline] finished at=$(date -Is)"
