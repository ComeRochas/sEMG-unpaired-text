#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/cr4206/sEMGencoderJEPA/logs/evaluate_%j.out
#SBATCH --error=/scratch/cr4206/sEMGencoderJEPA/logs/evaluate_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMGencoderJEPA/data}
SPLIT=${SPLIT:-test}
METHOD=${METHOD:-beam}
GRID_SEARCH=${GRID_SEARCH:-0}     # 1 = on, 0 = off

# Edit this list to evaluate multiple checkpoints in one job.
# Override from CLI: CHECKPOINTS="runs/a/best.pt runs/b/best.pt" sbatch slurm/evaluate.slurm
CHECKPOINTS=${CHECKPOINTS:-"runs/baseline/best_20260511_1651.pt runs/finetune_uml_gaddy_lambda0.5/best.pt"}

cd /scratch/cr4206/sEMGencoderJEPA
export PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA${PYTHONPATH:+:$PYTHONPATH}

mkdir -p /scratch/cr4206/sEMGencoderJEPA/logs

echo "[evaluate] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[evaluate] checkpoints=$CHECKPOINTS split=$SPLIT method=$METHOD grid_search=$GRID_SEARCH"

GRID_FLAG=""
[[ "$GRID_SEARCH" == "1" ]] && GRID_FLAG="--grid-search"

"$PYTHON" -u scripts/evaluate.py \
  --cache-dir "$CACHE_DIR" \
  --checkpoints $CHECKPOINTS \
  --split "$SPLIT" \
  --method "$METHOD" \
  $GRID_FLAG

echo "[evaluate] finished at=$(date -Is)"
