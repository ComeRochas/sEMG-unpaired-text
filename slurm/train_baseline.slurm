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
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/train_baseline_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/train_baseline_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data}
# Phase-2: override per (unit, resolution). Examples:
#   UNIT=char    CONV_STRIDES="2 2 2"   OUTPUT_DIR=runs/baseline_char_8x     sbatch slurm/train_baseline.slurm
#   UNIT=subword SUBWORD_MODEL=data/tokenizers/subword_500.model \
#                CONV_STRIDES="2 2 2 2" OUTPUT_DIR=runs/baseline_subword_16x sbatch slurm/train_baseline.slurm
UNIT=${UNIT:-char}
CONV_STRIDES=${CONV_STRIDES:-2 2 2}
SUBWORD_MODEL=${SUBWORD_MODEL:-}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMG-unpaired-text/runs/baseline_${UNIT}}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}

mkdir -p /scratch/cr4206/sEMG-unpaired-text/logs "$OUTPUT_DIR"

echo "[train_baseline] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[train_baseline] cache_dir=$CACHE_DIR output_dir=$OUTPUT_DIR unit=$UNIT conv_strides=$CONV_STRIDES"

EXTRA_ARGS=()
[ -n "$SUBWORD_MODEL" ] && EXTRA_ARGS+=(--subword-model "$SUBWORD_MODEL")

"$PYTHON" -u scripts/train_baseline.py \
  --config configs/train_baseline.yaml \
  --cache-dir "$CACHE_DIR" \
  --output-directory "$OUTPUT_DIR" \
  --unit "$UNIT" \
  --conv-strides $CONV_STRIDES \
  "${EXTRA_ARGS[@]}" \
  --wandb \
  --epochs 400 --lr-decay-milestones 250 300 350

echo "[train_baseline] finished at=$(date -Is)"
