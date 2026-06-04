#!/bin/bash
# Evaluate every completed sweep run on the test set and write runs/sweep_summary.md.
# Forward passes are fast on GPU; beam search is CPU-bound -> ~10-20 min for all runs.
#
# Usage:  sbatch slurm/analyze_sweep.slurm        (test split, all runs)
#
#SBATCH --job-name=analyze_sweep
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/analyze_sweep_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/analyze_sweep_%j.err

set -euo pipefail
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
SPLIT=${SPLIT:-test}
cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export NLTK_DATA=/home/cr4206/nltk_data

"$PYTHON" -u scripts/analyze_sweep_results.py \
  --split "$SPLIT" --k 6 --out runs/sweep_summary_${SPLIT}.md
