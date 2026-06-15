#!/bin/bash
# Phoneme CTC diagnostics: greedy PER + phone-level reconstructions for the 3 checkpoints.
# Usage:  sbatch slurm/phoneme_diag.slurm
#         K=12 SPLIT=test sbatch slurm/phoneme_diag.slurm
#
#SBATCH --job-name=ph_diag
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/ph_diag_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/ph_diag_%j.err

set -euo pipefail
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
SPLIT=${SPLIT:-test}
K=${K:-8}
cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export NLTK_DATA=/home/cr4206/nltk_data

"$PYTHON" -u scripts/phoneme_diag.py --split "$SPLIT" --k "$K"
