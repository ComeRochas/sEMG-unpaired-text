#!/bin/bash
# Decode a phoneme CTC checkpoint to WORDS (lexicon + KenLM) and report word WER.
# Usage:
#   CKPT=runs/baseline_phoneme_10x/last.pt FACTOR=10 sbatch slurm/phoneme_to_words.slurm
#
#SBATCH --job-name=p2w
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/p2w_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/p2w_%j.err

set -euo pipefail
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CKPT=${CKPT:-runs/baseline_phoneme_10x/last.pt}
FACTOR=${FACTOR:-10}
SPLIT=${SPLIT:-test}
cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export NLTK_DATA=/home/cr4206/nltk_data

"$PYTHON" -u scripts/phoneme_to_words.py \
  --checkpoint "$CKPT" --downsample-factor "$FACTOR" --split "$SPLIT" --k 8
