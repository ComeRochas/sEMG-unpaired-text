#!/bin/bash
# Build the open-vocab phoneme lexicon (g2p over LibriSpeech unigrams). CPU-only, ~once.
# Usage:  sbatch slurm/build_lexicon.slurm
#
#SBATCH --job-name=build_lex
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/build_lex_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/build_lex_%j.err

set -euo pipefail
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export NLTK_DATA=/home/cr4206/nltk_data

"$PYTHON" -u scripts/build_lexicon_from_unigrams.py
