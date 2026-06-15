#!/bin/bash
# Phonemes -> words via lexicon + KenLM beam over the phone lattice (torchaudio).
# Dev-tunes (lm_weight, word_score), reports test word WER for the 3 phoneme ckpts.
# Usage:  sbatch slurm/phoneme_to_words_lattice.slurm
#         SPLIT=test sbatch slurm/phoneme_to_words_lattice.slurm
#
#SBATCH --job-name=p2w_lat
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/p2w_lat_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/p2w_lat_%j.err

set -euo pipefail
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
SPLIT=${SPLIT:-test}
cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export NLTK_DATA=/home/cr4206/nltk_data

"$PYTHON" -u scripts/phoneme_to_words_lattice.py --split "$SPLIT" --k 8
