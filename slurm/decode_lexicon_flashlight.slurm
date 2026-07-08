#!/bin/bash
# Apples-to-apples: char + phoneme through the SAME flashlight lexicon+KenLM decoder.
# Usage:  sbatch slurm/decode_lexicon_flashlight.slurm
#
#SBATCH --job-name=flash_cmp
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/flash_cmp_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/flash_cmp_%j.err

set -euo pipefail
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
SPLIT=${SPLIT:-test}
CKPT_NAME=${CKPT_NAME:-last.pt}
cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
export NLTK_DATA=/home/cr4206/nltk_data

# RUNS="unit:run_dir:factor unit:run_dir:factor ..." overrides the script's DEFAULT_RUNS.
RUNS_FLAG=()
if [ -n "${RUNS:-}" ]; then
  RUNS_FLAG=(--runs $RUNS)
fi

"$PYTHON" -u scripts/decode_lexicon_flashlight.py --split "$SPLIT" --k 4 \
  --ckpt-name "$CKPT_NAME" "${RUNS_FLAG[@]}"
