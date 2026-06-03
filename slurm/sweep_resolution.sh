#!/bin/bash
# Launch a token-resolution sweep for ONE target unit: one train_baseline job per
# downsample factor. Each job uses configs/train_baseline.yaml (200 epochs, schedule
# [125,150,175]) and writes to runs/baseline_<tag>_<factor>x.
#
# Usage:
#   UNIT=char    FACTORS="10 16 20 25"        bash slurm/sweep_resolution.sh
#   UNIT=phoneme PHONEME_DICT=/home/cr4206/nltk_data/corpora/cmudict/cmudict \
#                FACTORS="8 10 16 20 25"      bash slurm/sweep_resolution.sh
#   UNIT=subword SUBWORD_MODEL=data/tokenizers/subword_500.model \
#                FACTORS="16 20 25 32 40"     bash slurm/sweep_resolution.sh
#
# Notes:
#   - char-8x is already trained (runs/baseline_char_8x); omit 8 from char FACTORS.
#   - Factors must divide fixed_raw_len=1600 (i.e. be 2^a*5^b): 8,10,16,20,25,32,40,50,...
#   - Does NOT poll the queue afterwards (HPC rule). Check logs/ when jobs finish.
set -euo pipefail

ROOT=/scratch/cr4206/sEMG-unpaired-text
UNIT=${UNIT:?set UNIT=char|subword|phoneme}
FACTORS=${FACTORS:?set FACTORS="16 20 25 ..."}
SUBWORD_MODEL=${SUBWORD_MODEL:-}
PHONEME_DICT=${PHONEME_DICT:-}

# Run-dir tag: subword vocab name if applicable, else the unit.
TAG=$UNIT
[ -n "$SUBWORD_MODEL" ] && TAG="$(basename "$SUBWORD_MODEL" .model)"   # e.g. subword_500

cd "$ROOT"
for F in $FACTORS; do
  OUT=$ROOT/runs/baseline_${TAG}_${F}x
  echo "[sweep] unit=$UNIT factor=${F}x -> $OUT"
  UNIT="$UNIT" DOWNSAMPLE_FACTOR="$F" \
    SUBWORD_MODEL="$SUBWORD_MODEL" PHONEME_DICT="$PHONEME_DICT" \
    OUTPUT_DIR="$OUT" \
    sbatch slurm/train_baseline.slurm
done
