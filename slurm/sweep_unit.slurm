#!/bin/bash
# Token-resolution sweep for ONE unit in a SINGLE slurm allocation: trains each
# factor sequentially (one queue wait, results land one factor at a time). A failed
# factor is logged and skipped so the rest of the sweep continues.
#
# Usage:
#   UNIT=char    FACTORS="10 16 20 25"        sbatch slurm/sweep_unit.slurm
#   UNIT=subword SUBWORD_MODEL=data/tokenizers/subword_500.model \
#                FACTORS="16 20 25 32 40"     sbatch slurm/sweep_unit.slurm
#   UNIT=phoneme PHONEME_DICT=data/tokenizers/phoneme_g2p.dict \
#                FACTORS="8 10 16 20 25"      sbatch slurm/sweep_unit.slurm
#
#SBATCH --job-name=sweep_unit
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/sweep_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/sweep_%j.err

set -uo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data}
UNIT=${UNIT:?set UNIT=char|subword|phoneme}
FACTORS=${FACTORS:?set FACTORS="16 20 25 ..."}
SUBWORD_MODEL=${SUBWORD_MODEL:-}
PHONEME_DICT=${PHONEME_DICT:-}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}
mkdir -p logs

TAG=$UNIT
[ -n "$SUBWORD_MODEL" ] && TAG="$(basename "$SUBWORD_MODEL" .model)"   # e.g. subword_500
EXTRA=()
[ -n "$SUBWORD_MODEL" ] && EXTRA+=(--subword-model "$SUBWORD_MODEL")
[ -n "$PHONEME_DICT" ] && EXTRA+=(--phoneme-dict "$PHONEME_DICT")

echo "[sweep] job=${SLURM_JOB_ID:-manual} host=$(hostname) unit=$UNIT tag=$TAG factors=[$FACTORS] start=$(date -Is)"

for F in $FACTORS; do
  OUT=/scratch/cr4206/sEMG-unpaired-text/runs/baseline_${TAG}_${F}x
  echo "[sweep] ===== factor=${F}x -> $OUT  ($(date -Is)) ====="
  mkdir -p "$OUT"
  if "$PYTHON" -u scripts/train_baseline.py \
      --config configs/train_baseline.yaml \
      --cache-dir "$CACHE_DIR" \
      --output-directory "$OUT" \
      --unit "$UNIT" \
      --downsample-factor "$F" \
      "${EXTRA[@]}" \
      --wandb --wandb-run-name "${TAG}_${F}x"; then
    echo "[sweep] factor=${F}x done at=$(date -Is)"
  else
    echo "[sweep] factor=${F}x FAILED (rc=$?), continuing"
  fi
done

echo "[sweep] all factors finished at=$(date -Is)"
