#!/bin/bash
#SBATCH --job-name=eval_hubert_resynth
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/eval_hubert_resynth_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/eval_hubert_resynth_%j.err

# Real word-WER for an EMG->HuBERT-units model, via re-synthesis + ASR. Three stages:
#   predict (project env) -> vocode (HuBERTVoc container) -> asr (project env).
# Re-run a subset with e.g. STAGES="asr".

set -euo pipefail

REPO=${REPO:-/scratch/cr4206/sEMG-unpaired-text}
PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
RUN_DIR=${RUN_DIR:-${REPO}/runs/uml_hubert}        # dir holding best_emg_branch.pt
CKPT=${CKPT:-best_emg_branch.pt}
SPLIT=${SPLIT:-test}
OUT=${OUT:-${REPO}/runs/hubert_resynth}
CACHE_DIR=${CACHE_DIR:-${REPO}/data}
UNITS_DIR=${UNITS_DIR:-${REPO}/data/hubert_units}
HUBERT_K=${HUBERT_K:-100}
FACTOR=${FACTOR:-8}
WHISPER_MODEL=${WHISPER_MODEL:-openai/whisper-small}
STAGES=${STAGES:-"predict vocode asr"}

HUBERTVOC_ROOT=${HUBERTVOC_ROOT:-/scratch/th3482/HuBERTVoc}
SIF=${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
OVERLAY=${OVERLAY:-${HUBERTVOC_ROOT}/singularity/overlay-50G-10M.ext3}
SINGULARITY_BIN=${SINGULARITY_BIN:-singularity}

cd "$REPO"
export PYTHONPATH=${REPO}${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=${HF_HOME:-/scratch/cr4206/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/cr4206/.cache/huggingface}
mkdir -p "${REPO}/logs" "$OUT" "$HF_HOME"

echo "[eval_hubert_resynth] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[eval_hubert_resynth] run_dir=$RUN_DIR ckpt=$CKPT split=$SPLIT factor=$FACTOR stages='$STAGES'"

# Stage A: EMG -> predicted units (+ gold units, refs, UER).
if [[ " $STAGES " == *" predict "* ]]; then
  "$PYTHON" -u scripts/eval_hubert_resynth.py --stage predict \
    --run-dir "$RUN_DIR" --ckpt "$CKPT" --split "$SPLIT" --out-dir "$OUT" \
    --cache-dir "$CACHE_DIR" --hubert-units-dir "$UNITS_DIR" \
    --hubert-k "$HUBERT_K" --downsample-factor "$FACTOR"
fi

# Stage B: units -> wavs, via HuBERTVoc's own unit2speech (no new vocoder code).
if [[ " $STAGES " == *" vocode "* ]]; then
  for kind in pred gold; do
    "$SINGULARITY_BIN" exec --nv \
      --overlay "${OVERLAY}:ro" \
      --bind "${HUBERTVOC_ROOT}:/workspace" \
      --bind /scratch:/scratch \
      "$SIF" \
      /workspace/scripts/container_python.sh scripts/hubert_pipeline.py unit2speech \
        --input "${OUT}/${SPLIT}/${kind}.units" \
        --output-dir "${OUT}/${SPLIT}/${kind}_wavs"
  done
fi

# Stage C: wavs -> Whisper -> word-WER (EMG->units + topline).
if [[ " $STAGES " == *" asr "* ]]; then
  "$PYTHON" -u scripts/eval_hubert_resynth.py --stage asr \
    --split "$SPLIT" --out-dir "$OUT" --whisper-model "$WHISPER_MODEL"
fi

echo "[eval_hubert_resynth] finished at=$(date -Is)"
