#!/bin/bash
#SBATCH --job-name=finetune_from_uml
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMG-unpaired-text/logs/finetune_from_uml_%j.out
#SBATCH --error=/scratch/cr4206/sEMG-unpaired-text/logs/finetune_from_uml_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMG-unpaired-text/data}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMG-unpaired-text/runs/finetune_from_uml}

# Either point at a *_emg_branch.pt (preferred) or a full UMLModel state dict.
EMG_BRANCH=${EMG_BRANCH:-/scratch/cr4206/sEMG-unpaired-text/runs/uml/best_emg_branch.pt}
UML_FULL_CKPT=${UML_FULL_CKPT:-}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

cd /scratch/cr4206/sEMG-unpaired-text
export PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text${PYTHONPATH:+:$PYTHONPATH}

mkdir -p /scratch/cr4206/sEMG-unpaired-text/logs "$OUTPUT_DIR"

CKPT_FLAG=""
if [ -n "$EMG_BRANCH" ] && [ -f "$EMG_BRANCH" ]; then
    CKPT_FLAG="--emg-branch $EMG_BRANCH"
elif [ -n "$UML_FULL_CKPT" ] && [ -f "$UML_FULL_CKPT" ]; then
    CKPT_FLAG="--uml-full-ckpt $UML_FULL_CKPT"
else
    echo "[error] neither EMG_BRANCH nor UML_FULL_CKPT points at an existing file"
    echo "        EMG_BRANCH=$EMG_BRANCH"
    echo "        UML_FULL_CKPT=$UML_FULL_CKPT"
    exit 1
fi

echo "[finetune_from_uml] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[finetune_from_uml] cache_dir=$CACHE_DIR output_dir=$OUTPUT_DIR ckpt=$CKPT_FLAG"

"$PYTHON" -u scripts/finetune_from_uml.py \
  --config configs/finetune_from_uml.yaml \
  --cache-dir "$CACHE_DIR" \
  $CKPT_FLAG \
  --output-directory "$OUTPUT_DIR" \
  $EXTRA_FLAGS \
  --wandb

echo "[finetune_from_uml] finished at=$(date -Is)"
