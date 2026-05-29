#!/bin/bash
#SBATCH --job-name=finetune_from_jepa
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cr4206/sEMGencoderJEPA/logs/finetune_from_jepa_%j.out
#SBATCH --error=/scratch/cr4206/sEMGencoderJEPA/logs/finetune_from_jepa_%j.err

set -euo pipefail

PYTHON=${PYTHON:-/scratch/cr4206/envs/silent_speech/bin/python}
CACHE_DIR=${CACHE_DIR:-/scratch/cr4206/sEMGencoderJEPA/data}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/cr4206/sEMGencoderJEPA/runs/jepa_finetune}
PRETRAINED_ENCODER=${PRETRAINED_ENCODER:-/scratch/cr4206/sEMGencoderJEPA/runs/jepa_pretrain/pretrained_encoder.pt}
# Optional fallback: if PRETRAINED_ENCODER doesn't exist but UML_BEST does,
# extract emg_encoder.* keys out of a full UMLModel checkpoint.
UML_BEST=${UML_BEST:-}

cd /scratch/cr4206/sEMGencoderJEPA
export PYTHONPATH=/scratch/cr4206/sEMGencoderJEPA${PYTHONPATH:+:$PYTHONPATH}

mkdir -p /scratch/cr4206/sEMGencoderJEPA/logs "$OUTPUT_DIR"

# Extraction fallback (used by UML→finetune): if pretrained_encoder.pt is
# missing but a UML best.pt exists, build pretrained_encoder.pt by
# stripping the "emg_encoder." prefix from the UML state dict.
if [ ! -f "$PRETRAINED_ENCODER" ] && [ -n "$UML_BEST" ] && [ -f "$UML_BEST" ]; then
    echo "[finetune_from_jepa] extracting encoder from $UML_BEST -> $PRETRAINED_ENCODER"
    "$PYTHON" -c "
import torch, os, sys
sd = torch.load(os.environ['UML_BEST'], map_location='cpu')
prefix = 'emg_encoder.'
enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
if not enc:
    sys.exit(f'no emg_encoder.* keys in {os.environ[\"UML_BEST\"]}')
out = os.environ['PRETRAINED_ENCODER']
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(enc, out)
print(f'wrote {len(enc)} encoder keys -> {out}')
"
fi

echo "[finetune_from_jepa] job=${SLURM_JOB_ID:-manual} host=$(hostname) start=$(date -Is)"
echo "[finetune_from_jepa] cache_dir=$CACHE_DIR output_dir=$OUTPUT_DIR pretrained_encoder=$PRETRAINED_ENCODER"

"$PYTHON" -u scripts/finetune_from_jepa.py \
  --config configs/finetune_from_jepa.yaml \
  --cache-dir "$CACHE_DIR" \
  --pretrained-encoder "$PRETRAINED_ENCODER" \
  --output-directory "$OUTPUT_DIR" \
  --wandb

echo "[finetune_from_jepa] finished at=$(date -Is)"
