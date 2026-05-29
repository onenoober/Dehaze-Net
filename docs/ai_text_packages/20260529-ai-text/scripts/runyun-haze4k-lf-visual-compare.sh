#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/Dehaze-Net"
CODE_DIR="$ROOT/code"
OUT_DIR="${OUT_DIR:-$ROOT/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-20260521}"
BASELINE_CKPT="${BASELINE_CKPT:-$ROOT/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"
LF_CKPT="${LF_CKPT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"

cd "$CODE_DIR"

echo "OUT_DIR=$OUT_DIR"
echo "BASELINE_CKPT=$BASELINE_CKPT"
echo "LF_CKPT=$LF_CKPT"
echo "NUM_SAMPLES=$NUM_SAMPLES"

/opt/anaconda/envs/py310/bin/python visual_compare_train_ckpt.py \
  --dataset HAZE4K \
  --split test \
  --output_dir "$OUT_DIR" \
  --baseline_checkpoint "$BASELINE_CKPT" \
  --lf_checkpoint "$LF_CKPT" \
  --num_samples "$NUM_SAMPLES"
