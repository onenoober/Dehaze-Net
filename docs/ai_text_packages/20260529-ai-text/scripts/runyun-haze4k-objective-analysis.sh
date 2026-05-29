#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/Dehaze-Net"
CODE_DIR="$ROOT/code"
COMPARE_DIR="${COMPARE_DIR:-$ROOT/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-20260522}"
OUTPUT_DIR="${OUTPUT_DIR:-$COMPARE_DIR/analysis}"
CURRENT_DIR_NAME="${CURRENT_DIR_NAME:-lf}"
CURRENT_LABEL="${CURRENT_LABEL:-DEA-Net-LF}"
BASELINE_LABEL="${BASELINE_LABEL:-DEA-Net-CR}"
TOP_K="${TOP_K:-8}"

cd "$CODE_DIR"

echo "COMPARE_DIR=$COMPARE_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "BASELINE_LABEL=$BASELINE_LABEL"
echo "CURRENT_DIR_NAME=$CURRENT_DIR_NAME"
echo "CURRENT_LABEL=$CURRENT_LABEL"
echo "TOP_K=$TOP_K"

/opt/anaconda/envs/py310/bin/python analyze_visual_compare.py \
  --compare_dir "$COMPARE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --baseline_label "$BASELINE_LABEL" \
  --current_dir_name "$CURRENT_DIR_NAME" \
  --current_label "$CURRENT_LABEL" \
  --top_k "$TOP_K" \
  --save_heatmaps \
  --save_diagnostic_panels
