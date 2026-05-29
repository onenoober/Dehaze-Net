#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CODE_DIR="$ROOT/code"
MATRIX_CSV="${MATRIX_CSV:-$ROOT/experiment/HAZE4K/route_evidence_review/HAZE4K-route-evidence-review-20260528/model_per_image_matrix.csv}"
HAZY_DIR="${HAZY_DIR:-$ROOT/dataset/HAZE4K/test/hazy}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/wavelet_preserve_proxy/HAZE4K-wavelet-preserve-proxy-$(date +%Y%m%d-%H%M%S)}"
SPLITS="${SPLITS:-5}"
VALID_FRACTION="${VALID_FRACTION:-0.30}"
LOGISTIC_STEPS="${LOGISTIC_STEPS:-400}"
MAX_IMAGES="${MAX_IMAGES:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x /opt/anaconda/envs/py310/bin/python ]]; then
    PYTHON_BIN="/opt/anaconda/envs/py310/bin/python"
  elif [[ -x /root/miniconda3/envs/py310/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/envs/py310/bin/python"
  elif [[ -x /home/ubuntu/miniconda3/envs/py310/bin/python ]]; then
    PYTHON_BIN="/home/ubuntu/miniconda3/envs/py310/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

cd "$CODE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "PYTHON_BIN=$PYTHON_BIN"
echo "MATRIX_CSV=$MATRIX_CSV"
echo "HAZY_DIR=$HAZY_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SPLITS=$SPLITS"
echo "VALID_FRACTION=$VALID_FRACTION"
echo "LOGISTIC_STEPS=$LOGISTIC_STEPS"
echo "MAX_IMAGES=$MAX_IMAGES"

"$PYTHON_BIN" analyze_wavelet_preserve_proxy.py \
  --matrix_csv "$MATRIX_CSV" \
  --hazy_dir "$HAZY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --splits "$SPLITS" \
  --valid_fraction "$VALID_FRACTION" \
  --logistic_steps "$LOGISTIC_STEPS" \
  --max_images "$MAX_IMAGES" \
  "$@"
