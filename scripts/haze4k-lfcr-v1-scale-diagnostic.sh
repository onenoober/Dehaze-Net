#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CODE_DIR="$ROOT/code"
CHECKPOINT="${CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"
MODEL_LABEL="${MODEL_LABEL:-DEA-Net-LF-v1}"
SPLIT="${SPLIT:-train}"
MAX_IMAGES="${MAX_IMAGES:-256}"
PATCH_SIZE="${PATCH_SIZE:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/loss_scale/lfcr-v1-lfv1-${SPLIT}${MAX_IMAGES}-$(date +%Y%m%d-%H%M%S)}"
LOWPASS_POOL="${LOWPASS_POOL:-8}"
TRAINING_MARGIN="${TRAINING_MARGIN:-0.02}"
FREQUENCY_WEIGHT="${FREQUENCY_WEIGHT:-0.1}"
LOWFREQ_WEIGHT="${LOWFREQ_WEIGHT:-0.1}"
CANDIDATE_WEIGHTS="${CANDIDATE_WEIGHTS:-0.001,0.003,0.005,0.01}"
NEGATIVE_MODES="${NEGATIVE_MODES:-hazy,hazy_lowpass,output_lowpass,under_dehazed_mix}"
SELECTED_NEGATIVE_MODES="${SELECTED_NEGATIVE_MODES:-hazy,output_lowpass,under_dehazed_mix}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x /root/miniconda3/envs/py310/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/envs/py310/bin/python"
  elif [[ -x /opt/anaconda/envs/py310/bin/python ]]; then
    PYTHON_BIN="/opt/anaconda/envs/py310/bin/python"
  elif [[ -x /home/ubuntu/miniconda3/envs/py310/bin/python ]]; then
    PYTHON_BIN="/home/ubuntu/miniconda3/envs/py310/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

cd "$CODE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "PYTHON_BIN=$PYTHON_BIN"
echo "CHECKPOINT=$CHECKPOINT"
echo "MODEL_LABEL=$MODEL_LABEL"
echo "SPLIT=$SPLIT"
echo "MAX_IMAGES=$MAX_IMAGES"
echo "PATCH_SIZE=$PATCH_SIZE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOWPASS_POOL=$LOWPASS_POOL"
echo "TRAINING_MARGIN=$TRAINING_MARGIN"
echo "FREQUENCY_WEIGHT=$FREQUENCY_WEIGHT"
echo "LOWFREQ_WEIGHT=$LOWFREQ_WEIGHT"
echo "CANDIDATE_WEIGHTS=$CANDIDATE_WEIGHTS"
echo "NEGATIVE_MODES=$NEGATIVE_MODES"
echo "SELECTED_NEGATIVE_MODES=$SELECTED_NEGATIVE_MODES"

"$PYTHON_BIN" analyze_crplus_v2_loss_scale.py \
  --dataset HAZE4K \
  --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" \
  --model_label "$MODEL_LABEL" \
  --output_dir "$OUTPUT_DIR" \
  --max_images "$MAX_IMAGES" \
  --patch_size "$PATCH_SIZE" \
  --lowpass_pool "$LOWPASS_POOL" \
  --training_margin "$TRAINING_MARGIN" \
  --frequency_weight "$FREQUENCY_WEIGHT" \
  --lowfreq_weight "$LOWFREQ_WEIGHT" \
  --candidate_weights "$CANDIDATE_WEIGHTS" \
  --negative_modes "$NEGATIVE_MODES" \
  --selected_negative_modes "$SELECTED_NEGATIVE_MODES" \
  --use_lf_prior \
  --lf_prior_channels "${LF_PRIOR_CHANNELS:-8}" \
  --lf_prior_pool "${LF_PRIOR_POOL:-8}" \
  --lf_prior_gate_init "${LF_PRIOR_GATE_INIT:-0.0}" \
  "$@"
