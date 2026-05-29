#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CODE_DIR="$ROOT/code"
if [[ -z "${DATASET_ROOT:-}" ]]; then
  if [[ -d "$ROOT/dataset/HAZE4K" ]]; then
    DATASET_ROOT="$ROOT/dataset/HAZE4K"
  else
    DATASET_ROOT="/root/workspace/Dehaze-Net/dataset/HAZE4K"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/supervised_preserve_proxy/HAZE4K-supervised-preserve-proxy-$(date +%Y%m%d-%H%M%S)}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"
LFV1_CHECKPOINT="${LFV1_CHECKPOINT:-/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"
PYTHON_BIN="${PYTHON_BIN:-}"
SPLIT="${SPLIT:-train}"
PATCH_SIZE="${PATCH_SIZE:-256}"
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-4}"
MAX_IMAGES="${MAX_IMAGES:-0}"
SPLITS="${SPLITS:-3}"
FEATURE_SETS="${FEATURE_SETS:-hazy_wavelet,teacher_output_proxy,hazy_wavelet_plus_teacher_outputs}"
HEADS="${HEADS:-logistic,mlp}"
HELDOUT_GROUPS="${HELDOUT_GROUPS-airlight_bin,beta_bin}"
MLP_EPOCHS="${MLP_EPOCHS:-40}"
HEAD_DEVICE="${HEAD_DEVICE:-cuda}"
MAIN_FEATURE_SET="${MAIN_FEATURE_SET:-hazy_wavelet_plus_teacher_outputs}"
MAIN_HEAD="${MAIN_HEAD:-logistic}"
LOGISTIC_STEPS="${LOGISTIC_STEPS:-500}"
FEATURES_CSV="${FEATURES_CSV:-}"
SKLEARN_MAX_ITER="${SKLEARN_MAX_ITER:-1000}"
SKLEARN_C="${SKLEARN_C:-1.0}"
SKLEARN_SOLVER="${SKLEARN_SOLVER:-lbfgs}"

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
echo "DATASET_ROOT=$DATASET_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "BASELINE_CHECKPOINT=$BASELINE_CHECKPOINT"
echo "LFV1_CHECKPOINT=$LFV1_CHECKPOINT"
echo "SPLIT=$SPLIT"
echo "PATCH_SIZE=$PATCH_SIZE"
echo "PATCHES_PER_IMAGE=$PATCHES_PER_IMAGE"
echo "MAX_IMAGES=$MAX_IMAGES"
echo "SPLITS=$SPLITS"
echo "FEATURE_SETS=$FEATURE_SETS"
echo "HEADS=$HEADS"
echo "HELDOUT_GROUPS=$HELDOUT_GROUPS"
echo "MLP_EPOCHS=$MLP_EPOCHS"
echo "HEAD_DEVICE=$HEAD_DEVICE"
echo "MAIN_FEATURE_SET=$MAIN_FEATURE_SET"
echo "MAIN_HEAD=$MAIN_HEAD"
echo "LOGISTIC_STEPS=$LOGISTIC_STEPS"
echo "FEATURES_CSV=$FEATURES_CSV"
echo "SKLEARN_MAX_ITER=$SKLEARN_MAX_ITER"
echo "SKLEARN_C=$SKLEARN_C"
echo "SKLEARN_SOLVER=$SKLEARN_SOLVER"

EXTRA_ARGS=()
if [[ -n "$FEATURES_CSV" ]]; then
  EXTRA_ARGS+=(--features_csv "$FEATURES_CSV")
fi

"$PYTHON_BIN" analyze_supervised_preserve_proxy.py \
  --dataset_root "$DATASET_ROOT" \
  --split "$SPLIT" \
  --baseline_checkpoint "$BASELINE_CHECKPOINT" \
  --lfv1_checkpoint "$LFV1_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --patch_size "$PATCH_SIZE" \
  --patches_per_image "$PATCHES_PER_IMAGE" \
  --max_images "$MAX_IMAGES" \
  --splits "$SPLITS" \
  --feature_sets "$FEATURE_SETS" \
  --heads "$HEADS" \
  --heldout_groups "$HELDOUT_GROUPS" \
  --logistic_steps "$LOGISTIC_STEPS" \
  --sklearn_max_iter "$SKLEARN_MAX_ITER" \
  --sklearn_c "$SKLEARN_C" \
  --sklearn_solver "$SKLEARN_SOLVER" \
  --mlp_epochs "$MLP_EPOCHS" \
  --head_device "$HEAD_DEVICE" \
  --main_feature_set "$MAIN_FEATURE_SET" \
  --main_head "$MAIN_HEAD" \
  "${EXTRA_ARGS[@]}" \
  "$@"
