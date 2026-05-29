#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-/root/miniconda3/envs/py310/bin/python}"
RUN_ID="${RUN_ID:-HAZE4K-depth-guided-lf-preflight-autodl-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/depth_guided_lf_preflight/$RUN_ID}"
DEPTH_CACHE_DIR="${DEPTH_CACHE_DIR:-$ROOT/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf}"
LOG_DIR="${LOG_DIR:-$ROOT/experiment/HAZE4K/_run_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"
LFV1_CHECKPOINT="${LFV1_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"
DEPTH_MODEL="${DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Small-hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$OUTPUT_DIR" "$DEPTH_CACHE_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RUN_ID=$RUN_ID"
echo "ROOT=$ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DEPTH_CACHE_DIR=$DEPTH_CACHE_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "BASELINE_CHECKPOINT=$BASELINE_CHECKPOINT"
echo "LFV1_CHECKPOINT=$LFV1_CHECKPOINT"
echo "DEPTH_MODEL=$DEPTH_MODEL"
echo "HF_ENDPOINT=$HF_ENDPOINT"
date

"$PY" - <<'PY' || "$PY" -m pip install scikit-learn==1.7.2 transformers safetensors huggingface_hub
import sklearn
import transformers
print("scikit-learn", sklearn.__version__)
print("transformers", transformers.__version__)
PY

test -f "$BASELINE_CHECKPOINT"
test -f "$LFV1_CHECKPOINT"

cd "$ROOT/code"

"$PY" analyze_depth_guided_lf_preflight.py \
  --dataset_root ../dataset/HAZE4K \
  --split train \
  --baseline_checkpoint "$BASELINE_CHECKPOINT" \
  --lfv1_checkpoint "$LFV1_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --depth_cache_dir "$DEPTH_CACHE_DIR" \
  --depth_model "$DEPTH_MODEL" \
  --patch_size "${PATCH_SIZE:-256}" \
  --patches_per_image "${PATCHES_PER_IMAGE:-4}" \
  --max_images "${MAX_IMAGES:-0}" \
  --splits "${SPLITS:-3}" \
  --heads "${HEADS:-sklearn_hgb,sklearn_ridge}" \
  --feature_sets "${FEATURE_SETS:-hazy_wavelet,hazy_wavelet_plus_teacher_outputs,depth_only,hazy_depth,hazy_depth_plus_teacher_outputs,hazy_shuffled_depth_plus_teacher_outputs}" \
  --main_feature_set "${MAIN_FEATURE_SET:-hazy_depth_plus_teacher_outputs}" \
  --main_head "${MAIN_HEAD:-sklearn_hgb}" \
  --min_gain "${MIN_GAIN:-0.25}" \
  --min_preserve_recall "${MIN_PRESERVE_RECALL:-0.68}" \
  --min_regression_improve_recall "${MIN_REGRESSION_IMPROVE_RECALL:-0.60}" \
  --min_intervention_precision "${MIN_INTERVENTION_PRECISION:-0.60}" \
  --min_strong_cr_regression_improve_recall "${MIN_STRONG_CR_REGRESSION_IMPROVE_RECALL:-0.60}" \
  --min_oracle_c_corr "${MIN_ORACLE_C_CORR:-0.45}" \
  --min_heldout_gain "${MIN_HELDOUT_GAIN:-0.0}" \
  --min_heldout_preserve_recall "${MIN_HELDOUT_PRESERVE_RECALL:-0.62}" \
  --min_heldout_regression_improve_recall "${MIN_HELDOUT_REGRESSION_IMPROVE_RECALL:-0.45}" \
  --min_heldout_oracle_c_corr "${MIN_HELDOUT_ORACLE_C_CORR:-0.30}"

echo "summary=$OUTPUT_DIR/summary.json"
date
