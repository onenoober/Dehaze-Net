#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-/root/miniconda3/envs/py310/bin/python}"
RUN_ID="${RUN_ID:-HAZE4K-brfrc-v2-representation-audit-autodl-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/brfrc_v2_representation_audit/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$ROOT/experiment/HAZE4K/_run_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"
LFV1_CHECKPOINT="${LFV1_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RUN_ID=$RUN_ID"
echo "ROOT=$ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "BASELINE_CHECKPOINT=$BASELINE_CHECKPOINT"
echo "LFV1_CHECKPOINT=$LFV1_CHECKPOINT"
date

"$PY" - <<'PY' || "$PY" -m pip install scikit-learn==1.7.2
import sklearn
print("scikit-learn", sklearn.__version__)
PY

test -f "$BASELINE_CHECKPOINT"
test -f "$LFV1_CHECKPOINT"

cd "$ROOT/code"

"$PY" analyze_brf_representation_audit.py \
  --dataset_root ../dataset/HAZE4K \
  --split train \
  --baseline_checkpoint "$BASELINE_CHECKPOINT" \
  --lfv1_checkpoint "$LFV1_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --pad_size "${PAD_SIZE:-4}" \
  --target_grid "${TARGET_GRID:-8}" \
  --spatial_feature_grid "${SPATIAL_FEATURE_GRID:-4}" \
  --feature_pool_grid "${FEATURE_POOL_GRID:-2}" \
  --max_images "${MAX_IMAGES:-0}" \
  --random_splits "${RANDOM_SPLITS:-3}" \
  --valid_fraction "${VALID_FRACTION:-0.25}" \
  --heads "${HEADS:-ridge,tiny_mlp}" \
  --feature_sets "${FEATURE_SETS:-A_output,B_cr_features,C_lfv1_features,D_feature_contrast,B_cr_features_shuffled,C_lfv1_features_shuffled,D_feature_contrast_shuffled}" \
  --ridge_alpha "${RIDGE_ALPHA:-10.0}" \
  --mlp_hidden "${MLP_HIDDEN:-64}" \
  --mlp_max_iter "${MLP_MAX_ITER:-160}" \
  --mlp_alpha "${MLP_ALPHA:-0.001}" \
  --min_residual_cosine "${MIN_RESIDUAL_COSINE:-0.20}" \
  --max_wrong_direction_rate "${MAX_WRONG_DIRECTION_RATE:-0.35}" \
  --min_lf_mse_improved_rate "${MIN_LF_MSE_IMPROVED_RATE:-0.55}" \
  --min_lfv1_gain_preservation "${MIN_LFV1_GAIN_PRESERVATION:-0.70}" \
  --min_strong_cr_preservation "${MIN_STRONG_CR_PRESERVATION:-0.70}" \
  --min_intervention_precision "${MIN_INTERVENTION_PRECISION:-0.60}" \
  --min_confidence_corr "${MIN_CONFIDENCE_CORR:-0.45}" \
  --min_output_cosine_gap "${MIN_OUTPUT_COSINE_GAP:-0.05}" \
  --min_shuffled_cosine_gap "${MIN_SHUFFLED_COSINE_GAP:-0.05}"

echo "summary=$OUTPUT_DIR/summary.json"
date
