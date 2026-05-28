#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-/opt/anaconda/envs/py310/bin/python}"
RUN_ID="${RUN_ID:-HAZE4K-residual-field-confidence-preflight-runyun-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/residual_field_confidence_preflight/$RUN_ID}"

BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"
LFV1_CHECKPOINT="${LFV1_CHECKPOINT:-/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"

"$PY" - <<'PY' || "$PY" -m pip install scikit-learn==1.7.2
import sklearn
print("scikit-learn", sklearn.__version__)
PY

mkdir -p "$OUTPUT_DIR"
cd "$ROOT/code"

echo "RUN_ID=$RUN_ID"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "BASELINE_CHECKPOINT=$BASELINE_CHECKPOINT"
echo "LFV1_CHECKPOINT=$LFV1_CHECKPOINT"

"$PY" analyze_residual_field_confidence_preflight.py \
  --dataset_root ../dataset/HAZE4K \
  --split train \
  --baseline_checkpoint "$BASELINE_CHECKPOINT" \
  --lfv1_checkpoint "$LFV1_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --patch_size "${PATCH_SIZE:-256}" \
  --patches_per_image "${PATCHES_PER_IMAGE:-4}" \
  --max_images "${MAX_IMAGES:-0}" \
  --splits "${SPLITS:-3}" \
  --heads "${HEADS:-sklearn_hgb,sklearn_ridge}" \
  --main_feature_set "${MAIN_FEATURE_SET:-hazy_wavelet_plus_teacher_outputs}" \
  --main_head "${MAIN_HEAD:-sklearn_hgb}"
