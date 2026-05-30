#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-/root/miniconda3/envs/py310/bin/python}"
RUN_ID="${RUN_ID:-HAZE4K-strong-official-depth-safety-audit-autodl-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/strong_official_depth_safety_audit/$RUN_ID}"
DEPTH_CACHE_DIR="${DEPTH_CACHE_DIR:-$ROOT/experiment/HAZE4K/depth_cache/strong_official_depth_safety}"
LOG_DIR="${LOG_DIR:-$ROOT/experiment/HAZE4K/_run_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

OFFICIAL_RUN="${OFFICIAL_RUN:-$ROOT/experiment/HAZE4K/DEA-Net-OfficialWarmStart-LFv1-H4K-local-scout100k-20260528-152808}"
OFFICIAL_STEP0_CHECKPOINT="${OFFICIAL_STEP0_CHECKPOINT:-$OFFICIAL_RUN/saved_model/official_warmstart_step0.pk}"
OFFICIAL_WARM_CHECKPOINT="${OFFICIAL_WARM_CHECKPOINT:-$OFFICIAL_RUN/saved_model/best.pk}"
COLD_CR_CHECKPOINT="${COLD_CR_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"
LFV1_CHECKPOINT="${LFV1_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"
RESIDUALCALIB_CHECKPOINT="${RESIDUALCALIB_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654/saved_model/best.pk}"
CRPLUS_CHECKPOINT="${CRPLUS_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540/saved_model/best.pk}"
CBRFRC_CHECKPOINT="${CBRFRC_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811/saved_model/best.pk}"
ROUTE_EVIDENCE_CSV="${ROUTE_EVIDENCE_CSV:-$ROOT/docs/ai_text_packages/20260530-cbrfrc-v1/experiment/HAZE4K/route_evidence_review/HAZE4K-route-evidence-review-20260528/model_per_image_matrix.csv}"

DEPTH_MODELS="${DEPTH_MODELS:-depthanything=depth-anything/Depth-Anything-V2-Small-hf,midas=Intel/dpt-hybrid-midas}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$OUTPUT_DIR" "$DEPTH_CACHE_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RUN_ID=$RUN_ID"
echo "ROOT=$ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DEPTH_CACHE_DIR=$DEPTH_CACHE_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "OFFICIAL_STEP0_CHECKPOINT=$OFFICIAL_STEP0_CHECKPOINT"
echo "OFFICIAL_WARM_CHECKPOINT=$OFFICIAL_WARM_CHECKPOINT"
echo "COLD_CR_CHECKPOINT=$COLD_CR_CHECKPOINT"
echo "LFV1_CHECKPOINT=$LFV1_CHECKPOINT"
echo "RESIDUALCALIB_CHECKPOINT=$RESIDUALCALIB_CHECKPOINT"
echo "CRPLUS_CHECKPOINT=$CRPLUS_CHECKPOINT"
echo "CBRFRC_CHECKPOINT=$CBRFRC_CHECKPOINT"
echo "ROUTE_EVIDENCE_CSV=$ROUTE_EVIDENCE_CSV"
echo "DEPTH_MODELS=$DEPTH_MODELS"
echo "HF_ENDPOINT=$HF_ENDPOINT"
date

"$PY" - <<'PY' || "$PY" -m pip install scikit-learn==1.7.2 transformers safetensors huggingface_hub
import sklearn
import transformers
print("scikit-learn", sklearn.__version__)
print("transformers", transformers.__version__)
PY

test -f "$OFFICIAL_STEP0_CHECKPOINT"
test -f "$OFFICIAL_WARM_CHECKPOINT"
test -f "$COLD_CR_CHECKPOINT"
test -f "$LFV1_CHECKPOINT"
test -f "$RESIDUALCALIB_CHECKPOINT"
test -f "$CBRFRC_CHECKPOINT"
test -f "$ROUTE_EVIDENCE_CSV"

CRPLUS_ARGS=()
if [[ -f "$CRPLUS_CHECKPOINT" ]]; then
  CRPLUS_ARGS=(--crplus_checkpoint "$CRPLUS_CHECKPOINT")
else
  echo "CRPlus-v2 checkpoint not found on this server; using route evidence CSV metrics only."
fi

cd "$ROOT/code"

"$PY" analyze_strong_official_depth_safety_audit.py \
  --dataset_root ../dataset/HAZE4K \
  --split "${SPLIT:-test}" \
  --official_step0_checkpoint "$OFFICIAL_STEP0_CHECKPOINT" \
  --official_warm_checkpoint "$OFFICIAL_WARM_CHECKPOINT" \
  --cold_cr_checkpoint "$COLD_CR_CHECKPOINT" \
  --lfv1_checkpoint "$LFV1_CHECKPOINT" \
  --residualcalib_checkpoint "$RESIDUALCALIB_CHECKPOINT" \
  "${CRPLUS_ARGS[@]}" \
  --cbrfrc_checkpoint "$CBRFRC_CHECKPOINT" \
  --route_evidence_csv "$ROUTE_EVIDENCE_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --depth_cache_dir "$DEPTH_CACHE_DIR" \
  --depth_models "$DEPTH_MODELS" \
  --candidate_depth_names "${CANDIDATE_DEPTH_NAMES:-warm,cold_cr,lfv1,residualcalib,crplus,cbrfrc}" \
  --pad_size "${PAD_SIZE:-4}" \
  --spatial_feature_grid "${SPATIAL_FEATURE_GRID:-4}" \
  --feature_pool_grid "${FEATURE_POOL_GRID:-2}" \
  --max_images "${MAX_IMAGES:-0}" \
  --random_splits "${RANDOM_SPLITS:-3}" \
  --valid_fraction "${VALID_FRACTION:-0.25}" \
  --content_clusters "${CONTENT_CLUSTERS:-5}" \
  --target_margins "${TARGET_MARGINS:-0.20}" \
  --nochange_margin "${NOCHANGE_MARGIN:-0.05}" \
  --heads "${HEADS:-logistic,decision_tree,hgb}" \
  --tree_max_depth "${TREE_MAX_DEPTH:-3}" \
  --min_samples_leaf "${MIN_SAMPLES_LEAF:-20}" \
  --threshold_min_precision "${THRESHOLD_MIN_PRECISION:-0.55}" \
  --threshold_min_gain "${THRESHOLD_MIN_GAIN:-0.0}" \
  --threshold_min_intervention_rate "${THRESHOLD_MIN_INTERVENTION_RATE:-0.02}"

echo "decision=$OUTPUT_DIR/decision_summary.json"
echo "summary=$OUTPUT_DIR/summary.json"
date
