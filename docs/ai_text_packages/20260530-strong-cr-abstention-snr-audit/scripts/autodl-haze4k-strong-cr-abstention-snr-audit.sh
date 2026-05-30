#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-/root/miniconda3/envs/py310/bin/python}"
RUN_ID="${RUN_ID:-HAZE4K-strong-cr-abstention-snr-audit-autodl-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/strong_cr_abstention_snr_audit/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$ROOT/experiment/HAZE4K/_run_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"
LFV1_CHECKPOINT="${LFV1_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"
RESIDUALCALIB_CHECKPOINT="${RESIDUALCALIB_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654/saved_model/best.pk}"
CRPLUS_CHECKPOINT="${CRPLUS_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540/saved_model/best.pk}"
CBRFRC_CHECKPOINT="${CBRFRC_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811/saved_model/best.pk}"
BRFRC_V2_SUMMARY="${BRFRC_V2_SUMMARY:-$ROOT/experiment/HAZE4K/brfrc_v2_representation_audit/HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028/summary.json}"
ROUTE_EVIDENCE_CSV="${ROUTE_EVIDENCE_CSV:-$ROOT/docs/ai_text_packages/20260530-cbrfrc-v1/experiment/HAZE4K/route_evidence_review/HAZE4K-route-evidence-review-20260528/model_per_image_matrix.csv}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RUN_ID=$RUN_ID"
echo "ROOT=$ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "BASELINE_CHECKPOINT=$BASELINE_CHECKPOINT"
echo "LFV1_CHECKPOINT=$LFV1_CHECKPOINT"
echo "RESIDUALCALIB_CHECKPOINT=$RESIDUALCALIB_CHECKPOINT"
echo "CRPLUS_CHECKPOINT=$CRPLUS_CHECKPOINT"
echo "CBRFRC_CHECKPOINT=$CBRFRC_CHECKPOINT"
echo "BRFRC_V2_SUMMARY=$BRFRC_V2_SUMMARY"
echo "ROUTE_EVIDENCE_CSV=$ROUTE_EVIDENCE_CSV"
date

"$PY" - <<'PY' || "$PY" -m pip install scikit-learn==1.7.2
import sklearn
print("scikit-learn", sklearn.__version__)
PY

test -f "$BASELINE_CHECKPOINT"
test -f "$LFV1_CHECKPOINT"
test -f "$RESIDUALCALIB_CHECKPOINT"
test -f "$CBRFRC_CHECKPOINT"
test -f "$ROUTE_EVIDENCE_CSV"

CRPLUS_ARGS=()
if [[ -f "$CRPLUS_CHECKPOINT" ]]; then
  CRPLUS_ARGS=(--crplus_checkpoint "$CRPLUS_CHECKPOINT")
else
  echo "CRPlus-v2 checkpoint not found on this server; using route evidence CSV metrics."
fi

cd "$ROOT/code"

"$PY" analyze_strong_cr_abstention_residual_snr_audit.py \
  --dataset_root ../dataset/HAZE4K \
  --split "${SPLIT:-test}" \
  --baseline_checkpoint "$BASELINE_CHECKPOINT" \
  --lfv1_checkpoint "$LFV1_CHECKPOINT" \
  --residualcalib_checkpoint "$RESIDUALCALIB_CHECKPOINT" \
  "${CRPLUS_ARGS[@]}" \
  --cbrfrc_checkpoint "$CBRFRC_CHECKPOINT" \
  --brfrc_v2_summary "$BRFRC_V2_SUMMARY" \
  --route_evidence_csv "$ROUTE_EVIDENCE_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --pad_size "${PAD_SIZE:-4}" \
  --target_grid "${TARGET_GRID:-8}" \
  --spatial_feature_grid "${SPATIAL_FEATURE_GRID:-4}" \
  --feature_pool_grid "${FEATURE_POOL_GRID:-2}" \
  --max_images "${MAX_IMAGES:-0}" \
  --random_splits "${RANDOM_SPLITS:-3}" \
  --valid_fraction "${VALID_FRACTION:-0.25}" \
  --heads "${HEADS:-logistic,ridge_classifier,hgb,calibrated_logistic}" \
  --feature_sets "${FEATURE_SETS:-A_output,D_feature_contrast,E_abstention_risk,E_abstention_risk_shuffled,F_diagnostic_leakage}" \
  --min_strong_preserve_recall "${MIN_STRONG_PRESERVE_RECALL:-0.75}" \
  --max_strong_false_intervention "${MAX_STRONG_FALSE_INTERVENTION:-0.20}" \
  --min_intervention_precision "${MIN_INTERVENTION_PRECISION:-0.60}" \
  --min_lfv1_gain_preservation "${MIN_LFV1_GAIN_PRESERVATION:-0.70}" \
  --min_confidence_corr "${MIN_CONFIDENCE_CORR:-0.45}" \
  --min_sim_psnr_margin_vs_lfv1 "${MIN_SIM_PSNR_MARGIN_VS_LFV1:--0.02}" \
  --preferred_sim_psnr_margin_vs_lfv1 "${PREFERRED_SIM_PSNR_MARGIN_VS_LFV1:-0.05}"

echo "summary=$OUTPUT_DIR/summary.json"
date
