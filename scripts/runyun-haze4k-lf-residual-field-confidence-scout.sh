#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CODE_DIR="$ROOT/code"
PY="${PY:-/opt/anaconda/envs/py310/bin/python}"

W_LOSS_CR_REF_RESIDUAL_VALUE="${W_LOSS_CR_REF_RESIDUAL:-0.001}"
MODEL_NAME="${MODEL_NAME:-DEA-Net-LF-RFC-v1-w${W_LOSS_CR_REF_RESIDUAL_VALUE}-H4K-scout100k-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT/experiment/HAZE4K/_run_logs"
LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"

EPOCHS_VALUE="${EPOCHS:-20}"
ITERS_PER_EPOCH_VALUE="${ITERS_PER_EPOCH:-5000}"
BS_VALUE="${BS:-16}"
PATCH_SIZE_VALUE="${PATCH_SIZE:-256}"
W_LOSS_CR_VALUE="${W_LOSS_CR:-0.1}"
START_LR_VALUE="${START_LR:-0.0001}"
END_LR_VALUE="${END_LR:-0.000001}"
CHECKPOINT_INTERVAL_STEPS_VALUE="${CHECKPOINT_INTERVAL_STEPS:-10000}"
EVAL_INTERVAL_STEPS_VALUE="${EVAL_INTERVAL_STEPS:-10000}"
CR_REF_CHECKPOINT="${CR_REF_CHECKPOINT:-/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"

if ! [[ "$EPOCHS_VALUE" =~ ^[0-9]+$ && "$ITERS_PER_EPOCH_VALUE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EPOCHS and ITERS_PER_EPOCH must be integers." >&2
  exit 2
fi

TOTAL_STEPS=$((EPOCHS_VALUE * ITERS_PER_EPOCH_VALUE))
FAIR_PROTOCOL=1
if [[ "$TOTAL_STEPS" -ne 100000 ]]; then
  FAIR_PROTOCOL=0
fi
if [[ "$BS_VALUE" != "16" || "$PATCH_SIZE_VALUE" != "256" || "$W_LOSS_CR_VALUE" != "0.1" || "$START_LR_VALUE" != "0.0001" || "$END_LR_VALUE" != "0.000001" || "$CHECKPOINT_INTERVAL_STEPS_VALUE" != "10000" || "$EVAL_INTERVAL_STEPS_VALUE" != "10000" ]]; then
  FAIR_PROTOCOL=0
fi

if [[ "$FAIR_PROTOCOL" -ne 1 && "${ALLOW_NONFAIR_PROTOCOL:-0}" != "1" ]]; then
  cat >&2 <<EOF
ERROR: Refusing to launch a non-fair HAZE4K LF-RFC scout.
Formal comparison protocol is:
  epochs=20, iters_per_epoch=5000, total=100000
  bs=16, patch_size=256, w_loss_CR=0.1
  start_lr=0.0001, end_lr=0.000001
  checkpoint/eval interval=10000

Current request:
  epochs=$EPOCHS_VALUE, iters_per_epoch=$ITERS_PER_EPOCH_VALUE, total=$TOTAL_STEPS
  bs=$BS_VALUE, patch_size=$PATCH_SIZE_VALUE, w_loss_CR=$W_LOSS_CR_VALUE
  start_lr=$START_LR_VALUE, end_lr=$END_LR_VALUE
  checkpoint_interval=$CHECKPOINT_INTERVAL_STEPS_VALUE, eval_interval=$EVAL_INTERVAL_STEPS_VALUE
EOF
  exit 2
fi

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "MODEL_NAME=$MODEL_NAME"
echo "LOG_FILE=$LOG_FILE"
echo "TOTAL_STEPS=$TOTAL_STEPS"
echo "CR_REF_CHECKPOINT=$CR_REF_CHECKPOINT"

"$PY" train.py \
  --use_lf_prior \
  --lf_prior_channels "${LF_PRIOR_CHANNELS:-8}" \
  --lf_prior_pool "${LF_PRIOR_POOL:-8}" \
  --lf_prior_gate_init "${LF_PRIOR_GATE_INIT:-0.0}" \
  --lf_prior_injection pre_mix \
  --lf_residual_calibration \
  --lf_calib_hidden_channels "${LF_CALIB_HIDDEN_CHANNELS:-8}" \
  --lf_calib_alpha_max "${LF_CALIB_ALPHA_MAX:-1.0}" \
  --w_loss_cr_ref_residual "$W_LOSS_CR_REF_RESIDUAL_VALUE" \
  --cr_ref_checkpoint "$CR_REF_CHECKPOINT" \
  --cr_ref_residual_pool "${CR_REF_RESIDUAL_POOL:-8}" \
  --cr_ref_residual_warmup_steps "${CR_REF_RESIDUAL_WARMUP_STEPS:-0}" \
  --cr_ref_residual_target_norm_floor "${CR_REF_RESIDUAL_TARGET_NORM_FLOOR:-0.0}" \
  --cr_ref_residual_magnitude_weight "${CR_REF_RESIDUAL_MAGNITUDE_WEIGHT:-0.25}" \
  --cr_ref_residual_magnitude_cap "${CR_REF_RESIDUAL_MAGNITUDE_CAP:-2.0}" \
  ${LF_EXTRA_ARGS:-} \
  --model_name "$MODEL_NAME" \
  --dataset HAZE4K \
  --epochs "$EPOCHS_VALUE" \
  --iters_per_epoch "$ITERS_PER_EPOCH_VALUE" \
  --bs "$BS_VALUE" \
  --patch_size "$PATCH_SIZE_VALUE" \
  --num_workers "${NUM_WORKERS:-12}" \
  --test_num_workers "${TEST_NUM_WORKERS:-4}" \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor "${PREFETCH_FACTOR:-2}" \
  --w_loss_L1 1.0 \
  --w_loss_CR "$W_LOSS_CR_VALUE" \
  --start_lr "$START_LR_VALUE" \
  --end_lr "$END_LR_VALUE" \
  --exp_dir ../experiment/ \
  --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS_VALUE" \
  --eval_interval_steps "$EVAL_INTERVAL_STEPS_VALUE" \
  --save_epoch_checkpoints false \
  --no_pdf_plots \
  --no_tqdm \
  2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
