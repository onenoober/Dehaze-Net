#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CODE_DIR="$ROOT/code"
MODEL_NAME="${MODEL_NAME:-DEA-Net-OfficialWarmStart-LFv1-H4K-finetune100k-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT/experiment/HAZE4K/_run_logs"
LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"
OFFICIAL_CHECKPOINT="${OFFICIAL_CHECKPOINT:-$ROOT/trained_models/HAZE4K/PSNR3426_SSIM9885.pth}"
EPOCHS_VALUE="${EPOCHS:-20}"
ITERS_PER_EPOCH_VALUE="${ITERS_PER_EPOCH:-5000}"
BS_VALUE="${BS:-16}"
PATCH_SIZE_VALUE="${PATCH_SIZE:-256}"
START_LR_VALUE="${START_LR:-0.00002}"
END_LR_VALUE="${END_LR:-0.000001}"
W_LOSS_CR_VALUE="${W_LOSS_CR:-0.1}"
CHECKPOINT_INTERVAL_STEPS_VALUE="${CHECKPOINT_INTERVAL_STEPS:-10000}"
EVAL_INTERVAL_STEPS_VALUE="${EVAL_INTERVAL_STEPS:-10000}"
USE_LF_PRIOR_VALUE="${USE_LF_PRIOR:-1}"
CHECK_FORWARD_VALUE="${CHECK_FORWARD:-1}"
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

if ! [[ "$EPOCHS_VALUE" =~ ^[0-9]+$ && "$ITERS_PER_EPOCH_VALUE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EPOCHS and ITERS_PER_EPOCH must be integers." >&2
  exit 2
fi

TOTAL_STEPS=$((EPOCHS_VALUE * ITERS_PER_EPOCH_VALUE))
if [[ "$TOTAL_STEPS" -ne 100000 && "${ALLOW_SHORT_WARMSTART:-0}" != "1" ]]; then
  cat >&2 <<EOF
ERROR: Refusing to launch a short-horizon official warm-start fine-tune.
This isolated route still keeps train.py's LR horizon at 100000 steps by
default, then uses written gates to stop early if the route is not useful.

Current request:
  epochs=$EPOCHS_VALUE, iters_per_epoch=$ITERS_PER_EPOCH_VALUE, total=$TOTAL_STEPS

For smoke/diagnostic only, rerun with ALLOW_SHORT_WARMSTART=1 and label the
artifact as invalid for both cold-start and warm-start route comparison.
EOF
  exit 2
fi

if [[ ! -f "$OFFICIAL_CHECKPOINT" ]]; then
  echo "ERROR: official checkpoint not found: $OFFICIAL_CHECKPOINT" >&2
  echo "Set OFFICIAL_CHECKPOINT to the actual HAZE4K .pth filename." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

ARCH_FLAGS=()
if [[ "$USE_LF_PRIOR_VALUE" == "1" ]]; then
  ARCH_FLAGS+=(
    --use_lf_prior
    --lf_prior_channels "${LF_PRIOR_CHANNELS:-8}"
    --lf_prior_pool "${LF_PRIOR_POOL:-8}"
    --lf_prior_gate_init "${LF_PRIOR_GATE_INIT:-0.0}"
    --lf_prior_injection "${LF_PRIOR_INJECTION:-pre_mix}"
  )
  if [[ "${LF_PRIOR_RESIDUAL_CENTER:-0}" == "1" ]]; then
    ARCH_FLAGS+=(--lf_prior_residual_center)
  fi
  if [[ "${LF_CONDITIONAL_MASK:-0}" == "1" ]]; then
    ARCH_FLAGS+=(--lf_conditional_mask)
  fi
  if [[ "${LF_HAZE_AWARE_MASK:-0}" == "1" ]]; then
    ARCH_FLAGS+=(--lf_haze_aware_mask)
  fi
  if [[ "${LF_RESIDUAL_CALIBRATION:-0}" == "1" ]]; then
    ARCH_FLAGS+=(--lf_residual_calibration)
  fi
  if [[ "${LF_RESIDUAL_SELECTOR:-0}" == "1" ]]; then
    ARCH_FLAGS+=(--lf_residual_selector)
  fi
fi

PREPARE_FLAGS=("${ARCH_FLAGS[@]}")
if [[ "$CHECK_FORWARD_VALUE" == "1" ]]; then
  PREPARE_FLAGS+=(--check_forward)
fi

echo "MODEL_NAME=$MODEL_NAME"
echo "LOG_FILE=$LOG_FILE"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "OFFICIAL_CHECKPOINT=$OFFICIAL_CHECKPOINT"
echo "TOTAL_STEPS=$TOTAL_STEPS"
echo "START_LR=$START_LR_VALUE"
echo "END_LR=$END_LR_VALUE"
echo "USE_LF_PRIOR=$USE_LF_PRIOR_VALUE"
echo "CHECK_FORWARD=$CHECK_FORWARD_VALUE"
if [[ "$TOTAL_STEPS" -ne 100000 ]]; then
  echo "WARNING: ALLOW_SHORT_WARMSTART=1; this run is diagnostic only."
fi

{
  "$PYTHON_BIN" prepare_official_warmstart_checkpoint.py \
    --official_checkpoint "$OFFICIAL_CHECKPOINT" \
    --model_name "$MODEL_NAME" \
    --dataset HAZE4K \
    --exp_dir ../experiment/ \
    --start_lr "$START_LR_VALUE" \
    "${PREPARE_FLAGS[@]}"

  "$PYTHON_BIN" train.py \
    --resume true \
    --pre_trained_model official_warmstart_step0.pk \
    --model_name "$MODEL_NAME" \
    --dataset HAZE4K \
    --epochs "$EPOCHS_VALUE" \
    --iters_per_epoch "$ITERS_PER_EPOCH_VALUE" \
    --bs "$BS_VALUE" \
    --patch_size "$PATCH_SIZE_VALUE" \
    --num_workers "${NUM_WORKERS:-12}" \
    --test_num_workers "${TEST_NUM_WORKERS:-4}" \
    --max_train_batches "${MAX_TRAIN_BATCHES:-0}" \
    --max_test_batches "${MAX_TEST_BATCHES:-0}" \
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
    "${ARCH_FLAGS[@]}" \
    "$@"
} 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
