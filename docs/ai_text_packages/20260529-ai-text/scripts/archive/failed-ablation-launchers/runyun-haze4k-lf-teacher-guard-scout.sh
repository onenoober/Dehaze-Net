#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/Dehaze-Net"
CODE_DIR="$ROOT/code"
MODEL_NAME="${MODEL_NAME:-DEA-Net-LF-TeacherGuard-H4K-scout-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT/experiment/HAZE4K/_run_logs"
LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$ROOT/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "MODEL_NAME=$MODEL_NAME"
echo "LOG_FILE=$LOG_FILE"
echo "TEACHER_CHECKPOINT=$TEACHER_CHECKPOINT"

/opt/anaconda/envs/py310/bin/python train.py \
  --use_lf_prior \
  --lf_prior_channels "${LF_PRIOR_CHANNELS:-8}" \
  --lf_prior_pool "${LF_PRIOR_POOL:-8}" \
  --lf_prior_gate_init "${LF_PRIOR_GATE_INIT:-0.0}" \
  --lf_prior_injection "${LF_PRIOR_INJECTION:-pre_mix}" \
  --w_loss_teacher_guard "${W_LOSS_TEACHER_GUARD:-0.05}" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --teacher_guard_margin "${TEACHER_GUARD_MARGIN:-0.0}" \
  --teacher_guard_warmup_steps "${TEACHER_GUARD_WARMUP_STEPS:-20000}" \
  --teacher_guard_max_weight "${TEACHER_GUARD_MAX_WEIGHT:-2.0}" \
  --teacher_guard_patch_pool "${TEACHER_GUARD_PATCH_POOL:-0}" \
  --model_name "$MODEL_NAME" \
  --dataset HAZE4K \
  --epochs "${EPOCHS:-20}" \
  --iters_per_epoch "${ITERS_PER_EPOCH:-5000}" \
  --bs "${BS:-16}" \
  --patch_size "${PATCH_SIZE:-256}" \
  --num_workers "${NUM_WORKERS:-12}" \
  --test_num_workers "${TEST_NUM_WORKERS:-4}" \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor 2 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --checkpoint_interval_steps 10000 \
  --eval_interval_steps 10000 \
  --save_epoch_checkpoints false \
  --no_pdf_plots \
  "$@" \
  2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
