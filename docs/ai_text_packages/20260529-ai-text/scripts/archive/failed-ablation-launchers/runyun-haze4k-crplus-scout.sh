#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/Dehaze-Net"
CODE_DIR="$ROOT/code"
MODEL_NAME="${MODEL_NAME:-DEA-Net-CRPlus-P1-w005-H4K-scout-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT/experiment/HAZE4K/_run_logs"
LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "MODEL_NAME=$MODEL_NAME"
echo "LOG_FILE=$LOG_FILE"

/opt/anaconda/envs/py310/bin/python train.py \
  --model_name "$MODEL_NAME" \
  --dataset HAZE4K \
  --epochs 20 \
  --iters_per_epoch 5000 \
  --bs 16 \
  --patch_size 256 \
  --num_workers 12 \
  --test_num_workers 4 \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor 2 \
  --w_loss_L1 1.0 \
  --w_loss_CR "${W_LOSS_CR:-0.05}" \
  --cr_negative_mode hazy_lowpass \
  --cr_lowpass_pool "${CR_LOWPASS_POOL:-8}" \
  --cr_lowpass_weight "${CR_LOWPASS_WEIGHT:-1.0}" \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --checkpoint_interval_steps 10000 \
  --eval_interval_steps 10000 \
  --save_epoch_checkpoints false \
  --no_pdf_plots \
  2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
