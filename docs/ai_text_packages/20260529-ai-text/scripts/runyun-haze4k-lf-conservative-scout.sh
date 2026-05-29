#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/Dehaze-Net"
CODE_DIR="$ROOT/code"
MODEL_NAME="${MODEL_NAME:-DEA-Net-LF-Conservative-H4K-scout-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT/experiment/HAZE4K/_run_logs"
LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "MODEL_NAME=$MODEL_NAME"
echo "LOG_FILE=$LOG_FILE"

/opt/anaconda/envs/py310/bin/python train.py \
  --use_lf_prior \
  --lf_prior_channels "${LF_PRIOR_CHANNELS:-4}" \
  --lf_prior_pool "${LF_PRIOR_POOL:-8}" \
  --lf_prior_gate_init "${LF_PRIOR_GATE_INIT:-0.0}" \
  --lf_prior_residual_center \
  --lf_prior_train_dropout "${LF_PRIOR_TRAIN_DROPOUT:-0.25}" \
  --lf_prior_gate_max "${LF_PRIOR_GATE_MAX:-0.02}" \
  --w_loss_lf_gate "${W_LOSS_LF_GATE:-0.01}" \
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
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --checkpoint_interval_steps 10000 \
  --eval_interval_steps 10000 \
  --save_epoch_checkpoints false \
  --no_pdf_plots \
  2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
