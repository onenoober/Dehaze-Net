#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/Dehaze-Net"
CODE_DIR="$ROOT/code"
MODEL_NAME="${MODEL_NAME:-DEA-Net-LF-ConditionalMask-H4K-scout-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT/experiment/HAZE4K/_run_logs"
LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "MODEL_NAME=$MODEL_NAME"
echo "LOG_FILE=$LOG_FILE"

/opt/anaconda/envs/py310/bin/python train.py \
  --use_lf_prior \
  --lf_prior_channels "${LF_PRIOR_CHANNELS:-8}" \
  --lf_prior_pool "${LF_PRIOR_POOL:-8}" \
  --lf_prior_gate_init "${LF_PRIOR_GATE_INIT:-0.0}" \
  --lf_prior_injection pre_mix \
  --lf_conditional_mask \
  --lf_mask_hidden_channels "${LF_MASK_HIDDEN_CHANNELS:-8}" \
  --lf_mask_init_bias "${LF_MASK_INIT_BIAS:-2.0}" \
  ${LF_EXTRA_ARGS:-} \
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
  --prefetch_factor "${PREFETCH_FACTOR:-2}" \
  --w_loss_L1 1.0 \
  --w_loss_CR "${W_LOSS_CR:-0.1}" \
  --start_lr "${START_LR:-0.0001}" \
  --end_lr "${END_LR:-0.000001}" \
  --exp_dir ../experiment/ \
  --checkpoint_interval_steps "${CHECKPOINT_INTERVAL_STEPS:-10000}" \
  --eval_interval_steps "${EVAL_INTERVAL_STEPS:-10000}" \
  --save_epoch_checkpoints false \
  --no_pdf_plots \
  2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
