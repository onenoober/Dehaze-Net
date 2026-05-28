#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CODE_DIR="$ROOT/code"
PY="${PY:-/opt/anaconda/envs/py310/bin/python}"
RUN_ID="${RUN_ID:-HAZE4K-lf-v2-mbr-preflight-runyun-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiment/HAZE4K/lf_v2_mbr_preflight/$RUN_ID}"

LF_PRIOR_CHANNELS="${LF_PRIOR_CHANNELS:-8}"
LF_PRIOR_POOL="${LF_PRIOR_POOL:-8}"
LF_PRIOR_GATE_INIT="${LF_PRIOR_GATE_INIT:-0.0}"
LF_MBR_CHANNELS="${LF_MBR_CHANNELS:-8}"
LF_MBR_POOL_SIZES="${LF_MBR_POOL_SIZES:-4,8,16}"

mkdir -p "$OUTPUT_DIR"
cd "$CODE_DIR"

echo "RUN_ID=$RUN_ID"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LF_MBR_CHANNELS=$LF_MBR_CHANNELS"
echo "LF_MBR_POOL_SIZES=$LF_MBR_POOL_SIZES"

"$PY" preflight_lf_v2_mbr.py \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
  --batch_size "${PREFLIGHT_BATCH_SIZE:-1}" \
  --image_size "${PREFLIGHT_IMAGE_SIZE:-256}" \
  --latency_warmup "${LATENCY_WARMUP:-20}" \
  --latency_iters "${LATENCY_ITERS:-80}" \
  --lf_prior_channels "$LF_PRIOR_CHANNELS" \
  --lf_prior_pool "$LF_PRIOR_POOL" \
  --lf_prior_gate_init "$LF_PRIOR_GATE_INIT" \
  --lf_mbr_channels "$LF_MBR_CHANNELS" \
  --lf_mbr_pool_sizes "$LF_MBR_POOL_SIZES" \
  --param_budget_pct "${PARAM_BUDGET_PCT:-3.0}" \
  --latency_budget_pct "${LATENCY_BUDGET_PCT:-8.0}" \
  --neutral_tolerance "${NEUTRAL_TOLERANCE:-1e-7}" \
  --branch_std_floor "${BRANCH_STD_FLOOR:-1e-8}" \
  --strict_exit

SMOKE_MODEL_NAME="${SMOKE_MODEL_NAME:-smoke-H4K-LF-v2-MBR-runyun-$(date +%Y%m%d-%H%M%S)}"
SMOKE_LOG="$OUTPUT_DIR/${SMOKE_MODEL_NAME}.log"

echo "SMOKE_MODEL_NAME=$SMOKE_MODEL_NAME"
echo "SMOKE_LOG=$SMOKE_LOG"

"$PY" train.py \
  --use_lf_prior \
  --lf_prior_channels "$LF_PRIOR_CHANNELS" \
  --lf_prior_pool "$LF_PRIOR_POOL" \
  --lf_prior_gate_init "$LF_PRIOR_GATE_INIT" \
  --lf_prior_injection pre_mix \
  --lf_multiscale_refiner \
  --lf_mbr_channels "$LF_MBR_CHANNELS" \
  --lf_mbr_pool_sizes "$LF_MBR_POOL_SIZES" \
  --model_name "$SMOKE_MODEL_NAME" \
  --dataset HAZE4K \
  --epochs 1 \
  --iters_per_epoch 2 \
  --bs 2 \
  --patch_size 64 \
  --max_train_batches 1 \
  --max_test_batches 1 \
  --num_workers 2 \
  --test_num_workers 1 \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor 2 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --checkpoint_interval_steps 2 \
  --eval_interval_steps 2 \
  --save_epoch_checkpoints false \
  --no_pdf_plots \
  --no_tensorboard \
  --no_tqdm \
  2>&1 | tee "$SMOKE_LOG"
SMOKE_EXIT="${PIPESTATUS[0]}"
if [[ "$SMOKE_EXIT" -ne 0 ]]; then
  exit "$SMOKE_EXIT"
fi

SMOKE_CKPT="$ROOT/experiment/HAZE4K/$SMOKE_MODEL_NAME/saved_model/latest.pk"
"$PY" - "$SMOKE_CKPT" "$OUTPUT_DIR/smoke_summary.json" <<'PY'
import json
import sys
import torch

ckpt_path, output_path = sys.argv[1], sys.argv[2]
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
loss_log = checkpoint.get("loss_log", {})
values = loss_log.get("LF_mbr_std", [])
summary = {
    "checkpoint": ckpt_path,
    "step": int(checkpoint.get("step", -1)),
    "lf_mbr_std_tail": float(values[-1]) if values else None,
    "lf_mbr_stats_present": bool(values),
    "recommendation": "preflight_passed_launch_allowed" if values and float(values[-1]) > 1e-8 else "do_not_train_lf_v2_mbr_yet"
}
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["recommendation"] != "preflight_passed_launch_allowed":
    raise SystemExit(1)
PY

echo "LF-v2 MBR preflight passed. 100k launch is allowed by preflight gates."
