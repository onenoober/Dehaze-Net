#!/usr/bin/env bash
set -u

ROOT="/root/workspace/Dehaze-Net"
CODE_DIR="$ROOT/code"
OUT_ROOT="$ROOT/experiment/HAZE4K/bs_speed_benchmark_$(date +%Y%m%d-%H%M%S)"
STEPS="${STEPS:-600}"
BATCHES="${BATCHES:-16 24 32}"

source /opt/anaconda/etc/profile.d/conda.sh
conda activate py310

mkdir -p "$OUT_ROOT"
cd "$CODE_DIR" || exit 1

echo "OUT_ROOT=$OUT_ROOT"
echo "STEPS=$STEPS"
echo "BATCHES=$BATCHES"

for bs in $BATCHES; do
  model="bench-H4K-speed-bs${bs}-$(date +%Y%m%d-%H%M%S)"
  run_log="$OUT_ROOT/${model}.log"
  mem_log="$OUT_ROOT/${model}.mem.log"
  time_file="$OUT_ROOT/${model}.time"
  summary_file="$OUT_ROOT/${model}.summary"

  echo "=== START bs=$bs model=$model ==="
  (
    while true; do
      date '+%F %T'
      nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits
      sleep 2
    done
  ) > "$mem_log" 2>&1 &
  mon_pid=$!

  start_ts=$(python - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)

  python train.py \
    --epochs 1 \
    --iters_per_epoch "$STEPS" \
    --bs "$bs" \
    --patch_size 256 \
    --num_workers 16 \
    --test_num_workers 4 \
    --pin_memory \
    --persistent_workers \
    --prefetch_factor 4 \
    --max_test_batches 1 \
    --w_loss_L1 1.0 \
    --w_loss_CR 0.1 \
    --start_lr 0.0001 \
    --end_lr 0.000001 \
    --exp_dir ../experiment/ \
    --model_name "$model" \
    --dataset HAZE4K \
    --checkpoint_interval_steps 0 \
    --eval_interval_steps "$STEPS" \
    --save_epoch_checkpoints false \
    --no_pdf_plots \
    --tb_log_interval 50 \
    2>&1 | tee "$run_log"
  rc=${PIPESTATUS[0]}

  end_ts=$(python - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)
  python - <<PY > "$time_file"
start = float("$start_ts")
end = float("$end_ts")
print(f"wall_seconds={end - start:.3f}")
PY

  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true

  python - <<PY | tee "$summary_file"
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

bs = $bs
steps = $STEPS
rc = $rc
model = "$model"
model_dir = Path("$ROOT/experiment/HAZE4K") / model
events = sorted((model_dir / "tensorboard").glob("events.out.tfevents.*"))
time_file = Path("$time_file")
mem_log = Path("$mem_log")
wall = None
if time_file.exists():
    for line in time_file.read_text().splitlines():
        if line.startswith("wall_seconds="):
            wall = float(line.split("=", 1)[1])

peak_mem = 0
if mem_log.exists():
    for line in mem_log.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit():
            peak_mem = max(peak_mem, int(parts[0]))

step_s = 0.0
img_s = 0.0
first_step = last_step = None
event_seconds = None
if events:
    ea = event_accumulator.EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "train/loss_total" in tags:
        vals = ea.Scalars("train/loss_total")
        if len(vals) >= 2:
            first, last = vals[0], vals[-1]
            first_step, last_step = first.step, last.step
            event_seconds = last.wall_time - first.wall_time
            if event_seconds > 0:
                step_s = (last.step - first.step) / event_seconds
                img_s = step_s * bs

print(f"SUMMARY bs={bs} rc={rc} model={model}")
print(f"SUMMARY wall_seconds={wall}")
print(f"SUMMARY event_steps={first_step}->{last_step} event_seconds={event_seconds}")
print(f"SUMMARY step_per_s={step_s:.4f} images_per_s={img_s:.2f} peak_mem_mib={peak_mem}")
print(f"SUMMARY log={Path('$run_log')}")
PY

  echo "=== END bs=$bs rc=$rc ==="
done

echo "=== ALL SUMMARIES ==="
cat "$OUT_ROOT"/*.summary
echo "OUT_ROOT=$OUT_ROOT"
