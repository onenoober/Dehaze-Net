#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/Dehaze-Net"
CODE_DIR="$ROOT/code"
BASELINE_CKPT="${BASELINE_CKPT:-$ROOT/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk}"
LF_CKPT="${LF_CKPT:-$ROOT/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk}"
BASE_COMPARE_DIR="${BASE_COMPARE_DIR:-$ROOT/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-20260522}"
SAMPLE_LIST="${SAMPLE_LIST:-$BASE_COMPARE_DIR/samples.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-gate-sweep-20260522}"
GATE_SCALES="${GATE_SCALES:-0 0.25 0.5 0.75 1.0}"
TOP_K="${TOP_K:-8}"

cd "$CODE_DIR"

mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "BASELINE_CKPT=$BASELINE_CKPT"
echo "LF_CKPT=$LF_CKPT"
echo "SAMPLE_LIST=$SAMPLE_LIST"
echo "GATE_SCALES=$GATE_SCALES"

for scale in $GATE_SCALES; do
  tag="${scale//./p}"
  out_dir="$OUT_ROOT/scale_$tag"
  label="DEA-Net-LF gate x$scale"

  echo "Running gate scale $scale -> $out_dir"
  /opt/anaconda/envs/py310/bin/python visual_compare_train_ckpt.py \
    --dataset HAZE4K \
    --split test \
    --output_dir "$out_dir" \
    --baseline_checkpoint "$BASELINE_CKPT" \
    --lf_checkpoint "$LF_CKPT" \
    --sample_list "$SAMPLE_LIST" \
    --lf_gate_scale "$scale" \
    --lf_label "$label"

  /opt/anaconda/envs/py310/bin/python analyze_visual_compare.py \
    --compare_dir "$out_dir" \
    --output_dir "$out_dir/analysis" \
    --baseline_label "DEA-Net-CR" \
    --current_dir_name lf \
    --current_label "$label" \
    --top_k "$TOP_K" \
    --save_heatmaps \
    --save_diagnostic_panels
done

OUT_ROOT="$OUT_ROOT" /opt/anaconda/envs/py310/bin/python - <<'PY'
import csv
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
rows = []
for scale_dir in sorted(out_root.glob("scale_*")):
    summary_path = scale_dir / "summary.json"
    analysis_path = scale_dir / "analysis" / "analysis_summary.json"
    if not summary_path.exists() or not analysis_path.exists():
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    rows.append({
        "scale_dir": scale_dir.name,
        "lf_gate_scale": summary.get("lf_gate_scale"),
        "lf_original_gate": summary.get("lf_original_gate"),
        "lf_effective_gate": summary.get("lf_effective_gate"),
        "mean_delta_psnr": summary.get("mean_delta_psnr"),
        "mean_delta_ssim": summary.get("mean_delta_ssim"),
        "analysis_mean_delta_psnr": analysis.get("mean_delta_psnr"),
        "analysis_mean_delta_ssim": analysis.get("mean_delta_ssim"),
        "mean_delta_mae_improvement": analysis.get("mean_delta_mae_improvement"),
        "mean_delta_e_improvement": analysis.get("mean_delta_e_improvement"),
        "mean_luma_abs_bias_improvement": analysis.get("mean_luma_abs_bias_improvement"),
        "mean_saturation_abs_bias_improvement": analysis.get("mean_saturation_abs_bias_improvement"),
        "mean_dark_channel_abs_bias_improvement": analysis.get("mean_dark_channel_abs_bias_improvement"),
        "mean_edge_error_improvement": analysis.get("mean_edge_error_improvement"),
        "current_better_objective": analysis.get("tag_counts", {}).get("current_better_objective", 0),
        "current_worse_objective": analysis.get("tag_counts", {}).get("current_worse_objective", 0),
        "mixed_or_neutral": analysis.get("tag_counts", {}).get("mixed_or_neutral", 0),
    })

csv_path = out_root / "gate_sweep_summary.csv"
if rows:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

md_path = out_root / "gate_sweep_summary.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# LF Gate Scale Sweep Summary\n\n")
    f.write("| scale | effective_gate | delta_psnr | delta_ssim | delta_E_improvement | better | worse | mixed |\n")
    f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for row in rows:
        f.write(
            "| {scale} | {gate:.6f} | {psnr:.4f} | {ssim:.6f} | {de:.4f} | {better} | {worse} | {mixed} |\n".format(
                scale=row["lf_gate_scale"],
                gate=float(row["lf_effective_gate"]),
                psnr=float(row["mean_delta_psnr"]),
                ssim=float(row["mean_delta_ssim"]),
                de=float(row["mean_delta_e_improvement"]),
                better=row["current_better_objective"],
                worse=row["current_worse_objective"],
                mixed=row["mixed_or_neutral"],
            )
        )

print(json.dumps(rows, indent=2))
PY
