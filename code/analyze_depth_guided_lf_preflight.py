import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from analyze_residual_field_confidence_preflight import (
    add_blend_targets,
    add_strength_bins,
    crop_np,
    run_audit,
    summarize,
)
from analyze_supervised_preserve_proxy import (
    add_meta_bins,
    image_stats_features,
    infer_one,
    load_model,
    parse_haze4k_name,
    patch_coords,
    psnr_np,
    read_feature_rows,
    residual_features,
    tensor_to_image,
)
from analyze_wavelet_preserve_proxy import is_number, parse_value, write_csv
from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether frozen relative depth features improve LF-v1 residual "
            "confidence/correction risk before any Depth-Guided LF training."
        )
    )
    parser.add_argument("--dataset_root", type=str, default="../dataset/HAZE4K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hazy_dir", type=str, default="")
    parser.add_argument("--clear_dir", type=str, default="")
    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--lfv1_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--features_csv", type=str, default="")
    parser.add_argument("--depth_cache_dir", type=str, default="")
    parser.add_argument("--depth_model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--depth_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--patches_per_image", type=int, default=4)
    parser.add_argument("--lowfreq_pool", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--valid_fraction", type=float, default=0.25)
    parser.add_argument("--heads", type=str, default="sklearn_hgb,sklearn_ridge")
    parser.add_argument(
        "--feature_sets",
        type=str,
        default=(
            "hazy_wavelet,hazy_wavelet_plus_teacher_outputs,depth_only,"
            "hazy_depth,hazy_depth_plus_teacher_outputs,"
            "hazy_shuffled_depth_plus_teacher_outputs"
        ),
    )
    parser.add_argument("--main_feature_set", type=str, default="hazy_depth_plus_teacher_outputs")
    parser.add_argument("--main_head", type=str, default="sklearn_hgb")
    parser.add_argument("--heldout_groups", type=str, default="airlight_bin,beta_bin")
    parser.add_argument("--hgb_max_iter", type=int, default=180)
    parser.add_argument("--hgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--hgb_l2", type=float, default=0.01)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--min_gain", type=float, default=0.25)
    parser.add_argument("--min_oracle_recovery", type=float, default=0.20)
    parser.add_argument("--min_preserve_recall", type=float, default=0.68)
    parser.add_argument("--min_regression_improve_recall", type=float, default=0.60)
    parser.add_argument("--min_intervention_precision", type=float, default=0.60)
    parser.add_argument("--min_strong_cr_regression_improve_recall", type=float, default=0.60)
    parser.add_argument("--min_oracle_c_corr", type=float, default=0.45)
    parser.add_argument("--min_heldout_gain", type=float, default=0.0)
    parser.add_argument("--min_heldout_preserve_recall", type=float, default=0.62)
    parser.add_argument("--min_heldout_regression_improve_recall", type=float, default=0.45)
    parser.add_argument("--min_heldout_oracle_c_corr", type=float, default=0.30)
    return parser.parse_args()


def resolve_dirs(args):
    if args.hazy_dir and args.clear_dir:
        return args.hazy_dir, args.clear_dir
    split_root = Path(args.dataset_root) / args.split
    return resolve_pair_dirs(split_root)


class DepthEstimator:
    def __init__(self, model_id, device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
            depth = output.predicted_depth
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        return depth.detach().float().cpu().numpy().astype(np.float32)


def robust_normalize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros_like(depth, dtype=np.float32)
    values = depth[finite]
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi - lo <= EPS:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo <= EPS:
        return np.zeros_like(depth, dtype=np.float32)
    depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    depth[~finite] = 0.0
    return depth.astype(np.float32)


def depth_cache_path(cache_dir, split, name):
    return Path(cache_dir) / split / (name.replace("/", "__") + ".npy")


def load_or_generate_depth(name, hazy_img, args, estimator):
    cache_dir = args.depth_cache_dir
    if not cache_dir:
        raise ValueError("--depth_cache_dir is required unless --features_csv is used")
    path = depth_cache_path(cache_dir, args.split, name)
    if path.exists():
        return np.load(path).astype(np.float32), False
    if estimator is None:
        raise RuntimeError("Depth cache miss but no estimator is available: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = estimator.predict(hazy_img)
    np.save(path, depth.astype(np.float32))
    return depth, True


def scalar_stats(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.zeros((1,), dtype=np.float32)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10.0)),
        "p25": float(np.percentile(values, 25.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p75": float(np.percentile(values, 75.0)),
        "p90": float(np.percentile(values, 90.0)),
    }


def add_prefixed_stats(row, prefix, values):
    for key, value in scalar_stats(values).items():
        row["{}_{}".format(prefix, key)] = value


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 2:
        return 0.0
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def add_depth_features(row, depth_patch, hazy_patch, residual_patch):
    depth = np.asarray(depth_patch, dtype=np.float32)
    depth = np.where(np.isfinite(depth), depth, 0.0)
    depth = np.clip(depth, 0.0, 1.0)
    inv_depth = 1.0 - depth
    luma = (
        0.299 * hazy_patch[:, :, 0]
        + 0.587 * hazy_patch[:, :, 1]
        + 0.114 * hazy_patch[:, :, 2]
    ).astype(np.float32)
    dark = np.min(hazy_patch.astype(np.float32), axis=2)
    residual_norm = np.sqrt(np.sum(residual_patch.astype(np.float32) ** 2, axis=2))

    for prefix, values in (("depth", depth), ("inv_depth", inv_depth)):
        gy, gx = np.gradient(values)
        grad = np.sqrt(gx * gx + gy * gy)
        add_prefixed_stats(row, prefix, values)
        add_prefixed_stats(row, prefix + "_grad", grad)
        row[prefix + "_luma_corr"] = safe_corr(values, luma)
        row[prefix + "_dark_corr"] = safe_corr(values, dark)
        row[prefix + "_residual_norm_corr"] = safe_corr(values, residual_norm)
        add_prefixed_stats(row, prefix + "_x_luma", values * luma)
        add_prefixed_stats(row, prefix + "_x_dark", values * dark)
        add_prefixed_stats(row, prefix + "_x_residual_norm", values * residual_norm)


def collect_rows(args):
    hazy_dir, clear_dir = resolve_dirs(args)
    names = list_image_files(hazy_dir)
    if args.max_images > 0:
        names = names[:args.max_images]
    if args.model_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for model inference but not available")
    if args.depth_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for depth inference but not available")

    baseline_model, baseline_ckpt = load_model(args.baseline_checkpoint, False, args.model_device)
    lfv1_model, lfv1_ckpt = load_model(args.lfv1_checkpoint, True, args.model_device)
    estimator = DepthEstimator(args.depth_model, args.depth_device)
    to_tensor = ToTensor()
    rng = random.Random(args.seed)
    rows = []
    depth_generated = 0
    depth_loaded = 0
    depth_seconds = 0.0

    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy_path = Path(hazy_dir) / name
            clear_path = Path(find_clear_image(clear_dir, name))
            hazy_img = Image.open(hazy_path).convert("RGB")
            clear_img = Image.open(clear_path).convert("RGB")
            start = time.time()
            depth_raw, generated = load_or_generate_depth(name, hazy_img, args, estimator)
            depth_seconds += time.time() - start
            depth_generated += 1 if generated else 0
            depth_loaded += 0 if generated else 1

            hazy = to_tensor(hazy_img).unsqueeze(0).to(args.model_device)
            clear = to_tensor(clear_img).unsqueeze(0).to(args.model_device)
            cr = infer_one(baseline_model, hazy, args.pad_size)
            lfv1 = infer_one(lfv1_model, hazy, args.pad_size)

            hazy_np = tensor_to_image(hazy)
            clear_np = tensor_to_image(clear)
            cr_np = tensor_to_image(cr)
            lfv1_np = tensor_to_image(lfv1)
            height, width = hazy_np.shape[:2]
            if depth_raw.shape[:2] != (height, width):
                raise RuntimeError("Depth shape mismatch for {}".format(name))
            depth_norm = robust_normalize_depth(depth_raw)
            meta = parse_haze4k_name(name)
            image_cr_psnr = psnr_np(cr_np, clear_np)
            image_lfv1_psnr = psnr_np(lfv1_np, clear_np)
            for patch_idx, coord in enumerate(
                patch_coords(height, width, args.patch_size, args.patches_per_image, rng)
            ):
                y, x, ph, pw = coord
                hazy_patch = crop_np(hazy_np, coord)
                clear_patch = crop_np(clear_np, coord)
                cr_patch = crop_np(cr_np, coord)
                lfv1_patch = crop_np(lfv1_np, coord)
                depth_patch = depth_norm[y:y + ph, x:x + pw]
                cr_psnr = psnr_np(cr_patch, clear_patch)
                lfv1_psnr = psnr_np(lfv1_patch, clear_patch)
                row = {
                    "filename": name,
                    "patch_id": "{}:{}:{}:{}:{}".format(name, y, x, ph, pw),
                    "image_index": idx,
                    "patch_index": patch_idx,
                    "image_id": meta["image_id"],
                    "airlight": meta["airlight"],
                    "beta": meta["beta"],
                    "height": height,
                    "width": width,
                    "patch_y": y,
                    "patch_x": x,
                    "patch_h": ph,
                    "patch_w": pw,
                    "patch_y_norm": float(y / max(1, height - ph)),
                    "patch_x_norm": float(x / max(1, width - pw)),
                    "image_cr_psnr": image_cr_psnr,
                    "image_lfv1_psnr": image_lfv1_psnr,
                    "image_delta_psnr": image_lfv1_psnr - image_cr_psnr,
                    "cr_psnr": cr_psnr,
                    "lfv1_psnr": lfv1_psnr,
                    "lfv1_delta_cr_psnr": lfv1_psnr - cr_psnr,
                }
                add_meta_bins(row)
                add_blend_targets(row, cr_patch, lfv1_patch, clear_patch, args.lowfreq_pool)
                row.update(image_stats_features(hazy_patch, "hazy", include_wavelet=True))
                row.update(image_stats_features(cr_patch, "cr_out", include_wavelet=False))
                row.update(image_stats_features(lfv1_patch, "lfv1_out", include_wavelet=False))
                row.update(residual_features(lfv1_patch - cr_patch, "lfv1_minus_cr"))
                row.update(residual_features(cr_patch - hazy_patch, "cr_minus_hazy"))
                row.update(residual_features(lfv1_patch - hazy_patch, "lfv1_minus_hazy"))
                add_depth_features(row, depth_patch, hazy_patch, lfv1_patch - cr_patch)
                rows.append(row)
            if idx % 25 == 0 or idx == len(names):
                print(
                    "depth-guided labeled {}/{} images, {} patches, depth generated/loaded {}/{}".format(
                        idx, len(names), len(rows), depth_generated, depth_loaded
                    ),
                    flush=True,
                )

    meta_payload = {
        "baseline_checkpoint_step": baseline_ckpt.get("step"),
        "lfv1_checkpoint_step": lfv1_ckpt.get("step"),
        "hazy_dir": str(hazy_dir),
        "clear_dir": str(clear_dir),
        "images": len(names),
        "patches": len(rows),
        "lowfreq_pool": args.lowfreq_pool,
        "depth_model": args.depth_model,
        "depth_cache_dir": args.depth_cache_dir,
        "depth_generated": depth_generated,
        "depth_loaded": depth_loaded,
        "depth_seconds": depth_seconds,
    }
    return rows, meta_payload


def numeric_feature_names(rows, predicate):
    excluded = {
        "filename",
        "patch_id",
        "image_id",
        "airlight_bin",
        "beta_bin",
        "airlight_beta_bin",
        "cr_strength_bin",
    }
    target_names = {
        "image_cr_psnr",
        "image_lfv1_psnr",
        "image_delta_psnr",
        "cr_psnr",
        "lfv1_psnr",
        "lfv1_delta_cr_psnr",
        "cr_mse",
        "lfv1_mse",
        "err_cross",
        "blend_oracle_c",
        "blend_oracle_psnr",
        "blend_oracle_gain_vs_lfv1",
        "lf_residual_cosine",
        "lf_residual_norm_ratio",
        "lf_mse_delta",
    }
    names = []
    for key in sorted(rows[0].keys()):
        if key in excluded or key in target_names:
            continue
        if not predicate(key):
            continue
        values = [row.get(key, "") for row in rows]
        if sum(1 for value in values if is_number(value)) >= max(5, len(rows) // 10):
            names.append(key)
    return names


def add_shuffled_depth_features(rows, seed):
    rng = random.Random(seed)
    depth_keys = [
        key for key in rows[0].keys()
        if key.startswith("depth_") or key.startswith("inv_depth_")
    ]
    shuffled = list(range(len(rows)))
    rng.shuffle(shuffled)
    for row_idx, source_idx in enumerate(shuffled):
        source = rows[source_idx]
        target = rows[row_idx]
        for key in depth_keys:
            target["shuf_" + key] = source.get(key, 0.0)


def build_feature_sets(rows):
    hazy = numeric_feature_names(rows, lambda key: key.startswith("hazy_"))
    teacher = numeric_feature_names(
        rows,
        lambda key: key.startswith(
            ("cr_out_", "lfv1_out_", "lfv1_minus_cr_", "cr_minus_hazy_", "lfv1_minus_hazy_")
        ),
    )
    depth = numeric_feature_names(
        rows,
        lambda key: key.startswith("depth_") or key.startswith("inv_depth_"),
    )
    shuffled_depth = numeric_feature_names(
        rows,
        lambda key: key.startswith("shuf_depth_") or key.startswith("shuf_inv_depth_"),
    )
    metadata = numeric_feature_names(
        rows,
        lambda key: key in ("airlight", "beta", "height", "width", "patch_x_norm", "patch_y_norm"),
    )
    return {
        "hazy_wavelet": hazy,
        "hazy_wavelet_plus_teacher_outputs": sorted(set(hazy + teacher)),
        "depth_only": depth,
        "hazy_depth": sorted(set(hazy + depth)),
        "hazy_depth_plus_teacher_outputs": sorted(set(hazy + depth + teacher)),
        "hazy_shuffled_depth_plus_teacher_outputs": sorted(set(hazy + shuffled_depth + teacher)),
        "metadata_diagnostic": metadata,
        "hazy_depth_plus_metadata": sorted(set(hazy + depth + metadata)),
        "hazy_depth_teacher_outputs_plus_metadata": sorted(set(hazy + depth + teacher + metadata)),
    }


def write_report(path, summary, meta, recommendation, args):
    lines = [
        "# HAZE4K Depth-Guided LF Preflight",
        "",
        "## Recommendation",
        "",
        "- `{}`".format(recommendation),
        "",
        "## Pass Line",
        "",
        "- Main feature/head: `{}` / `{}`".format(args.main_feature_set, args.main_head),
        "- Simulated gain vs LF-v1 >= `{:.4f}` dB".format(args.min_gain),
        "- LF-v1 gain preserve recall >= `{:.4f}`".format(args.min_preserve_recall),
        "- Intervention precision >= `{:.4f}`".format(args.min_intervention_precision),
        "- Strong-CR regression recall >= `{:.4f}`".format(args.min_strong_cr_regression_improve_recall),
        "- Confidence/oracle correlation >= `{:.4f}`".format(args.min_oracle_c_corr),
        "- Airlight/beta held-out rows must also satisfy stability floors.",
        "",
        "## Source",
        "",
        "- Images: `{}`".format(meta["images"]),
        "- Patches: `{}`".format(meta["patches"]),
        "- Baseline checkpoint step: `{}`".format(meta.get("baseline_checkpoint_step")),
        "- LF-v1 checkpoint step: `{}`".format(meta.get("lfv1_checkpoint_step")),
        "- Depth model: `{}`".format(meta.get("depth_model")),
        "- Depth cache: `{}`".format(meta.get("depth_cache_dir")),
        "- Depth generated/loaded: `{}` / `{}`".format(meta.get("depth_generated"), meta.get("depth_loaded")),
        "- Depth cache seconds: `{:.2f}`".format(float(meta.get("depth_seconds") or 0.0)),
        "",
        "## Summary",
        "",
        "| Feature Set | Head | Split | Gain | Recovery | Preserve Recall | Regression Improve | Strong CR Improve | Intervene Precision | c Corr | Main | Stable |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary:
        lines.append(
            "| {feature_set} | {head} | {split} | {gain:.4f} | {recovery:.4f} | {preserve:.4f} | {regress:.4f} | {strong:.4f} | {precision:.4f} | {corr:.4f} | {main} | {stable} |".format(
                feature_set=row["feature_set"],
                head=row["head"],
                split=row["split_family"],
                gain=float(row.get("gain_vs_lfv1_mean") or 0.0),
                recovery=float(row.get("oracle_recovery_mean") or 0.0),
                preserve=float(row.get("preserve_recall_mean") or 0.0),
                regress=float(row.get("regression_improve_recall_mean") or 0.0),
                strong=float(row.get("strong_cr_regression_improve_recall_mean") or 0.0),
                precision=float(row.get("intervention_precision_mean") or 0.0),
                corr=float(row.get("pred_oracle_c_corr_mean") or 0.0),
                main="yes" if row.get("passes_main_line") else "no",
                stable="yes" if row.get("passes_stability_line") else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.features_csv:
        rows = read_feature_rows(args.features_csv)
        rows = [{key: parse_value(value) for key, value in row.items()} for row in rows]
        meta = {
            "baseline_checkpoint_step": "reused_features",
            "lfv1_checkpoint_step": "reused_features",
            "hazy_dir": "reused_features",
            "clear_dir": "reused_features",
            "images": len({row["filename"] for row in rows}),
            "patches": len(rows),
            "lowfreq_pool": "reused_features",
            "features_csv": args.features_csv,
            "depth_model": "reused_features",
            "depth_cache_dir": "reused_features",
            "depth_generated": "reused_features",
            "depth_loaded": "reused_features",
            "depth_seconds": 0.0,
        }
    else:
        rows, meta = collect_rows(args)

    add_strength_bins(rows)
    add_shuffled_depth_features(rows, args.seed + 17)
    feature_sets = build_feature_sets(rows)
    write_csv(output_dir / "depth_guided_lf_features.csv", rows)
    results = run_audit(rows, feature_sets, args)
    summary = summarize(results, args)
    pass_rows = [
        row for row in summary
        if row.get("passes_main_line") and row.get("passes_stability_line")
    ]
    recommendation = (
        "preflight_passed_depth_guided_lf_scout_allowed"
        if pass_rows else
        "do_not_train_depth_guided_lf_yet"
    )
    write_csv(output_dir / "split_results.csv", results)
    write_csv(output_dir / "summary.csv", summary)
    payload = {
        "recommendation": recommendation,
        "args": vars(args),
        "teacher_meta": meta,
        "feature_counts": {name: len(values) for name, values in feature_sets.items()},
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / "analysis_report.md", summary, meta, recommendation, args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
