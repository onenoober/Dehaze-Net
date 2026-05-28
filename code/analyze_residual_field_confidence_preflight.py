import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from analyze_supervised_preserve_proxy import (
    EPS,
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
from analyze_wavelet_preserve_proxy import corr, is_number, mean, parse_value, std, write_csv
from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a continuous LF-v1/CR residual-field confidence "
            "target is learnable before launching a residual-field calibration scout."
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--patches_per_image", type=int, default=4)
    parser.add_argument("--lowfreq_pool", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--valid_fraction", type=float, default=0.25)
    parser.add_argument("--heads", type=str, default="sklearn_hgb,sklearn_ridge")
    parser.add_argument(
        "--feature_sets",
        type=str,
        default="hazy_wavelet,teacher_output_proxy,hazy_wavelet_plus_teacher_outputs",
    )
    parser.add_argument("--main_feature_set", type=str, default="hazy_wavelet_plus_teacher_outputs")
    parser.add_argument("--main_head", type=str, default="sklearn_hgb")
    parser.add_argument("--heldout_groups", type=str, default="airlight_bin,beta_bin")
    parser.add_argument("--hgb_max_iter", type=int, default=180)
    parser.add_argument("--hgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--hgb_l2", type=float, default=0.01)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--min_gain", type=float, default=0.05)
    parser.add_argument("--min_oracle_recovery", type=float, default=0.20)
    parser.add_argument("--min_preserve_recall", type=float, default=0.68)
    parser.add_argument("--min_regression_improve_recall", type=float, default=0.55)
    parser.add_argument("--min_intervention_precision", type=float, default=0.60)
    parser.add_argument("--min_strong_cr_regression_improve_recall", type=float, default=0.55)
    parser.add_argument("--min_oracle_c_corr", type=float, default=0.35)
    parser.add_argument("--min_heldout_gain", type=float, default=0.0)
    parser.add_argument("--min_heldout_preserve_recall", type=float, default=0.62)
    parser.add_argument("--min_heldout_regression_improve_recall", type=float, default=0.45)
    parser.add_argument("--min_heldout_oracle_c_corr", type=float, default=0.25)
    return parser.parse_args()


def resolve_dirs(args):
    if args.hazy_dir and args.clear_dir:
        return args.hazy_dir, args.clear_dir
    split_root = Path(args.dataset_root) / args.split
    return resolve_pair_dirs(split_root)


def crop_np(image, coord):
    y, x, h, w = coord
    return image[y:y + h, x:x + w, :]


def mse_np(pred, gt):
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    gt = np.clip(gt.astype(np.float32), 0.0, 1.0)
    return float(np.mean((pred - gt) ** 2))


def psnr_from_mse(value):
    value = max(float(value), EPS)
    return float(20.0 * math.log10(1.0 / math.sqrt(value)))


def lowpass_np(image, pool):
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    low = F.avg_pool2d(tensor, kernel_size=pool, stride=pool, ceil_mode=True)
    low = F.interpolate(low, size=image.shape[:2], mode="bilinear", align_corners=False)
    return low.squeeze(0).numpy().transpose(1, 2, 0)


def residual_cosine(current, target):
    a = current.reshape(-1).astype(np.float32)
    b = target.reshape(-1).astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def add_blend_targets(row, cr_patch, lfv1_patch, clear_patch, lowfreq_pool):
    cr_err = cr_patch.astype(np.float32) - clear_patch.astype(np.float32)
    lfv1_err = lfv1_patch.astype(np.float32) - clear_patch.astype(np.float32)
    direction = lfv1_patch.astype(np.float32) - cr_patch.astype(np.float32)
    direction_energy = float(np.mean(direction * direction))
    cr_direction_dot = float(np.mean(cr_err * direction))
    if direction_energy <= EPS:
        oracle_c = 0.0
    else:
        oracle_c = float(np.clip(-cr_direction_dot / direction_energy, 0.0, 1.0))
    cr_mse = mse_np(cr_patch, clear_patch)
    lfv1_mse = mse_np(lfv1_patch, clear_patch)
    err_cross = float(np.mean(cr_err * lfv1_err))
    oracle_mse = (
        ((1.0 - oracle_c) ** 2) * cr_mse
        + (oracle_c ** 2) * lfv1_mse
        + 2.0 * oracle_c * (1.0 - oracle_c) * err_cross
    )
    row.update({
        "cr_mse": cr_mse,
        "lfv1_mse": lfv1_mse,
        "err_cross": err_cross,
        "blend_oracle_c": oracle_c,
        "blend_oracle_psnr": psnr_from_mse(oracle_mse),
        "blend_oracle_gain_vs_lfv1": psnr_from_mse(oracle_mse) - row["lfv1_psnr"],
    })

    cr_low = lowpass_np(cr_patch, lowfreq_pool)
    lfv1_low = lowpass_np(lfv1_patch, lowfreq_pool)
    clear_low = lowpass_np(clear_patch, lowfreq_pool)
    target_residual = clear_low - cr_low
    lfv1_residual = lfv1_low - cr_low
    target_norm = float(np.linalg.norm(target_residual.reshape(-1)))
    lfv1_norm = float(np.linalg.norm(lfv1_residual.reshape(-1)))
    row.update({
        "lf_residual_cosine": residual_cosine(lfv1_residual, target_residual),
        "lf_residual_norm_ratio": lfv1_norm / (target_norm + EPS),
        "lf_mse_delta": mse_np(lfv1_low, clear_low) - mse_np(cr_low, clear_low),
    })


def collect_rows(args):
    hazy_dir, clear_dir = resolve_dirs(args)
    names = list_image_files(hazy_dir)
    if args.max_images > 0:
        names = names[:args.max_images]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    baseline_model, baseline_ckpt = load_model(args.baseline_checkpoint, False, args.device)
    lfv1_model, lfv1_ckpt = load_model(args.lfv1_checkpoint, True, args.device)
    to_tensor = ToTensor()
    rng = random.Random(args.seed)
    rows = []

    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy_path = Path(hazy_dir) / name
            clear_path = Path(find_clear_image(clear_dir, name))
            hazy_img = Image.open(hazy_path).convert("RGB")
            clear_img = Image.open(clear_path).convert("RGB")
            hazy = to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = to_tensor(clear_img).unsqueeze(0).to(args.device)
            cr = infer_one(baseline_model, hazy, args.pad_size)
            lfv1 = infer_one(lfv1_model, hazy, args.pad_size)

            hazy_np = tensor_to_image(hazy)
            clear_np = tensor_to_image(clear)
            cr_np = tensor_to_image(cr)
            lfv1_np = tensor_to_image(lfv1)
            height, width = hazy_np.shape[:2]
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
                rows.append(row)
            if idx % 25 == 0 or idx == len(names):
                print("residual-field labeled {}/{} images, {} patches".format(idx, len(names), len(rows)), flush=True)

    meta_payload = {
        "baseline_checkpoint_step": baseline_ckpt.get("step"),
        "lfv1_checkpoint_step": lfv1_ckpt.get("step"),
        "hazy_dir": str(hazy_dir),
        "clear_dir": str(clear_dir),
        "images": len(names),
        "patches": len(rows),
        "lowfreq_pool": args.lowfreq_pool,
    }
    return rows, meta_payload


def add_strength_bins(rows):
    sorted_rows = sorted(rows, key=lambda row: row["cr_psnr"])
    weak_cutoff = sorted_rows[max(0, len(rows) // 4 - 1)]["cr_psnr"]
    strong_cutoff = sorted_rows[min(len(rows) - 1, (len(rows) * 3) // 4)]["cr_psnr"]
    for row in rows:
        if row["cr_psnr"] <= weak_cutoff:
            row["cr_strength_bin"] = "cr_weakest_25"
        elif row["cr_psnr"] >= strong_cutoff:
            row["cr_strength_bin"] = "cr_strongest_25"
        else:
            row["cr_strength_bin"] = "cr_middle_50"


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


def build_feature_sets(rows):
    hazy = numeric_feature_names(rows, lambda key: key.startswith("hazy_"))
    teacher = numeric_feature_names(
        rows,
        lambda key: key.startswith(
            ("cr_out_", "lfv1_out_", "lfv1_minus_cr_", "cr_minus_hazy_", "lfv1_minus_hazy_")
        ),
    )
    metadata = numeric_feature_names(
        rows,
        lambda key: key in ("airlight", "beta", "height", "width", "patch_x_norm", "patch_y_norm"),
    )
    return {
        "hazy_wavelet": hazy,
        "teacher_output_proxy": teacher,
        "hazy_wavelet_plus_teacher_outputs": sorted(set(hazy + teacher)),
        "metadata_diagnostic": metadata,
        "hazy_wavelet_plus_metadata": sorted(set(hazy + metadata)),
        "hazy_teacher_outputs_plus_metadata": sorted(set(hazy + teacher + metadata)),
    }


def matrix(rows, features):
    data = []
    for row in rows:
        values = []
        for key in features:
            value = row.get(key, 0.0)
            values.append(float(value) if is_number(value) else 0.0)
        data.append(values)
    return np.asarray(data, dtype=np.float32)


def standardize(train_rows, valid_rows, features):
    x_train = matrix(train_rows, features)
    x_valid = matrix(valid_rows, features)
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma < EPS] = 1.0
    return (x_train - mu) / sigma, (x_valid - mu) / sigma


def targets(rows):
    return np.asarray([float(row["blend_oracle_c"]) for row in rows], dtype=np.float32)


def fit_predict_sklearn(x_train, y_train, x_valid, head, args):
    if head == "sklearn_hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            l2_regularization=args.hgb_l2,
            random_state=args.seed,
        )
    elif head == "sklearn_ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=args.ridge_alpha)
    else:
        raise ValueError("Unsupported head: {}".format(head))
    model.fit(x_train, y_train)
    return np.clip(model.predict(x_valid).astype(np.float32), 0.0, 1.0)


def blend_psnr(row, c):
    cr_mse = float(row["cr_mse"])
    lfv1_mse = float(row["lfv1_mse"])
    err_cross = float(row["err_cross"])
    value = ((1.0 - c) ** 2) * cr_mse + (c ** 2) * lfv1_mse + 2.0 * c * (1.0 - c) * err_cross
    return psnr_from_mse(value)


def evaluate_rows(rows, pred_c):
    pred_c = np.clip(np.asarray(pred_c, dtype=np.float32), 0.0, 1.0)
    blend = [blend_psnr(row, float(c)) for row, c in zip(rows, pred_c)]
    lfv1 = [float(row["lfv1_psnr"]) for row in rows]
    cr = [float(row["cr_psnr"]) for row in rows]
    oracle = [float(row["blend_oracle_psnr"]) for row in rows]
    oracle_c = [float(row["blend_oracle_c"]) for row in rows]
    delta = [float(row["lfv1_delta_cr_psnr"]) for row in rows]
    pred_values = [float(value) for value in pred_c]

    preserve_flags = []
    regression_improve_flags = []
    strong_regression_improve_flags = []
    interventions = []
    intervention_true = []
    for row, c, psnr_value in zip(rows, pred_c, blend):
        is_preserve = float(row["lfv1_delta_cr_psnr"]) >= 0.30
        is_regression = float(row["lfv1_delta_cr_psnr"]) <= -0.30
        is_strong = row.get("cr_strength_bin") == "cr_strongest_25"
        if is_preserve:
            preserve_flags.append(1 if psnr_value >= float(row["lfv1_psnr"]) - 0.10 else 0)
        if is_regression:
            regression_improve_flags.append(1 if psnr_value >= float(row["lfv1_psnr"]) + 0.10 else 0)
            if is_strong:
                strong_regression_improve_flags.append(1 if psnr_value >= float(row["lfv1_psnr"]) + 0.10 else 0)
        if c <= 0.5:
            interventions.append(1)
            intervention_true.append(1 if is_regression else 0)

    gain = mean(blend) - mean(lfv1)
    oracle_gain = mean(oracle) - mean(lfv1)
    return {
        "n": len(rows),
        "mean_pred_c": mean(pred_values),
        "std_pred_c": std(pred_values),
        "mean_oracle_c": mean(oracle_c),
        "blend_mean_psnr": mean(blend),
        "cr_mean_psnr": mean(cr),
        "lfv1_mean_psnr": mean(lfv1),
        "oracle_mean_psnr": mean(oracle),
        "gain_vs_lfv1": float(gain),
        "gain_vs_cr": mean(blend) - mean(cr),
        "oracle_gain_vs_lfv1": float(oracle_gain),
        "oracle_recovery": float(gain / (oracle_gain + EPS)) if oracle_gain > EPS else 0.0,
        "preserve_recall": mean(preserve_flags),
        "regression_improve_recall": mean(regression_improve_flags),
        "strong_cr_regression_improve_recall": mean(strong_regression_improve_flags),
        "intervention_rate": mean(interventions),
        "intervention_precision": mean(intervention_true),
        "pred_oracle_c_corr": corr(pred_c, oracle_c),
        "pred_lfv1_delta_corr": corr(pred_c, delta),
    }


def split_random_by_image(rows, valid_fraction, rng):
    images = sorted({row["filename"] for row in rows})
    rng.shuffle(images)
    valid_n = max(1, int(round(len(images) * valid_fraction)))
    valid_images = set(images[:valid_n])
    train = [row for row in rows if row["filename"] not in valid_images]
    valid = [row for row in rows if row["filename"] in valid_images]
    return train, valid


def split_group(rows, key, value):
    valid_images = {row["filename"] for row in rows if row.get(key) == value}
    valid = [row for row in rows if row["filename"] in valid_images]
    train = [row for row in rows if row["filename"] not in valid_images]
    return train, valid


def valid_split(train, valid):
    return len(train) >= 100 and len(valid) >= 40


def run_one(train, valid, features, head, args):
    x_train, x_valid = standardize(train, valid, features)
    pred = fit_predict_sklearn(x_train, targets(train), x_valid, head, args)
    return evaluate_rows(valid, pred)


def run_audit(rows, feature_sets_by_name, args):
    rng = random.Random(args.seed)
    wanted_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    if wanted_sets != ["all"]:
        feature_sets_by_name = {
            name: values for name, values in feature_sets_by_name.items()
            if name in set(wanted_sets)
        }
    heads = tuple(item.strip() for item in args.heads.split(",") if item.strip())
    heldout_groups = tuple(item.strip() for item in args.heldout_groups.split(",") if item.strip())
    results = []
    for feature_set_name, features in feature_sets_by_name.items():
        if not features:
            continue
        for head in heads:
            for split_idx in range(args.splits):
                train, valid = split_random_by_image(rows, args.valid_fraction, rng)
                if not valid_split(train, valid):
                    continue
                print("audit feature_set={} head={} split=random_image_{}".format(feature_set_name, head, split_idx + 1), flush=True)
                metrics = run_one(train, valid, features, head, args)
                metrics.update({
                    "feature_set": feature_set_name,
                    "head": head,
                    "split_family": "random_image",
                    "split": "random_image_{}".format(split_idx + 1),
                    "feature_count": len(features),
                })
                results.append(metrics)
            for group_key in heldout_groups:
                values = sorted({row.get(group_key, "") for row in rows if row.get(group_key, "") != ""})
                for value in values:
                    train, valid = split_group(rows, group_key, value)
                    if not valid_split(train, valid):
                        continue
                    print("audit feature_set={} head={} split={}={}".format(feature_set_name, head, group_key, value), flush=True)
                    metrics = run_one(train, valid, features, head, args)
                    metrics.update({
                        "feature_set": feature_set_name,
                        "head": head,
                        "split_family": group_key,
                        "split": "{}={}".format(group_key, value),
                        "feature_count": len(features),
                    })
                    results.append(metrics)
    return results


def passes_main(summary, args):
    if summary["feature_set"] != args.main_feature_set:
        return False
    if summary["head"] != args.main_head:
        return False
    if summary["split_family"] != "random_image":
        return False
    return (
        float(summary.get("gain_vs_lfv1_mean") or 0.0) >= args.min_gain
        and float(summary.get("oracle_recovery_mean") or 0.0) >= args.min_oracle_recovery
        and float(summary.get("preserve_recall_mean") or 0.0) >= args.min_preserve_recall
        and float(summary.get("regression_improve_recall_mean") or 0.0) >= args.min_regression_improve_recall
        and float(summary.get("intervention_precision_mean") or 0.0) >= args.min_intervention_precision
        and float(summary.get("strong_cr_regression_improve_recall_mean") or 0.0) >= args.min_strong_cr_regression_improve_recall
        and float(summary.get("pred_oracle_c_corr_mean") or 0.0) >= args.min_oracle_c_corr
    )


def summarize(results, args):
    grouped = {}
    for row in results:
        key = (row["feature_set"], row["head"], row["split_family"])
        grouped.setdefault(key, []).append(row)
    metrics = (
        "gain_vs_lfv1",
        "gain_vs_cr",
        "oracle_recovery",
        "preserve_recall",
        "regression_improve_recall",
        "strong_cr_regression_improve_recall",
        "intervention_rate",
        "intervention_precision",
        "pred_oracle_c_corr",
        "pred_lfv1_delta_corr",
        "mean_pred_c",
        "mean_oracle_c",
    )
    summaries = []
    for (feature_set, head, split_family), items in sorted(grouped.items()):
        summary = {
            "feature_set": feature_set,
            "head": head,
            "split_family": split_family,
            "splits": len(items),
        }
        for metric in metrics:
            values = [item[metric] for item in items if is_number(item.get(metric, ""))]
            summary[metric + "_mean"] = mean(values)
            summary[metric + "_std"] = std(values)
        summary["passes_main_line"] = passes_main(summary, args)
        summaries.append(summary)
    add_stability_flags(summaries, args)
    return summaries


def add_stability_flags(summaries, args):
    by_key = {}
    for row in summaries:
        key = (row["feature_set"], row["head"])
        by_key.setdefault(key, []).append(row)
    for row in summaries:
        row["passes_stability_line"] = False
        if not row["passes_main_line"]:
            continue
        siblings = by_key[(row["feature_set"], row["head"])]
        heldout = [item for item in siblings if item["split_family"] in ("airlight_bin", "beta_bin")]
        if not heldout:
            continue
        row["heldout_min_gain_vs_lfv1"] = min(float(item.get("gain_vs_lfv1_mean") or 0.0) for item in heldout)
        row["heldout_min_preserve_recall"] = min(float(item.get("preserve_recall_mean") or 0.0) for item in heldout)
        row["heldout_min_regression_improve_recall"] = min(float(item.get("regression_improve_recall_mean") or 0.0) for item in heldout)
        row["heldout_min_pred_oracle_c_corr"] = min(float(item.get("pred_oracle_c_corr_mean") or 0.0) for item in heldout)
        row["passes_stability_line"] = (
            row["heldout_min_gain_vs_lfv1"] >= args.min_heldout_gain
            and row["heldout_min_preserve_recall"] >= args.min_heldout_preserve_recall
            and row["heldout_min_regression_improve_recall"] >= args.min_heldout_regression_improve_recall
            and row["heldout_min_pred_oracle_c_corr"] >= args.min_heldout_oracle_c_corr
        )


def write_report(path, summary, meta, recommendation):
    lines = [
        "# HAZE4K Residual Field Confidence Preflight",
        "",
        "## Recommendation",
        "",
        "- `{}`".format(recommendation),
        "",
        "## Source",
        "",
        "- Images: `{}`".format(meta["images"]),
        "- Patches: `{}`".format(meta["patches"]),
        "- Baseline checkpoint step: `{}`".format(meta.get("baseline_checkpoint_step")),
        "- LF-v1 checkpoint step: `{}`".format(meta.get("lfv1_checkpoint_step")),
        "- Low-frequency pool: `{}`".format(meta.get("lowfreq_pool")),
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
        }
    else:
        rows, meta = collect_rows(args)
    add_strength_bins(rows)
    feature_sets = build_feature_sets(rows)
    write_csv(output_dir / "residual_field_confidence_features.csv", rows)
    results = run_audit(rows, feature_sets, args)
    summary = summarize(results, args)
    pass_rows = [row for row in summary if row.get("passes_main_line") and row.get("passes_stability_line")]
    recommendation = (
        "proceed_to_cr_ref_residual_field_scout"
        if pass_rows else
        "do_not_train_residual_field_confidence_yet"
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
    write_report(output_dir / "analysis_report.md", summary, meta, recommendation)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
