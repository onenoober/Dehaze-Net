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

from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs
from model import DEANet


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether frozen DEA-Net bottleneck/decoder representations "
            "predict baseline-relative low-frequency residual direction and "
            "reliability better than output-only features."
        )
    )
    parser.add_argument("--dataset_root", type=str, default="../dataset/HAZE4K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hazy_dir", type=str, default="")
    parser.add_argument("--clear_dir", type=str, default="")
    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--lfv1_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--features_npz", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_size", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--valid_fraction", type=float, default=0.25)
    parser.add_argument("--random_splits", type=int, default=3)
    parser.add_argument("--target_grid", type=int, default=8)
    parser.add_argument("--spatial_feature_grid", type=int, default=4)
    parser.add_argument("--feature_pool_grid", type=int, default=2)
    parser.add_argument("--heads", type=str, default="ridge,tiny_mlp")
    parser.add_argument(
        "--feature_sets",
        type=str,
        default="A_output,B_cr_features,C_lfv1_features,D_feature_contrast,B_cr_features_shuffled,C_lfv1_features_shuffled,D_feature_contrast_shuffled",
    )
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--mlp_hidden", type=int, default=64)
    parser.add_argument("--mlp_max_iter", type=int, default=160)
    parser.add_argument("--mlp_alpha", type=float, default=0.001)
    parser.add_argument("--positive_margin", type=float, default=0.30)
    parser.add_argument("--intervention_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--min_residual_cosine", type=float, default=0.20)
    parser.add_argument("--max_wrong_direction_rate", type=float, default=0.35)
    parser.add_argument("--min_lf_mse_improved_rate", type=float, default=0.55)
    parser.add_argument("--min_lfv1_gain_preservation", type=float, default=0.70)
    parser.add_argument("--min_strong_cr_preservation", type=float, default=0.70)
    parser.add_argument("--min_intervention_precision", type=float, default=0.60)
    parser.add_argument("--min_confidence_corr", type=float, default=0.45)
    parser.add_argument("--min_output_cosine_gap", type=float, default=0.05)
    parser.add_argument("--min_shuffled_cosine_gap", type=float, default=0.05)
    parser.add_argument("--min_heldout_residual_cosine", type=float, default=0.12)
    parser.add_argument("--max_heldout_wrong_direction_rate", type=float, default=0.45)
    parser.add_argument("--min_heldout_preservation", type=float, default=0.62)
    parser.add_argument("--min_heldout_confidence_corr", type=float, default=0.30)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_model(checkpoint_path, use_lf_prior, device):
    model = DEANet(
        base_dim=32,
        use_lf_prior=use_lf_prior,
        lf_prior_channels=8,
        lf_prior_pool=8,
        lf_prior_gate_init=0.0,
        lf_prior_injection="pre_mix",
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(strip_module_prefix(checkpoint_state_dict(checkpoint)))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    step = checkpoint.get("step", "") if isinstance(checkpoint, dict) else ""
    return model, step


def pad_tensor(x, pad_size):
    if pad_size <= 0:
        return x
    _, _, h, w = x.size()
    mod_pad_h = (pad_size - h % pad_size) % pad_size
    mod_pad_w = (pad_size - w % pad_size) % pad_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")


def image_tensor(path, device):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def infer_with_features(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_tensor(hazy, pad_size)
    out, features = model.forward_with_features(padded)
    out = out[:, :, :h, :w].clamp(0, 1)
    return out, features


def lowpass(x, pool_size):
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)


def psnr_tensor(pred, target):
    mse = float(F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).detach().cpu().item())
    if mse <= EPS:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def parse_haze4k_name(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    meta = {"image_id": parts[0] if parts else stem, "airlight": "", "beta": ""}
    if len(parts) >= 3:
        try:
            meta["airlight"] = float(parts[1])
            meta["beta"] = float(parts[2])
        except ValueError:
            pass
    return meta


def make_bin(value, edges, labels):
    if value == "":
        return "unknown"
    value = float(value)
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def add_meta_bins(row):
    a_edges = [0.65, 0.75, 0.85, 0.95]
    a_labels = ["A<0.65", "0.65<=A<0.75", "0.75<=A<0.85", "0.85<=A<0.95", "A>=0.95"]
    beta_edges = [0.8, 1.1, 1.4, 1.7]
    beta_labels = ["beta<0.8", "0.8<=beta<1.1", "1.1<=beta<1.4", "1.4<=beta<1.7", "beta>=1.7"]
    row["airlight_bin"] = make_bin(row["airlight"], a_edges, a_labels)
    row["beta_bin"] = make_bin(row["beta"], beta_edges, beta_labels)


def add_cr_strength_bins(rows):
    scores = np.asarray([row["cr_psnr"] for row in rows], dtype=np.float64)
    q1, q2, q3 = np.quantile(scores, [0.25, 0.50, 0.75])
    for row in rows:
        value = row["cr_psnr"]
        if value < q1:
            row["cr_strength_bin"] = "weak_cr_q1"
        elif value < q2:
            row["cr_strength_bin"] = "midlow_cr_q2"
        elif value < q3:
            row["cr_strength_bin"] = "midhigh_cr_q3"
        else:
            row["cr_strength_bin"] = "strong_cr_q4"


def add_stats(features, prefix, tensor):
    x = tensor.detach().float()
    flat = x.flatten(2)
    features[prefix + "_mean"] = scalar(x.mean())
    features[prefix + "_std"] = scalar(x.std(unbiased=False))
    features[prefix + "_abs_mean"] = scalar(x.abs().mean())
    features[prefix + "_rms"] = scalar(torch.sqrt(torch.mean(x * x) + EPS))
    features[prefix + "_min"] = scalar(x.min())
    features[prefix + "_max"] = scalar(x.max())
    channel_mean = x.mean(dim=(2, 3))[0]
    channel_std = flat.std(dim=2, unbiased=False)[0]
    for idx, value in enumerate(channel_mean):
        features["{}_c{:03d}_mean".format(prefix, idx)] = scalar(value)
    for idx, value in enumerate(channel_std):
        features["{}_c{:03d}_std".format(prefix, idx)] = scalar(value)


def add_spatial(features, prefix, tensor, grid):
    pooled = F.adaptive_avg_pool2d(tensor.detach().float(), (grid, grid))[0]
    flat = pooled.reshape(-1)
    for idx, value in enumerate(flat):
        features["{}_sp{:04d}".format(prefix, idx)] = scalar(value)


def add_image_features(features, prefix, tensor, grid):
    add_stats(features, prefix, tensor)
    add_spatial(features, prefix, tensor, grid)
    lp = lowpass(tensor, 8)
    hf = tensor - lp
    add_stats(features, prefix + "_lp8", lp)
    add_stats(features, prefix + "_hf8", hf)


def add_model_feature_block(features, prefix, fmap_by_name, pool_grid):
    for name in ("bottleneck", "dec3", "dec2", "dec1"):
        fmap = fmap_by_name[name]
        key = "{}_{}".format(prefix, name)
        add_stats(features, key, fmap)
        pooled = F.adaptive_avg_pool2d(fmap.detach().float(), (pool_grid, pool_grid))[0]
        flat = pooled.reshape(-1)
        for idx, value in enumerate(flat):
            features["{}_p{:04d}".format(key, idx)] = scalar(value)


def add_feature_contrast(features, cr_feats, lfv1_feats, pool_grid):
    for name in ("bottleneck", "dec3", "dec2", "dec1"):
        cr = cr_feats[name].detach().float()
        lf = lfv1_feats[name].detach().float()
        diff = lf - cr
        key = "contrast_" + name
        add_stats(features, key, diff)
        add_spatial(features, key, diff, pool_grid)
        cr_vec = cr.reshape(1, -1)
        lf_vec = lf.reshape(1, -1)
        numerator = (cr_vec * lf_vec).sum(dim=1)
        denominator = cr_vec.norm(dim=1) * lf_vec.norm(dim=1) + EPS
        features[key + "_cosine"] = scalar((numerator / denominator).mean())
        features[key + "_norm_ratio"] = scalar(lf_vec.norm(dim=1).mean() / (cr_vec.norm(dim=1).mean() + EPS))


def target_vectors(j0, gt, target_grid):
    j0_lp = lowpass(j0, 8)
    gt_lp = lowpass(gt, 8)
    residual = gt_lp - j0_lp
    target = F.adaptive_avg_pool2d(residual, (target_grid, target_grid))[0].reshape(-1)
    base = F.adaptive_avg_pool2d(j0_lp, (target_grid, target_grid))[0].reshape(-1)
    truth = F.adaptive_avg_pool2d(gt_lp, (target_grid, target_grid))[0].reshape(-1)
    return (
        target.detach().cpu().numpy().astype(np.float32),
        base.detach().cpu().numpy().astype(np.float32),
        truth.detach().cpu().numpy().astype(np.float32),
    )


def collect_rows(args):
    split_root = Path(args.dataset_root) / args.split
    hazy_dir, clear_dir = (
        (args.hazy_dir, args.clear_dir)
        if args.hazy_dir and args.clear_dir
        else resolve_pair_dirs(split_root)
    )
    hazy_dir = Path(hazy_dir)
    clear_dir = Path(clear_dir)
    image_names = list_image_files(hazy_dir)
    if args.max_images > 0:
        image_names = image_names[: args.max_images]

    cr_model, cr_step = load_model(args.baseline_checkpoint, False, args.device)
    lfv1_model, lfv1_step = load_model(args.lfv1_checkpoint, True, args.device)

    rows = []
    y = []
    base_lp = []
    gt_lp = []
    total = len(image_names)
    with torch.no_grad():
        for idx, filename in enumerate(image_names, 1):
            hazy = image_tensor(hazy_dir / filename, args.device)
            clear = image_tensor(find_clear_image(clear_dir, filename), args.device)
            cr_out, cr_feats = infer_with_features(cr_model, hazy, args.pad_size)
            lfv1_out, lfv1_feats = infer_with_features(lfv1_model, hazy, args.pad_size)

            features = {}
            add_image_features(features, "input", hazy, args.spatial_feature_grid)
            add_image_features(features, "j0", cr_out, args.spatial_feature_grid)
            add_image_features(features, "input_minus_j0", hazy - cr_out, args.spatial_feature_grid)
            add_model_feature_block(features, "cr_feat", cr_feats, args.feature_pool_grid)
            add_model_feature_block(features, "lfv1_feat", lfv1_feats, args.feature_pool_grid)
            add_feature_contrast(features, cr_feats, lfv1_feats, args.feature_pool_grid)

            target, base, truth = target_vectors(cr_out, clear, args.target_grid)
            y.append(target)
            base_lp.append(base)
            gt_lp.append(truth)

            row = dict(features)
            row.update(parse_haze4k_name(filename))
            row["filename"] = filename
            row["cr_psnr"] = psnr_tensor(cr_out, clear)
            row["lfv1_psnr"] = psnr_tensor(lfv1_out, clear)
            row["lfv1_delta_cr_psnr"] = row["lfv1_psnr"] - row["cr_psnr"]
            row["target_lf_mse"] = float(np.mean(target * target))
            add_meta_bins(row)
            rows.append(row)

            if idx % 25 == 0 or idx == total:
                print("features {}/{}".format(idx, total), flush=True)

    add_cr_strength_bins(rows)
    meta = {
        "images": len(rows),
        "hazy_dir": str(hazy_dir),
        "clear_dir": str(clear_dir),
        "baseline_checkpoint": args.baseline_checkpoint,
        "lfv1_checkpoint": args.lfv1_checkpoint,
        "baseline_checkpoint_step": cr_step,
        "lfv1_checkpoint_step": lfv1_step,
        "target_grid": args.target_grid,
    }
    arrays = {
        "y": np.stack(y, axis=0).astype(np.float32),
        "base_lp": np.stack(base_lp, axis=0).astype(np.float32),
        "gt_lp": np.stack(gt_lp, axis=0).astype(np.float32),
    }
    return rows, arrays, meta


def numeric_feature_keys(rows):
    excluded = {
        "filename",
        "image_id",
        "airlight_bin",
        "beta_bin",
        "cr_strength_bin",
    }
    keys = []
    for key in sorted(rows[0].keys()):
        if key in excluded:
            continue
        if isinstance(rows[0].get(key), (int, float, np.floating)):
            keys.append(key)
    return keys


def build_feature_sets(rows, args):
    all_keys = numeric_feature_keys(rows)
    output_prefixes = (
        "input_",
        "j0_",
        "input_minus_j0_",
        "airlight",
        "beta",
    )
    output_keys = [key for key in all_keys if key.startswith(output_prefixes) or key in ("airlight", "beta")]
    cr_keys = [key for key in all_keys if key.startswith("cr_feat_")]
    lfv1_keys = [key for key in all_keys if key.startswith("lfv1_feat_")]
    contrast_keys = [key for key in all_keys if key.startswith("contrast_")]
    return {
        "A_output": output_keys,
        "B_cr_features": output_keys + cr_keys,
        "C_lfv1_features": output_keys + lfv1_keys,
        "D_feature_contrast": output_keys + contrast_keys,
        "B_cr_features_shuffled": output_keys + ["shuf_" + key for key in cr_keys],
        "C_lfv1_features_shuffled": output_keys + ["shuf_" + key for key in lfv1_keys],
        "D_feature_contrast_shuffled": output_keys + ["shuf_" + key for key in contrast_keys],
    }


def add_shuffled_feature_blocks(rows, args):
    rng = random.Random(args.seed + 137)
    all_keys = numeric_feature_keys(rows)
    for prefix in ("cr_feat_", "lfv1_feat_", "contrast_"):
        keys = [key for key in all_keys if key.startswith(prefix)]
        order = list(range(len(rows)))
        rng.shuffle(order)
        source = [{key: rows[i][key] for key in keys} for i in order]
        for row, values in zip(rows, source):
            for key, value in values.items():
                row["shuf_" + key] = value


def matrix(rows, indices, features):
    x = np.zeros((len(indices), len(features)), dtype=np.float32)
    for row_pos, idx in enumerate(indices):
        row = rows[idx]
        for col, key in enumerate(features):
            value = row.get(key, 0.0)
            if value == "":
                value = 0.0
            x[row_pos, col] = float(value)
    return x


def standardize(x_train, x_valid):
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    keep = sigma > 1e-6
    if not np.any(keep):
        keep = np.ones_like(sigma, dtype=bool)
    sigma[sigma < 1e-6] = 1.0
    return (x_train[:, keep] - mu[keep]) / sigma[keep], (x_valid[:, keep] - mu[keep]) / sigma[keep], int(np.sum(keep))


def split_random(n, valid_fraction, rng):
    order = list(range(n))
    rng.shuffle(order)
    valid_n = max(1, int(round(n * valid_fraction)))
    valid = sorted(order[:valid_n])
    train = sorted(order[valid_n:])
    return train, valid


def split_group(rows, key, value):
    valid = [idx for idx, row in enumerate(rows) if row.get(key) == value]
    train = [idx for idx, row in enumerate(rows) if row.get(key) != value]
    return train, valid


def valid_split(train, valid):
    return len(train) >= 50 and len(valid) >= 25


def fit_predict_head(head, x_train, y_train, x_valid, args, seed):
    if head == "ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=args.ridge_alpha, random_state=seed)
    elif head == "tiny_mlp":
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(
            hidden_layer_sizes=(args.mlp_hidden,),
            activation="relu",
            solver="adam",
            alpha=args.mlp_alpha,
            batch_size=min(256, max(32, len(x_train))),
            learning_rate_init=0.001,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=seed,
        )
    else:
        raise ValueError("Unknown head {}".format(head))
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train).astype(np.float32)
    valid_pred = model.predict(x_valid).astype(np.float32)
    return train_pred, valid_pred


def fit_confidence(x_train, train_gain, x_valid, args, seed):
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=args.ridge_alpha, random_state=seed)
    model.fit(x_train, train_gain)
    return model.predict(x_valid).astype(np.float32)


def eval_predictions(rows, indices, pred, conf, arrays, args):
    y = arrays["y"][indices]
    base = arrays["base_lp"][indices]
    truth = arrays["gt_lp"][indices]
    base_mse = np.mean((base - truth) ** 2, axis=1)
    pred_mse = np.mean((base + pred - truth) ** 2, axis=1)
    actual_gain = base_mse - pred_mse
    improved = actual_gain > 0.0
    dot = np.sum(pred * y, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(y, axis=1)
    cosine = dot / (pred_norm * target_norm + EPS)
    error_ratio = np.linalg.norm(pred - y, axis=1) / (target_norm + EPS)
    norm_ratio = pred_norm / (target_norm + EPS)
    selected = conf > args.intervention_confidence_threshold

    gain_cases = [
        pos for pos, idx in enumerate(indices)
        if rows[idx]["lfv1_delta_cr_psnr"] >= args.positive_margin
    ]
    strong_cases = [
        pos for pos, idx in enumerate(indices)
        if rows[idx]["cr_strength_bin"] == "strong_cr_q4"
    ]

    return {
        "n": len(indices),
        "mean_residual_cosine": float(np.mean(cosine)),
        "median_residual_cosine": float(np.median(cosine)),
        "wrong_direction_count": int(np.sum(cosine < 0.0)),
        "wrong_direction_rate": float(np.mean(cosine < 0.0)),
        "lf_mse_improved_count": int(np.sum(improved)),
        "lf_mse_regressed_count": int(np.sum(~improved)),
        "lf_mse_improved_rate": float(np.mean(improved)),
        "lf_mse_gain_mean": float(np.mean(actual_gain)),
        "residual_norm_ratio_mean": float(np.mean(norm_ratio)),
        "residual_error_ratio_mean": float(np.mean(error_ratio)),
        "intervention_count": int(np.sum(selected)),
        "intervention_precision": float(np.mean(improved[selected])) if np.any(selected) else 0.0,
        "confidence_corr": pearson(conf.tolist(), actual_gain.tolist()),
        "lfv1_gain_case_count": len(gain_cases),
        "lfv1_gain_preservation_recall": float(np.mean(improved[gain_cases])) if gain_cases else 0.0,
        "strong_cr_case_count": len(strong_cases),
        "strong_cr_preservation_recall": float(np.mean(improved[strong_cases])) if strong_cases else 0.0,
    }, actual_gain


def run_one(rows, arrays, train_idx, valid_idx, feature_set, features, head, args, seed):
    x_train = matrix(rows, train_idx, features)
    x_valid = matrix(rows, valid_idx, features)
    x_train, x_valid, active_features = standardize(x_train, x_valid)
    y_train = arrays["y"][train_idx]
    train_pred, valid_pred = fit_predict_head(head, x_train, y_train, x_valid, args, seed)
    train_metrics, train_gain = eval_predictions(rows, train_idx, train_pred, np.ones(len(train_idx)), arrays, args)
    valid_conf = fit_confidence(x_train, train_gain, x_valid, args, seed)
    metrics, _ = eval_predictions(rows, valid_idx, valid_pred, valid_conf, arrays, args)
    metrics.update({
        "feature_set": feature_set,
        "head": head,
        "feature_count": len(features),
        "active_feature_count": active_features,
        "train_mean_residual_cosine": train_metrics["mean_residual_cosine"],
        "train_lf_mse_improved_rate": train_metrics["lf_mse_improved_rate"],
    })
    return metrics


def run_audit(rows, arrays, feature_sets, args):
    requested_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    requested_heads = [item.strip() for item in args.heads.split(",") if item.strip()]
    rng = random.Random(args.seed)
    results = []
    split_specs = []
    for split_idx in range(args.random_splits):
        train, valid = split_random(len(rows), args.valid_fraction, rng)
        split_specs.append(("random", "random_{}".format(split_idx + 1), train, valid))
    for key in ("airlight_bin", "beta_bin", "cr_strength_bin"):
        values = sorted({row.get(key, "") for row in rows if row.get(key, "")})
        for value in values:
            train, valid = split_group(rows, key, value)
            split_specs.append((key, "{}={}".format(key, value), train, valid))

    for feature_set in requested_sets:
        features = feature_sets.get(feature_set, [])
        if not features:
            continue
        for head in requested_heads:
            for split_num, (split_family, split_name, train, valid) in enumerate(split_specs):
                if not valid_split(train, valid):
                    continue
                print("{} {} {}".format(feature_set, head, split_name), flush=True)
                metrics = run_one(
                    rows,
                    arrays,
                    train,
                    valid,
                    feature_set,
                    features,
                    head,
                    args,
                    args.seed + split_num,
                )
                metrics.update({
                    "split_family": split_family,
                    "split": split_name,
                    "train_n": len(train),
                    "valid_n": len(valid),
                })
                results.append(metrics)
    return results


def mean(values):
    values = [float(value) for value in values if value != ""]
    if not values:
        return 0.0
    return float(np.mean(values))


def std(values):
    values = [float(value) for value in values if value != ""]
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def summarize(results, args):
    grouped = {}
    for row in results:
        key = (row["feature_set"], row["head"], row["split_family"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (feature_set, head, split_family), items in sorted(grouped.items()):
        summary = {
            "feature_set": feature_set,
            "head": head,
            "split_family": split_family,
            "splits": len(items),
        }
        for metric in (
            "mean_residual_cosine",
            "median_residual_cosine",
            "wrong_direction_rate",
            "lf_mse_improved_rate",
            "lf_mse_gain_mean",
            "intervention_precision",
            "confidence_corr",
            "lfv1_gain_preservation_recall",
            "strong_cr_preservation_recall",
            "residual_norm_ratio_mean",
            "residual_error_ratio_mean",
        ):
            vals = [item[metric] for item in items]
            summary[metric + "_mean"] = mean(vals)
            summary[metric + "_std"] = std(vals)
        summaries.append(summary)
    by_key = {(row["feature_set"], row["head"], row["split_family"]): row for row in summaries}
    for row in summaries:
        feature_set = row["feature_set"]
        head = row["head"]
        split_family = row["split_family"]
        output = by_key.get(("A_output", head, split_family), {})
        control_name = feature_set + "_shuffled"
        if feature_set.endswith("_shuffled"):
            control_name = ""
        control = by_key.get((control_name, head, split_family), {})
        row["residual_cosine_gap_vs_output"] = (
            row["mean_residual_cosine_mean"] - output.get("mean_residual_cosine_mean", row["mean_residual_cosine_mean"])
        )
        row["residual_cosine_gap_vs_shuffled"] = (
            row["mean_residual_cosine_mean"] - control.get("mean_residual_cosine_mean", row["mean_residual_cosine_mean"])
            if control else 0.0
        )
        row["passes_random_line"] = passes_random(row, args)
        row["passes_heldout_line"] = passes_heldout(row, args)
    return summaries


def passes_random(row, args):
    if row["split_family"] != "random":
        return False
    if row["feature_set"] in ("A_output",) or row["feature_set"].endswith("_shuffled"):
        return False
    return (
        row["mean_residual_cosine_mean"] >= args.min_residual_cosine
        and row["wrong_direction_rate_mean"] <= args.max_wrong_direction_rate
        and row["lf_mse_improved_rate_mean"] >= args.min_lf_mse_improved_rate
        and row["lfv1_gain_preservation_recall_mean"] >= args.min_lfv1_gain_preservation
        and row["strong_cr_preservation_recall_mean"] >= args.min_strong_cr_preservation
        and row["intervention_precision_mean"] >= args.min_intervention_precision
        and row["confidence_corr_mean"] >= args.min_confidence_corr
        and row["residual_cosine_gap_vs_output"] >= args.min_output_cosine_gap
        and row["residual_cosine_gap_vs_shuffled"] >= args.min_shuffled_cosine_gap
    )


def passes_heldout(row, args):
    if row["split_family"] == "random":
        return False
    if row["feature_set"] in ("A_output",) or row["feature_set"].endswith("_shuffled"):
        return False
    return (
        row["mean_residual_cosine_mean"] >= args.min_heldout_residual_cosine
        and row["wrong_direction_rate_mean"] <= args.max_heldout_wrong_direction_rate
        and row["lfv1_gain_preservation_recall_mean"] >= args.min_heldout_preservation
        and row["strong_cr_preservation_recall_mean"] >= args.min_heldout_preservation
        and row["confidence_corr_mean"] >= args.min_heldout_confidence_corr
    )


def final_recommendation(summaries):
    random_pass = [row for row in summaries if row["passes_random_line"]]
    if not random_pass:
        return "do_not_train_brf_v2_representation_yet"
    for row in random_pass:
        feature_set = row["feature_set"]
        head = row["head"]
        heldout = [
            item for item in summaries
            if item["feature_set"] == feature_set and item["head"] == head and item["split_family"] != "random"
        ]
        if heldout and all(item["passes_heldout_line"] for item in heldout):
            return "stage0_passed_write_brf_v2_representation_model_card"
    return "do_not_train_brf_v2_representation_yet"


def serializable_feature_rows(rows):
    out = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key.startswith("shuf_"):
                continue
            if isinstance(value, (int, float, str, np.floating)):
                item[key] = value
        out.append(item)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, summaries, recommendation, meta, args):
    lines = [
        "# HAZE4K BRFRC-v2 Representation Audit",
        "",
        "## Recommendation",
        "",
        "- `{}`".format(recommendation),
        "",
        "## Source",
        "",
        "- Images: `{}`".format(meta["images"]),
        "- Baseline checkpoint step: `{}`".format(meta.get("baseline_checkpoint_step")),
        "- LF-v1 checkpoint step: `{}`".format(meta.get("lfv1_checkpoint_step")),
        "- Target grid: `{}`".format(meta.get("target_grid")),
        "",
        "## Summary",
        "",
        "| Feature Set | Head | Split | Cos | Wrong | LF Improve | LF-v1 Preserve | Strong-CR Preserve | Precision | Conf Corr | Gap Output | Gap Shuffled | Random Pass | Heldout Pass |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {feature_set} | {head} | {split} | {cos:.4f} | {wrong:.4f} | {imp:.4f} | {gain:.4f} | {strong:.4f} | {prec:.4f} | {corr:.4f} | {go:.4f} | {gs:.4f} | {rp} | {hp} |".format(
                feature_set=row["feature_set"],
                head=row["head"],
                split=row["split_family"],
                cos=row["mean_residual_cosine_mean"],
                wrong=row["wrong_direction_rate_mean"],
                imp=row["lf_mse_improved_rate_mean"],
                gain=row["lfv1_gain_preservation_recall_mean"],
                strong=row["strong_cr_preservation_recall_mean"],
                prec=row["intervention_precision_mean"],
                corr=row["confidence_corr_mean"],
                go=row["residual_cosine_gap_vs_output"],
                gs=row["residual_cosine_gap_vs_shuffled"],
                rp="yes" if row["passes_random_line"] else "no",
                hp="yes" if row["passes_heldout_line"] else "no",
            )
        )
    lines.extend([
        "",
        "## Pass Line",
        "",
        "- Random residual cosine >= `{:.2f}` and output/shuffled cosine gap >= `{:.2f}`.".format(
            args.min_residual_cosine, args.min_output_cosine_gap
        ),
        "- Wrong-direction rate <= `{:.2f}`.".format(args.max_wrong_direction_rate),
        "- LF MSE improved rate >= `{:.2f}`.".format(args.min_lf_mse_improved_rate),
        "- LF-v1 gain and strong-CR preservation recall >= `{:.2f}`.".format(args.min_lfv1_gain_preservation),
        "- Intervention precision >= `{:.2f}` and confidence correlation >= `{:.2f}`.".format(
            args.min_intervention_precision, args.min_confidence_corr
        ),
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_npz(path, rows, arrays, feature_sets, meta):
    feature_keys = numeric_feature_keys(rows)
    x = matrix(rows, list(range(len(rows))), feature_keys)
    payload = {
        "x": x.astype(np.float32),
        "y": arrays["y"],
        "base_lp": arrays["base_lp"],
        "gt_lp": arrays["gt_lp"],
        "feature_keys": np.asarray(feature_keys, dtype=object),
        "filenames": np.asarray([row["filename"] for row in rows], dtype=object),
        "feature_sets": json.dumps(feature_sets, sort_keys=True),
        "meta": json.dumps(meta, sort_keys=True),
    }
    np.savez_compressed(path, **payload)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.features_npz:
        raise NotImplementedError("Reusing feature npz is not implemented for this first audit script.")

    rows, arrays, meta = collect_rows(args)
    add_shuffled_feature_blocks(rows, args)
    feature_sets = build_feature_sets(rows, args)
    results = run_audit(rows, arrays, feature_sets, args)
    summaries = summarize(results, args)
    recommendation = final_recommendation(summaries)

    write_csv(output_dir / "feature_rows_compact.csv", serializable_feature_rows(rows))
    write_csv(output_dir / "split_results.csv", results)
    write_csv(output_dir / "summary.csv", summaries)
    save_npz(output_dir / "feature_matrix_targets.npz", rows, arrays, feature_sets, meta)
    payload = {
        "recommendation": recommendation,
        "args": vars(args),
        "meta": meta,
        "feature_counts": {name: len(values) for name, values in feature_sets.items()},
        "summary": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / "analysis_report.md", summaries, recommendation, meta, args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
