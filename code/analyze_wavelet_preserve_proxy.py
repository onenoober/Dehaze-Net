import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - activation audit is optional.
    torch = None
    F = None


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether hazy-input wavelet/degradation features can predict "
            "when LF-v1 should be preserved versus suppressed or calibrated."
        )
    )
    parser.add_argument(
        "--matrix_csv",
        type=str,
        default=(
            "../experiment/HAZE4K/route_evidence_review/"
            "HAZE4K-route-evidence-review-20260528/model_per_image_matrix.csv"
        ),
    )
    parser.add_argument(
        "--hazy_dir",
        type=str,
        default="../dataset/HAZE4K/test/hazy",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--valid_fraction", type=float, default=0.30)
    parser.add_argument("--logistic_steps", type=int, default=400)
    parser.add_argument("--logistic_lr", type=float, default=0.08)
    parser.add_argument("--logistic_l2", type=float, default=0.05)
    parser.add_argument("--positive_margin", type=float, default=0.30)
    parser.add_argument("--negative_margin", type=float, default=-0.30)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--baseline_checkpoint", type=str, default="")
    parser.add_argument("--lfv1_checkpoint", type=str, default="")
    parser.add_argument("--activation_device", type=str, default="")
    parser.add_argument("--activation_pad_size", type=int, default=4)
    parser.add_argument("--min_gain", type=float, default=0.10)
    parser.add_argument("--min_oracle_recovery", type=float, default=0.15)
    parser.add_argument("--min_precision", type=float, default=0.60)
    parser.add_argument("--min_preserve_recall", type=float, default=0.60)
    parser.add_argument("--min_balanced_accuracy", type=float, default=0.60)
    return parser.parse_args()


def parse_value(value):
    if value is None:
        return ""
    value = value.strip()
    if value == "":
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values):
    values = [float(value) for value in values if is_number(value)]
    return float(np.mean(values)) if values else ""


def std(values):
    values = [float(value) for value in values if is_number(value)]
    return float(np.std(values)) if values else ""


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{key: parse_value(value) for key, value in raw.items()} for raw in reader]
    if not rows:
        raise ValueError("No rows found in {}".format(path))
    return rows


def safe_float(row, key):
    value = row.get(key, "")
    if not is_number(value):
        raise ValueError("Missing numeric {} for {}".format(key, row.get("filename", "")))
    return float(value)


def stats(prefix, arr):
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(-1)
    return {
        prefix + "_mean": float(np.mean(flat)),
        prefix + "_std": float(np.std(flat)),
        prefix + "_min": float(np.min(flat)),
        prefix + "_max": float(np.max(flat)),
        prefix + "_p10": float(np.percentile(flat, 10)),
        prefix + "_p50": float(np.percentile(flat, 50)),
        prefix + "_p90": float(np.percentile(flat, 90)),
        prefix + "_abs_mean": float(np.mean(np.abs(flat))),
        prefix + "_energy": float(np.mean(flat * flat)),
    }


def haar_subbands(channel):
    h = channel.shape[0] - channel.shape[0] % 2
    w = channel.shape[1] - channel.shape[1] % 2
    channel = channel[:h, :w]
    a = channel[0::2, 0::2]
    b = channel[0::2, 1::2]
    c = channel[1::2, 0::2]
    d = channel[1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def add_wavelet_stats(features, prefix, channel, levels=2):
    current = channel
    for level in range(1, levels + 1):
        ll, lh, hl, hh = haar_subbands(current)
        features.update(stats("{}_l{}_ll".format(prefix, level), ll))
        features.update(stats("{}_l{}_lh".format(prefix, level), lh))
        features.update(stats("{}_l{}_hl".format(prefix, level), hl))
        features.update(stats("{}_l{}_hh".format(prefix, level), hh))
        high_energy = np.mean(lh * lh + hl * hl + hh * hh)
        ll_energy = np.mean(ll * ll)
        features["{}_l{}_high_to_ll_energy".format(prefix, level)] = float(
            high_energy / (ll_energy + EPS)
        )
        features["{}_l{}_directional_balance".format(prefix, level)] = float(
            (np.mean(np.abs(lh)) - np.mean(np.abs(hl))) /
            (np.mean(np.abs(lh)) + np.mean(np.abs(hl)) + EPS)
        )
        current = ll


def image_features(path):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError("Could not read image {}".format(path))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    dark = np.min(image, axis=2)
    maxc = np.max(image, axis=2)
    minc = np.min(image, axis=2)
    saturation = (maxc - minc) / (maxc + EPS)
    blur = cv2.GaussianBlur(luma, (0, 0), 3.0)
    high = luma - blur
    sobel_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    lap = cv2.Laplacian(luma, cv2.CV_32F, ksize=3)
    low8 = cv2.resize(
        cv2.resize(luma, (max(1, luma.shape[1] // 8), max(1, luma.shape[0] // 8)), interpolation=cv2.INTER_AREA),
        (luma.shape[1], luma.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    density = 1.0 - dark

    features = {
        "height": float(image.shape[0]),
        "width": float(image.shape[1]),
        "aspect": float(image.shape[1] / (image.shape[0] + EPS)),
        "rgb_mean_r": float(np.mean(r)),
        "rgb_mean_g": float(np.mean(g)),
        "rgb_mean_b": float(np.mean(b)),
        "rgb_std_r": float(np.std(r)),
        "rgb_std_g": float(np.std(g)),
        "rgb_std_b": float(np.std(b)),
        "rg_mean_delta": float(np.mean(r - g)),
        "gb_mean_delta": float(np.mean(g - b)),
        "rb_mean_delta": float(np.mean(r - b)),
        "lap_var": float(np.var(lap)),
        "high_abs_mean": float(np.mean(np.abs(high))),
        "high_std": float(np.std(high)),
        "low8_std": float(np.std(low8)),
        "low8_luma_delta_abs_mean": float(np.mean(np.abs(luma - low8))),
        "density_dark_gap_mean": float(np.mean(density - (1.0 - luma))),
        "luma_dark_gap_mean": float(np.mean(luma - dark)),
        "edge_density_corr": corr(edge.reshape(-1), density.reshape(-1)),
        "low_luma_density_corr": corr(low8.reshape(-1), density.reshape(-1)),
    }
    for name, arr in (
        ("luma", luma),
        ("dark", dark),
        ("saturation", saturation),
        ("density", density),
        ("edge", edge),
        ("high", high),
        ("low8", low8),
    ):
        features.update(stats(name, arr))
    add_wavelet_stats(features, "luma_haar", luma, levels=2)
    add_wavelet_stats(features, "dark_haar", dark, levels=2)
    add_wavelet_stats(features, "density_haar", density, levels=2)
    return features


def corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= EPS:
        return 0.0
    return float(np.sum(a * b) / denom)


def enrich_rows(rows, hazy_dir, max_images=0):
    hazy_dir = Path(hazy_dir)
    out = []
    selected = rows[:max_images] if max_images > 0 else rows
    total = len(selected)
    for idx, row in enumerate(selected, 1):
        filename = str(row["filename"])
        features = image_features(hazy_dir / filename)
        item = dict(row)
        item.update(features)
        item["best_alt_psnr"] = max(
            safe_float(item, "cr_psnr"),
            safe_float(item, "rescalib_psnr"),
            safe_float(item, "crplusv2_psnr"),
            safe_float(item, "lfcr_v1_w005_psnr"),
        )
        item["best_alt_label"] = best_alt_label(item)
        item["best_alt_gain_vs_lfv1"] = (
            float(item["best_alt_psnr"]) - safe_float(item, "lfv1_psnr")
        )
        if idx % 50 == 0 or idx == total:
            print("extracted {}/{}".format(idx, total), flush=True)
        out.append(item)
    return out


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(checkpoint_path, use_lf_prior, device):
    from model import DEANet

    model = DEANet(
        base_dim=32,
        use_lf_prior=use_lf_prior,
        lf_prior_channels=8,
        lf_prior_pool=8,
        lf_prior_gate_init=0.0,
        lf_prior_injection="pre_mix",
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def pad_tensor(x, patch_size):
    if patch_size <= 0:
        return x
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")


def image_tensor(path, device):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError("Could not read image {}".format(path))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def torch_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def torch_tensor_stats(prefix, tensor):
    x = tensor.detach().float()
    channel_mean = x.mean(dim=(2, 3))
    spatial_std = x.flatten(2).std(dim=2, unbiased=False)
    return {
        prefix + "_mean": torch_scalar(x.mean()),
        prefix + "_std": torch_scalar(x.std(unbiased=False)),
        prefix + "_abs_mean": torch_scalar(x.abs().mean()),
        prefix + "_rms": torch_scalar(torch.sqrt(torch.mean(x * x) + EPS)),
        prefix + "_min": torch_scalar(x.min()),
        prefix + "_max": torch_scalar(x.max()),
        prefix + "_channel_mean_std": torch_scalar(channel_mean.std(dim=1, unbiased=False).mean()),
        prefix + "_spatial_std_mean": torch_scalar(spatial_std.mean()),
        prefix + "_active_frac_001": torch_scalar((x.abs() > 0.01).float().mean()),
    }


class ActivationCollector:
    def __init__(self, model, prefix):
        self.prefix = prefix
        self.values = {}
        self.handles = []
        for name in (
            "down3",
            "level3_block4",
            "level3_block8",
            "mix1",
            "mix2",
        ):
            module = getattr(model, name, None)
            if module is not None:
                self.handles.append(module.register_forward_hook(self._hook(name)))
        lf_prior = getattr(model, "lf_prior", None)
        if lf_prior is not None:
            self.handles.append(lf_prior.register_forward_hook(self._hook("lf_prior")))
            adapter = getattr(lf_prior, "adapter", None)
            if adapter is not None:
                self.handles.append(adapter.register_forward_hook(self._hook("lf_adapter")))

    def _hook(self, name):
        def handle(_module, _inputs, output):
            if torch.is_tensor(output):
                self.values[name] = output.detach()
        return handle

    def clear(self):
        self.values = {}

    def row(self):
        out = {}
        for name, value in self.values.items():
            out.update(torch_tensor_stats("act_{}_{}".format(self.prefix, name), value))
        return out

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def add_activation_features(rows, hazy_dir, args):
    if not args.baseline_checkpoint and not args.lfv1_checkpoint:
        return rows
    if torch is None:
        raise ImportError("torch is required for activation feature extraction")
    device = args.activation_device or ("cuda" if torch.cuda.is_available() else "cpu")
    specs = []
    if args.baseline_checkpoint:
        specs.append(("cr", args.baseline_checkpoint, False))
    if args.lfv1_checkpoint:
        specs.append(("lfv1", args.lfv1_checkpoint, True))
    models = [(label, load_model(path, use_lf, device)) for label, path, use_lf in specs]
    collectors = [(label, model, ActivationCollector(model, label)) for label, model in models]
    hazy_dir = Path(hazy_dir)
    total = len(rows)
    with torch.no_grad():
        for idx, row in enumerate(rows, 1):
            x = pad_tensor(image_tensor(hazy_dir / str(row["filename"]), device), args.activation_pad_size)
            for _label, model, collector in collectors:
                collector.clear()
                _ = model(x)
                row.update(collector.row())
            if idx % 25 == 0 or idx == total:
                print("activation {}/{}".format(idx, total), flush=True)
    for _label, _model, collector in collectors:
        collector.close()
    return rows


def best_alt_label(row):
    pairs = (
        ("cr", safe_float(row, "cr_psnr")),
        ("rescalib", safe_float(row, "rescalib_psnr")),
        ("crplusv2", safe_float(row, "crplusv2_psnr")),
        ("lfcr_v1_w005", safe_float(row, "lfcr_v1_w005_psnr")),
    )
    return max(pairs, key=lambda item: item[1])[0]


def feature_sets(rows):
    excluded = {
        "filename",
        "image_id",
        "airlight_bin",
        "beta_bin",
        "airlight_beta_bin",
        "winner_by_psnr",
        "winner_label",
        "cr_strength_bin",
        "best_alt_label",
    }
    unsafe_or_target_prefixes = (
        "input_",
        "cr_",
        "lfv1_",
        "rescalib_",
        "crplusv2_",
        "lfcr_",
        "oracle",
        "best_alt",
    )
    all_numeric = []
    for key in sorted(rows[0].keys()):
        if key in excluded or key in ("airlight", "beta"):
            continue
        if key.startswith(unsafe_or_target_prefixes):
            continue
        values = [row.get(key, "") for row in rows]
        if sum(1 for value in values if is_number(value)) >= max(5, len(rows) // 10):
            all_numeric.append(key)
    basic_tokens = (
        "height",
        "width",
        "aspect",
        "rgb_",
        "luma_",
        "dark_",
        "saturation_",
        "edge_",
        "lap_",
        "high_",
        "low8_",
        "density_dark",
        "luma_dark",
    )
    activation = [key for key in all_numeric if key.startswith("act_")]
    non_activation = [key for key in all_numeric if not key.startswith("act_")]
    basic = [
        key for key in non_activation
        if not ("haar" in key) and key.startswith(basic_tokens)
    ]
    wavelet = non_activation
    metadata = ["airlight", "beta"]
    sets = {
        "hazy_basic": sorted(set(basic)),
        "hazy_wavelet": sorted(set(wavelet)),
        "metadata_diagnostic": metadata,
        "hazy_wavelet_plus_metadata": sorted(set(wavelet + metadata)),
    }
    if activation:
        sets.update({
            "activation_only": sorted(set(activation)),
            "hazy_wavelet_plus_activation": sorted(set(wavelet + activation)),
            "hazy_wavelet_activation_plus_metadata": sorted(set(wavelet + activation + metadata)),
        })
    return sets


def build_tasks(rows, positive_margin, negative_margin):
    tasks = {}
    extreme = []
    for row in rows:
        delta = safe_float(row, "lfv1_delta_cr_psnr")
        if delta >= positive_margin:
            item = dict(row)
            item["_target"] = 1
            item["_task_name"] = "extreme_preserve_vs_intervene"
            extreme.append(item)
        elif delta <= negative_margin:
            item = dict(row)
            item["_target"] = 0
            item["_task_name"] = "extreme_preserve_vs_intervene"
            extreme.append(item)
    tasks["extreme_preserve_vs_intervene"] = extreme

    oracle = []
    for row in rows:
        item = dict(row)
        item["_target"] = 1 if safe_float(item, "lfv1_psnr") >= float(item["best_alt_psnr"]) else 0
        item["_task_name"] = "oracle_lfv1_vs_best_alt"
        oracle.append(item)
    tasks["oracle_lfv1_vs_best_alt"] = oracle
    return tasks


def standardize(train_rows, valid_rows, features):
    x_train = matrix(train_rows, features)
    x_valid = matrix(valid_rows, features)
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma < EPS] = 1.0
    return (x_train - mu) / sigma, (x_valid - mu) / sigma


def matrix(rows, features):
    data = []
    for row in rows:
        values = []
        for key in features:
            value = row.get(key, 0.0)
            values.append(float(value) if is_number(value) else 0.0)
        data.append(values)
    return np.asarray(data, dtype=np.float32)


def targets(rows):
    return np.asarray([int(row["_target"]) for row in rows], dtype=np.float32)


def fit_logistic(x, y, steps, lr, l2):
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    pos = max(1.0, float(np.sum(y == 1)))
    neg = max(1.0, float(np.sum(y == 0)))
    sample_weights = np.where(y == 1, (pos + neg) / (2.0 * pos), (pos + neg) / (2.0 * neg))
    for _ in range(steps):
        logits = np.clip(x.dot(weights) + bias, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        errors = (probs - y) * sample_weights
        grad_w = x.T.dot(errors) / len(y) + l2 * weights
        grad_b = float(np.mean(errors))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def predict_proba(x, weights, bias):
    logits = np.clip(x.dot(weights) + bias, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate_rows(rows, probs):
    pred = (probs >= 0.5).astype(np.int32)
    y = np.asarray([int(row["_target"]) for row in rows], dtype=np.int32)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    pos_recall = tp / (tp + fn + EPS)
    neg_recall = tn / (tn + fp + EPS)
    selected = []
    oracle = []
    lfv1 = []
    preserve_gain_flags = []
    intervene_regress_flags = []
    for row, pred_value in zip(rows, pred):
        lfv1_psnr = safe_float(row, "lfv1_psnr")
        alt_psnr = float(row["best_alt_psnr"])
        selected.append(lfv1_psnr if pred_value == 1 else alt_psnr)
        oracle.append(max(lfv1_psnr, alt_psnr))
        lfv1.append(lfv1_psnr)
        if safe_float(row, "lfv1_delta_cr_psnr") >= 0.30:
            preserve_gain_flags.append(1 if pred_value == 1 else 0)
        if safe_float(row, "lfv1_delta_cr_psnr") <= -0.30:
            intervene_regress_flags.append(1 if pred_value == 0 else 0)
    selected_gain = mean(selected) - mean(lfv1)
    oracle_gain = mean(oracle) - mean(lfv1)
    return {
        "n": len(rows),
        "positive_n": int(np.sum(y == 1)),
        "negative_n": int(np.sum(y == 0)),
        "accuracy": float((tp + tn) / (len(rows) + EPS)),
        "balanced_accuracy": float((pos_recall + neg_recall) * 0.5),
        "preserve_precision": float(tp / (tp + fp + EPS)),
        "preserve_recall": float(pos_recall),
        "intervene_precision": float(tn / (tn + fn + EPS)),
        "intervene_recall": float(neg_recall),
        "selected_mean_psnr": mean(selected),
        "lfv1_mean_psnr": mean(lfv1),
        "oracle_mean_psnr": mean(oracle),
        "gain_vs_lfv1": float(selected_gain),
        "oracle_gain_vs_lfv1": float(oracle_gain),
        "oracle_recovery": float(selected_gain / (oracle_gain + EPS)) if oracle_gain > EPS else 0.0,
        "lfv1_gain_preserve_recall": mean(preserve_gain_flags),
        "lfv1_regression_intervene_recall": mean(intervene_regress_flags),
    }


def split_random(rows, valid_fraction, rng):
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    valid_n = max(1, int(round(len(rows) * valid_fraction)))
    valid_idx = set(idx[:valid_n])
    train = [row for i, row in enumerate(rows) if i not in valid_idx]
    valid = [row for i, row in enumerate(rows) if i in valid_idx]
    return train, valid


def split_group(rows, key, value):
    valid = [row for row in rows if row.get(key) == value]
    train = [row for row in rows if row.get(key) != value]
    return train, valid


def valid_binary_split(train, valid):
    if len(train) < 20 or len(valid) < 10:
        return False
    return len({row["_target"] for row in train}) == 2 and len({row["_target"] for row in valid}) == 2


def run_one(train, valid, features, args):
    x_train, x_valid = standardize(train, valid, features)
    y_train = targets(train)
    weights, bias = fit_logistic(
        x_train,
        y_train,
        steps=args.logistic_steps,
        lr=args.logistic_lr,
        l2=args.logistic_l2,
    )
    probs = predict_proba(x_valid, weights, bias)
    return evaluate_rows(valid, probs)


def run_audit(tasks, feature_sets_by_name, args):
    rng = random.Random(args.seed)
    results = []
    for task_name, rows in tasks.items():
        for feature_set_name, features in feature_sets_by_name.items():
            if not features:
                continue
            for split_idx in range(args.splits):
                train, valid = split_random(rows, args.valid_fraction, rng)
                if not valid_binary_split(train, valid):
                    continue
                metrics = run_one(train, valid, features, args)
                metrics.update({
                    "task": task_name,
                    "feature_set": feature_set_name,
                    "split_family": "random",
                    "split": "random_{}".format(split_idx + 1),
                    "feature_count": len(features),
                })
                results.append(metrics)
            for group_key in ("airlight_bin", "beta_bin"):
                values = sorted({row.get(group_key, "") for row in rows if row.get(group_key, "") != ""})
                for value in values:
                    train, valid = split_group(rows, group_key, value)
                    if not valid_binary_split(train, valid):
                        continue
                    metrics = run_one(train, valid, features, args)
                    metrics.update({
                        "task": task_name,
                        "feature_set": feature_set_name,
                        "split_family": group_key,
                        "split": "{}={}".format(group_key, value),
                        "feature_count": len(features),
                    })
                    results.append(metrics)
    return results


def summarize(results, args):
    grouped = {}
    for row in results:
        key = (row["task"], row["feature_set"], row["split_family"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (task, feature_set, split_family), items in sorted(grouped.items()):
        summary = {
            "task": task,
            "feature_set": feature_set,
            "split_family": split_family,
            "splits": len(items),
        }
        for metric in (
            "gain_vs_lfv1",
            "oracle_recovery",
            "balanced_accuracy",
            "preserve_precision",
            "preserve_recall",
            "intervene_precision",
            "intervene_recall",
            "lfv1_gain_preserve_recall",
            "lfv1_regression_intervene_recall",
        ):
            values = [item[metric] for item in items if is_number(item.get(metric, ""))]
            summary[metric + "_mean"] = mean(values)
            summary[metric + "_std"] = std(values)
        summary["passes_preflight_line"] = passes(summary, args)
        summaries.append(summary)
    return summaries


def passes(summary, args):
    if summary["feature_set"] not in ("hazy_wavelet", "hazy_wavelet_plus_activation"):
        return False
    if summary["task"] != "extreme_preserve_vs_intervene":
        return False
    if summary["split_family"] != "random":
        return False
    return (
        summary.get("gain_vs_lfv1_mean", 0) >= args.min_gain
        and summary.get("oracle_recovery_mean", 0) >= args.min_oracle_recovery
        and summary.get("intervene_precision_mean", 0) >= args.min_precision
        and summary.get("lfv1_gain_preserve_recall_mean", 0) >= args.min_preserve_recall
        and summary.get("balanced_accuracy_mean", 0) >= args.min_balanced_accuracy
    )


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = sorted(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, summary, task_counts, recommendation):
    lines = [
        "# HAZE4K Wavelet Preserve Proxy Audit",
        "",
        "## Recommendation",
        "",
        "- `{}`".format(recommendation),
        "",
        "## Task Counts",
        "",
        "| Task | Samples | Preserve | Intervene |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task, counts in task_counts.items():
        lines.append(
            "| {} | {} | {} | {} |".format(
                task, counts["n"], counts["positive_n"], counts["negative_n"]
            )
        )
    lines.extend([
        "",
        "## Summary",
        "",
        "| Task | Feature Set | Split Family | Gain vs LF-v1 | Recovery | Balanced Acc | Preserve Recall | Intervene Precision | Pass |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in summary:
        lines.append(
            "| {task} | {feature_set} | {split_family} | {gain:.4f} | {recovery:.4f} | {bal:.4f} | {recall:.4f} | {iprec:.4f} | {passed} |".format(
                task=row["task"],
                feature_set=row["feature_set"],
                split_family=row["split_family"],
                gain=float(row.get("gain_vs_lfv1_mean") or 0.0),
                recovery=float(row.get("oracle_recovery_mean") or 0.0),
                bal=float(row.get("balanced_accuracy_mean") or 0.0),
                recall=float(row.get("lfv1_gain_preserve_recall_mean") or 0.0),
                iprec=float(row.get("intervene_precision_mean") or 0.0),
                passed="yes" if row["passes_preflight_line"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.matrix_csv)
    rows = enrich_rows(rows, args.hazy_dir, max_images=args.max_images)
    rows = add_activation_features(rows, args.hazy_dir, args)
    sets = feature_sets(rows)
    tasks = build_tasks(rows, args.positive_margin, args.negative_margin)
    results = run_audit(tasks, sets, args)
    summary = summarize(results, args)
    task_counts = {
        name: {
            "n": len(task_rows),
            "positive_n": sum(1 for row in task_rows if row["_target"] == 1),
            "negative_n": sum(1 for row in task_rows if row["_target"] == 0),
        }
        for name, task_rows in tasks.items()
    }
    pass_rows = [row for row in summary if row["passes_preflight_line"]]
    recommendation = (
        "proceed_to_wavelet_preserve_route_card"
        if pass_rows else
        "do_not_train_wavelet_preserve_yet"
    )

    feature_fieldnames = sorted(rows[0].keys())
    write_csv(output_dir / "wavelet_preserve_features.csv", rows, feature_fieldnames)
    write_csv(output_dir / "split_results.csv", results)
    write_csv(output_dir / "summary.csv", summary)
    payload = {
        "recommendation": recommendation,
        "args": vars(args),
        "task_counts": task_counts,
        "feature_counts": {name: len(values) for name, values in sets.items()},
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / "analysis_report.md", summary, task_counts, recommendation)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
