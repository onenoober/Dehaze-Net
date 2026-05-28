import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from analyze_wavelet_preserve_proxy import (
    EPS,
    add_wavelet_stats,
    corr,
    fit_logistic,
    is_number,
    mean,
    parse_value,
    predict_proba,
    stats,
    std,
    write_csv,
)
from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs
from model import DEANet


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build train-split teacher labels from CR and LF-v1 checkpoints, "
            "then audit whether a small preservation head can learn when to "
            "preserve LF-v1 versus fall back to CR."
        )
    )
    parser.add_argument("--dataset_root", type=str, default="../dataset/HAZE4K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hazy_dir", type=str, default="")
    parser.add_argument("--clear_dir", type=str, default="")
    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--lfv1_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--features_csv",
        type=str,
        default="",
        help="Reuse an existing supervised_preserve_features.csv and skip teacher forward.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--head_device", type=str, default="cpu")
    parser.add_argument("--pad_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--patches_per_image", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--valid_fraction", type=float, default=0.25)
    parser.add_argument("--positive_margin", type=float, default=0.30)
    parser.add_argument("--negative_margin", type=float, default=-0.30)
    parser.add_argument("--logistic_steps", type=int, default=500)
    parser.add_argument("--logistic_lr", type=float, default=0.08)
    parser.add_argument("--logistic_l2", type=float, default=0.05)
    parser.add_argument("--mlp_epochs", type=int, default=80)
    parser.add_argument("--mlp_hidden", type=int, default=64)
    parser.add_argument("--mlp_batch_size", type=int, default=512)
    parser.add_argument("--mlp_lr", type=float, default=0.003)
    parser.add_argument("--mlp_weight_decay", type=float, default=0.01)
    parser.add_argument("--mlp_dropout", type=float, default=0.10)
    parser.add_argument(
        "--feature_sets",
        type=str,
        default="hazy_wavelet,teacher_output_proxy,hazy_wavelet_plus_teacher_outputs",
        help="Comma-separated feature sets to audit, or 'all'.",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="logistic,mlp",
        help="Comma-separated heads to audit: logistic, sklearn_logistic, mlp, or a combination.",
    )
    parser.add_argument(
        "--heldout_groups",
        type=str,
        default="airlight_bin,beta_bin",
        help="Comma-separated metadata groups for held-out splits.",
    )
    parser.add_argument("--main_feature_set", type=str, default="hazy_wavelet_plus_teacher_outputs")
    parser.add_argument("--main_head", type=str, default="logistic")
    parser.add_argument("--sklearn_max_iter", type=int, default=1000)
    parser.add_argument("--sklearn_c", type=float, default=1.0)
    parser.add_argument(
        "--sklearn_solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
    )
    parser.add_argument("--min_gain", type=float, default=0.05)
    parser.add_argument("--min_oracle_recovery", type=float, default=0.20)
    parser.add_argument("--min_intervene_precision", type=float, default=0.62)
    parser.add_argument("--min_preserve_recall", type=float, default=0.65)
    parser.add_argument("--min_balanced_accuracy", type=float, default=0.62)
    parser.add_argument("--min_strong_cr_regression_recall", type=float, default=0.60)
    parser.add_argument("--min_heldout_intervene_precision", type=float, default=0.56)
    parser.add_argument("--min_heldout_preserve_recall", type=float, default=0.56)
    parser.add_argument("--min_heldout_balanced_accuracy", type=float, default=0.56)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint


def pad_img(x, patch_size):
    if patch_size <= 0:
        return x
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")


def infer_one(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    pred = model(pad_img(hazy, pad_size)).clamp(0, 1)
    return pred[:, :, :h, :w]


def psnr_np(pred, gt):
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
    gt = np.clip(gt, 0.0, 1.0).astype(np.float32)
    rmse = math.sqrt(float(np.mean((pred - gt) ** 2)))
    if rmse <= EPS:
        return 100.0
    return 20.0 * math.log10(1.0 / rmse)


def tensor_to_image(tensor):
    array = tensor.detach().float().cpu().numpy()[0]
    return np.transpose(array, (1, 2, 0))


def parse_haze4k_name(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    meta = {
        "image_id": parts[0] if parts else stem,
        "airlight": "",
        "beta": "",
    }
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
    row["airlight_beta_bin"] = row["airlight_bin"] + " | " + row["beta_bin"]


def image_stats_features(image, prefix, include_wavelet=True):
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
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
        cv2.resize(
            luma,
            (max(1, luma.shape[1] // 8), max(1, luma.shape[0] // 8)),
            interpolation=cv2.INTER_AREA,
        ),
        (luma.shape[1], luma.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    density = 1.0 - dark
    features = {
        prefix + "_rgb_mean_r": float(np.mean(r)),
        prefix + "_rgb_mean_g": float(np.mean(g)),
        prefix + "_rgb_mean_b": float(np.mean(b)),
        prefix + "_rgb_std_r": float(np.std(r)),
        prefix + "_rgb_std_g": float(np.std(g)),
        prefix + "_rgb_std_b": float(np.std(b)),
        prefix + "_rg_mean_delta": float(np.mean(r - g)),
        prefix + "_gb_mean_delta": float(np.mean(g - b)),
        prefix + "_rb_mean_delta": float(np.mean(r - b)),
        prefix + "_lap_var": float(np.var(lap)),
        prefix + "_high_abs_mean": float(np.mean(np.abs(high))),
        prefix + "_high_std": float(np.std(high)),
        prefix + "_low8_std": float(np.std(low8)),
        prefix + "_low8_luma_delta_abs_mean": float(np.mean(np.abs(luma - low8))),
        prefix + "_density_dark_gap_mean": float(np.mean(density - (1.0 - luma))),
        prefix + "_luma_dark_gap_mean": float(np.mean(luma - dark)),
        prefix + "_edge_density_corr": corr(edge.reshape(-1), density.reshape(-1)),
        prefix + "_low_luma_density_corr": corr(low8.reshape(-1), density.reshape(-1)),
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
        features.update(stats(prefix + "_" + name, arr))
    if include_wavelet:
        add_wavelet_stats(features, prefix + "_luma_haar", luma, levels=2)
        add_wavelet_stats(features, prefix + "_dark_haar", dark, levels=2)
        add_wavelet_stats(features, prefix + "_density_haar", density, levels=2)
    return features


def residual_features(array, prefix):
    array = array.astype(np.float32)
    luma = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    channel_mean = np.mean(array, axis=(0, 1))
    channel_std = np.std(array, axis=(0, 1))
    features = {
        prefix + "_mean_r": float(channel_mean[0]),
        prefix + "_mean_g": float(channel_mean[1]),
        prefix + "_mean_b": float(channel_mean[2]),
        prefix + "_std_r": float(channel_std[0]),
        prefix + "_std_g": float(channel_std[1]),
        prefix + "_std_b": float(channel_std[2]),
        prefix + "_abs_mean": float(np.mean(np.abs(array))),
        prefix + "_rms": float(np.sqrt(np.mean(array * array) + EPS)),
        prefix + "_luma_abs_mean": float(np.mean(np.abs(luma))),
        prefix + "_luma_rms": float(np.sqrt(np.mean(luma * luma) + EPS)),
        prefix + "_luma_p10": float(np.percentile(luma.reshape(-1), 10)),
        prefix + "_luma_p50": float(np.percentile(luma.reshape(-1), 50)),
        prefix + "_luma_p90": float(np.percentile(luma.reshape(-1), 90)),
    }
    add_wavelet_stats(features, prefix + "_luma_haar", luma, levels=2)
    return features


def patch_coords(height, width, patch_size, count, rng):
    ph = min(patch_size, height)
    pw = min(patch_size, width)
    max_y = max(0, height - ph)
    max_x = max(0, width - pw)
    coords = []
    center = (max_y // 2, max_x // 2, ph, pw)
    coords.append(center)
    attempts = 0
    while len(coords) < count and attempts < count * 20:
        attempts += 1
        y = rng.randint(0, max_y) if max_y > 0 else 0
        x = rng.randint(0, max_x) if max_x > 0 else 0
        coord = (y, x, ph, pw)
        if coord not in coords:
            coords.append(coord)
    return coords[:count]


def crop_np(image, coord):
    y, x, h, w = coord
    return image[y:y + h, x:x + w, :]


def resolve_dirs(args):
    if args.hazy_dir and args.clear_dir:
        return args.hazy_dir, args.clear_dir
    split_root = Path(args.dataset_root) / args.split
    return resolve_pair_dirs(split_root)


def read_feature_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{key: parse_value(value) for key, value in raw.items()} for raw in reader]
    if not rows:
        raise ValueError("No feature rows found in {}".format(path))
    return rows


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
    rows = []
    rng = random.Random(args.seed)

    with torch.no_grad():
        for idx, name in enumerate(names, 1):
            hazy_path = Path(hazy_dir) / name
            clear_path = Path(find_clear_image(clear_dir, name))
            hazy_img = Image.open(hazy_path).convert("RGB")
            clear_img = Image.open(clear_path).convert("RGB")
            hazy = to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = to_tensor(clear_img).unsqueeze(0).to(args.device)

            baseline = infer_one(baseline_model, hazy, args.pad_size)
            lfv1 = infer_one(lfv1_model, hazy, args.pad_size)
            hazy_np = tensor_to_image(hazy)
            clear_np = tensor_to_image(clear)
            baseline_np = tensor_to_image(baseline)
            lfv1_np = tensor_to_image(lfv1)
            meta = parse_haze4k_name(name)
            image_cr_psnr = psnr_np(baseline_np, clear_np)
            image_lfv1_psnr = psnr_np(lfv1_np, clear_np)
            height, width = hazy_np.shape[:2]

            for patch_idx, coord in enumerate(
                patch_coords(height, width, args.patch_size, args.patches_per_image, rng)
            ):
                y, x, ph, pw = coord
                hazy_patch = crop_np(hazy_np, coord)
                clear_patch = crop_np(clear_np, coord)
                cr_patch = crop_np(baseline_np, coord)
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
                row.update(image_stats_features(hazy_patch, "hazy", include_wavelet=True))
                row.update(image_stats_features(cr_patch, "cr_out", include_wavelet=False))
                row.update(image_stats_features(lfv1_patch, "lfv1_out", include_wavelet=False))
                row.update(residual_features(lfv1_patch - cr_patch, "lfv1_minus_cr"))
                row.update(residual_features(cr_patch - hazy_patch, "cr_minus_hazy"))
                row.update(residual_features(lfv1_patch - hazy_patch, "lfv1_minus_hazy"))
                rows.append(row)

            if idx % 25 == 0 or idx == len(names):
                print("teacher labeled {}/{} images, {} patches".format(idx, len(names), len(rows)), flush=True)

    meta_payload = {
        "baseline_checkpoint_step": baseline_ckpt.get("step"),
        "lfv1_checkpoint_step": lfv1_ckpt.get("step"),
        "baseline_checkpoint_max_psnr": baseline_ckpt.get("max_psnr"),
        "lfv1_checkpoint_max_psnr": lfv1_ckpt.get("max_psnr"),
        "hazy_dir": str(hazy_dir),
        "clear_dir": str(clear_dir),
        "images": len(names),
        "patches": len(rows),
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


def build_tasks(rows, positive_margin, negative_margin):
    decisive = []
    for row in rows:
        delta = float(row["lfv1_delta_cr_psnr"])
        if delta >= positive_margin:
            item = dict(row)
            item["_target"] = 1
            item["_task_name"] = "patch_preserve_vs_intervene"
            decisive.append(item)
        elif delta <= negative_margin:
            item = dict(row)
            item["_target"] = 0
            item["_task_name"] = "patch_preserve_vs_intervene"
            decisive.append(item)
    return {"patch_preserve_vs_intervene": decisive}


def numeric_feature_names(rows, predicate):
    names = []
    excluded = {
        "filename",
        "patch_id",
        "image_id",
        "airlight_bin",
        "beta_bin",
        "airlight_beta_bin",
        "cr_strength_bin",
        "_target",
        "_task_name",
    }
    target_prefixes = ("image_cr_psnr", "image_lfv1_psnr", "image_delta_psnr", "cr_psnr", "lfv1_psnr", "lfv1_delta_cr_psnr")
    for key in sorted(rows[0].keys()):
        if key in excluded or key.startswith(target_prefixes):
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


def targets(rows):
    return np.asarray([int(row["_target"]) for row in rows], dtype=np.float32)


def standardize(train_rows, valid_rows, features):
    x_train = matrix(train_rows, features)
    x_valid = matrix(valid_rows, features)
    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma < EPS] = 1.0
    return (x_train - mu) / sigma, (x_valid - mu) / sigma


class SmallHead(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super(SmallHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_predict_mlp(x_train, y_train, x_valid, args, seed):
    device = torch.device(args.head_device)
    torch.manual_seed(seed)
    model = SmallHead(x_train.shape[1], args.mlp_hidden, args.mlp_dropout).to(device)
    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    pos = max(1.0, float(torch.sum(y_train_t == 1).item()))
    neg = max(1.0, float(torch.sum(y_train_t == 0).item()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    batch_size = max(16, int(args.mlp_batch_size))
    model.train()
    for _epoch in range(args.mlp_epochs):
        perm = torch.randperm(len(x_train_t), generator=generator)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x_valid).float().to(device))).cpu().numpy()
    return probs.astype(np.float32)


def fit_predict_sklearn_logistic(x_train, y_train, x_valid, args):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for head=sklearn_logistic"
        ) from exc
    model = LogisticRegression(
        C=args.sklearn_c,
        class_weight="balanced",
        max_iter=args.sklearn_max_iter,
        solver=args.sklearn_solver,
    )
    model.fit(x_train, y_train.astype(np.int32))
    return model.predict_proba(x_valid)[:, 1].astype(np.float32)


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
    strong_true_regressions = []
    strong_interventions = []
    strong_false_interventions = []
    for row, pred_value in zip(rows, pred):
        cr_psnr = float(row["cr_psnr"])
        lfv1_psnr = float(row["lfv1_psnr"])
        selected.append(lfv1_psnr if pred_value == 1 else cr_psnr)
        oracle.append(max(lfv1_psnr, cr_psnr))
        lfv1.append(lfv1_psnr)
        is_strong = row.get("cr_strength_bin") == "cr_strongest_25"
        if is_strong and int(row["_target"]) == 0:
            strong_true_regressions.append(1 if pred_value == 0 else 0)
        if is_strong and pred_value == 0:
            strong_interventions.append(1 if int(row["_target"]) == 0 else 0)
        if is_strong and int(row["_target"]) == 1:
            strong_false_interventions.append(1 if pred_value == 0 else 0)

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
        "strong_cr_regression_intervene_recall": mean(strong_true_regressions),
        "strong_cr_intervention_precision": mean(strong_interventions),
        "strong_cr_gain_false_intervene_rate": mean(strong_false_interventions),
    }


def valid_binary_split(train, valid):
    if len(train) < 40 or len(valid) < 20:
        return False
    return len({row["_target"] for row in train}) == 2 and len({row["_target"] for row in valid}) == 2


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


def run_one(train, valid, features, head, args, seed):
    x_train, x_valid = standardize(train, valid, features)
    y_train = targets(train)
    if head == "logistic":
        weights, bias = fit_logistic(
            x_train,
            y_train,
            steps=args.logistic_steps,
            lr=args.logistic_lr,
            l2=args.logistic_l2,
        )
        probs = predict_proba(x_valid, weights, bias)
    elif head == "sklearn_logistic":
        probs = fit_predict_sklearn_logistic(x_train, y_train, x_valid, args)
    elif head == "mlp":
        probs = fit_predict_mlp(x_train, y_train, x_valid, args, seed)
    else:
        raise ValueError("Unsupported head: {}".format(head))
    return evaluate_rows(valid, probs)


def run_audit(tasks, feature_sets_by_name, args):
    rng = random.Random(args.seed)
    results = []
    wanted_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    if wanted_sets != ["all"]:
        feature_sets_by_name = {
            name: values for name, values in feature_sets_by_name.items()
            if name in set(wanted_sets)
        }
    heads = tuple(item.strip() for item in args.heads.split(",") if item.strip())
    for head in heads:
        if head not in ("logistic", "sklearn_logistic", "mlp"):
            raise ValueError("Unsupported head in --heads: {}".format(head))
    heldout_groups = tuple(item.strip() for item in args.heldout_groups.split(",") if item.strip())
    for task_name, rows in tasks.items():
        for feature_set_name, features in feature_sets_by_name.items():
            if not features:
                continue
            for head in heads:
                for split_idx in range(args.splits):
                    train, valid = split_random_by_image(rows, args.valid_fraction, rng)
                    if not valid_binary_split(train, valid):
                        continue
                    print(
                        "audit task={} feature_set={} head={} split=random_image_{}".format(
                            task_name, feature_set_name, head, split_idx + 1
                        ),
                        flush=True,
                    )
                    metrics = run_one(
                        train,
                        valid,
                        features,
                        head,
                        args,
                        seed=args.seed + split_idx + len(results),
                    )
                    metrics.update({
                        "task": task_name,
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
                        if not valid_binary_split(train, valid):
                            continue
                        print(
                            "audit task={} feature_set={} head={} split={}={}".format(
                                task_name, feature_set_name, head, group_key, value
                            ),
                            flush=True,
                        )
                        metrics = run_one(
                            train,
                            valid,
                            features,
                            head,
                            args,
                            seed=args.seed + len(results),
                        )
                        metrics.update({
                            "task": task_name,
                            "feature_set": feature_set_name,
                            "head": head,
                            "split_family": group_key,
                            "split": "{}={}".format(group_key, value),
                            "feature_count": len(features),
                        })
                        results.append(metrics)
    return results


def summarize(results, args):
    grouped = {}
    for row in results:
        key = (row["task"], row["feature_set"], row["head"], row["split_family"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    metrics = (
        "gain_vs_lfv1",
        "oracle_recovery",
        "balanced_accuracy",
        "preserve_precision",
        "preserve_recall",
        "intervene_precision",
        "intervene_recall",
        "strong_cr_regression_intervene_recall",
        "strong_cr_intervention_precision",
        "strong_cr_gain_false_intervene_rate",
    )
    for (task, feature_set, head, split_family), items in sorted(grouped.items()):
        summary = {
            "task": task,
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


def passes_main(summary, args):
    if summary["task"] != "patch_preserve_vs_intervene":
        return False
    if summary["feature_set"] != args.main_feature_set:
        return False
    if summary["head"] != args.main_head:
        return False
    if summary["split_family"] != "random_image":
        return False
    return (
        summary.get("gain_vs_lfv1_mean", 0) >= args.min_gain
        and summary.get("oracle_recovery_mean", 0) >= args.min_oracle_recovery
        and summary.get("balanced_accuracy_mean", 0) >= args.min_balanced_accuracy
        and summary.get("intervene_precision_mean", 0) >= args.min_intervene_precision
        and summary.get("preserve_recall_mean", 0) >= args.min_preserve_recall
        and summary.get("strong_cr_regression_intervene_recall_mean", 0) >= args.min_strong_cr_regression_recall
    )


def add_stability_flags(summaries, args):
    by_key = {}
    for row in summaries:
        key = (row["task"], row["feature_set"], row["head"])
        by_key.setdefault(key, []).append(row)
    for row in summaries:
        row["passes_stability_line"] = False
        if not row["passes_main_line"]:
            continue
        siblings = by_key[(row["task"], row["feature_set"], row["head"])]
        heldout = [
            item for item in siblings
            if item["split_family"] in ("airlight_bin", "beta_bin")
        ]
        if not heldout:
            continue
        min_intervene = min(float(item.get("intervene_precision_mean") or 0.0) for item in heldout)
        min_preserve = min(float(item.get("preserve_recall_mean") or 0.0) for item in heldout)
        min_balanced = min(float(item.get("balanced_accuracy_mean") or 0.0) for item in heldout)
        row["heldout_min_intervene_precision"] = min_intervene
        row["heldout_min_preserve_recall"] = min_preserve
        row["heldout_min_balanced_accuracy"] = min_balanced
        row["passes_stability_line"] = (
            min_intervene >= args.min_heldout_intervene_precision
            and min_preserve >= args.min_heldout_preserve_recall
            and min_balanced >= args.min_heldout_balanced_accuracy
        )


def task_counts(tasks):
    return {
        name: {
            "n": len(rows),
            "positive_n": sum(1 for row in rows if row["_target"] == 1),
            "negative_n": sum(1 for row in rows if row["_target"] == 0),
        }
        for name, rows in tasks.items()
    }


def write_report(path, summary, counts, meta, recommendation):
    lines = [
        "# HAZE4K Supervised Preserve Proxy Audit",
        "",
        "## Recommendation",
        "",
        "- `{}`".format(recommendation),
        "",
        "## Teacher Label Source",
        "",
        "- Images: `{}`".format(meta["images"]),
        "- Patches: `{}`".format(meta["patches"]),
        "- Baseline checkpoint step: `{}`".format(meta.get("baseline_checkpoint_step")),
        "- LF-v1 checkpoint step: `{}`".format(meta.get("lfv1_checkpoint_step")),
        "",
        "## Task Counts",
        "",
        "| Task | Samples | Preserve | Intervene |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task, row in counts.items():
        lines.append("| {} | {} | {} | {} |".format(
            task,
            row["n"],
            row["positive_n"],
            row["negative_n"],
        ))
    lines.extend([
        "",
        "## Summary",
        "",
        "| Task | Feature Set | Head | Split | Gain | Recovery | Bal Acc | Preserve Recall | Intervene Precision | Strong CR Recall | Main | Stable |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in summary:
        lines.append(
            "| {task} | {feature_set} | {head} | {split} | {gain:.4f} | {recovery:.4f} | {bal:.4f} | {preserve:.4f} | {intervene:.4f} | {strong:.4f} | {main} | {stable} |".format(
                task=row["task"],
                feature_set=row["feature_set"],
                head=row["head"],
                split=row["split_family"],
                gain=float(row.get("gain_vs_lfv1_mean") or 0.0),
                recovery=float(row.get("oracle_recovery_mean") or 0.0),
                bal=float(row.get("balanced_accuracy_mean") or 0.0),
                preserve=float(row.get("preserve_recall_mean") or 0.0),
                intervene=float(row.get("intervene_precision_mean") or 0.0),
                strong=float(row.get("strong_cr_regression_intervene_recall_mean") or 0.0),
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
        meta = {
            "baseline_checkpoint_step": "reused_features",
            "lfv1_checkpoint_step": "reused_features",
            "baseline_checkpoint_max_psnr": "",
            "lfv1_checkpoint_max_psnr": "",
            "hazy_dir": "reused_features",
            "clear_dir": "reused_features",
            "images": len({row["filename"] for row in rows}),
            "patches": len(rows),
            "features_csv": args.features_csv,
        }
        print("loaded feature rows: {}".format(args.features_csv), flush=True)
    else:
        rows, meta = collect_rows(args)
    add_strength_bins(rows)
    tasks = build_tasks(rows, args.positive_margin, args.negative_margin)
    feature_sets = build_feature_sets(rows)
    write_csv(output_dir / "supervised_preserve_features.csv", rows)
    print("wrote feature rows before audit: {}".format(output_dir / "supervised_preserve_features.csv"), flush=True)
    results = run_audit(tasks, feature_sets, args)
    summary = summarize(results, args)
    counts = task_counts(tasks)
    pass_rows = [row for row in summary if row.get("passes_main_line") and row.get("passes_stability_line")]
    recommendation = (
        "proceed_to_distilled_preserve_route_card"
        if pass_rows else
        "do_not_train_supervised_preserve_yet"
    )

    write_csv(output_dir / "split_results.csv", results)
    write_csv(output_dir / "summary.csv", summary)
    payload = {
        "recommendation": recommendation,
        "args": vars(args),
        "teacher_meta": meta,
        "task_counts": counts,
        "feature_counts": {name: len(values) for name, values in feature_sets.items()},
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / "analysis_report.md", summary, counts, meta, recommendation)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
