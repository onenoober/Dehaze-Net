import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze_brf_representation_audit import (
    EPS,
    add_image_features,
    add_meta_bins,
    add_model_feature_block,
    image_tensor,
    infer_with_features,
    lowpass,
    parse_haze4k_name,
    pearson,
    psnr_tensor,
    scalar,
)
from analyze_strong_cr_abstention_residual_snr_audit import (
    add_feature_contrast,
    infer_plain,
    load_cbrfrc,
    load_deanet_variant,
    residual_cosine,
    sign_flip_rate,
    tensor_norm,
    try_ssim,
)
from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs


CANDIDATES = ("warm", "cold_cr", "lfv1", "residualcalib", "crplus", "cbrfrc")
FEATURE_CANDIDATES = ("warm", "cold_cr", "lfv1", "residualcalib", "cbrfrc")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Official-centric marginal gain audit: join official step0, "
            "official warm-start, and cold-start candidate outputs; then test "
            "whether official failure cases are deployably separable."
        )
    )
    parser.add_argument("--dataset_root", type=str, default="../dataset/HAZE4K")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--hazy_dir", type=str, default="")
    parser.add_argument("--clear_dir", type=str, default="")
    parser.add_argument("--official_step0_checkpoint", type=str, required=True)
    parser.add_argument("--official_warm_checkpoint", type=str, required=True)
    parser.add_argument("--cold_cr_checkpoint", type=str, required=True)
    parser.add_argument("--lfv1_checkpoint", type=str, required=True)
    parser.add_argument("--residualcalib_checkpoint", type=str, required=True)
    parser.add_argument("--crplus_checkpoint", type=str, default="")
    parser.add_argument("--cbrfrc_checkpoint", type=str, required=True)
    parser.add_argument("--route_evidence_csv", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_size", type=int, default=4)
    parser.add_argument("--spatial_feature_grid", type=int, default=4)
    parser.add_argument("--feature_pool_grid", type=int, default=2)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--target_margins", type=str, default="0.15,0.20")
    parser.add_argument("--nochange_margin", type=float, default=0.05)
    parser.add_argument("--random_splits", type=int, default=3)
    parser.add_argument("--valid_fraction", type=float, default=0.25)
    parser.add_argument("--content_clusters", type=int, default=5)
    parser.add_argument("--heads", type=str, default="logistic,ridge_classifier,hgb")
    parser.add_argument(
        "--feature_sets",
        type=str,
        default=(
            "A_official_output,A_output_plus_meta,B_candidate_output_deltas,"
            "C_official_internal,D_internal_contrast,E_all_deployable,"
            "E_all_deployable_shuffled,Z_diagnostic_leakage"
        ),
    )
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--logistic_c", type=float, default=0.5)
    parser.add_argument("--hgb_max_iter", type=int, default=160)
    parser.add_argument("--hgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--hgb_l2", type=float, default=0.01)
    parser.add_argument("--bootstrap_rounds", type=int, default=300)
    parser.add_argument("--min_oracle_gain", type=float, default=0.05)
    parser.add_argument("--min_intervention_precision", type=float, default=0.75)
    parser.add_argument("--max_strong_nochange_false_intervention", type=float, default=0.05)
    parser.add_argument("--max_nochange_false_intervention", type=float, default=0.10)
    parser.add_argument("--min_bootstrap_gain_p05", type=float, default=0.0)
    parser.add_argument("--min_mean_gain", type=float, default=0.02)
    parser.add_argument("--min_shuffled_precision_gap", type=float, default=0.05)
    return parser.parse_args()


def parse_float(value, default=""):
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_route_evidence(path):
    if not path:
        return {}
    evidence_path = Path(path)
    if not evidence_path.is_file():
        return {}
    with evidence_path.open("r", encoding="utf-8", newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f) if row.get("filename")}


def candidate_metrics(row, name, out, clear, official):
    psnr = psnr_tensor(out, clear)
    ssim = try_ssim(out, clear)
    row[name + "_psnr"] = psnr
    row[name + "_ssim"] = ssim
    row[name + "_delta_official_psnr"] = psnr - row["official_psnr"]
    row[name + "_delta_official_ssim"] = ssim - row["official_ssim"] if ssim != "" and row["official_ssim"] != "" else ""
    official_lp8 = lowpass(official, 8)
    out_lp8 = lowpass(out, 8)
    row[name + "_residual_norm_lf8"] = tensor_norm(out_lp8 - official_lp8)
    row[name + "_residual_cosine_to_target_lf8"] = residual_cosine(out_lp8 - official_lp8, lowpass(clear, 8) - official_lp8)


def add_route_crplus_metrics(row, evidence):
    psnr = parse_float(evidence.get("crplusv2_psnr"))
    ssim = parse_float(evidence.get("crplusv2_ssim"))
    row["crplus_psnr"] = psnr
    row["crplus_ssim"] = ssim
    row["crplus_delta_official_psnr"] = psnr - row["official_psnr"] if psnr != "" else ""
    row["crplus_delta_official_ssim"] = ssim - row["official_ssim"] if ssim != "" and row["official_ssim"] != "" else ""
    row["crplus_metric_source"] = "route_evidence_csv"


def add_output_delta_features(row, name, out, official, grid):
    prefix = "delta_{}_official".format(name)
    add_image_features(row, prefix, out - official, grid)
    row[prefix + "_lf8_norm"] = tensor_norm(lowpass(out, 8) - lowpass(official, 8))


def add_named_feature_contrast(row, prefix, first_feats, second_feats, pool_grid):
    before = set(row.keys())
    add_feature_contrast(row, first_feats, second_feats, pool_grid)
    for key in list(row.keys()):
        if key in before or not key.startswith("contrast_"):
            continue
        row[prefix + key[len("contrast"):]] = row.pop(key)


def add_quantile_bins(rows, source_key, dest_key, labels):
    values = np.asarray([row[source_key] for row in rows], dtype=np.float64)
    q1, q2, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    for row in rows:
        value = row[source_key]
        if value < q1:
            row[dest_key] = labels[0]
        elif value < q2:
            row[dest_key] = labels[1]
        elif value < q3:
            row[dest_key] = labels[2]
        else:
            row[dest_key] = labels[3]


def add_content_clusters(rows, args):
    from sklearn.cluster import KMeans

    keys = [
        key for key in numeric_feature_keys(rows, include_labels=False)
        if key.startswith(("input_", "official_", "input_minus_official_"))
        and key not in ("official_psnr", "official_ssim")
    ]
    if not keys or args.content_clusters <= 1:
        for row in rows:
            row["content_cluster"] = "content_all"
        return
    x = matrix_from_keys(rows, list(range(len(rows))), keys)
    x = standardize_single(x)
    clusters = KMeans(n_clusters=args.content_clusters, random_state=args.seed, n_init=20).fit_predict(x)
    for row, cluster in zip(rows, clusters):
        row["content_cluster"] = "content_k{}_c{}".format(args.content_clusters, int(cluster))


def add_oracle_labels(rows, args):
    for row in rows:
        present = [name for name in CANDIDATES if row.get(name + "_psnr", "") != ""]
        if not present:
            row["best_candidate"] = ""
            row["best_candidate_psnr"] = ""
            row["best_candidate_gain_vs_official"] = ""
            row["oracle_with_nochange_gain_vs_official"] = 0.0
            continue
        best = max(present, key=lambda name: row[name + "_psnr"])
        gain = row[best + "_psnr"] - row["official_psnr"]
        row["best_candidate"] = best
        row["best_candidate_psnr"] = row[best + "_psnr"]
        row["best_candidate_gain_vs_official"] = gain
        row["oracle_with_nochange_gain_vs_official"] = max(0.0, gain)
        row["label_nochange"] = int(gain <= args.nochange_margin)
        row["label_strong_official_nochange"] = int(
            row["official_strength_bin"] == "strong_official_q4" and gain <= args.nochange_margin
        )
        row["label_residual_low_energy"] = int(row["residual_energy_bin"] == "residual_low_q1")
        for margin in target_margins(args):
            suffix = margin_suffix(margin)
            intervene = gain >= margin
            preserve = gain <= args.nochange_margin or row["label_strong_official_nochange"]
            ignore = not intervene and not preserve
            row["label_intervene_" + suffix] = int(intervene)
            row["label_preserve_" + suffix] = int(preserve and not intervene)
            row["label_ignore_" + suffix] = int(ignore)
            row["label_class_" + suffix] = 1 if intervene else (0 if preserve else -1)


def target_margins(args):
    return [float(item.strip()) for item in args.target_margins.split(",") if item.strip()]


def margin_suffix(margin):
    return "m{:03d}".format(int(round(float(margin) * 100)))


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

    official_model, official_step = load_deanet_variant(args.official_step0_checkpoint, "lfv1", args.device)
    warm_model, warm_step = load_deanet_variant(args.official_warm_checkpoint, "lfv1", args.device)
    cold_cr_model, cold_cr_step = load_deanet_variant(args.cold_cr_checkpoint, "cr", args.device)
    lfv1_model, lfv1_step = load_deanet_variant(args.lfv1_checkpoint, "lfv1", args.device)
    residualcalib_model, residualcalib_step = load_deanet_variant(args.residualcalib_checkpoint, "residualcalib", args.device)
    crplus_model = None
    crplus_step = ""
    if args.crplus_checkpoint and Path(args.crplus_checkpoint).is_file():
        crplus_model, crplus_step = load_deanet_variant(args.crplus_checkpoint, "crplus", args.device)
    cbrfrc_model, cbrfrc_step = load_cbrfrc(args.cbrfrc_checkpoint, args.device)
    route_evidence = load_route_evidence(args.route_evidence_csv)

    rows = []
    total = len(image_names)
    with torch.no_grad():
        for idx, filename in enumerate(image_names, 1):
            hazy = image_tensor(hazy_dir / filename, args.device)
            clear = image_tensor(find_clear_image(clear_dir, filename), args.device)

            official_out, official_feats = infer_with_features(official_model, hazy, args.pad_size)
            warm_out, warm_feats = infer_with_features(warm_model, hazy, args.pad_size)
            cold_cr_out, cold_cr_feats = infer_with_features(cold_cr_model, hazy, args.pad_size)
            lfv1_out, lfv1_feats = infer_with_features(lfv1_model, hazy, args.pad_size)
            residualcalib_out, residualcalib_feats = infer_with_features(residualcalib_model, hazy, args.pad_size)
            cbrfrc_out = infer_plain(cbrfrc_model, hazy, args.pad_size)

            row = {}
            add_image_features(row, "input", hazy, args.spatial_feature_grid)
            add_image_features(row, "official", official_out, args.spatial_feature_grid)
            add_image_features(row, "input_minus_official", hazy - official_out, args.spatial_feature_grid)
            add_model_feature_block(row, "official_feat", official_feats, args.feature_pool_grid)
            for name, out in (
                ("warm", warm_out),
                ("cold_cr", cold_cr_out),
                ("lfv1", lfv1_out),
                ("residualcalib", residualcalib_out),
                ("cbrfrc", cbrfrc_out),
            ):
                add_output_delta_features(row, name, out, official_out, args.spatial_feature_grid)
            for name, feats in (
                ("warm", warm_feats),
                ("cold_cr", cold_cr_feats),
                ("lfv1", lfv1_feats),
                ("residualcalib", residualcalib_feats),
            ):
                add_named_feature_contrast(row, "contrast_{}_official".format(name), official_feats, feats, args.feature_pool_grid)

            row.update(parse_haze4k_name(filename))
            row["filename"] = filename
            row["official_psnr"] = psnr_tensor(official_out, clear)
            row["official_ssim"] = try_ssim(official_out, clear)
            row["diag_target_residual_norm_lf8"] = tensor_norm(lowpass(clear, 8) - lowpass(official_out, 8))
            row["diag_lfv1_target_residual_cosine_lf8"] = residual_cosine(
                lowpass(lfv1_out, 8) - lowpass(official_out, 8),
                lowpass(clear, 8) - lowpass(official_out, 8),
            )
            row["diag_lfv1_target_sign_flip_rate"] = sign_flip_rate(
                lowpass(lfv1_out, 8) - lowpass(official_out, 8),
                lowpass(clear, 8) - lowpass(official_out, 8),
            )

            candidate_metrics(row, "warm", warm_out, clear, official_out)
            candidate_metrics(row, "cold_cr", cold_cr_out, clear, official_out)
            candidate_metrics(row, "lfv1", lfv1_out, clear, official_out)
            candidate_metrics(row, "residualcalib", residualcalib_out, clear, official_out)
            candidate_metrics(row, "cbrfrc", cbrfrc_out, clear, official_out)
            if crplus_model is not None:
                crplus_out = infer_plain(crplus_model, hazy, args.pad_size)
                candidate_metrics(row, "crplus", crplus_out, clear, official_out)
                add_output_delta_features(row, "crplus", crplus_out, official_out, args.spatial_feature_grid)
            elif filename in route_evidence:
                add_route_crplus_metrics(row, route_evidence[filename])

            residual_norms = [
                row.get(name + "_residual_norm_lf8", "")
                for name in CANDIDATES
                if row.get(name + "_residual_norm_lf8", "") != ""
            ]
            row["risk_max_candidate_residual_norm_lf8"] = max(residual_norms) if residual_norms else 0.0
            row["risk_mean_candidate_residual_norm_lf8"] = float(np.mean(residual_norms)) if residual_norms else 0.0
            row["risk_warm_lfv1_delta_agreement_lf8"] = residual_cosine(
                lowpass(warm_out, 8) - lowpass(official_out, 8),
                lowpass(lfv1_out, 8) - lowpass(official_out, 8),
            )
            add_meta_bins(row)
            row["airlight_beta_bin"] = "{} | {}".format(row["airlight_bin"], row["beta_bin"])
            rows.append(row)

            if idx % 25 == 0 or idx == total:
                print("official audit rows {}/{}".format(idx, total), flush=True)

    add_quantile_bins(rows, "official_psnr", "official_strength_bin", (
        "weak_official_q1",
        "midlow_official_q2",
        "midhigh_official_q3",
        "strong_official_q4",
    ))
    add_quantile_bins(rows, "cold_cr_psnr", "cr_strength_bin", (
        "weak_cr_q1",
        "midlow_cr_q2",
        "midhigh_cr_q3",
        "strong_cr_q4",
    ))
    add_quantile_bins(rows, "risk_max_candidate_residual_norm_lf8", "residual_energy_bin", (
        "residual_low_q1",
        "residual_midlow_q2",
        "residual_midhigh_q3",
        "residual_high_q4",
    ))
    add_content_clusters(rows, args)
    add_oracle_labels(rows, args)

    meta = {
        "images": len(rows),
        "hazy_dir": str(hazy_dir),
        "clear_dir": str(clear_dir),
        "official_step0_checkpoint": args.official_step0_checkpoint,
        "official_warm_checkpoint": args.official_warm_checkpoint,
        "cold_cr_checkpoint": args.cold_cr_checkpoint,
        "lfv1_checkpoint": args.lfv1_checkpoint,
        "residualcalib_checkpoint": args.residualcalib_checkpoint,
        "crplus_checkpoint": args.crplus_checkpoint,
        "route_evidence_csv": args.route_evidence_csv,
        "cbrfrc_checkpoint": args.cbrfrc_checkpoint,
        "official_step0_step": official_step,
        "official_warm_step": warm_step,
        "cold_cr_step": cold_cr_step,
        "lfv1_step": lfv1_step,
        "residualcalib_step": residualcalib_step,
        "crplus_step": crplus_step,
        "cbrfrc_step": cbrfrc_step,
    }
    return rows, meta


def is_number(value):
    return isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) and math.isfinite(float(value))


def numeric_feature_keys(rows, include_labels=False):
    excluded = {
        "filename",
        "image_id",
        "airlight_bin",
        "beta_bin",
        "airlight_beta_bin",
        "official_strength_bin",
        "cr_strength_bin",
        "content_cluster",
        "residual_energy_bin",
        "best_candidate",
        "crplus_metric_source",
    }
    metric_suffixes = ("_psnr", "_ssim", "_delta_official_psnr", "_delta_official_ssim")
    keys = []
    for key in sorted(rows[0].keys()):
        if key in excluded:
            continue
        if not include_labels and (
            key.startswith(("label_", "diag_"))
            or key in ("best_candidate_psnr", "best_candidate_gain_vs_official", "oracle_with_nochange_gain_vs_official")
            or key.endswith(metric_suffixes)
        ):
            continue
        if is_number(rows[0].get(key)):
            keys.append(key)
    return keys


def build_feature_sets(rows):
    keys = numeric_feature_keys(rows)
    output = [
        key for key in keys
        if key.startswith(("input_", "official_", "input_minus_official_"))
        and key not in ("official_psnr", "official_ssim")
    ]
    meta = [key for key in ("airlight", "beta") if key in keys]
    candidate_delta = [key for key in keys if key.startswith("delta_")]
    official_internal = [key for key in keys if key.startswith("official_feat_")]
    contrast = [key for key in keys if key.startswith("contrast_")]
    risk = [key for key in keys if key.startswith("risk_")]
    deployable = output + candidate_delta + official_internal + contrast + risk
    shuffled = output + ["shuf_" + key for key in candidate_delta + official_internal + contrast + risk]
    diagnostic = numeric_feature_keys(rows, include_labels=True)
    return {
        "A_official_output": output,
        "A_output_plus_meta": output + meta,
        "B_candidate_output_deltas": output + candidate_delta + risk,
        "C_official_internal": output + official_internal,
        "D_internal_contrast": output + contrast + risk,
        "E_all_deployable": deployable,
        "E_all_deployable_shuffled": shuffled,
        "Z_diagnostic_leakage": diagnostic,
    }


def add_shuffled_feature_blocks(rows, args):
    rng = random.Random(args.seed + 917)
    keys = [
        key for key in numeric_feature_keys(rows)
        if key.startswith(("delta_", "official_feat_", "contrast_", "risk_"))
    ]
    order = list(range(len(rows)))
    rng.shuffle(order)
    source = [{key: rows[idx][key] for key in keys} for idx in order]
    for row, values in zip(rows, source):
        for key, value in values.items():
            row["shuf_" + key] = value


def matrix_from_keys(rows, indices, features):
    x = np.zeros((len(indices), len(features)), dtype=np.float32)
    for row_pos, idx in enumerate(indices):
        row = rows[idx]
        for col, key in enumerate(features):
            value = row.get(key, 0.0)
            if value == "" or value is None:
                value = 0.0
            x[row_pos, col] = float(value)
    return x


def standardize_pair(x_train, x_valid):
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    keep = sigma > 1e-6
    if not np.any(keep):
        keep = np.ones_like(sigma, dtype=bool)
    sigma[sigma < 1e-6] = 1.0
    return (x_train[:, keep] - mu[keep]) / sigma[keep], (x_valid[:, keep] - mu[keep]) / sigma[keep], int(np.sum(keep))


def standardize_single(x):
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return (x - mu) / sigma


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))).astype(np.float32)


def fit_scores(head, x_train, y_train, x_valid, args, seed):
    if head == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=args.logistic_c, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_valid)[:, 1]
    if head == "ridge_classifier":
        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier(alpha=args.ridge_alpha, class_weight="balanced")
        model.fit(x_train, y_train)
        return sigmoid(model.decision_function(x_train)), sigmoid(model.decision_function(x_valid))
    if head == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            l2_regularization=args.hgb_l2,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_valid)[:, 1]
    raise ValueError("Unknown head {}".format(head))


def split_random(n, valid_fraction, rng):
    order = list(range(n))
    rng.shuffle(order)
    valid_n = max(1, int(round(n * valid_fraction)))
    return sorted(order[valid_n:]), sorted(order[:valid_n])


def split_group(rows, key, value):
    valid = [idx for idx, row in enumerate(rows) if row.get(key) == value]
    train = [idx for idx, row in enumerate(rows) if row.get(key) != value]
    return train, valid


def label_key(margin):
    return "label_class_" + margin_suffix(margin)


def valid_split(rows, train, valid, margin):
    key = label_key(margin)
    train_labels = [rows[idx][key] for idx in train if rows[idx][key] >= 0]
    valid_labels = [rows[idx][key] for idx in valid if rows[idx][key] >= 0]
    return len(train_labels) >= 50 and len(valid_labels) >= 25 and len(set(train_labels)) == 2


def bootstrap_p05(values, rounds, seed):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(1, rounds)):
        sample = values[rng.integers(0, len(values), len(values))]
        means.append(float(np.mean(sample)))
    return float(np.percentile(np.asarray(means, dtype=np.float64), 5))


def eval_threshold(rows, indices, scores, threshold, margin, args, seed):
    key = label_key(margin)
    intervene = scores >= threshold
    labels = np.asarray([rows[idx][key] for idx in indices], dtype=np.int32)
    labeled = labels >= 0
    safe = labels == 1
    nochange = np.asarray([bool(rows[idx]["label_nochange"]) for idx in indices], dtype=bool)
    strong_nochange = np.asarray([bool(rows[idx]["label_strong_official_nochange"]) for idx in indices], dtype=bool)
    strong_official = np.asarray([rows[idx]["official_strength_bin"] == "strong_official_q4" for idx in indices], dtype=bool)
    residual_low = np.asarray([bool(rows[idx]["label_residual_low_energy"]) for idx in indices], dtype=bool)
    official_psnr = np.asarray([rows[idx]["official_psnr"] for idx in indices], dtype=np.float64)
    best_psnr = np.asarray([rows[idx]["best_candidate_psnr"] for idx in indices], dtype=np.float64)
    best_gain = np.asarray([rows[idx]["best_candidate_gain_vs_official"] for idx in indices], dtype=np.float64)
    per_image_gain = np.where(intervene, best_psnr - official_psnr, 0.0)
    predicted_labeled = intervene & labeled

    return {
        "eval_n": int(len(indices)),
        "labeled_n": int(np.sum(labeled)),
        "ignored_n": int(np.sum(~labeled)),
        "intervention_rate": float(np.mean(intervene)) if len(indices) else 0.0,
        "intervention_precision": float(np.mean(safe[predicted_labeled])) if np.any(predicted_labeled) else 0.0,
        "nochange_count": int(np.sum(nochange)),
        "nochange_false_intervention_rate": float(np.mean(intervene[nochange])) if np.any(nochange) else 0.0,
        "strong_official_count": int(np.sum(strong_official)),
        "strong_official_intervention_rate": float(np.mean(intervene[strong_official])) if np.any(strong_official) else 0.0,
        "strong_nochange_count": int(np.sum(strong_nochange)),
        "strong_nochange_false_intervention_rate": float(np.mean(intervene[strong_nochange])) if np.any(strong_nochange) else 0.0,
        "residual_low_count": int(np.sum(residual_low)),
        "residual_low_false_intervention_rate": float(np.mean(intervene[residual_low])) if np.any(residual_low) else 0.0,
        "mean_oracle_gated_gain": float(np.mean(per_image_gain)),
        "bootstrap_gain_p05": bootstrap_p05(per_image_gain, args.bootstrap_rounds, seed),
        "official_psnr": float(np.mean(official_psnr)),
        "sim_psnr": float(np.mean(official_psnr + per_image_gain)),
        "sim_delta_vs_official": float(np.mean(per_image_gain)),
        "oracle_recovery": float(np.mean(per_image_gain) / (np.mean(np.maximum(best_gain, 0.0)) + EPS)),
        "confidence_corr": pearson(scores.tolist(), best_gain.tolist()),
    }


def choose_threshold(rows, indices, scores, margin, args):
    best = None
    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = eval_threshold(rows, indices, scores, float(threshold), margin, args, args.seed + 19)
        feasible = (
            metrics["intervention_precision"] >= args.min_intervention_precision
            and metrics["strong_nochange_false_intervention_rate"] <= args.max_strong_nochange_false_intervention
            and metrics["nochange_false_intervention_rate"] <= args.max_nochange_false_intervention
        )
        key = (
            1 if feasible else 0,
            metrics["mean_oracle_gated_gain"],
            metrics["intervention_precision"],
            -metrics["strong_nochange_false_intervention_rate"],
            metrics["intervention_rate"],
        )
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def run_one(rows, train_idx, valid_idx, feature_set, features, head, margin, args, seed):
    key = label_key(margin)
    train_labeled = [idx for idx in train_idx if rows[idx][key] >= 0]
    x_train = matrix_from_keys(rows, train_labeled, features)
    x_valid = matrix_from_keys(rows, valid_idx, features)
    x_train, x_valid, active_features = standardize_pair(x_train, x_valid)
    y_train = np.asarray([rows[idx][key] for idx in train_labeled], dtype=np.int32)
    train_scores, valid_scores = fit_scores(head, x_train, y_train, x_valid, args, seed)
    threshold = choose_threshold(rows, train_labeled, train_scores, margin, args)
    metrics = eval_threshold(rows, valid_idx, valid_scores, threshold, margin, args, seed + 101)
    metrics.update({
        "feature_set": feature_set,
        "head": head,
        "target_margin": margin,
        "threshold": threshold,
        "feature_count": len(features),
        "active_feature_count": active_features,
        "train_labeled_n": len(train_labeled),
        "eligible_for_pass": int(feature_set not in ("Z_diagnostic_leakage",) and not feature_set.endswith("_shuffled")),
    })
    return metrics


def run_audit(rows, feature_sets, args):
    requested_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    requested_heads = [item.strip() for item in args.heads.split(",") if item.strip()]
    rng = random.Random(args.seed)
    split_specs = []
    for split_idx in range(args.random_splits):
        train, valid = split_random(len(rows), args.valid_fraction, rng)
        split_specs.append(("random", "random_{}".format(split_idx + 1), train, valid))
    for key in (
        "official_strength_bin",
        "cr_strength_bin",
        "airlight_bin",
        "beta_bin",
        "content_cluster",
        "residual_energy_bin",
    ):
        for value in sorted({row.get(key, "") for row in rows if row.get(key, "")}):
            train, valid = split_group(rows, key, value)
            split_specs.append((key, "{}={}".format(key, value), train, valid))

    results = []
    for margin in target_margins(args):
        for feature_set in requested_sets:
            features = feature_sets.get(feature_set, [])
            if not features:
                continue
            for head in requested_heads:
                for split_num, (split_family, split_name, train, valid) in enumerate(split_specs):
                    if not valid_split(rows, train, valid, margin):
                        continue
                    print("{} {} margin {:.2f} {}".format(feature_set, head, margin, split_name), flush=True)
                    metrics = run_one(
                        rows,
                        train,
                        valid,
                        feature_set,
                        features,
                        head,
                        margin,
                        args,
                        args.seed + split_num + int(round(margin * 1000)),
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
    values = [float(v) for v in values if v != ""]
    return float(np.mean(values)) if values else 0.0


def std(values):
    values = [float(v) for v in values if v != ""]
    return float(np.std(values)) if len(values) >= 2 else 0.0


def summarize_results(results, args):
    grouped = {}
    for row in results:
        split_group = "random" if row["split_family"] == "random" else row["split"]
        key = (row["target_margin"], row["feature_set"], row["head"], row["split_family"], split_group)
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (margin, feature_set, head, split_family, split_group), items in sorted(grouped.items()):
        summary = {
            "target_margin": margin,
            "feature_set": feature_set,
            "head": head,
            "split_family": split_family,
            "split_group": split_group,
            "splits": len(items),
            "eligible_for_pass": items[0]["eligible_for_pass"],
        }
        for metric in (
            "intervention_precision",
            "nochange_false_intervention_rate",
            "strong_nochange_false_intervention_rate",
            "residual_low_false_intervention_rate",
            "mean_oracle_gated_gain",
            "bootstrap_gain_p05",
            "oracle_recovery",
            "confidence_corr",
            "intervention_rate",
            "labeled_n",
            "ignored_n",
            "sim_delta_vs_official",
        ):
            vals = [item[metric] for item in items]
            summary[metric + "_mean"] = mean(vals)
            summary[metric + "_std"] = std(vals)
        summaries.append(summary)

    by_key = {(row["target_margin"], row["head"], row["split_group"]): row for row in summaries if row["feature_set"] == "E_all_deployable_shuffled"}
    for row in summaries:
        control = by_key.get((row["target_margin"], row["head"], row["split_group"]), {})
        row["precision_gap_vs_shuffled"] = (
            row["intervention_precision_mean"] - control.get("intervention_precision_mean", row["intervention_precision_mean"])
        )
        row["passes_line"] = passes_line(row, args)
    return summaries


def passes_line(row, args):
    if not row["eligible_for_pass"]:
        return False
    return (
        row["intervention_precision_mean"] >= args.min_intervention_precision
        and row["strong_nochange_false_intervention_rate_mean"] <= args.max_strong_nochange_false_intervention
        and row["nochange_false_intervention_rate_mean"] <= args.max_nochange_false_intervention
        and row["bootstrap_gain_p05_mean"] > args.min_bootstrap_gain_p05
        and row["mean_oracle_gated_gain_mean"] >= args.min_mean_gain
        and row["precision_gap_vs_shuffled"] >= args.min_shuffled_precision_gap
    )


def oracle_summary(rows, args):
    summary = {
        "num_images": len(rows),
        "mean_official_psnr": mean([row["official_psnr"] for row in rows]),
        "mean_oracle_with_nochange_psnr": mean([
            row["official_psnr"] + row["oracle_with_nochange_gain_vs_official"] for row in rows
        ]),
        "mean_oracle_with_nochange_gain_vs_official": mean([
            row["oracle_with_nochange_gain_vs_official"] for row in rows
        ]),
        "mean_best_candidate_gain_vs_official": mean([row["best_candidate_gain_vs_official"] for row in rows]),
        "median_best_candidate_gain_vs_official": float(np.median([
            row["best_candidate_gain_vs_official"] for row in rows
        ])),
        "nochange_margin": args.nochange_margin,
        "nochange_count": int(sum(row["label_nochange"] for row in rows)),
        "strong_official_nochange_count": int(sum(row["label_strong_official_nochange"] for row in rows)),
        "best_candidate_counts": {
            name: int(sum(row["best_candidate"] == name for row in rows)) for name in CANDIDATES
        },
    }
    for name in ("warm", "cold_cr", "lfv1", "residualcalib", "crplus", "cbrfrc"):
        vals = [row[name + "_psnr"] for row in rows if row.get(name + "_psnr", "") != ""]
        summary["mean_{}_psnr".format(name)] = mean(vals)
        summary["mean_{}_gain_vs_official".format(name)] = mean([
            row[name + "_delta_official_psnr"] for row in rows if row.get(name + "_delta_official_psnr", "") != ""
        ])
    gains = np.asarray([row["best_candidate_gain_vs_official"] for row in rows], dtype=np.float64)
    clipped = np.maximum(gains, 0.0)
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.30):
        summary["candidate_gain_ge_{:.2f}_count".format(threshold)] = int(np.sum(gains >= threshold))
    for pct in (10, 25, 50, 75, 90, 95):
        summary["best_gain_p{:02d}".format(pct)] = float(np.percentile(gains, pct))
        summary["oracle_nochange_gain_p{:02d}".format(pct)] = float(np.percentile(clipped, pct))
    groups = {}
    for key in ("official_strength_bin", "cr_strength_bin", "airlight_bin", "beta_bin", "content_cluster", "residual_energy_bin"):
        for value in sorted({row[key] for row in rows}):
            subset = [row for row in rows if row[key] == value]
            groups["{}={}".format(key, value)] = {
                "n": len(subset),
                "mean_oracle_gain": mean([row["oracle_with_nochange_gain_vs_official"] for row in subset]),
                "mean_best_gain": mean([row["best_candidate_gain_vs_official"] for row in subset]),
                "positive_ge_015": int(sum(row["best_candidate_gain_vs_official"] >= 0.15 for row in subset)),
                "nochange_count": int(sum(row["label_nochange"] for row in subset)),
            }
    summary["groups"] = groups
    summary["top_gain_cases"] = [
        {
            "filename": row["filename"],
            "best_candidate": row["best_candidate"],
            "official_psnr": row["official_psnr"],
            "best_candidate_psnr": row["best_candidate_psnr"],
            "gain": row["best_candidate_gain_vs_official"],
        }
        for row in sorted(rows, key=lambda item: item["best_candidate_gain_vs_official"], reverse=True)[:20]
    ]
    return summary


def final_recommendation(oracle, summaries):
    if oracle["mean_oracle_with_nochange_gain_vs_official"] < 0.05:
        return "stop_official_headroom_too_small"
    required = {
        "official_strength_bin",
        "cr_strength_bin",
        "airlight_bin",
        "beta_bin",
        "content_cluster",
        "residual_energy_bin",
    }
    random_pass = [row for row in summaries if row["split_family"] == "random" and row["passes_line"]]
    for row in random_pass:
        matching = [
            item for item in summaries
            if item["target_margin"] == row["target_margin"]
            and item["feature_set"] == row["feature_set"]
            and item["head"] == row["head"]
            and item["split_family"] in required
        ]
        families = {item["split_family"] for item in matching}
        if families == required and all(item["passes_line"] for item in matching):
            return "stage0_passed_write_tiny_official_adapter_card"
    return "do_not_train_official_adapter_yet"


def compact_rows(rows, args):
    keys = [
        "filename",
        "image_id",
        "airlight",
        "beta",
        "airlight_bin",
        "beta_bin",
        "airlight_beta_bin",
        "official_strength_bin",
        "cr_strength_bin",
        "content_cluster",
        "residual_energy_bin",
        "official_psnr",
        "official_ssim",
        "best_candidate",
        "best_candidate_psnr",
        "best_candidate_gain_vs_official",
        "oracle_with_nochange_gain_vs_official",
        "label_nochange",
        "label_strong_official_nochange",
        "label_residual_low_energy",
    ]
    for name in CANDIDATES:
        keys.extend([name + "_psnr", name + "_ssim", name + "_delta_official_psnr"])
    for margin in target_margins(args):
        suffix = margin_suffix(margin)
        keys.extend(["label_intervene_" + suffix, "label_preserve_" + suffix, "label_ignore_" + suffix, "label_class_" + suffix])
    return [{key: row.get(key, "") for key in keys} for row in rows]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, oracle, summaries, recommendation, meta, args):
    lines = [
        "# HAZE4K Official-Centric Marginal Gain Audit",
        "",
        "## Recommendation",
        "",
        "- `{}`".format(recommendation),
        "",
        "## Oracle Over Official",
        "",
        "- Images: `{}`".format(oracle["num_images"]),
        "- Mean official PSNR: `{:.4f}`".format(oracle["mean_official_psnr"]),
        "- Mean oracle-with-no-change PSNR / gain: `{:.4f}` / `+{:.4f}`".format(
            oracle["mean_oracle_with_nochange_psnr"],
            oracle["mean_oracle_with_nochange_gain_vs_official"],
        ),
        "- Median best-candidate gain: `{:.4f}`".format(oracle["median_best_candidate_gain_vs_official"]),
        "- Candidate gain counts >= 0.15 / 0.20 dB: `{}` / `{}`".format(
            oracle["candidate_gain_ge_0.15_count"],
            oracle["candidate_gain_ge_0.20_count"],
        ),
        "- No-change / strong-official no-change counts: `{}` / `{}`".format(
            oracle["nochange_count"],
            oracle["strong_official_nochange_count"],
        ),
        "- Best-candidate counts: `{}`".format(oracle["best_candidate_counts"]),
        "",
        "## Source",
        "",
        "- Official step0 checkpoint: `{}` step `{}`".format(meta["official_step0_checkpoint"], meta["official_step0_step"]),
        "- Official warm checkpoint: `{}` step `{}`".format(meta["official_warm_checkpoint"], meta["official_warm_step"]),
        "- Cold CR checkpoint: `{}` step `{}`".format(meta["cold_cr_checkpoint"], meta["cold_cr_step"]),
        "- LF-v1 checkpoint: `{}` step `{}`".format(meta["lfv1_checkpoint"], meta["lfv1_step"]),
        "- ResidualCalib checkpoint: `{}` step `{}`".format(meta["residualcalib_checkpoint"], meta["residualcalib_step"]),
        "- CRPlus checkpoint/source: `{}` / `{}`".format(meta["crplus_checkpoint"], meta["route_evidence_csv"]),
        "- CBRFRC checkpoint: `{}` step `{}`".format(meta["cbrfrc_checkpoint"], meta["cbrfrc_step"]),
        "",
        "## Probe Summary",
        "",
        "| Margin | Feature Set | Head | Split | Precision | Strong No-change False Int | No-change False Int | Low-energy False Int | Mean Gain | Bootstrap p05 | Recovery | Shuffle Gap | Pass |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ordered = sorted(
        summaries,
        key=lambda row: (
            -int(row["passes_line"]),
            row["target_margin"],
            row["feature_set"],
            row["head"],
            row["split_group"],
        ),
    )
    for row in ordered:
        lines.append(
            "| {margin:.2f} | {feature_set} | {head} | {split} | {precision:.4f} | {strong_false:.4f} | {false:.4f} | {low_false:.4f} | {gain:.4f} | {p05:.4f} | {recovery:.4f} | {gap:.4f} | {passed} |".format(
                margin=row["target_margin"],
                feature_set=row["feature_set"],
                head=row["head"],
                split=row["split_group"],
                precision=row["intervention_precision_mean"],
                strong_false=row["strong_nochange_false_intervention_rate_mean"],
                false=row["nochange_false_intervention_rate_mean"],
                low_false=row["residual_low_false_intervention_rate_mean"],
                gain=row["mean_oracle_gated_gain_mean"],
                p05=row["bootstrap_gain_p05_mean"],
                recovery=row["oracle_recovery_mean"],
                gap=row["precision_gap_vs_shuffled"],
                passed="yes" if row["passes_line"] else "no",
            )
        )
    lines.extend([
        "",
        "## Pass Line",
        "",
        "- intervention precision >= `{:.2f}`".format(args.min_intervention_precision),
        "- strong-official no-change false intervention <= `{:.2f}`".format(args.max_strong_nochange_false_intervention),
        "- all no-change false intervention <= `{:.2f}`".format(args.max_nochange_false_intervention),
        "- bootstrap gain p05 > `{:.2f}` and mean gain >= `{:.2f}`".format(
            args.min_bootstrap_gain_p05,
            args.min_mean_gain,
        ),
        "- shuffled precision gap >= `{:.2f}`".format(args.min_shuffled_precision_gap),
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, meta = collect_rows(args)
    add_shuffled_feature_blocks(rows, args)
    feature_sets = build_feature_sets(rows)
    results = run_audit(rows, feature_sets, args)
    summaries = summarize_results(results, args)
    oracle = oracle_summary(rows, args)
    recommendation = final_recommendation(oracle, summaries)

    write_csv(output_dir / "official_candidate_matrix.csv", compact_rows(rows, args))
    write_csv(output_dir / "split_results.csv", results)
    write_csv(output_dir / "summary.csv", summaries)
    payload = {
        "recommendation": recommendation,
        "args": vars(args),
        "meta": meta,
        "oracle": oracle,
        "feature_counts": {key: len(value) for key, value in feature_sets.items()},
        "label_counts": {
            margin_suffix(margin): {
                "intervene": int(sum(row["label_intervene_" + margin_suffix(margin)] for row in rows)),
                "preserve": int(sum(row["label_preserve_" + margin_suffix(margin)] for row in rows)),
                "ignore": int(sum(row["label_ignore_" + margin_suffix(margin)] for row in rows)),
            }
            for margin in target_margins(args)
        },
        "summary": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "oracle_summary.json").write_text(json.dumps(oracle, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir / "analysis_report.md", oracle, summaries, recommendation, meta, args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
