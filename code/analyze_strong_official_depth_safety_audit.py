import argparse
import copy
import csv
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
)
from analyze_depth_guided_lf_preflight import DepthEstimator, robust_normalize_depth, safe_corr
from analyze_official_marginal_gain_audit import (
    CANDIDATES,
    add_content_clusters,
    add_named_feature_contrast,
    add_oracle_labels,
    add_output_delta_features,
    add_quantile_bins,
    add_route_crplus_metrics,
    bootstrap_p05,
    candidate_metrics,
    load_route_evidence,
    margin_suffix,
    mean,
    split_group,
    split_random,
    standardize_pair,
    target_margins,
    write_csv,
)
from analyze_strong_cr_abstention_residual_snr_audit import (
    infer_plain,
    load_cbrfrc,
    load_deanet_variant,
    residual_cosine,
    sign_flip_rate,
    tensor_norm,
    try_ssim,
)
from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs


DEFAULT_DEPTH_MODELS = (
    "depthanything=depth-anything/Depth-Anything-V2-Small-hf,"
    "midas=Intel/dpt-hybrid-midas"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Strong-official/no-change depth safety audit. This is a scoped "
            "probe: it tests whether predicted depth reduces false intervention "
            "on strong-official no-change HAZE4K samples beyond C0/C1 features."
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
    parser.add_argument("--depth_cache_dir", type=str, required=True)
    parser.add_argument("--depth_models", type=str, default=DEFAULT_DEPTH_MODELS)
    parser.add_argument("--candidate_depth_names", type=str, default="warm,cold_cr,lfv1,residualcalib,crplus,cbrfrc")
    parser.add_argument("--model_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_size", type=int, default=4)
    parser.add_argument("--spatial_feature_grid", type=int, default=4)
    parser.add_argument("--feature_pool_grid", type=int, default=2)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--target_margins", type=str, default="0.20")
    parser.add_argument("--nochange_margin", type=float, default=0.05)
    parser.add_argument("--random_splits", type=int, default=3)
    parser.add_argument("--valid_fraction", type=float, default=0.25)
    parser.add_argument("--content_clusters", type=int, default=5)
    parser.add_argument("--heads", type=str, default="logistic,decision_tree,hgb")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--logistic_c", type=float, default=0.5)
    parser.add_argument("--tree_max_depth", type=int, default=3)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--hgb_max_iter", type=int, default=160)
    parser.add_argument("--hgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--hgb_l2", type=float, default=0.01)
    parser.add_argument("--bootstrap_rounds", type=int, default=300)
    parser.add_argument("--threshold_min_precision", type=float, default=0.55)
    parser.add_argument("--threshold_min_gain", type=float, default=0.0)
    parser.add_argument("--threshold_min_intervention_rate", type=float, default=0.02)
    parser.add_argument("--min_relative_strong_fi_reduction", type=float, default=0.30)
    parser.add_argument("--min_relative_nochange_fi_reduction", type=float, default=0.20)
    parser.add_argument("--min_precision_delta", type=float, default=-0.03)
    parser.add_argument("--min_strong_absolute_precision", type=float, default=0.70)
    parser.add_argument("--max_strong_absolute_fi", type=float, default=0.10)
    parser.add_argument("--max_nochange_absolute_fi", type=float, default=0.15)
    parser.add_argument("--label_permutation_rounds", type=int, default=3)
    return parser.parse_args()


def slugify(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def parse_depth_models(value):
    specs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            alias, model_id = item.split("=", 1)
            alias = slugify(alias)
        else:
            model_id = item
            alias = slugify(model_id.split("/")[-1])
        specs.append((alias, model_id))
    if not specs:
        raise ValueError("--depth_models did not contain any model specs")
    return specs


def resolve_dirs(args):
    if args.hazy_dir and args.clear_dir:
        return Path(args.hazy_dir), Path(args.clear_dir)
    split_root = Path(args.dataset_root) / args.split
    hazy_dir, clear_dir = resolve_pair_dirs(split_root)
    return Path(hazy_dir), Path(clear_dir)


def tensor_to_pil(tensor):
    array = tensor.detach().float().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    array = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def tensor_to_hwc(tensor):
    return tensor.detach().float().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)


def depth_cache_path(cache_dir, estimator_alias, split, variant, filename):
    return (
        Path(cache_dir)
        / estimator_alias
        / split
        / slugify(variant)
        / (filename.replace("/", "__") + ".npy")
    )


def load_or_generate_depth(cache_dir, estimator_alias, split, variant, filename, pil_image, estimator):
    path = depth_cache_path(cache_dir, estimator_alias, split, variant, filename)
    if path.is_file():
        return np.load(path).astype(np.float32), False
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = estimator.predict(pil_image)
    np.save(path, depth.astype(np.float32))
    return depth.astype(np.float32), True


def entropy01(values, bins=32):
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0
    hist, _ = np.histogram(np.clip(flat, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    probs = hist.astype(np.float64)
    probs = probs / max(float(np.sum(probs)), EPS)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def gradient_energy(values):
    x = np.asarray(values, dtype=np.float32)
    if min(x.shape[:2]) < 2:
        return 0.0
    gy, gx = np.gradient(x)
    return float(np.mean(gx * gx + gy * gy))


def gradient_magnitude(values):
    x = np.asarray(values, dtype=np.float32)
    if min(x.shape[:2]) < 2:
        return np.zeros_like(x, dtype=np.float32)
    gy, gx = np.gradient(x)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def add_selected_depth_stats(row, prefix, depth):
    values = np.asarray(depth, dtype=np.float32)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, 1.0)
    flat = values.reshape(-1)
    row[prefix + "_mean"] = float(np.mean(flat))
    row[prefix + "_std"] = float(np.std(flat))
    row[prefix + "_p10"] = float(np.percentile(flat, 10.0))
    row[prefix + "_p50"] = float(np.percentile(flat, 50.0))
    row[prefix + "_p90"] = float(np.percentile(flat, 90.0))
    row[prefix + "_gradient_energy"] = gradient_energy(values)
    row[prefix + "_entropy"] = entropy01(values)


def add_depth_pair_features(row, prefix, first, second):
    diff = np.abs(np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32))
    row[prefix + "_abs_mean"] = float(np.mean(diff))
    row[prefix + "_abs_p90"] = float(np.percentile(diff.reshape(-1), 90.0))
    row[prefix + "_corr"] = safe_corr(first, second)
    row[prefix + "_structure_change"] = float(np.mean(np.abs(gradient_magnitude(first) - gradient_magnitude(second))))


def add_candidate_depth_features(row, estimator_alias, candidate_depths, depth_hazy, depth_official):
    official_abs = []
    official_abs_p90 = []
    hazy_abs = []
    structure = []
    gradient_change = []
    for name, depth_candidate in sorted(candidate_depths.items()):
        prefix = "{}_cand_{}".format(estimator_alias, name)
        add_depth_pair_features(row, prefix + "_official", depth_candidate, depth_official)
        add_depth_pair_features(row, prefix + "_hazy", depth_candidate, depth_hazy)
        cand_grad = gradient_energy(depth_candidate)
        off_grad = gradient_energy(depth_official)
        row[prefix + "_gradient_change"] = abs(cand_grad - off_grad)
        official_abs.append(row[prefix + "_official_abs_mean"])
        official_abs_p90.append(row[prefix + "_official_abs_p90"])
        hazy_abs.append(row[prefix + "_hazy_abs_mean"])
        structure.append(row[prefix + "_official_structure_change"])
        gradient_change.append(row[prefix + "_gradient_change"])

    aggregates = {
        "official_abs_mean": official_abs,
        "official_abs_p90": official_abs_p90,
        "hazy_abs_mean": hazy_abs,
        "official_structure_change": structure,
        "gradient_change": gradient_change,
    }
    for key, values in aggregates.items():
        prefix = "{}_candidate_depth_{}".format(estimator_alias, key)
        if values:
            row[prefix + "_min"] = float(np.min(values))
            row[prefix + "_mean"] = float(np.mean(values))
            row[prefix + "_max"] = float(np.max(values))
        else:
            row[prefix + "_min"] = 0.0
            row[prefix + "_mean"] = 0.0
            row[prefix + "_max"] = 0.0


def add_basic_image_stats(row, prefix, tensor):
    image = tensor_to_hwc(tensor)
    luma = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float32)
    rgb_mean = image.reshape(-1, 3).mean(axis=0)
    saturation = image.max(axis=2) - image.min(axis=2)
    dark = image.min(axis=2)
    grad = gradient_magnitude(luma)
    row[prefix + "_brightness"] = float(np.mean(luma))
    row[prefix + "_contrast"] = float(np.std(luma))
    row[prefix + "_saturation_mean"] = float(np.mean(saturation))
    row[prefix + "_saturation_std"] = float(np.std(saturation))
    row[prefix + "_color_cast_rg"] = float(abs(rgb_mean[0] - rgb_mean[1]))
    row[prefix + "_color_cast_rb"] = float(abs(rgb_mean[0] - rgb_mean[2]))
    row[prefix + "_color_cast_gb"] = float(abs(rgb_mean[1] - rgb_mean[2]))
    row[prefix + "_dark_channel_mean"] = float(np.mean(dark))
    row[prefix + "_dark_channel_p10"] = float(np.percentile(dark.reshape(-1), 10.0))
    row[prefix + "_edge_density"] = float(np.mean(grad > 0.05))
    row[prefix + "_entropy"] = entropy01(luma)
    row[prefix + "_low_texture_ratio"] = float(np.mean(grad < 0.01))
    row[prefix + "_gradient_energy"] = float(np.mean(grad * grad))


def add_basic_delta_stats(row, prefix, first, second):
    first_hwc = tensor_to_hwc(first)
    second_hwc = tensor_to_hwc(second)
    diff = first_hwc - second_hwc
    luma_diff = (
        0.299 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.114 * diff[:, :, 2]
    ).astype(np.float32)
    row[prefix + "_luma_abs_mean"] = float(np.mean(np.abs(luma_diff)))
    row[prefix + "_luma_abs_p90"] = float(np.percentile(np.abs(luma_diff).reshape(-1), 90.0))
    row[prefix + "_gradient_change"] = float(
        np.mean(np.abs(gradient_magnitude(first_hwc.mean(axis=2)) - gradient_magnitude(second_hwc.mean(axis=2))))
    )


def collect_rows(args):
    if args.model_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for DEA-Net inference but not available")
    if args.depth_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for depth inference but not available")

    hazy_dir, clear_dir = resolve_dirs(args)
    image_names = list_image_files(hazy_dir)
    if args.max_images > 0:
        image_names = image_names[: args.max_images]

    official_model, official_step = load_deanet_variant(args.official_step0_checkpoint, "lfv1", args.model_device)
    warm_model, warm_step = load_deanet_variant(args.official_warm_checkpoint, "lfv1", args.model_device)
    cold_cr_model, cold_cr_step = load_deanet_variant(args.cold_cr_checkpoint, "cr", args.model_device)
    lfv1_model, lfv1_step = load_deanet_variant(args.lfv1_checkpoint, "lfv1", args.model_device)
    residualcalib_model, residualcalib_step = load_deanet_variant(
        args.residualcalib_checkpoint,
        "residualcalib",
        args.model_device,
    )
    crplus_model = None
    crplus_step = ""
    if args.crplus_checkpoint and Path(args.crplus_checkpoint).is_file():
        crplus_model, crplus_step = load_deanet_variant(args.crplus_checkpoint, "crplus", args.model_device)
    cbrfrc_model, cbrfrc_step = load_cbrfrc(args.cbrfrc_checkpoint, args.model_device)
    route_evidence = load_route_evidence(args.route_evidence_csv)
    candidate_depth_names = {item.strip() for item in args.candidate_depth_names.split(",") if item.strip()}

    depth_specs = parse_depth_models(args.depth_models)
    estimators = []
    for alias, model_id in depth_specs:
        start = time.time()
        estimators.append((alias, model_id, DepthEstimator(model_id, args.depth_device)))
        print("loaded depth estimator {}={} in {:.2f}s".format(alias, model_id, time.time() - start), flush=True)

    rows = []
    depth_generated = {alias: 0 for alias, _ in depth_specs}
    depth_loaded = {alias: 0 for alias, _ in depth_specs}
    depth_seconds = {alias: 0.0 for alias, _ in depth_specs}
    total = len(image_names)

    with torch.no_grad():
        for idx, filename in enumerate(image_names, 1):
            hazy = image_tensor(hazy_dir / filename, args.model_device)
            clear = image_tensor(find_clear_image(clear_dir, filename), args.model_device)

            official_out, official_feats = infer_with_features(official_model, hazy, args.pad_size)
            warm_out, warm_feats = infer_with_features(warm_model, hazy, args.pad_size)
            cold_cr_out, cold_cr_feats = infer_with_features(cold_cr_model, hazy, args.pad_size)
            lfv1_out, lfv1_feats = infer_with_features(lfv1_model, hazy, args.pad_size)
            residualcalib_out, residualcalib_feats = infer_with_features(residualcalib_model, hazy, args.pad_size)
            cbrfrc_out = infer_plain(cbrfrc_model, hazy, args.pad_size)
            crplus_out = None
            if crplus_model is not None:
                crplus_out = infer_plain(crplus_model, hazy, args.pad_size)

            row = {}
            add_image_features(row, "official", official_out, args.spatial_feature_grid)
            add_image_features(row, "input_minus_official", hazy - official_out, args.spatial_feature_grid)
            add_model_feature_block(row, "official_feat", official_feats, args.feature_pool_grid)
            add_basic_image_stats(row, "basic_hazy", hazy)
            add_basic_image_stats(row, "basic_official", official_out)
            add_basic_delta_stats(row, "basic_hazy_official", hazy, official_out)

            candidate_outputs = {
                "warm": warm_out,
                "cold_cr": cold_cr_out,
                "lfv1": lfv1_out,
                "residualcalib": residualcalib_out,
                "cbrfrc": cbrfrc_out,
            }
            if crplus_out is not None:
                candidate_outputs["crplus"] = crplus_out

            for name, out in candidate_outputs.items():
                add_output_delta_features(row, name, out, official_out, args.spatial_feature_grid)
            for name, feats in (
                ("warm", warm_feats),
                ("cold_cr", cold_cr_feats),
                ("lfv1", lfv1_feats),
                ("residualcalib", residualcalib_feats),
            ):
                add_named_feature_contrast(
                    row,
                    "contrast_{}_official".format(name),
                    official_feats,
                    feats,
                    args.feature_pool_grid,
                )

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
            if crplus_out is not None:
                candidate_metrics(row, "crplus", crplus_out, clear, official_out)
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

            pil_inputs = {
                "hazy": tensor_to_pil(hazy),
                "official": tensor_to_pil(official_out),
            }
            for name, out in candidate_outputs.items():
                if name in candidate_depth_names:
                    pil_inputs["candidate_" + name] = tensor_to_pil(out)

            for estimator_alias, model_id, estimator in estimators:
                raw_depths = {}
                for variant, pil_image in pil_inputs.items():
                    start = time.time()
                    raw, generated = load_or_generate_depth(
                        args.depth_cache_dir,
                        estimator_alias,
                        args.split,
                        variant,
                        filename,
                        pil_image,
                        estimator,
                    )
                    depth_seconds[estimator_alias] += time.time() - start
                    depth_generated[estimator_alias] += 1 if generated else 0
                    depth_loaded[estimator_alias] += 0 if generated else 1
                    raw_depths[variant] = robust_normalize_depth(raw)

                depth_hazy = raw_depths["hazy"]
                depth_official = raw_depths["official"]
                add_selected_depth_stats(row, "{}_depth_hazy".format(estimator_alias), depth_hazy)
                add_selected_depth_stats(row, "{}_depth_official".format(estimator_alias), depth_official)
                add_depth_pair_features(
                    row,
                    "{}_depth_hazy_official".format(estimator_alias),
                    depth_hazy,
                    depth_official,
                )
                candidate_depths = {
                    key[len("candidate_"):]: value
                    for key, value in raw_depths.items()
                    if key.startswith("candidate_")
                }
                add_candidate_depth_features(row, estimator_alias, candidate_depths, depth_hazy, depth_official)

            rows.append(row)
            if idx % 10 == 0 or idx == total:
                depth_msg = ", ".join(
                    "{} gen/load {}/{}".format(alias, depth_generated[alias], depth_loaded[alias])
                    for alias, _ in depth_specs
                )
                print("depth safety rows {}/{} ({})".format(idx, total, depth_msg), flush=True)

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
    if len(rows) < max(2, args.content_clusters):
        for row in rows:
            row["content_cluster"] = "content_all"
    else:
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
        "depth_models": [{"alias": alias, "model_id": model_id} for alias, model_id in depth_specs],
        "depth_cache_dir": args.depth_cache_dir,
        "depth_generated": depth_generated,
        "depth_loaded": depth_loaded,
        "depth_seconds": depth_seconds,
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
    min_numeric = min(len(rows), max(5, len(rows) // 10))
    for key in sorted(rows[0].keys()):
        if key in excluded:
            continue
        if not include_labels and (
            key.startswith(("label_", "diag_"))
            or key in (
                "best_candidate_psnr",
                "best_candidate_gain_vs_official",
                "oracle_with_nochange_gain_vs_official",
            )
            or key.endswith(metric_suffixes)
        ):
            continue
        values = [row.get(key, "") for row in rows]
        if sum(1 for value in values if is_number(value)) >= min_numeric:
            keys.append(key)
    return keys


def depth_prefixes(rows):
    aliases = []
    for key in rows[0].keys():
        if key.endswith("_depth_hazy_mean"):
            aliases.append(key[:-len("_depth_hazy_mean")])
    return sorted(set(aliases))


def build_feature_sets(rows):
    keys = numeric_feature_keys(rows)
    official_output = [key for key in keys if key.startswith(("official_", "input_minus_official_"))]
    basic = [key for key in keys if key.startswith("basic_")]
    candidate_delta = [key for key in keys if key.startswith("delta_")]
    official_internal = [key for key in keys if key.startswith("official_feat_")]
    contrast = [key for key in keys if key.startswith("contrast_")]
    risk = [key for key in keys if key.startswith("risk_")]
    c0 = sorted(set(official_output + candidate_delta + official_internal + contrast + risk))
    c1 = sorted(set(c0 + basic))

    feature_sets = {
        "C0_deployable": c0,
        "C1_basic": c1,
    }
    aliases = depth_prefixes(rows)
    all_candidate_free = []
    all_candidate_aware = []
    for alias in aliases:
        candidate_free = [
            key for key in keys
            if key.startswith(
                (
                    alias + "_depth_hazy_",
                    alias + "_depth_official_",
                    alias + "_depth_hazy_official_",
                )
            )
        ]
        candidate_aware = [
            key for key in keys
            if key.startswith((alias + "_cand_", alias + "_candidate_depth_"))
        ]
        all_candidate_free.extend(candidate_free)
        all_candidate_aware.extend(candidate_aware)
        feature_sets["C2_" + alias] = sorted(set(c1 + candidate_free))
        feature_sets["C3_" + alias] = sorted(set(c1 + candidate_free + candidate_aware))
    if len(aliases) > 1:
        feature_sets["C2_both"] = sorted(set(c1 + all_candidate_free))
        feature_sets["C3_both"] = sorted(set(c1 + all_candidate_free + all_candidate_aware))
    return feature_sets


def add_shuffled_depth_features(rows, seed):
    rng = random.Random(seed)
    depth_keys = [
        key for key in numeric_feature_keys(rows)
        if any(
            marker in key
            for marker in (
                "_depth_hazy_",
                "_depth_official_",
                "_depth_hazy_official_",
                "_cand_",
                "_candidate_depth_",
            )
        )
    ]
    order = list(range(len(rows)))
    rng.shuffle(order)
    for row_idx, source_idx in enumerate(order):
        source = rows[source_idx]
        target = rows[row_idx]
        for key in depth_keys:
            target["shuf_" + key] = source.get(key, 0.0)


def add_shuffled_feature_sets(feature_sets):
    additions = {}
    c1 = feature_sets["C1_basic"]
    c1_set = set(c1)
    for name, features in feature_sets.items():
        if not name.startswith(("C2_", "C3_")):
            continue
        depth = [key for key in features if key not in c1_set]
        shuffled_depth = ["shuf_" + key for key in depth]
        additions[name + "_shuffled_depth"] = sorted(set(c1 + shuffled_depth))
    feature_sets.update(additions)


def matrix_from_keys(rows, indices, features):
    x = np.zeros((len(indices), len(features)), dtype=np.float32)
    for row_pos, idx in enumerate(indices):
        row = rows[idx]
        for col, key in enumerate(features):
            value = row.get(key, 0.0)
            if value == "" or value is None or not is_number(value):
                value = 0.0
            x[row_pos, col] = float(value)
    return x


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))).astype(np.float32)


def fit_scores(head, x_train, y_train, x_valid, args, seed):
    if head == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=args.logistic_c, class_weight="balanced", max_iter=2000, random_state=seed)
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_valid)[:, 1]
    if head == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(
            max_depth=args.tree_max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_valid)[:, 1]
    if head == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            l2_regularization=args.hgb_l2,
            max_depth=args.tree_max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_valid)[:, 1]
    raise ValueError("Unknown head {}".format(head))


def label_key(margin):
    return "label_class_" + margin_suffix(margin)


def permuted_label_key(margin):
    return "label_permuted_class_" + margin_suffix(margin)


def valid_split(rows, train, valid, margin):
    key = label_key(margin)
    train_labels = [rows[idx][key] for idx in train if rows[idx][key] >= 0]
    valid_labels = [rows[idx][key] for idx in valid if rows[idx][key] >= 0]
    return len(train_labels) >= 50 and len(valid_labels) >= 25 and len(set(train_labels)) == 2


def eval_threshold(rows, indices, scores, threshold, margin, args, seed, eval_key=None):
    key = eval_key or label_key(margin)
    intervene = scores >= threshold
    labels = np.asarray([rows[idx][key] for idx in indices], dtype=np.int32)
    labeled = labels >= 0
    positive = labels == 1
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
        "intervention_precision": float(np.mean(positive[predicted_labeled])) if np.any(predicted_labeled) else 0.0,
        "intervention_recall": float(np.mean(intervene[positive])) if np.any(positive) else 0.0,
        "nochange_count": int(np.sum(nochange)),
        "nochange_false_intervention_rate": float(np.mean(intervene[nochange])) if np.any(nochange) else 0.0,
        "strong_official_count": int(np.sum(strong_official)),
        "strong_official_intervention_rate": float(np.mean(intervene[strong_official])) if np.any(strong_official) else 0.0,
        "official_preserve_rate": 1.0 - (float(np.mean(intervene[strong_official])) if np.any(strong_official) else 0.0),
        "strong_nochange_count": int(np.sum(strong_nochange)),
        "strong_nochange_false_intervention_rate": float(np.mean(intervene[strong_nochange])) if np.any(strong_nochange) else 0.0,
        "residual_low_count": int(np.sum(residual_low)),
        "residual_low_false_intervention_rate": float(np.mean(intervene[residual_low])) if np.any(residual_low) else 0.0,
        "low_residual_preserve_rate": 1.0 - (float(np.mean(intervene[residual_low])) if np.any(residual_low) else 0.0),
        "simulated_mean_psnr_gain": float(np.mean(per_image_gain)),
        "bootstrap_gain_p05": bootstrap_p05(per_image_gain, args.bootstrap_rounds, seed),
        "official_psnr": float(np.mean(official_psnr)),
        "sim_psnr": float(np.mean(official_psnr + per_image_gain)),
        "oracle_recovery": float(np.mean(per_image_gain) / (np.mean(np.maximum(best_gain, 0.0)) + EPS)),
        "confidence_corr": pearson(scores.tolist(), best_gain.tolist()),
    }


def choose_threshold(rows, indices, scores, margin, args):
    best = None
    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = eval_threshold(rows, indices, scores, float(threshold), margin, args, args.seed + 19)
        feasible = (
            metrics["intervention_precision"] >= args.threshold_min_precision
            and metrics["simulated_mean_psnr_gain"] >= args.threshold_min_gain
            and metrics["intervention_rate"] >= args.threshold_min_intervention_rate
        )
        key = (
            1 if feasible else 0,
            -metrics["strong_nochange_false_intervention_rate"] if feasible else metrics["simulated_mean_psnr_gain"],
            -metrics["nochange_false_intervention_rate"] if feasible else metrics["intervention_precision"],
            metrics["simulated_mean_psnr_gain"] if feasible else metrics["intervention_recall"],
            metrics["intervention_precision"],
            metrics["intervention_recall"],
        )
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def split_specs(rows, args):
    rng = random.Random(args.seed)
    specs = []
    for split_idx in range(args.random_splits):
        train, valid = split_random(len(rows), args.valid_fraction, rng)
        specs.append(("random", "random_{}".format(split_idx + 1), train, valid))
    for family, key, value, name in (
        ("official_strength_heldout", "official_strength_bin", "strong_official_q4", "test_strong_official"),
        ("cr_strength_heldout", "cr_strength_bin", "strong_cr_q4", "test_strong_cr"),
        ("residual_low_energy_heldout", "residual_energy_bin", "residual_low_q1", "test_residual_low"),
    ):
        train, valid = split_group(rows, key, value)
        specs.append((family, name, train, valid))
    return specs


def run_one(rows, train_idx, valid_idx, feature_set, features, head, margin, args, seed, train_key=None, fixed_threshold=None):
    train_key = train_key or label_key(margin)
    eval_key = label_key(margin)
    train_labeled = [idx for idx in train_idx if rows[idx][train_key] >= 0]
    x_train = matrix_from_keys(rows, train_labeled, features)
    x_valid = matrix_from_keys(rows, valid_idx, features)
    x_train, x_valid, active_features = standardize_pair(x_train, x_valid)
    y_train = np.asarray([rows[idx][train_key] for idx in train_labeled], dtype=np.int32)
    train_scores, valid_scores = fit_scores(head, x_train, y_train, x_valid, args, seed)
    threshold = fixed_threshold if fixed_threshold is not None else choose_threshold(rows, train_labeled, train_scores, margin, args)
    metrics = eval_threshold(rows, valid_idx, valid_scores, threshold, margin, args, seed + 101, eval_key=eval_key)
    metrics.update({
        "feature_set": feature_set,
        "head": head,
        "target_margin": margin,
        "threshold": threshold,
        "feature_count": len(features),
        "active_feature_count": active_features,
        "train_labeled_n": len(train_labeled),
    })
    return metrics


def run_audit(rows, feature_sets, args):
    requested_heads = [item.strip() for item in args.heads.split(",") if item.strip()]
    results = []
    for margin in target_margins(args):
        for feature_set, features in feature_sets.items():
            if not features:
                continue
            for head in requested_heads:
                for split_num, (split_family, split_name, train, valid) in enumerate(split_specs(rows, args)):
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
                        "control": "real",
                    })
                    results.append(metrics)
    return results


def add_label_permutation(rows, margin, seed):
    rng = random.Random(seed)
    key = label_key(margin)
    values = [row[key] for row in rows]
    shuffled = values[:]
    rng.shuffle(shuffled)
    perm_key = permuted_label_key(margin)
    for row, value in zip(rows, shuffled):
        row[perm_key] = value


def run_label_permutation_control(rows, feature_sets, args):
    requested_heads = [item.strip() for item in args.heads.split(",") if item.strip()]
    requested_sets = [
        name for name in ("C1_basic", "C2_depthanything", "C2_midas", "C2_both", "C3_both")
        if name in feature_sets
    ]
    results = []
    for margin in target_margins(args):
        for round_idx in range(args.label_permutation_rounds):
            add_label_permutation(rows, margin, args.seed + 701 + round_idx)
            train_key = permuted_label_key(margin)
            for feature_set in requested_sets:
                features = feature_sets[feature_set]
                for head in requested_heads:
                    for split_num, (split_family, split_name, train, valid) in enumerate(split_specs(rows, args)):
                        if split_family != "random":
                            continue
                        train_labels = [rows[idx][train_key] for idx in train if rows[idx][train_key] >= 0]
                        if len(train_labels) < 50 or len(set(train_labels)) != 2:
                            continue
                        metrics = run_one(
                            rows,
                            train,
                            valid,
                            feature_set,
                            features,
                            head,
                            margin,
                            args,
                            args.seed + 900 + round_idx + split_num,
                            train_key=train_key,
                            fixed_threshold=0.5,
                        )
                        metrics.update({
                            "split_family": split_family,
                            "split": split_name,
                            "train_n": len(train),
                            "valid_n": len(valid),
                            "control": "label_permutation",
                            "permutation_round": round_idx + 1,
                        })
                        results.append(metrics)
    return results


def feature_group(feature_set):
    if feature_set.startswith("C0_"):
        return "C0"
    if feature_set.startswith("C1_"):
        return "C1"
    if feature_set.startswith("C2_"):
        return "C2"
    if feature_set.startswith("C3_"):
        return "C3"
    return "unknown"


def estimator_group(feature_set):
    name = feature_set
    for prefix in ("C2_", "C3_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace("_shuffled_depth", "")
    return name if name not in ("deployable", "basic") else ""


def summarize_results(results):
    grouped = {}
    for row in results:
        split_group_name = "random" if row["split_family"] == "random" else row["split"]
        key = (
            row["control"],
            row["target_margin"],
            row["feature_set"],
            row["head"],
            row["split_family"],
            split_group_name,
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (control, margin, feature_set, head, split_family, split_group_name), items in sorted(grouped.items()):
        summary = {
            "control": control,
            "target_margin": margin,
            "feature_group": feature_group(feature_set),
            "estimator_group": estimator_group(feature_set),
            "feature_set": feature_set,
            "head": head,
            "split_family": split_family,
            "split_group": split_group_name,
            "splits": len(items),
        }
        for metric in (
            "intervention_rate",
            "intervention_precision",
            "intervention_recall",
            "nochange_false_intervention_rate",
            "strong_nochange_false_intervention_rate",
            "residual_low_false_intervention_rate",
            "official_preserve_rate",
            "low_residual_preserve_rate",
            "simulated_mean_psnr_gain",
            "bootstrap_gain_p05",
            "oracle_recovery",
            "confidence_corr",
            "labeled_n",
            "ignored_n",
        ):
            vals = [item[metric] for item in items]
            summary[metric + "_mean"] = mean(vals)
            summary[metric + "_std"] = float(np.std([float(v) for v in vals])) if len(vals) >= 2 else 0.0
        summaries.append(summary)
    return summaries


def compact_rows(rows):
    keys = [
        "filename",
        "image_id",
        "airlight",
        "beta",
        "airlight_bin",
        "beta_bin",
        "official_strength_bin",
        "cr_strength_bin",
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
    return [{key: row.get(key, "") for key in keys} for row in rows]


def feature_group_rows(feature_sets):
    rows = []
    for name, features in sorted(feature_sets.items()):
        rows.append({
            "feature_set": name,
            "feature_group": feature_group(name),
            "estimator_group": estimator_group(name),
            "feature_count": len(features),
            "uses_shuffled_depth": int(name.endswith("_shuffled_depth")),
        })
    return rows


def best_random(summary, feature_group_name, allow_shuffled=False):
    rows = [
        row for row in summary
        if row["control"] == "real"
        and row["split_family"] == "random"
        and row["feature_group"] == feature_group_name
        and (allow_shuffled or not row["feature_set"].endswith("_shuffled_depth"))
    ]
    if not rows:
        return {}

    def key(row):
        useful = row["intervention_precision_mean"] > 0.0 and row["simulated_mean_psnr_gain_mean"] >= 0.0
        return (
            1 if useful else 0,
            -row["strong_nochange_false_intervention_rate_mean"],
            -row["nochange_false_intervention_rate_mean"],
            row["intervention_precision_mean"],
            row["simulated_mean_psnr_gain_mean"],
            row["intervention_recall_mean"],
        )

    return copy.deepcopy(max(rows, key=key))


def matching_row(summary, feature_set, head, split_family, split_group, control="real"):
    for row in summary:
        if (
            row["control"] == control
            and row["feature_set"] == feature_set
            and row["head"] == head
            and row["split_family"] == split_family
            and row["split_group"] == split_group
        ):
            return row
    return {}


def relative_reduction(before, after):
    before = float(before)
    after = float(after)
    if before <= EPS:
        return 0.0 if after <= EPS else -1.0
    return (before - after) / before


def compare_rows(candidate, baseline):
    if not candidate or not baseline:
        return {}
    return {
        "strong_official_FI_relative_reduction": relative_reduction(
            baseline["strong_nochange_false_intervention_rate_mean"],
            candidate["strong_nochange_false_intervention_rate_mean"],
        ),
        "all_nochange_FI_relative_reduction": relative_reduction(
            baseline["nochange_false_intervention_rate_mean"],
            candidate["nochange_false_intervention_rate_mean"],
        ),
        "precision_delta": candidate["intervention_precision_mean"] - baseline["intervention_precision_mean"],
        "simulated_gain_delta": candidate["simulated_mean_psnr_gain_mean"] - baseline["simulated_mean_psnr_gain_mean"],
        "bootstrap_gain_p05_delta": candidate["bootstrap_gain_p05_mean"] - baseline["bootstrap_gain_p05_mean"],
    }


def build_comparison_tables(summary, args):
    strong_rows = []
    heldout_rows = []
    shuffle_rows = []
    estimator_rows = []
    cfree_caware_rows = []
    random_real = [
        row for row in summary
        if row["control"] == "real"
        and row["split_family"] == "random"
        and not row["feature_set"].endswith("_shuffled_depth")
    ]
    for row in summary:
        if row["control"] != "real":
            continue
        base = matching_row(summary, "C1_basic", row["head"], row["split_family"], row["split_group"])
        comparison = compare_rows(row, base) if base else {}
        out = copy.deepcopy(row)
        out.update({key: comparison.get(key, "") for key in (
            "strong_official_FI_relative_reduction",
            "all_nochange_FI_relative_reduction",
            "precision_delta",
            "simulated_gain_delta",
            "bootstrap_gain_p05_delta",
        )})
        out["passes_relative_depth_line"] = int(
            row["feature_group"] == "C2"
            and comparison
            and comparison["strong_official_FI_relative_reduction"] >= args.min_relative_strong_fi_reduction
            and comparison["all_nochange_FI_relative_reduction"] >= args.min_relative_nochange_fi_reduction
            and comparison["precision_delta"] >= args.min_precision_delta
            and comparison["simulated_gain_delta"] >= 0.0
            and row["bootstrap_gain_p05_mean"] >= 0.0
        )
        out["passes_strong_absolute_line"] = int(
            row["strong_nochange_false_intervention_rate_mean"] <= args.max_strong_absolute_fi
            and row["nochange_false_intervention_rate_mean"] <= args.max_nochange_absolute_fi
            and row["intervention_precision_mean"] >= args.min_strong_absolute_precision
        )
        if row["split_family"] == "random":
            strong_rows.append(out)
        elif row["split_family"] in ("official_strength_heldout", "cr_strength_heldout", "residual_low_energy_heldout"):
            heldout_rows.append(out)

    for row in random_real:
        shuffled = matching_row(
            summary,
            row["feature_set"] + "_shuffled_depth",
            row["head"],
            row["split_family"],
            row["split_group"],
        )
        if not shuffled:
            continue
        shuffle_rows.append({
            "target_margin": row["target_margin"],
            "feature_set": row["feature_set"],
            "head": row["head"],
            "split_group": row["split_group"],
            "real_strong_nochange_fi": row["strong_nochange_false_intervention_rate_mean"],
            "shuffled_strong_nochange_fi": shuffled["strong_nochange_false_intervention_rate_mean"],
            "real_nochange_fi": row["nochange_false_intervention_rate_mean"],
            "shuffled_nochange_fi": shuffled["nochange_false_intervention_rate_mean"],
            "real_precision": row["intervention_precision_mean"],
            "shuffled_precision": shuffled["intervention_precision_mean"],
            "real_gain": row["simulated_mean_psnr_gain_mean"],
            "shuffled_gain": shuffled["simulated_mean_psnr_gain_mean"],
            "strong_fi_gap_real_minus_shuffled": row["strong_nochange_false_intervention_rate_mean"] - shuffled["strong_nochange_false_intervention_rate_mean"],
            "precision_gap_real_minus_shuffled": row["intervention_precision_mean"] - shuffled["intervention_precision_mean"],
            "gain_gap_real_minus_shuffled": row["simulated_mean_psnr_gain_mean"] - shuffled["simulated_mean_psnr_gain_mean"],
        })

    for alias in ("depthanything", "midas"):
        row = best_random(
            [item for item in summary if item["feature_set"] == "C2_" + alias or item["feature_set"] == "C1_basic"],
            "C2",
        )
        base = matching_row(summary, "C1_basic", row.get("head", ""), "random", "random") if row else {}
        comparison = compare_rows(row, base) if row and base else {}
        estimator_rows.append({
            "estimator": alias,
            "best_feature_set": row.get("feature_set", ""),
            "head": row.get("head", ""),
            "strong_official_FI_relative_reduction": comparison.get("strong_official_FI_relative_reduction", ""),
            "all_nochange_FI_relative_reduction": comparison.get("all_nochange_FI_relative_reduction", ""),
            "precision_delta": comparison.get("precision_delta", ""),
            "simulated_gain_delta": comparison.get("simulated_gain_delta", ""),
            "same_direction_effective": int(
                bool(comparison)
                and comparison["strong_official_FI_relative_reduction"] > 0.0
                and comparison["all_nochange_FI_relative_reduction"] >= 0.0
                and comparison["precision_delta"] >= args.min_precision_delta
            ),
        })

    for row in random_real:
        if not row["feature_set"].startswith("C3_"):
            continue
        c2_name = row["feature_set"].replace("C3_", "C2_", 1)
        c2 = matching_row(summary, c2_name, row["head"], row["split_family"], row["split_group"])
        if not c2:
            continue
        comparison = compare_rows(row, c2)
        cfree_caware_rows.append({
            "target_margin": row["target_margin"],
            "candidate_free_feature_set": c2_name,
            "candidate_aware_feature_set": row["feature_set"],
            "head": row["head"],
            "split_group": row["split_group"],
            **comparison,
        })
    return strong_rows, heldout_rows, shuffle_rows, estimator_rows, cfree_caware_rows


def label_permutation_table(real_summary, perm_summary):
    rows = []
    for perm in perm_summary:
        if perm["control"] != "label_permutation" or perm["split_family"] != "random":
            continue
        real = matching_row(real_summary, perm["feature_set"], perm["head"], "random", "random")
        if not real:
            continue
        rows.append({
            "feature_set": perm["feature_set"],
            "head": perm["head"],
            "permuted_precision": perm["intervention_precision_mean"],
            "real_precision": real["intervention_precision_mean"],
            "permuted_gain": perm["simulated_mean_psnr_gain_mean"],
            "real_gain": real["simulated_mean_psnr_gain_mean"],
            "permuted_strong_nochange_fi": perm["strong_nochange_false_intervention_rate_mean"],
            "real_strong_nochange_fi": real["strong_nochange_false_intervention_rate_mean"],
            "precision_delta_real_minus_permuted": real["intervention_precision_mean"] - perm["intervention_precision_mean"],
            "gain_delta_real_minus_permuted": real["simulated_mean_psnr_gain_mean"] - perm["simulated_mean_psnr_gain_mean"],
        })
    return rows


def heldout_status(summary, feature_set, head, args):
    status = {}
    for family in ("official_strength_heldout", "cr_strength_heldout", "residual_low_energy_heldout"):
        rows = [
            row for row in summary
            if row["control"] == "real"
            and row["feature_set"] == feature_set
            and row["head"] == head
            and row["split_family"] == family
        ]
        if not rows:
            status[family] = "missing"
            continue
        row = rows[0]
        base = matching_row(summary, "C1_basic", head, family, row["split_group"])
        comp = compare_rows(row, base) if base else {}
        ok = (
            bool(comp)
            and comp["strong_official_FI_relative_reduction"] >= 0.0
            and comp["all_nochange_FI_relative_reduction"] >= 0.0
            and comp["precision_delta"] >= args.min_precision_delta
            and row["bootstrap_gain_p05_mean"] >= 0.0
        )
        status[family] = "stable" if ok else "collapsed"
    return status


def build_decision_summary(summary, shuffle_rows, perm_rows, estimator_rows, args):
    best_c1 = best_random(summary, "C1")
    best_c2 = best_random(summary, "C2")
    best_c3 = best_random(summary, "C3")
    c2_vs_c1 = compare_rows(best_c2, best_c1)
    c3_vs_c1 = compare_rows(best_c3, best_c1)

    c2_pass = (
        bool(c2_vs_c1)
        and c2_vs_c1["strong_official_FI_relative_reduction"] >= args.min_relative_strong_fi_reduction
        and c2_vs_c1["all_nochange_FI_relative_reduction"] >= args.min_relative_nochange_fi_reduction
        and c2_vs_c1["precision_delta"] >= args.min_precision_delta
        and c2_vs_c1["simulated_gain_delta"] >= 0.0
        and best_c2.get("bootstrap_gain_p05_mean", -1.0) >= 0.0
    )
    c3_useful = (
        bool(c3_vs_c1)
        and c3_vs_c1["strong_official_FI_relative_reduction"] >= args.min_relative_strong_fi_reduction
        and c3_vs_c1["precision_delta"] >= args.min_precision_delta
        and c3_vs_c1["simulated_gain_delta"] >= 0.0
    )
    heldout = heldout_status(summary, best_c2.get("feature_set", ""), best_c2.get("head", ""), args) if best_c2 else {}
    heldout_ok = bool(heldout) and all(value == "stable" for value in heldout.values())
    shuffled_ok = any(
        row["feature_set"] == best_c2.get("feature_set", "")
        and (
            row["precision_gap_real_minus_shuffled"] > 0.03
            or row["gain_gap_real_minus_shuffled"] > 0.01
            or row["strong_fi_gap_real_minus_shuffled"] < -0.05
        )
        for row in shuffle_rows
    )
    perm_ok = any(
        row["feature_set"] == best_c2.get("feature_set", "")
        and (
            row["precision_delta_real_minus_permuted"] > 0.03
            or row["gain_delta_real_minus_permuted"] > 0.01
        )
        for row in perm_rows
    )
    consistency_ok = bool(estimator_rows) and all(
        int(row.get("same_direction_effective") or 0) == 1
        for row in estimator_rows
        if row.get("estimator") in ("depthanything", "midas")
    )

    if c2_pass and heldout_ok and shuffled_ok and perm_ok and consistency_ok:
        decision = "continue_depth_safety_route"
        recommend_next_step = "Write a tiny depth-aware official-default gate card before any model training."
    elif (not c2_pass) and c3_useful:
        decision = "candidate_aware_only"
        recommend_next_step = "Treat depth as candidate-selection evidence only; do not prioritize a depth-aware backbone."
    else:
        decision = "stop_depth_route"
        recommend_next_step = "Do not launch depth-aware DEA training from this audit."

    return {
        "decision": decision,
        "main_question": (
            "Does depth reduce strong-official/no-change false intervention beyond "
            "existing deployable and basic image features?"
        ),
        "best_C1": best_c1,
        "best_C2": best_c2,
        "best_C3": best_c3,
        "C2_vs_C1": {
            "strong_official_FI_relative_reduction": c2_vs_c1.get("strong_official_FI_relative_reduction"),
            "all_nochange_FI_relative_reduction": c2_vs_c1.get("all_nochange_FI_relative_reduction"),
            "precision_delta": c2_vs_c1.get("precision_delta"),
            "simulated_gain_delta": c2_vs_c1.get("simulated_gain_delta"),
            "bootstrap_gain_p05_delta": c2_vs_c1.get("bootstrap_gain_p05_delta"),
        },
        "heldout_results": {
            "official_strength": heldout.get("official_strength_heldout", ""),
            "CR_strength": heldout.get("cr_strength_heldout", ""),
            "residual_low_energy": heldout.get("residual_low_energy_heldout", ""),
        },
        "controls": {
            "shuffled_depth": "pass" if shuffled_ok else "fail_or_close",
            "label_permutation": "pass" if perm_ok else "fail_or_close",
            "basic_stat_control": "pass" if c2_pass else "fail",
            "depth_estimator_consistency": "pass" if consistency_ok else "fail",
        },
        "recommend_next_step": recommend_next_step,
    }


def oracle_summary(rows, args):
    gains = np.asarray([row["best_candidate_gain_vs_official"] for row in rows], dtype=np.float64)
    return {
        "num_images": len(rows),
        "mean_official_psnr": mean([row["official_psnr"] for row in rows]),
        "mean_best_candidate_gain_vs_official": mean([row["best_candidate_gain_vs_official"] for row in rows]),
        "mean_oracle_with_nochange_gain_vs_official": mean([
            row["oracle_with_nochange_gain_vs_official"] for row in rows
        ]),
        "nochange_margin": args.nochange_margin,
        "nochange_count": int(sum(row["label_nochange"] for row in rows)),
        "strong_official_nochange_count": int(sum(row["label_strong_official_nochange"] for row in rows)),
        "intervention_worthy_count": int(sum(gain >= target_margins(args)[0] for gain in gains)),
        "best_candidate_counts": {
            name: int(sum(row["best_candidate"] == name for row in rows)) for name in CANDIDATES
        },
    }


def write_protocol(path):
    lines = [
        "# Strong-Official No-Change Depth Safety Audit Protocol",
        "",
        "Main question: Can predicted depth reduce false intervention on strong-official/no-change samples beyond C1 basic image stats?",
        "",
        "Labels:",
        "",
        "- strong-official no-change: official_strength_bin == strong_official_q4 and best_candidate_gain_vs_official <= 0.05",
        "- all no-change: best_candidate_gain_vs_official <= 0.05",
        "- intervention-worthy: best_candidate_gain_vs_official >= 0.20",
        "",
        "Feature groups:",
        "",
        "- C0: deployable official/candidate output, residual, and DEA activation features.",
        "- C1: C0 plus basic image statistics.",
        "- C2: C1 plus candidate-free depth from hazy and official output.",
        "- C3: C1 plus candidate-free and candidate-aware depth features.",
        "",
        "Splits:",
        "",
        "- random image split for sanity check.",
        "- train non-strong official, test strong official.",
        "- train weak/mid CR, test strong CR.",
        "- train normal/high residual energy, test low residual energy.",
        "",
        "Controls:",
        "",
        "- shuffled-depth image-id control.",
        "- C2 vs C1 basic-stat control.",
        "- label-permutation control.",
        "",
        "Primary metric: strong-official no-change false intervention rate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path, oracle, summary, decision, meta, args):
    lines = [
        "# HAZE4K Strong-Official Depth Safety Audit",
        "",
        "## Decision",
        "",
        "- `{}`".format(decision["decision"]),
        "- {}".format(decision["recommend_next_step"]),
        "",
        "## Oracle And Labels",
        "",
        "- Images: `{}`".format(oracle["num_images"]),
        "- Mean official PSNR: `{:.4f}`".format(oracle["mean_official_psnr"]),
        "- Mean best-candidate gain vs official: `{:.4f}`".format(oracle["mean_best_candidate_gain_vs_official"]),
        "- No-change / strong-official no-change / intervention-worthy counts: `{}` / `{}` / `{}`".format(
            oracle["nochange_count"],
            oracle["strong_official_nochange_count"],
            oracle["intervention_worthy_count"],
        ),
        "- Best-candidate counts: `{}`".format(oracle["best_candidate_counts"]),
        "",
        "## Depth Source",
        "",
        "- Depth models: `{}`".format(meta["depth_models"]),
        "- Depth generated: `{}`".format(meta["depth_generated"]),
        "- Depth loaded: `{}`".format(meta["depth_loaded"]),
        "- Depth seconds: `{}`".format(meta["depth_seconds"]),
        "",
        "## C2 vs C1",
        "",
        "- strong-official FI relative reduction: `{}`".format(decision["C2_vs_C1"]["strong_official_FI_relative_reduction"]),
        "- all no-change FI relative reduction: `{}`".format(decision["C2_vs_C1"]["all_nochange_FI_relative_reduction"]),
        "- precision delta: `{}`".format(decision["C2_vs_C1"]["precision_delta"]),
        "- simulated gain delta: `{}`".format(decision["C2_vs_C1"]["simulated_gain_delta"]),
        "- bootstrap gain p05 delta: `{}`".format(decision["C2_vs_C1"]["bootstrap_gain_p05_delta"]),
        "",
        "## Heldout And Controls",
        "",
        "- Heldout: `{}`".format(decision["heldout_results"]),
        "- Controls: `{}`".format(decision["controls"]),
        "",
        "## Summary",
        "",
        "| Group | Feature Set | Head | Split | Precision | Recall | Strong FI | No-change FI | Gain | p05 | Preserve | Low Preserve |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        if row["control"] != "real":
            continue
        lines.append(
            "| {group} | {feature_set} | {head} | {split} | {precision:.4f} | {recall:.4f} | {strong:.4f} | {nochange:.4f} | {gain:.4f} | {p05:.4f} | {preserve:.4f} | {low_preserve:.4f} |".format(
                group=row["feature_group"],
                feature_set=row["feature_set"],
                head=row["head"],
                split=row["split_group"],
                precision=row["intervention_precision_mean"],
                recall=row["intervention_recall_mean"],
                strong=row["strong_nochange_false_intervention_rate_mean"],
                nochange=row["nochange_false_intervention_rate_mean"],
                gain=row["simulated_mean_psnr_gain_mean"],
                p05=row["bootstrap_gain_p05_mean"],
                preserve=row["official_preserve_rate_mean"],
                low_preserve=row["low_residual_preserve_rate_mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, meta = collect_rows(args)
    add_shuffled_depth_features(rows, args.seed + 17)
    feature_sets = build_feature_sets(rows)
    add_shuffled_feature_sets(feature_sets)

    write_csv(output_dir / "official_candidate_depth_join.csv", compact_rows(rows))
    write_csv(output_dir / "feature_group_summary.csv", feature_group_rows(feature_sets))

    results = run_audit(rows, feature_sets, args)
    summary = summarize_results(results)
    perm_results = run_label_permutation_control(rows, feature_sets, args)
    perm_summary = summarize_results(perm_results)

    strong_rows, heldout_rows, shuffle_rows, estimator_rows, cfree_caware_rows = build_comparison_tables(summary, args)
    perm_rows = label_permutation_table(summary, perm_summary)
    decision = build_decision_summary(summary, shuffle_rows, perm_rows, estimator_rows, args)
    oracle = oracle_summary(rows, args)

    write_csv(output_dir / "split_results.csv", results)
    write_csv(output_dir / "strong_nochange_safety_summary.csv", strong_rows)
    write_csv(output_dir / "heldout_safety_summary.csv", heldout_rows)
    write_csv(output_dir / "shuffle_depth_control.csv", shuffle_rows)
    write_csv(output_dir / "label_permutation_control.csv", perm_rows)
    write_csv(output_dir / "depth_estimator_consistency.csv", estimator_rows)
    write_csv(output_dir / "candidate_free_vs_candidate_aware.csv", cfree_caware_rows)
    write_protocol(output_dir / "depth_safety_audit_protocol.md")
    (output_dir / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "decision": decision,
        "args": vars(args),
        "meta": meta,
        "oracle": oracle,
        "feature_counts": {name: len(values) for name, values in feature_sets.items()},
        "summary": summary,
        "label_permutation_summary": perm_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir / "analysis_report.md", oracle, summary, decision, meta, args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
