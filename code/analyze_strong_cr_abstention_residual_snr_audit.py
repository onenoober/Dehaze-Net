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

from analyze_brf_representation_audit import (
    EPS,
    add_cr_strength_bins,
    add_feature_contrast,
    add_image_features,
    add_meta_bins,
    add_model_feature_block,
    image_tensor,
    infer_with_features,
    lowpass,
    parse_haze4k_name,
    psnr_tensor,
    scalar,
)
from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs
from model import BaselineRelativeFrequencyResidualCorrector, DEANet, DEANetCBRFRC


MODEL_NAMES = ("lfv1", "residualcalib", "crplus", "cbrfrc")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether strong-CR abstention/no-regression risk is readable "
            "before training another baseline-relative residual corrector."
        )
    )
    parser.add_argument("--dataset_root", type=str, default="../dataset/HAZE4K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hazy_dir", type=str, default="")
    parser.add_argument("--clear_dir", type=str, default="")
    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--lfv1_checkpoint", type=str, required=True)
    parser.add_argument("--residualcalib_checkpoint", type=str, required=True)
    parser.add_argument("--crplus_checkpoint", type=str, default="")
    parser.add_argument("--cbrfrc_checkpoint", type=str, required=True)
    parser.add_argument("--brfrc_v2_summary", type=str, default="")
    parser.add_argument("--route_evidence_csv", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_size", type=int, default=4)
    parser.add_argument("--target_grid", type=int, default=8)
    parser.add_argument("--spatial_feature_grid", type=int, default=4)
    parser.add_argument("--feature_pool_grid", type=int, default=2)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--random_splits", type=int, default=3)
    parser.add_argument("--valid_fraction", type=float, default=0.25)
    parser.add_argument("--heads", type=str, default="logistic,ridge_classifier,hgb,calibrated_logistic")
    parser.add_argument(
        "--feature_sets",
        type=str,
        default="A_output,D_feature_contrast,E_abstention_risk,E_abstention_risk_shuffled,F_diagnostic_leakage",
    )
    parser.add_argument("--positive_margin", type=float, default=0.30)
    parser.add_argument("--ambiguous_margin", type=float, default=0.10)
    parser.add_argument("--regression_margin", type=float, default=0.10)
    parser.add_argument("--severe_regression_margin", type=float, default=0.30)
    parser.add_argument("--max_lf_mse_delta_for_safe", type=float, default=0.0)
    parser.add_argument("--max_bias_delta_for_safe", type=float, default=0.0025)
    parser.add_argument("--max_ssim_drop_for_safe", type=float, default=0.0005)
    parser.add_argument("--lowpass_pools", type=str, default="4,8,16")
    parser.add_argument("--small_residual_quantile", type=float, default=0.15)
    parser.add_argument("--max_ignore_sign_flip", type=float, default=0.40)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--logistic_c", type=float, default=0.5)
    parser.add_argument("--hgb_max_iter", type=int, default=180)
    parser.add_argument("--hgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--hgb_l2", type=float, default=0.01)
    parser.add_argument("--min_strong_preserve_recall", type=float, default=0.75)
    parser.add_argument("--max_strong_false_intervention", type=float, default=0.20)
    parser.add_argument("--min_intervention_precision", type=float, default=0.60)
    parser.add_argument("--min_lfv1_gain_preservation", type=float, default=0.70)
    parser.add_argument("--min_confidence_corr", type=float, default=0.45)
    parser.add_argument("--min_shuffled_precision_gap", type=float, default=0.05)
    parser.add_argument("--min_sim_psnr_margin_vs_lfv1", type=float, default=-0.02)
    parser.add_argument("--preferred_sim_psnr_margin_vs_lfv1", type=float, default=0.05)
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


def set_eval_frozen(model, device):
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_deanet_variant(checkpoint_path, variant, device):
    kwargs = {"base_dim": 32}
    if variant == "lfv1":
        kwargs.update({"use_lf_prior": True, "lf_prior_channels": 8, "lf_prior_pool": 8, "lf_prior_gate_init": 0.0})
    elif variant == "residualcalib":
        kwargs.update({
            "use_lf_prior": True,
            "lf_prior_channels": 8,
            "lf_prior_pool": 8,
            "lf_prior_gate_init": 0.0,
            "lf_residual_calibration": True,
            "lf_calib_hidden_channels": 8,
            "lf_calib_alpha_max": 1.0,
        })
    elif variant in ("cr", "crplus"):
        pass
    else:
        raise ValueError("Unknown DEANet variant {}".format(variant))
    model = DEANet(**kwargs)
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(strip_module_prefix(checkpoint_state_dict(checkpoint)))
    step = checkpoint.get("step", "") if isinstance(checkpoint, dict) else ""
    return set_eval_frozen(model, device), step


def load_cbrfrc(checkpoint_path, device):
    baseline = DEANet(base_dim=32)
    corrector = BaselineRelativeFrequencyResidualCorrector(
        hidden_channels=16,
        gate_init=-4.0,
        hf_gate_init=-5.0,
        max_residual=0.08,
        max_color_residual=0.04,
        max_hf_residual=0.03,
        hf_scale=0.1,
        use_haze_prior=False,
        preserve_highfreq=True,
        pyramid_type="laplacian",
        lf_pool=8,
        mid_pool=4,
    )
    model = DEANetCBRFRC(baseline=baseline, corrector=corrector, freeze_baseline=True, use_baseline_detach=True)
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(strip_module_prefix(checkpoint_state_dict(checkpoint)))
    step = checkpoint.get("step", "") if isinstance(checkpoint, dict) else ""
    return set_eval_frozen(model, device), step


def pad_tensor(x, pad_size):
    if pad_size <= 0:
        return x
    _, _, h, w = x.size()
    mod_pad_h = (pad_size - h % pad_size) % pad_size
    mod_pad_w = (pad_size - w % pad_size) % pad_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")


def infer_plain(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_tensor(hazy, pad_size)
    out = model(padded)
    if isinstance(out, dict):
        out = out["out"]
    return out[:, :, :h, :w].clamp(0, 1)


def luma(x):
    return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114


def tensor_norm(x):
    return scalar(x.detach().float().reshape(x.shape[0], -1).norm(dim=1).mean())


def tensor_rms(x):
    return scalar(torch.sqrt(torch.mean(x.detach().float() * x.detach().float()) + EPS))


def tensor_mean_abs(x):
    return scalar(x.detach().float().abs().mean())


def mse_tensor(a, b):
    return scalar(torch.mean((a.detach().float() - b.detach().float()) ** 2))


def try_ssim(pred, target):
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return ""
    pred_np = pred.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    try:
        return float(structural_similarity(target_np, pred_np, channel_axis=2, data_range=1.0))
    except TypeError:
        return float(structural_similarity(target_np, pred_np, multichannel=True, data_range=1.0))


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x != "" and y != ""]
    if len(pairs) < 2:
        return 0.0
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def sign_flip_rate(reference, other):
    ref = reference.detach().float()
    oth = other.detach().float()
    scale = torch.maximum(ref.abs(), oth.abs())
    mask = scale > max(1e-4, scalar(scale.mean()) * 0.05)
    if scalar(mask.float().sum()) <= 0:
        return 0.0
    flips = torch.sign(ref[mask]) != torch.sign(oth[mask])
    return scalar(flips.float().mean())


def residual_cosine(a, b):
    avec = a.detach().float().reshape(1, -1)
    bvec = b.detach().float().reshape(1, -1)
    return scalar((avec * bvec).sum(dim=1) / (avec.norm(dim=1) * bvec.norm(dim=1) + EPS))


def add_risk_features(row, hazy, clear, j0, lfv1, pools):
    j0_lp8 = lowpass(j0, 8)
    gt_lp8 = lowpass(clear, 8)
    lf_lp8 = lowpass(lfv1, 8)
    hazy_lp8 = lowpass(hazy, 8)
    target_res = gt_lp8 - j0_lp8
    lfv1_res = lf_lp8 - j0_lp8
    row["risk_lfv1_residual_norm_lf8"] = tensor_norm(lfv1_res)
    row["risk_lfv1_residual_rms_lf8"] = tensor_rms(lfv1_res)
    row["risk_lfv1_residual_abs_mean_lf8"] = tensor_mean_abs(lfv1_res)
    row["risk_hazy_j0_residual_norm_lf8"] = tensor_norm(hazy_lp8 - j0_lp8)
    row["risk_j0_lf_variance"] = scalar(torch.var(j0_lp8.detach().float(), unbiased=False))
    row["risk_hazy_lf_variance"] = scalar(torch.var(hazy_lp8.detach().float(), unbiased=False))
    row["risk_j0_luma_lf_mean"] = scalar(luma(j0_lp8).mean())
    row["risk_hazy_luma_lf_mean"] = scalar(luma(hazy_lp8).mean())
    row["risk_hazy_minus_j0_luma_lf_bias"] = scalar((luma(hazy_lp8) - luma(j0_lp8)).mean())
    row["risk_hazy_minus_j0_color_lf_bias"] = scalar((hazy_lp8 - j0_lp8).mean(dim=(2, 3)).abs().mean())
    row["diag_target_residual_norm_lf8"] = tensor_norm(target_res)
    row["diag_target_residual_rms_lf8"] = tensor_rms(target_res)
    row["diag_lfv1_target_residual_cosine_lf8"] = residual_cosine(lfv1_res, target_res)
    row["diag_lfv1_residual_error_ratio_lf8"] = tensor_norm(lfv1_res - target_res) / (row["diag_target_residual_norm_lf8"] + EPS)
    row["diag_cr_lf_mse_lf8"] = mse_tensor(j0_lp8, gt_lp8)
    row["diag_lfv1_lf_mse_lf8"] = mse_tensor(lf_lp8, gt_lp8)
    row["diag_lfv1_lf_mse_gain_lf8"] = row["diag_cr_lf_mse_lf8"] - row["diag_lfv1_lf_mse_lf8"]

    flips = []
    cosines = []
    for pool in pools:
        if pool == 8:
            continue
        tgt_other = lowpass(clear, pool) - lowpass(j0, pool)
        flips.append(sign_flip_rate(target_res, tgt_other))
        cosines.append(residual_cosine(target_res, tgt_other))
    row["diag_target_sign_flip_rate"] = float(np.mean(flips)) if flips else 0.0
    row["diag_target_kernel_cosine_mean"] = float(np.mean(cosines)) if cosines else 1.0


def candidate_metrics(row, name, out, clear, j0):
    psnr = psnr_tensor(out, clear)
    ssim = try_ssim(out, clear)
    lp_out = lowpass(out, 8)
    lp_j0 = lowpass(j0, 8)
    lp_gt = lowpass(clear, 8)
    lf_mse = mse_tensor(lp_out, lp_gt)
    cr_lf_mse = mse_tensor(lp_j0, lp_gt)
    luma_bias = abs(scalar((luma(lp_out) - luma(lp_gt)).mean()))
    cr_luma_bias = abs(scalar((luma(lp_j0) - luma(lp_gt)).mean()))
    color_bias = scalar((lp_out - lp_gt).mean(dim=(2, 3)).abs().mean())
    cr_color_bias = scalar((lp_j0 - lp_gt).mean(dim=(2, 3)).abs().mean())
    row[name + "_psnr"] = psnr
    row[name + "_delta_cr_psnr"] = psnr - row["cr_psnr"]
    row[name + "_ssim"] = ssim
    row[name + "_delta_cr_ssim"] = ssim - row["cr_ssim"] if ssim != "" and row["cr_ssim"] != "" else ""
    row[name + "_lf_mse_delta_vs_cr"] = lf_mse - cr_lf_mse
    row[name + "_luma_bias_delta_vs_cr"] = luma_bias - cr_luma_bias
    row[name + "_color_bias_delta_vs_cr"] = color_bias - cr_color_bias


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
    pools = [int(item) for item in args.lowpass_pools.split(",") if item.strip()]

    cr_model, cr_step = load_deanet_variant(args.baseline_checkpoint, "cr", args.device)
    lfv1_model, lfv1_step = load_deanet_variant(args.lfv1_checkpoint, "lfv1", args.device)
    residualcalib_model, residualcalib_step = load_deanet_variant(args.residualcalib_checkpoint, "residualcalib", args.device)
    crplus_model = None
    crplus_step = ""
    if args.crplus_checkpoint and Path(args.crplus_checkpoint).is_file():
        crplus_model, crplus_step = load_deanet_variant(args.crplus_checkpoint, "crplus", args.device)
    cbrfrc_model, cbrfrc_step = load_cbrfrc(args.cbrfrc_checkpoint, args.device)
    models = {
        "residualcalib": residualcalib_model,
        "cbrfrc": cbrfrc_model,
    }
    if crplus_model is not None:
        models["crplus"] = crplus_model
    route_evidence = read_route_evidence(args.route_evidence_csv)

    rows = []
    total = len(image_names)
    with torch.no_grad():
        for idx, filename in enumerate(image_names, 1):
            hazy = image_tensor(hazy_dir / filename, args.device)
            clear = image_tensor(find_clear_image(clear_dir, filename), args.device)
            cr_out, cr_feats = infer_with_features(cr_model, hazy, args.pad_size)
            lfv1_out, lfv1_feats = infer_with_features(lfv1_model, hazy, args.pad_size)

            row = {}
            add_image_features(row, "input", hazy, args.spatial_feature_grid)
            add_image_features(row, "j0", cr_out, args.spatial_feature_grid)
            add_image_features(row, "input_minus_j0", hazy - cr_out, args.spatial_feature_grid)
            add_model_feature_block(row, "cr_feat", cr_feats, args.feature_pool_grid)
            add_model_feature_block(row, "lfv1_feat", lfv1_feats, args.feature_pool_grid)
            add_feature_contrast(row, cr_feats, lfv1_feats, args.feature_pool_grid)
            add_risk_features(row, hazy, clear, cr_out, lfv1_out, pools)

            row.update(parse_haze4k_name(filename))
            row["filename"] = filename
            row["cr_psnr"] = psnr_tensor(cr_out, clear)
            row["cr_ssim"] = try_ssim(cr_out, clear)
            candidate_metrics(row, "lfv1", lfv1_out, clear, cr_out)
            for name, model in models.items():
                out = infer_plain(model, hazy, args.pad_size)
                candidate_metrics(row, name, out, clear, cr_out)
            if "crplus_delta_cr_psnr" not in row and filename in route_evidence:
                add_crplus_route_evidence(row, route_evidence[filename])
            add_meta_bins(row)
            rows.append(row)

            if idx % 25 == 0 or idx == total:
                print("rows {}/{}".format(idx, total), flush=True)

    add_cr_strength_bins(rows)
    add_labels(rows, args)
    meta = {
        "images": len(rows),
        "hazy_dir": str(hazy_dir),
        "clear_dir": str(clear_dir),
        "baseline_checkpoint": args.baseline_checkpoint,
        "lfv1_checkpoint": args.lfv1_checkpoint,
        "residualcalib_checkpoint": args.residualcalib_checkpoint,
        "crplus_checkpoint": args.crplus_checkpoint,
        "route_evidence_csv": args.route_evidence_csv,
        "cbrfrc_checkpoint": args.cbrfrc_checkpoint,
        "brfrc_v2_summary": args.brfrc_v2_summary,
        "baseline_checkpoint_step": cr_step,
        "lfv1_checkpoint_step": lfv1_step,
        "residualcalib_checkpoint_step": residualcalib_step,
        "crplus_checkpoint_step": crplus_step,
        "cbrfrc_checkpoint_step": cbrfrc_step,
    }
    return rows, meta


def read_route_evidence(path):
    if not path:
        return {}
    route_path = Path(path)
    if not route_path.is_file():
        return {}
    with route_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["filename"]: row for row in rows if row.get("filename")}


def parse_float(value, default=""):
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def add_crplus_route_evidence(row, evidence):
    row["crplus_psnr"] = parse_float(evidence.get("crplusv2_psnr"))
    row["crplus_delta_cr_psnr"] = parse_float(evidence.get("crplusv2_delta_cr_psnr"))
    row["crplus_ssim"] = parse_float(evidence.get("crplusv2_ssim"))
    row["crplus_delta_cr_ssim"] = parse_float(evidence.get("crplusv2_delta_cr_ssim"))
    # The route-evidence matrix has PSNR/SSIM but not LF/color deltas. Mark
    # those neutral so CRPlus-v2 can contribute to labels without pretending
    # this file contains unavailable LF-bias measurements.
    row["crplus_lf_mse_delta_vs_cr"] = 0.0
    row["crplus_luma_bias_delta_vs_cr"] = 0.0
    row["crplus_color_bias_delta_vs_cr"] = 0.0
    row["crplus_metric_source"] = "route_evidence_csv"


def safe_candidate(row, name, args):
    if row.get(name + "_delta_cr_psnr", "") == "":
        return False
    delta = row[name + "_delta_cr_psnr"]
    ssim_delta = row[name + "_delta_cr_ssim"]
    ssim_ok = ssim_delta == "" or ssim_delta >= -args.max_ssim_drop_for_safe
    return (
        delta >= args.positive_margin
        and row[name + "_lf_mse_delta_vs_cr"] <= args.max_lf_mse_delta_for_safe
        and row[name + "_luma_bias_delta_vs_cr"] <= args.max_bias_delta_for_safe
        and row[name + "_color_bias_delta_vs_cr"] <= args.max_bias_delta_for_safe
        and ssim_ok
    )


def add_labels(rows, args):
    target_norms = np.asarray([row["diag_target_residual_norm_lf8"] for row in rows], dtype=np.float64)
    small_thr = float(np.quantile(target_norms, args.small_residual_quantile))
    for row in rows:
        deltas = [
            row[name + "_delta_cr_psnr"]
            for name in MODEL_NAMES
            if row.get(name + "_delta_cr_psnr", "") != ""
        ]
        if not deltas:
            row["label_preserve"] = 0
            row["label_intervene"] = 0
            row["label_ignore"] = 1
            row["label_class"] = -1
            continue
        safe = [name for name in MODEL_NAMES if safe_candidate(row, name, args)]
        max_delta = max(deltas)
        min_delta = min(deltas)
        close = max(abs(delta) for delta in deltas) < args.ambiguous_margin
        small_residual = row["diag_target_residual_norm_lf8"] <= small_thr
        sign_unstable = row["diag_target_sign_flip_rate"] >= args.max_ignore_sign_flip
        strong = row["cr_strength_bin"] == "strong_cr_q4"
        regression_risk = min_delta <= -args.regression_margin
        severe_regression_risk = min_delta <= -args.severe_regression_margin
        no_stable_gain = max_delta < args.ambiguous_margin

        preserve = (
            (strong and (not safe or regression_risk or no_stable_gain))
            or (not safe and (no_stable_gain or regression_risk))
            or severe_regression_risk
        )
        intervene = bool(safe) and not (strong and regression_risk)
        ignore = False
        if preserve and intervene:
            intervene = False
        if not preserve and not intervene:
            ignore = close or small_residual or sign_unstable
        if small_residual and sign_unstable and not strong and not intervene:
            ignore = True

        row["label_preserve"] = int(preserve and not ignore)
        row["label_intervene"] = int(intervene and not ignore)
        row["label_ignore"] = int(ignore)
        row["label_class"] = 1 if row["label_intervene"] else (0 if row["label_preserve"] else -1)
        row["label_safe_candidate_count"] = len(safe)
        row["label_safe_candidates"] = ",".join(safe)
        row["label_max_candidate_delta_psnr"] = max_delta
        row["label_min_candidate_delta_psnr"] = min_delta
        row["label_regression_risk"] = int(regression_risk)
        row["label_severe_regression_risk"] = int(severe_regression_risk)
        row["label_small_residual_threshold"] = small_thr
        row["label_small_residual"] = int(small_residual)
        row["label_sign_unstable"] = int(sign_unstable)


def is_number(value):
    return isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)


def numeric_feature_keys(rows):
    excluded = {
        "filename",
        "image_id",
        "airlight_bin",
        "beta_bin",
        "cr_strength_bin",
        "label_safe_candidates",
    }
    keys = []
    for key in sorted(rows[0].keys()):
        if key in excluded:
            continue
        if is_number(rows[0].get(key)):
            keys.append(key)
    return keys


def build_feature_sets(rows):
    all_keys = numeric_feature_keys(rows)
    output_prefixes = ("input_", "j0_", "input_minus_j0_")
    output_keys = [key for key in all_keys if key.startswith(output_prefixes) or key in ("airlight", "beta")]
    cr_keys = [key for key in all_keys if key.startswith("cr_feat_")]
    lfv1_keys = [key for key in all_keys if key.startswith("lfv1_feat_")]
    contrast_keys = [key for key in all_keys if key.startswith("contrast_")]
    risk_keys = [key for key in all_keys if key.startswith("risk_")]
    diagnostic_keys = [
        key for key in all_keys
        if key.startswith("diag_")
        or key.startswith("label_")
        or key.endswith("_delta_cr_psnr")
        or key.endswith("_lf_mse_delta_vs_cr")
        or key.endswith("_luma_bias_delta_vs_cr")
        or key.endswith("_color_bias_delta_vs_cr")
    ]
    deployable = output_keys + cr_keys + lfv1_keys + contrast_keys + risk_keys
    shuffled = output_keys + ["shuf_" + key for key in cr_keys + lfv1_keys + contrast_keys + risk_keys]
    return {
        "A_output": output_keys,
        "B_cr_features": output_keys + cr_keys,
        "C_lfv1_features": output_keys + lfv1_keys,
        "D_feature_contrast": output_keys + contrast_keys,
        "E_abstention_risk": deployable,
        "E_abstention_risk_shuffled": shuffled,
        "F_diagnostic_leakage": deployable + diagnostic_keys,
    }


def add_shuffled_feature_blocks(rows, args):
    rng = random.Random(args.seed + 271)
    keys = [
        key for key in numeric_feature_keys(rows)
        if key.startswith(("cr_feat_", "lfv1_feat_", "contrast_", "risk_"))
    ]
    order = list(range(len(rows)))
    rng.shuffle(order)
    source = [{key: rows[idx][key] for key in keys} for idx in order]
    for row, values in zip(rows, source):
        for key, value in values.items():
            row["shuf_" + key] = value


def matrix(rows, indices, features):
    x = np.zeros((len(indices), len(features)), dtype=np.float32)
    for row_pos, idx in enumerate(indices):
        row = rows[idx]
        for col, key in enumerate(features):
            value = row.get(key, 0.0)
            if value == "" or value is None:
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


def valid_split(rows, train, valid):
    if len(train) < 50 or len(valid) < 25:
        return False
    labels = [rows[idx]["label_class"] for idx in train if rows[idx]["label_class"] >= 0]
    return len(labels) >= 50 and len(set(labels)) == 2


def fit_scores(head, x_train, y_train, x_valid, args, seed):
    if head == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=args.logistic_c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_valid)[:, 1]
    if head == "ridge_classifier":
        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier(alpha=args.ridge_alpha, class_weight="balanced")
        model.fit(x_train, y_train)
        train_score = model.decision_function(x_train)
        valid_score = model.decision_function(x_valid)
        return sigmoid(train_score), sigmoid(valid_score)
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
    if head == "calibrated_logistic":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(
            C=args.logistic_c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        model.fit(x_train, y_train)
        return model.predict_proba(x_train)[:, 1], model.predict_proba(x_valid)[:, 1]
    raise ValueError("Unknown head {}".format(head))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))).astype(np.float32)


def eval_threshold(rows, indices, score, threshold):
    intervene = score >= threshold
    labels = np.asarray([rows[idx]["label_class"] for idx in indices], dtype=np.int32)
    labeled = labels >= 0
    preserve = labels == 0
    safe = labels == 1
    strong = np.asarray([rows[idx]["cr_strength_bin"] == "strong_cr_q4" for idx in indices], dtype=bool)
    strong_preserve = strong & preserve
    lfv1_gain = np.asarray([rows[idx]["lfv1_delta_cr_psnr"] >= 0.30 for idx in indices], dtype=bool)
    lfv1_regress = np.asarray([rows[idx]["lfv1_delta_cr_psnr"] <= -0.30 for idx in indices], dtype=bool)
    cr_psnr = np.asarray([rows[idx]["cr_psnr"] for idx in indices], dtype=np.float64)
    lfv1_psnr = np.asarray([rows[idx]["lfv1_psnr"] for idx in indices], dtype=np.float64)
    sim_psnr = np.where(intervene, lfv1_psnr, cr_psnr)
    lfv1_delta = np.asarray([rows[idx]["lfv1_delta_cr_psnr"] for idx in indices], dtype=np.float64)

    return {
        "eval_n": int(len(indices)),
        "labeled_n": int(np.sum(labeled)),
        "ignored_n": int(np.sum(~labeled)),
        "intervention_rate": float(np.mean(intervene)) if len(indices) else 0.0,
        "intervention_precision": float(np.mean(safe[intervene & labeled])) if np.any(intervene & labeled) else 0.0,
        "strong_q4_count": int(np.sum(strong)),
        "strong_q4_preserve_count": int(np.sum(strong_preserve)),
        "strong_q4_preserve_recall": float(np.mean(~intervene[strong_preserve])) if np.any(strong_preserve) else 0.0,
        "strong_q4_false_intervention_rate": float(np.mean(intervene[strong_preserve])) if np.any(strong_preserve) else 0.0,
        "lfv1_gain_case_count": int(np.sum(lfv1_gain)),
        "lfv1_gain_preservation": float(np.mean(intervene[lfv1_gain])) if np.any(lfv1_gain) else 0.0,
        "lfv1_regression_case_count": int(np.sum(lfv1_regress)),
        "lfv1_regression_rescue": float(np.mean(~intervene[lfv1_regress])) if np.any(lfv1_regress) else 0.0,
        "confidence_corr": pearson(score.tolist(), lfv1_delta.tolist()),
        "sim_psnr": float(np.mean(sim_psnr)),
        "cr_psnr": float(np.mean(cr_psnr)),
        "lfv1_psnr": float(np.mean(lfv1_psnr)),
        "sim_delta_vs_cr": float(np.mean(sim_psnr - cr_psnr)),
        "sim_delta_vs_lfv1": float(np.mean(sim_psnr - lfv1_psnr)),
    }


def choose_threshold(rows, indices, score, args):
    best = None
    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = eval_threshold(rows, indices, score, float(threshold))
        feasible = (
            metrics["strong_q4_preserve_recall"] >= args.min_strong_preserve_recall
            and metrics["strong_q4_false_intervention_rate"] <= args.max_strong_false_intervention
            and metrics["intervention_precision"] >= args.min_intervention_precision
        )
        key = (
            1 if feasible else 0,
            metrics["sim_psnr"],
            metrics["intervention_precision"],
            metrics["lfv1_gain_preservation"],
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1]


def run_one(rows, train_idx, valid_idx, feature_set, features, head, args, seed):
    train_labeled = [idx for idx in train_idx if rows[idx]["label_class"] >= 0]
    x_train = matrix(rows, train_labeled, features)
    x_valid = matrix(rows, valid_idx, features)
    x_train, x_valid, active_features = standardize(x_train, x_valid)
    y_train = np.asarray([rows[idx]["label_class"] for idx in train_labeled], dtype=np.int32)
    train_score, valid_score = fit_scores(head, x_train, y_train, x_valid, args, seed)
    threshold = choose_threshold(rows, train_labeled, train_score, args)
    metrics = eval_threshold(rows, valid_idx, valid_score, threshold)
    metrics.update({
        "feature_set": feature_set,
        "head": head,
        "threshold": threshold,
        "feature_count": len(features),
        "active_feature_count": active_features,
        "train_labeled_n": len(train_labeled),
        "eligible_for_pass": int(feature_set not in ("F_diagnostic_leakage",) and not feature_set.endswith("_shuffled")),
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
    for key in ("airlight_bin", "beta_bin", "cr_strength_bin"):
        values = sorted({row.get(key, "") for row in rows if row.get(key, "")})
        for value in values:
            train, valid = split_group(rows, key, value)
            split_specs.append((key, "{}={}".format(key, value), train, valid))

    results = []
    for feature_set in requested_sets:
        features = feature_sets.get(feature_set, [])
        if not features:
            continue
        for head in requested_heads:
            for split_num, (split_family, split_name, train, valid) in enumerate(split_specs):
                if not valid_split(rows, train, valid):
                    continue
                print("{} {} {}".format(feature_set, head, split_name), flush=True)
                metrics = run_one(rows, train, valid, feature_set, features, head, args, args.seed + split_num)
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
            "eligible_for_pass": items[0]["eligible_for_pass"],
        }
        for metric in (
            "intervention_precision",
            "strong_q4_preserve_recall",
            "strong_q4_false_intervention_rate",
            "lfv1_gain_preservation",
            "lfv1_regression_rescue",
            "confidence_corr",
            "sim_psnr",
            "cr_psnr",
            "lfv1_psnr",
            "sim_delta_vs_cr",
            "sim_delta_vs_lfv1",
            "intervention_rate",
            "labeled_n",
            "ignored_n",
        ):
            vals = [item[metric] for item in items]
            summary[metric + "_mean"] = mean(vals)
            summary[metric + "_std"] = std(vals)
        summaries.append(summary)

    by_key = {(row["feature_set"], row["head"], row["split_family"]): row for row in summaries}
    for row in summaries:
        control = by_key.get(("E_abstention_risk_shuffled", row["head"], row["split_family"]), {})
        row["precision_gap_vs_shuffled"] = (
            row["intervention_precision_mean"] - control.get("intervention_precision_mean", row["intervention_precision_mean"])
        )
        row["passes_random_line"] = passes_random(row, args)
        row["passes_heldout_line"] = passes_heldout(row, args)
    return summaries


def passes_random(row, args):
    if row["split_family"] != "random" or not row["eligible_for_pass"]:
        return False
    return (
        row["strong_q4_preserve_recall_mean"] >= args.min_strong_preserve_recall
        and row["strong_q4_false_intervention_rate_mean"] <= args.max_strong_false_intervention
        and row["intervention_precision_mean"] >= args.min_intervention_precision
        and row["lfv1_gain_preservation_mean"] >= args.min_lfv1_gain_preservation
        and row["confidence_corr_mean"] >= args.min_confidence_corr
        and row["precision_gap_vs_shuffled"] >= args.min_shuffled_precision_gap
        and row["sim_delta_vs_lfv1_mean"] >= args.min_sim_psnr_margin_vs_lfv1
    )


def passes_heldout(row, args):
    if row["split_family"] == "random" or not row["eligible_for_pass"]:
        return False
    return (
        row["strong_q4_preserve_recall_mean"] >= max(0.68, args.min_strong_preserve_recall - 0.07)
        and row["strong_q4_false_intervention_rate_mean"] <= min(0.30, args.max_strong_false_intervention + 0.10)
        and row["intervention_precision_mean"] >= max(0.55, args.min_intervention_precision - 0.05)
        and row["lfv1_gain_preservation_mean"] >= max(0.62, args.min_lfv1_gain_preservation - 0.08)
    )


def final_recommendation(summaries):
    random_pass = [row for row in summaries if row["passes_random_line"]]
    if not random_pass:
        return "do_not_train_abstention_brf_v3_yet"
    for row in random_pass:
        heldout = [
            item for item in summaries
            if item["feature_set"] == row["feature_set"]
            and item["head"] == row["head"]
            and item["split_family"] != "random"
        ]
        if heldout and all(item["passes_heldout_line"] for item in heldout):
            return "stage0_passed_write_abstention_first_brf_v3_card"
    return "do_not_train_abstention_brf_v3_yet"


def snr_by_cr_strength(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["cr_strength_bin"], []).append(row)
    out = []
    for name, items in sorted(groups.items()):
        entry = {"cr_strength_bin": name, "n": len(items)}
        for metric in (
            "diag_target_residual_norm_lf8",
            "risk_lfv1_residual_norm_lf8",
            "diag_lfv1_target_residual_cosine_lf8",
            "diag_target_sign_flip_rate",
            "diag_lfv1_lf_mse_gain_lf8",
            "lfv1_delta_cr_psnr",
            "residualcalib_delta_cr_psnr",
            "crplus_delta_cr_psnr",
            "cbrfrc_delta_cr_psnr",
            "risk_hazy_minus_j0_luma_lf_bias",
            "risk_hazy_minus_j0_color_lf_bias",
        ):
            vals = [item[metric] for item in items if item.get(metric, "") != ""]
            entry[metric + "_mean"] = mean(vals)
            entry[metric + "_std"] = std(vals)
        out.append(entry)
    by_name = {row["cr_strength_bin"]: row for row in out}
    weak = by_name.get("weak_cr_q1", {})
    strong = by_name.get("strong_cr_q4", {})
    ratio = 0.0
    if weak.get("diag_target_residual_norm_lf8_mean", 0.0) > EPS:
        ratio = strong.get("diag_target_residual_norm_lf8_mean", 0.0) / weak["diag_target_residual_norm_lf8_mean"]
    return out, {
        "strong_q4_target_residual_norm_over_weak_q1": ratio,
        "strong_q4_sign_flip_rate": strong.get("diag_target_sign_flip_rate_mean", 0.0),
    }


def compact_label_rows(rows):
    keys = [
        "filename",
        "airlight",
        "beta",
        "airlight_bin",
        "beta_bin",
        "cr_strength_bin",
        "cr_psnr",
        "lfv1_psnr",
        "residualcalib_psnr",
        "crplus_psnr",
        "cbrfrc_psnr",
        "lfv1_delta_cr_psnr",
        "residualcalib_delta_cr_psnr",
        "crplus_delta_cr_psnr",
        "cbrfrc_delta_cr_psnr",
        "diag_target_residual_norm_lf8",
        "risk_lfv1_residual_norm_lf8",
        "diag_lfv1_target_residual_cosine_lf8",
        "diag_target_sign_flip_rate",
        "label_preserve",
        "label_intervene",
        "label_ignore",
        "label_safe_candidate_count",
        "label_safe_candidates",
        "label_max_candidate_delta_psnr",
        "label_min_candidate_delta_psnr",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


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


def write_report(path, summaries, recommendation, meta, snr_summary, args):
    lines = [
        "# HAZE4K Strong-CR Abstention / Residual-SNR Audit",
        "",
        "## Recommendation",
        "",
        "- `{}`".format(recommendation),
        "",
        "## Source",
        "",
        "- Images: `{}`".format(meta["images"]),
        "- CR checkpoint step: `{}`".format(meta.get("baseline_checkpoint_step")),
        "- LF-v1 checkpoint step: `{}`".format(meta.get("lfv1_checkpoint_step")),
        "- ResidualCalib checkpoint step: `{}`".format(meta.get("residualcalib_checkpoint_step")),
        "- CRPlus-v2 checkpoint step: `{}`".format(meta.get("crplus_checkpoint_step")),
        "- CBRFRC-v1 checkpoint step: `{}`".format(meta.get("cbrfrc_checkpoint_step")),
        "",
        "## Residual-SNR Key Readouts",
        "",
        "- strong_q4 target residual norm / weak_q1 target residual norm: `{:.4f}`".format(
            snr_summary["strong_q4_target_residual_norm_over_weak_q1"]
        ),
        "- strong_q4 sign flip rate: `{:.4f}`".format(snr_summary["strong_q4_sign_flip_rate"]),
        "",
        "## Probe Summary",
        "",
        "| Feature Set | Head | Split | Strong Preserve | Strong False Int | Precision | LF-v1 Gain Keep | LF-v1 Regr Rescue | Corr | Sim-CR | Sim-LFv1 | Shuffle Gap | Random Pass | Heldout Pass |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {feature_set} | {head} | {split} | {spr:.4f} | {sfi:.4f} | {prec:.4f} | {gain:.4f} | {rescue:.4f} | {corr:.4f} | {simcr:.4f} | {simlf:.4f} | {gap:.4f} | {rp} | {hp} |".format(
                feature_set=row["feature_set"],
                head=row["head"],
                split=row["split_family"],
                spr=row["strong_q4_preserve_recall_mean"],
                sfi=row["strong_q4_false_intervention_rate_mean"],
                prec=row["intervention_precision_mean"],
                gain=row["lfv1_gain_preservation_mean"],
                rescue=row["lfv1_regression_rescue_mean"],
                corr=row["confidence_corr_mean"],
                simcr=row["sim_delta_vs_cr_mean"],
                simlf=row["sim_delta_vs_lfv1_mean"],
                gap=row["precision_gap_vs_shuffled"],
                rp="yes" if row["passes_random_line"] else "no",
                hp="yes" if row["passes_heldout_line"] else "no",
            )
        )
    lines.extend([
        "",
        "## Pass Line",
        "",
        "- strong_q4 preserve recall >= `{:.2f}`.".format(args.min_strong_preserve_recall),
        "- strong_q4 false-intervention rate <= `{:.2f}`.".format(args.max_strong_false_intervention),
        "- intervention precision >= `{:.2f}`.".format(args.min_intervention_precision),
        "- LF-v1 gain preservation >= `{:.2f}`.".format(args.min_lfv1_gain_preservation),
        "- confidence correlation >= `{:.2f}`.".format(args.min_confidence_corr),
        "- simulated PSNR close to LF-v1, minimum delta `{:.2f}`.".format(args.min_sim_psnr_margin_vs_lfv1),
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
    summaries = summarize(results, args)
    recommendation = final_recommendation(summaries)
    snr_rows, snr_summary = snr_by_cr_strength(rows)

    write_csv(output_dir / "label_rows.csv", compact_label_rows(rows))
    write_csv(output_dir / "split_results.csv", results)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "snr_by_cr_strength.csv", snr_rows)
    payload = {
        "recommendation": recommendation,
        "args": vars(args),
        "meta": meta,
        "feature_counts": {name: len(values) for name, values in feature_sets.items()},
        "label_counts": {
            "preserve": int(sum(row["label_preserve"] for row in rows)),
            "intervene": int(sum(row["label_intervene"] for row in rows)),
            "ignore": int(sum(row["label_ignore"] for row in rows)),
        },
        "snr_summary": snr_summary,
        "summary": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / "analysis_report.md", summaries, recommendation, meta, snr_summary, args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
