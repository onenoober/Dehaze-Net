import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from metric import psnr, ssim
from model import DEANet


EPS = 1e-8
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HAZE4K')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--lfv1_checkpoint', type=str, required=True)
    parser.add_argument('--residual_checkpoint', type=str, required=True)
    parser.add_argument('--sample_list', type=str, default='')
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pad_size', type=int, default=4)
    parser.add_argument('--lowfreq_pool', type=int, default=8)
    parser.add_argument('--panel_top_k', type=int, default=12)
    parser.add_argument('--save_outputs', action='store_true')
    parser.add_argument('--save_all_panels', action='store_true')
    parser.add_argument('--save_heatmaps', action='store_true')
    parser.add_argument('--lf_prior_channels', type=int, default=8)
    parser.add_argument('--lf_prior_pool', type=int, default=8)
    parser.add_argument('--lf_prior_gate_init', type=float, default=0.0)
    parser.add_argument('--lf_prior_residual_center', action='store_true')
    parser.add_argument('--lf_prior_gate_max', type=float, default=0.0)
    parser.add_argument('--lf_prior_injection', type=str, default='pre_mix', choices=['pre_mix', 'post_mix'])
    parser.add_argument('--lf_calib_hidden_channels', type=int, default=8)
    parser.add_argument('--lf_calib_alpha_max', type=float, default=1.0)
    return parser.parse_args()


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_model(checkpoint_path, args, use_lf_prior=False, residual_calibration=False):
    model = DEANet(
        base_dim=32,
        use_lf_prior=use_lf_prior,
        lf_prior_channels=args.lf_prior_channels,
        lf_prior_pool=args.lf_prior_pool,
        lf_prior_gate_init=args.lf_prior_gate_init,
        lf_prior_residual_center=args.lf_prior_residual_center,
        lf_prior_gate_max=args.lf_prior_gate_max,
        lf_prior_injection=args.lf_prior_injection,
        lf_residual_calibration=residual_calibration,
        lf_calib_hidden_channels=args.lf_calib_hidden_channels,
        lf_calib_alpha_max=args.lf_calib_alpha_max,
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    return model, checkpoint


def list_image_files(path):
    return sorted(
        item.name for item in Path(path).iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def resolve_first_existing_dir(root, names, role):
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        'No {} directory found under {}. Tried: {}'.format(
            role, root, ', '.join(str(root / name) for name in names)
        )
    )


def resolve_pair_dirs(split_path):
    split_path = Path(split_path)
    hazy_path = resolve_first_existing_dir(split_path, ('hazy', 'haze'), 'hazy')
    clear_path = resolve_first_existing_dir(split_path, ('clear', 'gt', 'GT'), 'clear')
    return str(hazy_path), str(clear_path)


def find_clear_image(clear_path, hazy_image_name):
    hazy_stem = Path(hazy_image_name).stem
    stems = [hazy_stem]
    prefix_stem = hazy_stem.split('_')[0]
    if prefix_stem not in stems:
        stems.append(prefix_stem)

    for stem in stems:
        for suffix in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
            candidate = Path(clear_path) / f'{stem}{suffix}'
            if candidate.exists():
                return str(candidate)
    raise FileNotFoundError(f'No clear image found for {hazy_image_name} in {clear_path}')


def lf_gate_value(model):
    lf_prior = getattr(model, 'lf_prior', None)
    if lf_prior is None:
        return None
    return float(lf_prior.gate.detach().cpu().item())


def infer_one(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    pred = model(padded).clamp(0, 1)
    return pred[:, :, :h, :w]


def tensor_to_pil(x):
    x = x.detach().clamp(0, 1).squeeze(0).cpu()
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    return Image.fromarray(x)


def tensor_to_np(x):
    x = x.detach().clamp(0, 1).squeeze(0).cpu()
    return x.permute(1, 2, 0).numpy().astype(np.float32)


def pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def metric_value(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def psnr_np(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def luminance(img):
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def lab_image(img):
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)


def hsv_image(img):
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)


def dark_channel(img, kernel_size=15):
    min_rgb = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(min_rgb, kernel)


def edge_map(img):
    y = luminance(img).astype(np.float32)
    sobel_x = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)


def image_quality(pred, gt):
    diff = pred - gt
    abs_diff = np.abs(diff)
    pred_lab = lab_image(pred)
    gt_lab = lab_image(gt)
    delta_e = np.linalg.norm(pred_lab - gt_lab, axis=2)
    pred_y = luminance(pred)
    gt_y = luminance(gt)
    pred_hsv = hsv_image(pred)
    gt_hsv = hsv_image(gt)
    pred_dc = dark_channel(pred)
    gt_dc = dark_channel(gt)
    pred_edge = edge_map(pred)
    gt_edge = edge_map(gt)
    pred_lap = cv2.Laplacian(pred_y.astype(np.float32), cv2.CV_32F, ksize=3)
    gt_lap = cv2.Laplacian(gt_y.astype(np.float32), cv2.CV_32F, ksize=3)
    pred_high = pred_y - cv2.GaussianBlur(pred_y, (0, 0), 3.0)
    gt_high = gt_y - cv2.GaussianBlur(gt_y, (0, 0), 3.0)
    return {
        'np_psnr': psnr_np(pred, gt),
        'mae': float(np.mean(abs_diff)),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'delta_e_mean': float(np.mean(delta_e)),
        'delta_e_p95': float(np.percentile(delta_e, 95)),
        'luma_mean': float(np.mean(pred_y)),
        'luma_bias': float(np.mean(pred_y) - np.mean(gt_y)),
        'luma_abs_bias': float(abs(np.mean(pred_y) - np.mean(gt_y))),
        'luma_std': float(np.std(pred_y)),
        'luma_std_delta': float(np.std(pred_y) - np.std(gt_y)),
        'saturation_mean': float(np.mean(pred_hsv[:, :, 1])),
        'saturation_bias': float(np.mean(pred_hsv[:, :, 1]) - np.mean(gt_hsv[:, :, 1])),
        'saturation_abs_bias': float(abs(np.mean(pred_hsv[:, :, 1]) - np.mean(gt_hsv[:, :, 1]))),
        'dark_channel_mean': float(np.mean(pred_dc)),
        'dark_channel_abs_bias': float(abs(np.mean(pred_dc) - np.mean(gt_dc))),
        'edge_mean': float(np.mean(pred_edge)),
        'edge_error': float(np.mean(np.abs(pred_edge - gt_edge))),
        'lap_var': float(np.var(pred_lap)),
        'lap_var_ratio_gt': float(np.var(pred_lap) / max(float(np.var(gt_lap)), EPS)),
        'highfreq_abs_mean': float(np.mean(np.abs(pred_high))),
        'highfreq_abs_error': float(abs(np.mean(np.abs(pred_high)) - np.mean(np.abs(gt_high)))),
    }


def prefix(prefix_name, metrics):
    return {f'{prefix_name}_{key}': value for key, value in metrics.items()}


def parse_haze4k_name(filename):
    stem = Path(filename).stem
    parts = stem.split('_')
    meta = {
        'image_id': parts[0] if parts else stem,
        'airlight': '',
        'beta': '',
    }
    if len(parts) >= 3:
        try:
            meta['airlight'] = float(parts[1])
            meta['beta'] = float(parts[2])
        except ValueError:
            pass
    return meta


def lowpass(x, pool_size):
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def rgb_to_luma(x):
    return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114


def scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def cosine(a, b):
    a_vec = a.reshape(a.shape[0], -1)
    b_vec = b.reshape(b.shape[0], -1)
    numerator = (a_vec * b_vec).sum(dim=1)
    denominator = a_vec.norm(dim=1) * b_vec.norm(dim=1) + EPS
    return scalar((numerator / denominator).mean())


def l2_norm(x):
    return scalar(x.reshape(x.shape[0], -1).norm(dim=1).mean())


def mse(a, b):
    return scalar(F.mse_loss(a, b))


def safe_ratio(numerator, denominator):
    if denominator <= EPS:
        return ''
    return float(numerator / denominator)


def residual_pair_metrics(ref_low, current_low, clear_low, prefix_name):
    target = clear_low - ref_low
    current = current_low - ref_low
    error = current - target
    target_luma = rgb_to_luma(clear_low) - rgb_to_luma(ref_low)
    current_luma = rgb_to_luma(current_low) - rgb_to_luma(ref_low)
    target_norm = l2_norm(target)
    current_norm = l2_norm(current)
    error_norm = l2_norm(error)
    current_mse = mse(current_low, clear_low)
    ref_mse = mse(ref_low, clear_low)
    return {
        f'{prefix_name}_residual_cosine': cosine(current, target),
        f'{prefix_name}_luma_residual_cosine': cosine(current_luma, target_luma),
        f'{prefix_name}_residual_norm_ratio': safe_ratio(current_norm, target_norm),
        f'{prefix_name}_residual_error_ratio': safe_ratio(error_norm, target_norm),
        f'{prefix_name}_lf_mse_delta': current_mse - ref_mse,
    }


def choose_names(hazy_names, sample_list, max_images):
    if sample_list:
        with open(sample_list, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        missing = [name for name in names if name not in hazy_names]
        if missing:
            raise FileNotFoundError('Samples not found: {}'.format(', '.join(missing)))
        return names
    if max_images > 0:
        return hazy_names[:max_images]
    return hazy_names


def numeric(values):
    return [float(value) for value in values if value != '' and value is not None]


def mean(values):
    values = numeric(values)
    if not values:
        return ''
    return float(np.mean(values))


def median(values):
    values = numeric(values)
    if not values:
        return ''
    return float(np.median(values))


def percentile(values, q):
    values = numeric(values)
    if not values:
        return ''
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x != '' and y != '' and x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_vals = [float(x) for x, _ in pairs]
    y_vals = [float(y) for _, y in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    x_var = sum((x - x_mean) ** 2 for x in x_vals)
    y_var = sum((y - y_mean) ** 2 for y in y_vals)
    if x_var <= EPS or y_var <= EPS:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    return cov / math.sqrt(x_var * y_var)


def make_bin(value, edges, labels):
    if value == '':
        return 'unknown'
    value = float(value)
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def add_bins(rows):
    a_edges = [0.65, 0.75, 0.85, 0.95]
    a_labels = ['A<0.65', '0.65<=A<0.75', '0.75<=A<0.85', '0.85<=A<0.95', 'A>=0.95']
    beta_edges = [0.8, 1.1, 1.4, 1.7]
    beta_labels = ['beta<0.8', '0.8<=beta<1.1', '1.1<=beta<1.4', '1.4<=beta<1.7', 'beta>=1.7']
    baseline_sorted = sorted(rows, key=lambda row: row['baseline_psnr'])
    weak_cutoff = baseline_sorted[max(0, len(rows) // 4 - 1)]['baseline_psnr']
    strong_cutoff = baseline_sorted[min(len(rows) - 1, (len(rows) * 3) // 4)]['baseline_psnr']
    for row in rows:
        row['airlight_bin'] = make_bin(row['airlight'], a_edges, a_labels)
        row['beta_bin'] = make_bin(row['beta'], beta_edges, beta_labels)
        if row['baseline_psnr'] <= weak_cutoff:
            row['baseline_strength_bin'] = 'baseline_weakest_25'
        elif row['baseline_psnr'] >= strong_cutoff:
            row['baseline_strength_bin'] = 'baseline_strongest_25'
        else:
            row['baseline_strength_bin'] = 'baseline_middle_50'
        row['lfv1_relation'] = relation_label(row['lfv1_delta_baseline_psnr'])
        row['residual_relation'] = relation_label(row['residual_delta_baseline_psnr'])
        row['residual_vs_lfv1_relation'] = relation_label(row['residual_delta_lfv1_psnr'])
        row['winner_by_psnr'] = winner_by_psnr(row)
        row['pattern'] = pattern_label(row)
    return weak_cutoff, strong_cutoff


def relation_label(delta):
    if delta >= 0.30:
        return 'gain>=0.30'
    if delta <= -0.30:
        return 'regress<=-0.30'
    if delta > 0:
        return 'small_gain'
    if delta < 0:
        return 'small_regress'
    return 'tie'


def winner_by_psnr(row):
    vals = {
        'baseline': row['baseline_psnr'],
        'lfv1': row['lfv1_psnr'],
        'residual': row['residual_psnr'],
    }
    return max(vals, key=vals.get)


def pattern_label(row):
    lf_b = row['lfv1_delta_baseline_psnr']
    res_b = row['residual_delta_baseline_psnr']
    res_lf = row['residual_delta_lfv1_psnr']
    if res_b >= 0.30 and res_lf >= 0.30:
        return 'residual_beats_both'
    if lf_b >= 0.30 and res_lf <= -0.30:
        return 'lost_lfv1_gain'
    if lf_b <= -0.30 and res_lf >= 0.30:
        return 'mitigates_lfv1_regression'
    if res_b <= -0.30 and res_lf <= -0.30:
        return 'residual_worst'
    if lf_b >= 0.30 and res_b >= 0.30:
        return 'both_improve_baseline'
    if lf_b <= -0.30 and res_b <= -0.30:
        return 'both_regress_baseline'
    return 'mixed_or_small'


def group_stats(group_name, group_value, subset):
    return {
        'group_name': group_name,
        'group_value': group_value,
        'num_images': len(subset),
        'mean_baseline_psnr': mean([row['baseline_psnr'] for row in subset]),
        'mean_lfv1_psnr': mean([row['lfv1_psnr'] for row in subset]),
        'mean_residual_psnr': mean([row['residual_psnr'] for row in subset]),
        'mean_lfv1_delta_baseline_psnr': mean([row['lfv1_delta_baseline_psnr'] for row in subset]),
        'mean_residual_delta_baseline_psnr': mean([row['residual_delta_baseline_psnr'] for row in subset]),
        'mean_residual_delta_lfv1_psnr': mean([row['residual_delta_lfv1_psnr'] for row in subset]),
        'median_residual_delta_lfv1_psnr': median([row['residual_delta_lfv1_psnr'] for row in subset]),
        'lfv1_better_030db_count': sum(1 for row in subset if row['lfv1_delta_baseline_psnr'] >= 0.30),
        'lfv1_worse_030db_count': sum(1 for row in subset if row['lfv1_delta_baseline_psnr'] <= -0.30),
        'residual_better_030db_count': sum(1 for row in subset if row['residual_delta_baseline_psnr'] >= 0.30),
        'residual_worse_030db_count': sum(1 for row in subset if row['residual_delta_baseline_psnr'] <= -0.30),
        'residual_over_lfv1_030db_count': sum(1 for row in subset if row['residual_delta_lfv1_psnr'] >= 0.30),
        'residual_under_lfv1_030db_count': sum(1 for row in subset if row['residual_delta_lfv1_psnr'] <= -0.30),
        'mean_lfv1_residual_cosine': mean([row['lfv1_from_baseline_residual_cosine'] for row in subset]),
        'mean_rescalib_residual_cosine': mean([row['rescalib_from_baseline_residual_cosine'] for row in subset]),
        'mean_residual_vs_lfv1_cosine': mean([row['rescalib_from_lfv1_residual_cosine'] for row in subset]),
        'mean_delta_e_change_residual_vs_lfv1': mean([row['residual_delta_e_mean'] - row['lfv1_delta_e_mean'] for row in subset]),
        'mean_luma_abs_bias_change_residual_vs_lfv1': mean([row['residual_luma_abs_bias'] - row['lfv1_luma_abs_bias'] for row in subset]),
        'mean_dark_channel_abs_bias_change_residual_vs_lfv1': mean([row['residual_dark_channel_abs_bias'] - row['lfv1_dark_channel_abs_bias'] for row in subset]),
        'mean_edge_error_change_residual_vs_lfv1': mean([row['residual_edge_error'] - row['lfv1_edge_error'] for row in subset]),
    }


def build_group_rows(rows):
    groups = [group_stats('all', 'all', rows)]
    keys = [
        'baseline_strength_bin',
        'airlight_bin',
        'beta_bin',
        'lfv1_relation',
        'residual_relation',
        'residual_vs_lfv1_relation',
        'winner_by_psnr',
        'pattern',
    ]
    for key in keys:
        for value in sorted(set(row[key] for row in rows)):
            subset = [row for row in rows if row[key] == value]
            groups.append(group_stats(key, value, subset))
    return groups


def compact(row):
    keys = [
        'filename', 'airlight', 'beta', 'baseline_strength_bin', 'pattern',
        'baseline_psnr', 'lfv1_psnr', 'residual_psnr',
        'lfv1_delta_baseline_psnr', 'residual_delta_baseline_psnr',
        'residual_delta_lfv1_psnr',
        'lfv1_from_baseline_residual_cosine',
        'rescalib_from_baseline_residual_cosine',
        'rescalib_from_lfv1_residual_cosine',
        'rescalib_from_baseline_residual_norm_ratio',
        'rescalib_from_baseline_lf_mse_delta',
    ]
    return {key: row[key] for key in keys}


def top(rows, key, top_k, reverse=True, predicate=None):
    subset = [row for row in rows if predicate is None or predicate(row)]
    return [compact(row) for row in sorted(subset, key=lambda row: row[key], reverse=reverse)[:top_k]]


def build_hard_cases(rows, top_k):
    return {
        'residual_beats_both': top(rows, 'residual_delta_lfv1_psnr', top_k, True, lambda row: row['pattern'] == 'residual_beats_both'),
        'lost_lfv1_gain': top(rows, 'residual_delta_lfv1_psnr', top_k, False, lambda row: row['pattern'] == 'lost_lfv1_gain'),
        'mitigates_lfv1_regression': top(rows, 'residual_delta_lfv1_psnr', top_k, True, lambda row: row['pattern'] == 'mitigates_lfv1_regression'),
        'residual_worst': top(rows, 'residual_delta_baseline_psnr', top_k, False, lambda row: row['pattern'] == 'residual_worst'),
        'strong_baseline_residual_regressions': top(
            rows,
            'residual_delta_baseline_psnr',
            top_k,
            False,
            lambda row: row['baseline_strength_bin'] == 'baseline_strongest_25' and row['residual_delta_baseline_psnr'] <= -0.30,
        ),
        'weak_baseline_residual_gains': top(
            rows,
            'residual_delta_baseline_psnr',
            top_k,
            True,
            lambda row: row['baseline_strength_bin'] == 'baseline_weakest_25' and row['residual_delta_baseline_psnr'] >= 0.30,
        ),
        'rescalib_wrong_direction': top(
            rows,
            'rescalib_from_baseline_residual_cosine',
            top_k,
            False,
            lambda row: row['rescalib_from_baseline_residual_cosine'] < 0,
        ),
        'largest_color_regression_vs_lfv1': top(rows, 'residual_color_regression_vs_lfv1', top_k, True),
        'largest_tone_regression_vs_lfv1': top(rows, 'residual_luma_abs_bias_regression_vs_lfv1', top_k, True),
        'largest_dark_channel_regression_vs_lfv1': top(rows, 'residual_dark_channel_abs_bias_regression_vs_lfv1', top_k, True),
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_uint8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def normalize_map(values, percentile_value=98.0):
    vmax = float(np.percentile(values, percentile_value))
    if vmax <= EPS:
        vmax = float(np.max(values))
    if vmax <= EPS:
        return np.zeros_like(values, dtype=np.uint8)
    return np.clip(values / vmax * 255.0, 0, 255).astype(np.uint8)


def error_heatmap(img, gt):
    err = np.mean(np.abs(img - gt), axis=2)
    norm = normalize_map(err)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def improvement_heatmap(ref, current, gt):
    ref_err = np.mean(np.abs(ref - gt), axis=2)
    cur_err = np.mean(np.abs(current - gt), axis=2)
    delta = ref_err - cur_err
    scale = float(np.percentile(np.abs(delta), 98))
    if scale <= EPS:
        scale = float(np.max(np.abs(delta)))
    if scale <= EPS:
        return np.zeros((*delta.shape, 3), dtype=np.uint8)
    mag = np.clip(np.abs(delta) / scale, 0, 1)
    rgb = np.zeros((*delta.shape, 3), dtype=np.float32)
    rgb[:, :, 1] = mag * (delta > 0)
    rgb[:, :, 0] = mag * (delta < 0)
    rgb[:, :, 2] = 0.15 * mag
    return (rgb * 255.0).astype(np.uint8)


def draw_panel(images, labels, metrics):
    label_h = 58
    gap = 8
    widths, heights = zip(*(img.size for img in images))
    total_w = sum(widths) + gap * (len(images) - 1)
    total_h = max(heights) + label_h
    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)
    x = 0
    for img, label, metric in zip(images, labels, metrics):
        canvas.paste(img, (x, label_h))
        draw.text((x + 4, 5), label, fill=(0, 0, 0))
        if metric:
            draw.text((x + 4, 25), metric, fill=(0, 0, 0))
        x += img.size[0] + gap
    return canvas


def save_panel(row, arrays, panels_dir, heatmap_dir=None):
    stem = Path(row['filename']).stem
    images = [
        Image.fromarray(to_uint8(arrays['input'])),
        Image.fromarray(to_uint8(arrays['baseline'])),
        Image.fromarray(to_uint8(arrays['lfv1'])),
        Image.fromarray(to_uint8(arrays['residual'])),
        Image.fromarray(to_uint8(arrays['gt'])),
    ]
    metrics = [
        '',
        'PSNR {:.3f} SSIM {:.4f}'.format(row['baseline_psnr'], row['baseline_ssim']),
        'PSNR {:.3f} dB {:.3f}'.format(row['lfv1_psnr'], row['lfv1_delta_baseline_psnr']),
        'PSNR {:.3f} dB-LF {:.3f}'.format(row['residual_psnr'], row['residual_delta_lfv1_psnr']),
        '',
    ]
    panel = draw_panel(
        images,
        ['hazy input', 'DEA-Net-CR', 'LF-v1', 'ResidualCalib', 'clear GT'],
        metrics,
    )
    panel.save(panels_dir / f'{stem}.png')
    if heatmap_dir is not None:
        Image.fromarray(error_heatmap(arrays['baseline'], arrays['gt'])).save(heatmap_dir / f'{stem}_baseline_error.png')
        Image.fromarray(error_heatmap(arrays['lfv1'], arrays['gt'])).save(heatmap_dir / f'{stem}_lfv1_error.png')
        Image.fromarray(error_heatmap(arrays['residual'], arrays['gt'])).save(heatmap_dir / f'{stem}_residual_error.png')
        Image.fromarray(improvement_heatmap(arrays['lfv1'], arrays['residual'], arrays['gt'])).save(heatmap_dir / f'{stem}_residual_vs_lfv1.png')
        Image.fromarray(improvement_heatmap(arrays['baseline'], arrays['residual'], arrays['gt'])).save(heatmap_dir / f'{stem}_residual_vs_baseline.png')


def save_outputs(row, arrays, output_dir):
    stem = Path(row['filename']).stem
    for key in ('input', 'baseline', 'lfv1', 'residual', 'gt'):
        path = output_dir / key
        path.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_uint8(arrays[key])).save(path / f'{stem}.png')


def fmt(value, digits=4):
    if value == '' or value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}'


def markdown_table(rows, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, '')
            vals.append(fmt(val, 4) if isinstance(val, float) else str(val))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def write_report(path, summary, group_rows, hard_cases):
    lines = [
        '# Three-Way HAZE4K Output Analysis',
        '',
        '## Summary',
        '',
        f"- Images: `{summary['num_images']}`",
        f"- Baseline PSNR/SSIM: `{summary['mean_baseline_psnr']:.4f}` / `{summary['mean_baseline_ssim']:.6f}`",
        f"- LF-v1 PSNR/SSIM: `{summary['mean_lfv1_psnr']:.4f}` / `{summary['mean_lfv1_ssim']:.6f}`",
        f"- ResidualCalib PSNR/SSIM: `{summary['mean_residual_psnr']:.4f}` / `{summary['mean_residual_ssim']:.6f}`",
        f"- LF-v1 vs baseline mean delta: `{summary['mean_lfv1_delta_baseline_psnr']:.4f}` dB",
        f"- ResidualCalib vs baseline mean delta: `{summary['mean_residual_delta_baseline_psnr']:.4f}` dB",
        f"- ResidualCalib vs LF-v1 mean delta: `{summary['mean_residual_delta_lfv1_psnr']:.4f}` dB",
        f"- PSNR winner counts: `{summary['winner_counts']}`",
        f"- Pattern counts: `{summary['pattern_counts']}`",
        f"- ResidualCalib wrong-direction vs baseline: `{summary['rescalib_wrong_direction_count']}`",
        f"- LF-v1 wrong-direction vs baseline: `{summary['lfv1_wrong_direction_count']}`",
        '',
        '## Key Group Rows',
        '',
    ]
    highlight_groups = [
        ('baseline_strength_bin', 'baseline_weakest_25'),
        ('baseline_strength_bin', 'baseline_middle_50'),
        ('baseline_strength_bin', 'baseline_strongest_25'),
        ('pattern', 'residual_beats_both'),
        ('pattern', 'lost_lfv1_gain'),
        ('pattern', 'mitigates_lfv1_regression'),
        ('pattern', 'residual_worst'),
    ]
    for group_name, group_value in highlight_groups:
        match = next((row for row in group_rows if row['group_name'] == group_name and row['group_value'] == group_value), None)
        if not match:
            continue
        lines.append(
            f"- `{group_name}={group_value}` n=`{match['num_images']}` "
            f"LF-v1 delta `{fmt(match['mean_lfv1_delta_baseline_psnr'])}`, "
            f"Residual delta baseline `{fmt(match['mean_residual_delta_baseline_psnr'])}`, "
            f"Residual delta LF-v1 `{fmt(match['mean_residual_delta_lfv1_psnr'])}`"
        )
    lines.extend(['', '## Hard Case Lists', ''])
    cols = [
        'filename', 'pattern', 'baseline_strength_bin', 'baseline_psnr',
        'lfv1_delta_baseline_psnr', 'residual_delta_baseline_psnr',
        'residual_delta_lfv1_psnr', 'rescalib_from_baseline_residual_cosine',
    ]
    for title, rows in hard_cases.items():
        lines.extend([f'### {title}', ''])
        lines.append(markdown_table(rows[:10], cols))
        lines.append('')
    lines.extend([
        '## Reading Notes',
        '',
        '- `residual_delta_lfv1_psnr > 0` means ResidualCalib beats LF-v1 on that image.',
        '- `lost_lfv1_gain` means LF-v1 has a meaningful gain over baseline but ResidualCalib gives a meaningful loss relative to LF-v1.',
        '- `mitigates_lfv1_regression` means LF-v1 hurts baseline and ResidualCalib recovers at least part of that loss.',
        '- Heatmaps use green where the later model reduces absolute error and red where it increases absolute error.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    if args.lowfreq_pool <= 0:
        raise ValueError('--lowfreq_pool must be positive')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_root = output_dir / 'panels'
    heatmap_root = output_dir / 'heatmaps' if args.save_heatmaps else None
    panels_root.mkdir(parents=True, exist_ok=True)
    if heatmap_root is not None:
        heatmap_root.mkdir(parents=True, exist_ok=True)

    split_root = Path('../dataset') / args.dataset / args.split
    hazy_dir, clear_dir = resolve_pair_dirs(split_root)
    hazy_names = list_image_files(hazy_dir)
    sample_names = choose_names(hazy_names, args.sample_list, args.max_images)

    baseline_model, baseline_ckpt = load_model(args.baseline_checkpoint, args, use_lf_prior=False)
    lfv1_model, lfv1_ckpt = load_model(args.lfv1_checkpoint, args, use_lf_prior=True)
    residual_model, residual_ckpt = load_model(
        args.residual_checkpoint,
        args,
        use_lf_prior=True,
        residual_calibration=True,
    )

    rows = []
    arrays_by_name = {}
    with torch.no_grad():
        for idx, name in enumerate(sample_names, start=1):
            hazy_path = Path(hazy_dir) / name
            clear_path = Path(find_clear_image(clear_dir, name))
            hazy_img = Image.open(hazy_path).convert('RGB')
            clear_img = Image.open(clear_path).convert('RGB')
            hazy = pil_to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = pil_to_tensor(clear_img).unsqueeze(0).to(args.device)

            baseline = infer_one(baseline_model, hazy, args.pad_size)
            lfv1 = infer_one(lfv1_model, hazy, args.pad_size)
            residual = infer_one(residual_model, hazy, args.pad_size)

            baseline_np = tensor_to_np(baseline)
            lfv1_np = tensor_to_np(lfv1)
            residual_np = tensor_to_np(residual)
            gt_np = tensor_to_np(clear)
            input_np = tensor_to_np(hazy)

            baseline_psnr = metric_value(psnr(baseline, clear))
            baseline_ssim = metric_value(ssim(baseline, clear))
            lfv1_psnr = metric_value(psnr(lfv1, clear))
            lfv1_ssim = metric_value(ssim(lfv1, clear))
            residual_psnr = metric_value(psnr(residual, clear))
            residual_ssim = metric_value(ssim(residual, clear))

            baseline_low = lowpass(baseline, args.lowfreq_pool)
            lfv1_low = lowpass(lfv1, args.lowfreq_pool)
            residual_low = lowpass(residual, args.lowfreq_pool)
            clear_low = lowpass(clear, args.lowfreq_pool)

            meta = parse_haze4k_name(name)
            row = {
                'index': idx,
                'filename': name,
                'image_id': meta['image_id'],
                'airlight': meta['airlight'],
                'beta': meta['beta'],
                'height': hazy.shape[-2],
                'width': hazy.shape[-1],
                'input_psnr': metric_value(psnr(hazy, clear)),
                'input_ssim': metric_value(ssim(hazy, clear)),
                'baseline_psnr': baseline_psnr,
                'baseline_ssim': baseline_ssim,
                'lfv1_psnr': lfv1_psnr,
                'lfv1_ssim': lfv1_ssim,
                'residual_psnr': residual_psnr,
                'residual_ssim': residual_ssim,
                'lfv1_delta_baseline_psnr': lfv1_psnr - baseline_psnr,
                'lfv1_delta_baseline_ssim': lfv1_ssim - baseline_ssim,
                'residual_delta_baseline_psnr': residual_psnr - baseline_psnr,
                'residual_delta_baseline_ssim': residual_ssim - baseline_ssim,
                'residual_delta_lfv1_psnr': residual_psnr - lfv1_psnr,
                'residual_delta_lfv1_ssim': residual_ssim - lfv1_ssim,
            }
            row.update(prefix('baseline', image_quality(baseline_np, gt_np)))
            row.update(prefix('lfv1', image_quality(lfv1_np, gt_np)))
            row.update(prefix('residual', image_quality(residual_np, gt_np)))
            row.update(residual_pair_metrics(baseline_low, lfv1_low, clear_low, 'lfv1_from_baseline'))
            row.update(residual_pair_metrics(baseline_low, residual_low, clear_low, 'rescalib_from_baseline'))
            row.update(residual_pair_metrics(lfv1_low, residual_low, clear_low, 'rescalib_from_lfv1'))
            row['residual_color_regression_vs_lfv1'] = row['residual_delta_e_mean'] - row['lfv1_delta_e_mean']
            row['residual_luma_abs_bias_regression_vs_lfv1'] = row['residual_luma_abs_bias'] - row['lfv1_luma_abs_bias']
            row['residual_dark_channel_abs_bias_regression_vs_lfv1'] = (
                row['residual_dark_channel_abs_bias'] - row['lfv1_dark_channel_abs_bias']
            )
            row['residual_edge_error_regression_vs_lfv1'] = row['residual_edge_error'] - row['lfv1_edge_error']
            rows.append(row)

            arrays = {
                'input': input_np,
                'baseline': baseline_np,
                'lfv1': lfv1_np,
                'residual': residual_np,
                'gt': gt_np,
            }
            arrays_by_name[name] = arrays
            if args.save_outputs:
                save_outputs(row, arrays, output_dir)

            if idx % 50 == 0 or idx == len(sample_names):
                print('analyzed {}/{}'.format(idx, len(sample_names)), flush=True)

    weak_cutoff, strong_cutoff = add_bins(rows)
    group_rows = build_group_rows(rows)
    hard_cases = build_hard_cases(rows, args.panel_top_k)

    selected = set()
    if args.save_all_panels:
        selected.update(row['filename'] for row in rows)
    else:
        for case_rows in hard_cases.values():
            selected.update(row['filename'] for row in case_rows[:args.panel_top_k])

    for category, case_rows in hard_cases.items():
        category_dir = panels_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        category_heatmap_dir = None
        if heatmap_root is not None:
            category_heatmap_dir = heatmap_root / category
            category_heatmap_dir.mkdir(parents=True, exist_ok=True)
        for compact_row in case_rows[:args.panel_top_k]:
            name = compact_row['filename']
            full_row = next(row for row in rows if row['filename'] == name)
            save_panel(full_row, arrays_by_name[name], category_dir, category_heatmap_dir)

    if args.save_all_panels:
        all_dir = panels_root / 'all'
        all_dir.mkdir(parents=True, exist_ok=True)
        all_heatmap_dir = None
        if heatmap_root is not None:
            all_heatmap_dir = heatmap_root / 'all'
            all_heatmap_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            save_panel(row, arrays_by_name[row['filename']], all_dir, all_heatmap_dir)

    winner_counts = Counter(row['winner_by_psnr'] for row in rows)
    pattern_counts = Counter(row['pattern'] for row in rows)
    baseline_weak_rows = [row for row in rows if row['baseline_strength_bin'] == 'baseline_weakest_25']
    baseline_strong_rows = [row for row in rows if row['baseline_strength_bin'] == 'baseline_strongest_25']
    summary = {
        'dataset': args.dataset,
        'split': args.split,
        'num_images': len(rows),
        'sample_list': args.sample_list,
        'baseline_checkpoint': args.baseline_checkpoint,
        'lfv1_checkpoint': args.lfv1_checkpoint,
        'residual_checkpoint': args.residual_checkpoint,
        'baseline_checkpoint_step': baseline_ckpt.get('step'),
        'lfv1_checkpoint_step': lfv1_ckpt.get('step'),
        'residual_checkpoint_step': residual_ckpt.get('step'),
        'baseline_checkpoint_max_psnr': baseline_ckpt.get('max_psnr'),
        'lfv1_checkpoint_max_psnr': lfv1_ckpt.get('max_psnr'),
        'residual_checkpoint_max_psnr': residual_ckpt.get('max_psnr'),
        'lfv1_gate': lf_gate_value(lfv1_model),
        'residual_gate': lf_gate_value(residual_model),
        'weak_baseline_cutoff_psnr': weak_cutoff,
        'strong_baseline_cutoff_psnr': strong_cutoff,
        'mean_baseline_psnr': mean([row['baseline_psnr'] for row in rows]),
        'mean_baseline_ssim': mean([row['baseline_ssim'] for row in rows]),
        'mean_lfv1_psnr': mean([row['lfv1_psnr'] for row in rows]),
        'mean_lfv1_ssim': mean([row['lfv1_ssim'] for row in rows]),
        'mean_residual_psnr': mean([row['residual_psnr'] for row in rows]),
        'mean_residual_ssim': mean([row['residual_ssim'] for row in rows]),
        'mean_lfv1_delta_baseline_psnr': mean([row['lfv1_delta_baseline_psnr'] for row in rows]),
        'median_lfv1_delta_baseline_psnr': median([row['lfv1_delta_baseline_psnr'] for row in rows]),
        'mean_residual_delta_baseline_psnr': mean([row['residual_delta_baseline_psnr'] for row in rows]),
        'median_residual_delta_baseline_psnr': median([row['residual_delta_baseline_psnr'] for row in rows]),
        'mean_residual_delta_lfv1_psnr': mean([row['residual_delta_lfv1_psnr'] for row in rows]),
        'median_residual_delta_lfv1_psnr': median([row['residual_delta_lfv1_psnr'] for row in rows]),
        'p10_residual_delta_lfv1_psnr': percentile([row['residual_delta_lfv1_psnr'] for row in rows], 10),
        'p90_residual_delta_lfv1_psnr': percentile([row['residual_delta_lfv1_psnr'] for row in rows], 90),
        'lfv1_better_baseline_030db_count': sum(1 for row in rows if row['lfv1_delta_baseline_psnr'] >= 0.30),
        'lfv1_worse_baseline_030db_count': sum(1 for row in rows if row['lfv1_delta_baseline_psnr'] <= -0.30),
        'residual_better_baseline_030db_count': sum(1 for row in rows if row['residual_delta_baseline_psnr'] >= 0.30),
        'residual_worse_baseline_030db_count': sum(1 for row in rows if row['residual_delta_baseline_psnr'] <= -0.30),
        'residual_better_lfv1_030db_count': sum(1 for row in rows if row['residual_delta_lfv1_psnr'] >= 0.30),
        'residual_worse_lfv1_030db_count': sum(1 for row in rows if row['residual_delta_lfv1_psnr'] <= -0.30),
        'winner_counts': dict(sorted(winner_counts.items())),
        'pattern_counts': dict(sorted(pattern_counts.items())),
        'weak_baseline_mean_lfv1_delta': mean([row['lfv1_delta_baseline_psnr'] for row in baseline_weak_rows]),
        'weak_baseline_mean_residual_delta': mean([row['residual_delta_baseline_psnr'] for row in baseline_weak_rows]),
        'weak_baseline_mean_residual_delta_lfv1': mean([row['residual_delta_lfv1_psnr'] for row in baseline_weak_rows]),
        'strong_baseline_mean_lfv1_delta': mean([row['lfv1_delta_baseline_psnr'] for row in baseline_strong_rows]),
        'strong_baseline_mean_residual_delta': mean([row['residual_delta_baseline_psnr'] for row in baseline_strong_rows]),
        'strong_baseline_mean_residual_delta_lfv1': mean([row['residual_delta_lfv1_psnr'] for row in baseline_strong_rows]),
        'lfv1_wrong_direction_count': sum(1 for row in rows if row['lfv1_from_baseline_residual_cosine'] < 0),
        'rescalib_wrong_direction_count': sum(1 for row in rows if row['rescalib_from_baseline_residual_cosine'] < 0),
        'rescalib_vs_lfv1_wrong_direction_count': sum(1 for row in rows if row['rescalib_from_lfv1_residual_cosine'] < 0),
        'corr_residual_delta_lfv1_psnr_rescalib_from_baseline_cosine': pearson(
            [row['residual_delta_lfv1_psnr'] for row in rows],
            [row['rescalib_from_baseline_residual_cosine'] for row in rows],
        ),
        'corr_residual_delta_baseline_psnr_rescalib_from_baseline_cosine': pearson(
            [row['residual_delta_baseline_psnr'] for row in rows],
            [row['rescalib_from_baseline_residual_cosine'] for row in rows],
        ),
        'mean_residual_color_regression_vs_lfv1': mean([row['residual_color_regression_vs_lfv1'] for row in rows]),
        'mean_residual_luma_abs_bias_regression_vs_lfv1': mean(
            [row['residual_luma_abs_bias_regression_vs_lfv1'] for row in rows]
        ),
        'mean_residual_dark_channel_abs_bias_regression_vs_lfv1': mean(
            [row['residual_dark_channel_abs_bias_regression_vs_lfv1'] for row in rows]
        ),
        'mean_residual_edge_error_regression_vs_lfv1': mean(
            [row['residual_edge_error_regression_vs_lfv1'] for row in rows]
        ),
    }

    write_csv(output_dir / 'per_image_three_way_metrics.csv', rows)
    write_csv(output_dir / 'group_summary.csv', group_rows)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'hard_cases.json', 'w', encoding='utf-8') as f:
        json.dump(hard_cases, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'samples.txt', 'w', encoding='utf-8') as f:
        for name in sample_names:
            f.write(name + '\n')
    write_report(output_dir / 'analysis_report.md', summary, group_rows, hard_cases)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
