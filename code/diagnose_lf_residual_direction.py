import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
    parser.add_argument('--current_checkpoint', type=str, required=True)
    parser.add_argument('--baseline_label', type=str, default='baseline')
    parser.add_argument('--current_label', type=str, default='current')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pad_size', type=int, default=4)
    parser.add_argument('--lowfreq_pool', type=int, default=8)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--baseline_use_lf_prior', action='store_true')
    parser.add_argument('--current_use_lf_prior', action='store_true')
    parser.add_argument('--baseline_lf_conditional_mask', action='store_true')
    parser.add_argument('--baseline_lf_haze_aware_mask', action='store_true')
    parser.add_argument('--baseline_lf_residual_calibration', action='store_true')
    parser.add_argument('--lf_prior_channels', type=int, default=8)
    parser.add_argument('--lf_prior_pool', type=int, default=8)
    parser.add_argument('--lf_prior_gate_init', type=float, default=0.0)
    parser.add_argument('--lf_prior_residual_center', action='store_true')
    parser.add_argument('--lf_prior_gate_max', type=float, default=0.0)
    parser.add_argument('--lf_prior_injection', type=str, default='pre_mix', choices=['pre_mix', 'post_mix'])
    parser.add_argument('--lf_conditional_mask', action='store_true')
    parser.add_argument('--lf_mask_hidden_channels', type=int, default=8)
    parser.add_argument('--lf_mask_init_bias', type=float, default=2.0)
    parser.add_argument('--lf_haze_aware_mask', action='store_true')
    parser.add_argument('--lf_haze_mask_strength', type=float, default=1.0)
    parser.add_argument('--lf_residual_calibration', action='store_true')
    parser.add_argument('--lf_calib_hidden_channels', type=int, default=8)
    parser.add_argument('--lf_calib_alpha_max', type=float, default=1.0)
    parser.add_argument('--baseline_lf_residual_selector', action='store_true')
    parser.add_argument('--lf_residual_selector', action='store_true')
    parser.add_argument('--lf_selector_hidden_channels', type=int, default=8)
    parser.add_argument('--lf_selector_init_bias', type=float, default=2.0)
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


def load_model(checkpoint_path, use_lf_prior, args, conditional_mask=False, haze_aware_mask=False, residual_calibration=False, residual_selector=False):
    model = DEANet(
        base_dim=32,
        use_lf_prior=use_lf_prior,
        lf_prior_channels=args.lf_prior_channels,
        lf_prior_pool=args.lf_prior_pool,
        lf_prior_gate_init=args.lf_prior_gate_init,
        lf_prior_residual_center=args.lf_prior_residual_center,
        lf_prior_gate_max=args.lf_prior_gate_max,
        lf_prior_injection=args.lf_prior_injection,
        lf_conditional_mask=conditional_mask,
        lf_mask_hidden_channels=args.lf_mask_hidden_channels,
        lf_mask_init_bias=args.lf_mask_init_bias,
        lf_haze_aware_mask=haze_aware_mask,
        lf_haze_mask_strength=args.lf_haze_mask_strength,
        lf_residual_calibration=residual_calibration,
        lf_calib_hidden_channels=args.lf_calib_hidden_channels,
        lf_calib_alpha_max=args.lf_calib_alpha_max,
        lf_residual_selector=residual_selector,
        lf_selector_hidden_channels=args.lf_selector_hidden_channels,
        lf_selector_init_bias=args.lf_selector_init_bias
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    return model, checkpoint


def list_image_files(path):
    return sorted(
        name for name in Path(path).iterdir()
        if name.is_file() and name.suffix.lower() in IMAGE_SUFFIXES
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


def pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def infer_one(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    pred = model(padded).clamp(0, 1)
    return pred[:, :, :h, :w]


def lf_gate_value(model):
    module = getattr(model, 'module', model)
    lf_prior = getattr(module, 'lf_prior', None)
    if lf_prior is None:
        return None
    return float(lf_prior.gate.detach().cpu().item())


def parse_haze4k_name(filename):
    stem = Path(filename).stem
    parts = stem.split('_')
    meta = {
        'image_id': parts[0] if parts else stem,
        'airlight': '',
        'beta': ''
    }
    if len(parts) >= 3:
        try:
            meta['airlight'] = float(parts[1])
            meta['beta'] = float(parts[2])
        except ValueError:
            pass
    return meta


def metric_value(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def lowpass(x, pool_size):
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def rgb_to_luma(x):
    return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114


def flatten_channels(x):
    return x.reshape(x.shape[0], x.shape[1], -1)


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


def mae(a, b):
    return scalar(F.l1_loss(a, b))


def sign_agreement(a, b):
    same = torch.sign(a) == torch.sign(b)
    meaningful = (a.abs() > 1.0 / 255.0) | (b.abs() > 1.0 / 255.0)
    if meaningful.sum().item() == 0:
        return ''
    return scalar(same[meaningful].float().mean())


def channel_bias(x):
    means = x.mean(dim=(0, 2, 3))
    return [float(v.detach().cpu().item()) for v in means]


def safe_ratio(numerator, denominator):
    if denominator <= EPS:
        return ''
    return float(numerator / denominator)


def residual_metrics(baseline_low, current_low, clear_low):
    target_residual = clear_low - baseline_low
    current_residual = current_low - baseline_low
    residual_error = current_residual - target_residual
    target_luma_residual = rgb_to_luma(clear_low) - rgb_to_luma(baseline_low)
    current_luma_residual = rgb_to_luma(current_low) - rgb_to_luma(baseline_low)

    target_norm = l2_norm(target_residual)
    current_norm = l2_norm(current_residual)
    error_norm = l2_norm(residual_error)
    target_luma_norm = l2_norm(target_luma_residual)
    current_luma_norm = l2_norm(current_luma_residual)
    rgb_bias = channel_bias(residual_error)

    return {
        'lf_target_residual_norm': target_norm,
        'lf_current_residual_norm': current_norm,
        'lf_residual_error_norm': error_norm,
        'lf_residual_norm_ratio': safe_ratio(current_norm, target_norm),
        'lf_residual_error_ratio': safe_ratio(error_norm, target_norm),
        'lf_residual_cosine': cosine(current_residual, target_residual),
        'lf_residual_sign_agreement': sign_agreement(current_residual, target_residual),
        'lf_luma_target_residual_norm': target_luma_norm,
        'lf_luma_current_residual_norm': current_luma_norm,
        'lf_luma_norm_ratio': safe_ratio(current_luma_norm, target_luma_norm),
        'lf_luma_residual_cosine': cosine(current_luma_residual, target_luma_residual),
        'lf_luma_sign_agreement': sign_agreement(current_luma_residual, target_luma_residual),
        'lf_residual_error_bias_r': rgb_bias[0],
        'lf_residual_error_bias_g': rgb_bias[1],
        'lf_residual_error_bias_b': rgb_bias[2],
        'lf_residual_error_bias_luma': scalar(rgb_to_luma(residual_error).mean()),
        'lf_baseline_mse': mse(baseline_low, clear_low),
        'lf_current_mse': mse(current_low, clear_low),
        'lf_mse_delta': mse(current_low, clear_low) - mse(baseline_low, clear_low),
        'lf_baseline_mae': mae(baseline_low, clear_low),
        'lf_current_mae': mae(current_low, clear_low),
        'lf_mae_delta': mae(current_low, clear_low) - mae(baseline_low, clear_low),
        'lf_luma_baseline_mse': mse(rgb_to_luma(baseline_low), rgb_to_luma(clear_low)),
        'lf_luma_current_mse': mse(rgb_to_luma(current_low), rgb_to_luma(clear_low)),
        'lf_luma_mse_delta': (
            mse(rgb_to_luma(current_low), rgb_to_luma(clear_low)) -
            mse(rgb_to_luma(baseline_low), rgb_to_luma(clear_low))
        ),
    }


def numeric(values):
    return [float(v) for v in values if v != '' and v is not None]


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
    if x_var <= 1e-12 or y_var <= 1e-12:
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


def group_stats(group_name, group_value, rows):
    return {
        'group_name': group_name,
        'group_value': group_value,
        'num_images': len(rows),
        'mean_delta_psnr': mean([row['delta_psnr'] for row in rows]),
        'median_delta_psnr': median([row['delta_psnr'] for row in rows]),
        'better_030db_count': sum(1 for row in rows if row['delta_psnr'] >= 0.30),
        'worse_030db_count': sum(1 for row in rows if row['delta_psnr'] <= -0.30),
        'mean_lf_residual_cosine': mean([row['lf_residual_cosine'] for row in rows]),
        'median_lf_residual_cosine': median([row['lf_residual_cosine'] for row in rows]),
        'mean_lf_luma_residual_cosine': mean([row['lf_luma_residual_cosine'] for row in rows]),
        'mean_lf_residual_norm_ratio': mean([row['lf_residual_norm_ratio'] for row in rows]),
        'median_lf_residual_norm_ratio': median([row['lf_residual_norm_ratio'] for row in rows]),
        'p90_lf_residual_norm_ratio': percentile([row['lf_residual_norm_ratio'] for row in rows], 90),
        'mean_lf_luma_norm_ratio': mean([row['lf_luma_norm_ratio'] for row in rows]),
        'mean_lf_residual_error_ratio': mean([row['lf_residual_error_ratio'] for row in rows]),
        'mean_lf_mse_delta': mean([row['lf_mse_delta'] for row in rows]),
        'mean_lf_luma_mse_delta': mean([row['lf_luma_mse_delta'] for row in rows]),
        'lf_mse_improved_count': sum(1 for row in rows if row['lf_mse_delta'] < 0),
        'lf_mse_regressed_count': sum(1 for row in rows if row['lf_mse_delta'] > 0),
        'lf_luma_mse_improved_count': sum(1 for row in rows if row['lf_luma_mse_delta'] < 0),
        'lf_luma_mse_regressed_count': sum(1 for row in rows if row['lf_luma_mse_delta'] > 0),
        'wrong_direction_count': sum(1 for row in rows if row['lf_residual_cosine'] < 0),
        'possible_overshoot_count': sum(
            1 for row in rows
            if row['lf_residual_cosine'] > 0.5
            and row['lf_residual_norm_ratio'] != ''
            and row['lf_residual_norm_ratio'] > 1.5
        ),
        'possible_under_correction_count': sum(
            1 for row in rows
            if row['lf_residual_cosine'] > 0.5
            and row['lf_residual_norm_ratio'] != ''
            and row['lf_residual_norm_ratio'] < 0.5
        ),
    }


def build_group_rows(rows):
    a_edges = [0.65, 0.75, 0.85, 0.95]
    a_labels = ['A<0.65', '0.65<=A<0.75', '0.75<=A<0.85', '0.85<=A<0.95', 'A>=0.95']
    beta_edges = [0.8, 1.1, 1.4, 1.7]
    beta_labels = ['beta<0.8', '0.8<=beta<1.1', '1.1<=beta<1.4', '1.4<=beta<1.7', 'beta>=1.7']

    for row in rows:
        row['airlight_bin'] = make_bin(row['airlight'], a_edges, a_labels)
        row['beta_bin'] = make_bin(row['beta'], beta_edges, beta_labels)

    baseline_sorted = sorted(rows, key=lambda row: row['baseline_psnr'])
    weak_cutoff = baseline_sorted[max(0, len(rows) // 4 - 1)]['baseline_psnr']
    strong_cutoff = baseline_sorted[min(len(rows) - 1, (len(rows) * 3) // 4)]['baseline_psnr']
    for row in rows:
        if row['baseline_psnr'] <= weak_cutoff:
            row['baseline_strength_bin'] = 'baseline_weakest_25'
        elif row['baseline_psnr'] >= strong_cutoff:
            row['baseline_strength_bin'] = 'baseline_strongest_25'
        else:
            row['baseline_strength_bin'] = 'baseline_middle_50'

    group_rows = [group_stats('all', 'all', rows)]
    for key, name in [
        ('baseline_strength_bin', 'baseline_strength_bin'),
        ('airlight_bin', 'airlight_bin'),
        ('beta_bin', 'beta_bin')
    ]:
        values = sorted(set(row[key] for row in rows))
        for value in values:
            subset = [row for row in rows if row[key] == value]
            group_rows.append(group_stats(name, value, subset))
    return group_rows


def compact_rows(rows):
    keep = [
        'filename', 'image_id', 'airlight', 'beta', 'baseline_strength_bin',
        'baseline_psnr', 'current_psnr', 'delta_psnr', 'lf_residual_cosine',
        'lf_luma_residual_cosine', 'lf_residual_norm_ratio',
        'lf_luma_norm_ratio', 'lf_residual_error_ratio', 'lf_mse_delta',
        'lf_luma_mse_delta'
    ]
    return [{key: row[key] for key in keep} for row in rows]


def build_hard_cases(rows, top_k):
    return {
        'worst_delta_psnr': compact_rows(sorted(rows, key=lambda row: row['delta_psnr'])[:top_k]),
        'best_delta_psnr': compact_rows(sorted(rows, key=lambda row: row['delta_psnr'], reverse=True)[:top_k]),
        'worst_residual_cosine': compact_rows(sorted(rows, key=lambda row: row['lf_residual_cosine'])[:top_k]),
        'largest_residual_overshoot': compact_rows(sorted(
            [row for row in rows if row['lf_residual_norm_ratio'] != ''],
            key=lambda row: row['lf_residual_norm_ratio'],
            reverse=True
        )[:top_k]),
        'largest_lowfreq_mse_regression': compact_rows(sorted(
            rows,
            key=lambda row: row['lf_mse_delta'],
            reverse=True
        )[:top_k]),
        'largest_lowfreq_mse_improvement': compact_rows(sorted(
            rows,
            key=lambda row: row['lf_mse_delta']
        )[:top_k]),
        'strong_baseline_regressions': compact_rows(sorted(
            [
                row for row in rows
                if row['baseline_strength_bin'] == 'baseline_strongest_25'
                and row['delta_psnr'] <= -0.30
            ],
            key=lambda row: row['delta_psnr']
        )[:top_k]),
        'weak_baseline_gains': compact_rows(sorted(
            [
                row for row in rows
                if row['baseline_strength_bin'] == 'baseline_weakest_25'
                and row['delta_psnr'] >= 0.30
            ],
            key=lambda row: row['delta_psnr'],
            reverse=True
        )[:top_k]),
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    if value == '' or value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}'


def write_report(path, summary, group_rows, hard_cases):
    lines = [
        '# LF Residual Direction Diagnostic',
        '',
        '## Summary',
        '',
        f"- Dataset: `{summary['dataset']}` / `{summary['split']}`",
        f"- Images: `{summary['num_images']}`",
        f"- Baseline: `{summary['baseline_label']}` step `{summary['baseline_checkpoint_step']}`",
        f"- Current: `{summary['current_label']}` step `{summary['current_checkpoint_step']}`",
        f"- Low-frequency pool: `{summary['lowfreq_pool']}`",
        f"- Mean delta PSNR: `{fmt(summary['mean_delta_psnr'])}`",
        f"- Better / worse at 0.30 dB: `{summary['better_030db_count']}` / `{summary['worse_030db_count']}`",
        f"- Mean residual cosine: `{fmt(summary['mean_lf_residual_cosine'])}`",
        f"- Mean luma residual cosine: `{fmt(summary['mean_lf_luma_residual_cosine'])}`",
        f"- Mean residual norm ratio: `{fmt(summary['mean_lf_residual_norm_ratio'])}`",
        f"- Mean residual error ratio: `{fmt(summary['mean_lf_residual_error_ratio'])}`",
        f"- Low-frequency MSE improved / regressed: `{summary['lf_mse_improved_count']}` / `{summary['lf_mse_regressed_count']}`",
        f"- Luma low-frequency MSE improved / regressed: `{summary['lf_luma_mse_improved_count']}` / `{summary['lf_luma_mse_regressed_count']}`",
        f"- Wrong-direction count (cosine < 0): `{summary['wrong_direction_count']}`",
        f"- Possible overshoot / under-correction counts: `{summary['possible_overshoot_count']}` / `{summary['possible_under_correction_count']}`",
        f"- Corr(delta PSNR, residual cosine): `{summary['corr_delta_psnr_lf_residual_cosine']}`",
        f"- Corr(delta PSNR, residual norm ratio): `{summary['corr_delta_psnr_lf_residual_norm_ratio']}`",
        '',
        '## Baseline Strength Groups',
        '',
    ]
    for group in group_rows:
        if group['group_name'] == 'baseline_strength_bin':
            lines.append(
                f"- `{group['group_value']}` n=`{group['num_images']}` "
                f"delta `{fmt(group['mean_delta_psnr'])}`, "
                f"cos `{fmt(group['mean_lf_residual_cosine'])}`, "
                f"norm_ratio `{fmt(group['mean_lf_residual_norm_ratio'])}`, "
                f"LF MSE improved/regressed `{group['lf_mse_improved_count']}/{group['lf_mse_regressed_count']}`"
            )

    lines.extend([
        '',
        '## Reading Guide',
        '',
        '- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.',
        '- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.',
        '- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.',
        '- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.',
        '',
        '## Hard Cases',
        '',
        '### Worst Delta PSNR',
        ''
    ])
    for row in hard_cases['worst_delta_psnr'][:10]:
        lines.append(
            f"- `{row['filename']}` delta `{fmt(row['delta_psnr'])}`, "
            f"cos `{fmt(row['lf_residual_cosine'])}`, norm_ratio `{fmt(row['lf_residual_norm_ratio'])}`, "
            f"LF MSE delta `{fmt(row['lf_mse_delta'], 8)}`"
        )

    lines.extend(['', '### Worst Residual Cosine', ''])
    for row in hard_cases['worst_residual_cosine'][:10]:
        lines.append(
            f"- `{row['filename']}` cos `{fmt(row['lf_residual_cosine'])}`, "
            f"delta `{fmt(row['delta_psnr'])}`, norm_ratio `{fmt(row['lf_residual_norm_ratio'])}`"
        )

    lines.extend(['', '### Largest Low-Frequency MSE Regressions', ''])
    for row in hard_cases['largest_lowfreq_mse_regression'][:10]:
        lines.append(
            f"- `{row['filename']}` LF MSE delta `{fmt(row['lf_mse_delta'], 8)}`, "
            f"delta `{fmt(row['delta_psnr'])}`, cos `{fmt(row['lf_residual_cosine'])}`"
        )

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.lowfreq_pool <= 0:
        raise ValueError('--lowfreq_pool must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_root = Path('../dataset') / args.dataset / args.split
    hazy_dir, clear_dir = resolve_pair_dirs(split_root)
    hazy_names = [path.name for path in list_image_files(hazy_dir)]
    if args.max_images > 0:
        hazy_names = hazy_names[:args.max_images]

    baseline_model, baseline_ckpt = load_model(
        args.baseline_checkpoint,
        args.baseline_use_lf_prior,
        args,
        conditional_mask=args.baseline_lf_conditional_mask,
        haze_aware_mask=args.baseline_lf_haze_aware_mask,
        residual_calibration=args.baseline_lf_residual_calibration,
        residual_selector=args.baseline_lf_residual_selector
    )
    current_model, current_ckpt = load_model(
        args.current_checkpoint,
        args.current_use_lf_prior,
        args,
        conditional_mask=args.lf_conditional_mask,
        haze_aware_mask=args.lf_haze_aware_mask,
        residual_calibration=args.lf_residual_calibration,
        residual_selector=args.lf_residual_selector
    )

    rows = []
    with torch.no_grad():
        for idx, name in enumerate(hazy_names, start=1):
            hazy_path = Path(hazy_dir) / name
            clear_path = Path(find_clear_image(clear_dir, name))
            hazy_img = Image.open(hazy_path).convert('RGB')
            clear_img = Image.open(clear_path).convert('RGB')

            hazy = pil_to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = pil_to_tensor(clear_img).unsqueeze(0).to(args.device)

            baseline = infer_one(baseline_model, hazy, args.pad_size)
            current = infer_one(current_model, hazy, args.pad_size)

            baseline_psnr = metric_value(psnr(baseline, clear))
            baseline_ssim = metric_value(ssim(baseline, clear))
            current_psnr = metric_value(psnr(current, clear))
            current_ssim = metric_value(ssim(current, clear))

            baseline_low = lowpass(baseline, args.lowfreq_pool)
            current_low = lowpass(current, args.lowfreq_pool)
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
                'baseline_psnr': baseline_psnr,
                'baseline_ssim': baseline_ssim,
                'current_psnr': current_psnr,
                'current_ssim': current_ssim,
                'delta_psnr': current_psnr - baseline_psnr,
                'delta_ssim': current_ssim - baseline_ssim,
            }
            row.update(residual_metrics(baseline_low, current_low, clear_low))
            rows.append(row)

            if idx % 50 == 0 or idx == len(hazy_names):
                print('diagnosed {}/{}'.format(idx, len(hazy_names)), flush=True)

    group_rows = build_group_rows(rows)
    hard_cases = build_hard_cases(rows, args.top_k)
    weak_rows = [row for row in rows if row['baseline_strength_bin'] == 'baseline_weakest_25']
    strong_rows = [row for row in rows if row['baseline_strength_bin'] == 'baseline_strongest_25']

    summary = {
        'dataset': args.dataset,
        'split': args.split,
        'num_images': len(rows),
        'baseline_label': args.baseline_label,
        'current_label': args.current_label,
        'baseline_checkpoint': args.baseline_checkpoint,
        'current_checkpoint': args.current_checkpoint,
        'baseline_checkpoint_step': baseline_ckpt.get('step'),
        'current_checkpoint_step': current_ckpt.get('step'),
        'baseline_checkpoint_max_psnr': baseline_ckpt.get('max_psnr'),
        'current_checkpoint_max_psnr': current_ckpt.get('max_psnr'),
        'baseline_lf_gate': lf_gate_value(baseline_model),
        'current_lf_gate': lf_gate_value(current_model),
        'lowfreq_pool': args.lowfreq_pool,
        'mean_baseline_psnr': mean([row['baseline_psnr'] for row in rows]),
        'mean_current_psnr': mean([row['current_psnr'] for row in rows]),
        'mean_delta_psnr': mean([row['delta_psnr'] for row in rows]),
        'median_delta_psnr': median([row['delta_psnr'] for row in rows]),
        'mean_delta_ssim': mean([row['delta_ssim'] for row in rows]),
        'better_030db_count': sum(1 for row in rows if row['delta_psnr'] >= 0.30),
        'worse_030db_count': sum(1 for row in rows if row['delta_psnr'] <= -0.30),
        'mean_lf_residual_cosine': mean([row['lf_residual_cosine'] for row in rows]),
        'median_lf_residual_cosine': median([row['lf_residual_cosine'] for row in rows]),
        'p10_lf_residual_cosine': percentile([row['lf_residual_cosine'] for row in rows], 10),
        'mean_lf_luma_residual_cosine': mean([row['lf_luma_residual_cosine'] for row in rows]),
        'mean_lf_residual_norm_ratio': mean([row['lf_residual_norm_ratio'] for row in rows]),
        'median_lf_residual_norm_ratio': median([row['lf_residual_norm_ratio'] for row in rows]),
        'p90_lf_residual_norm_ratio': percentile([row['lf_residual_norm_ratio'] for row in rows], 90),
        'mean_lf_luma_norm_ratio': mean([row['lf_luma_norm_ratio'] for row in rows]),
        'mean_lf_residual_error_ratio': mean([row['lf_residual_error_ratio'] for row in rows]),
        'mean_lf_mse_delta': mean([row['lf_mse_delta'] for row in rows]),
        'mean_lf_luma_mse_delta': mean([row['lf_luma_mse_delta'] for row in rows]),
        'lf_mse_improved_count': sum(1 for row in rows if row['lf_mse_delta'] < 0),
        'lf_mse_regressed_count': sum(1 for row in rows if row['lf_mse_delta'] > 0),
        'lf_luma_mse_improved_count': sum(1 for row in rows if row['lf_luma_mse_delta'] < 0),
        'lf_luma_mse_regressed_count': sum(1 for row in rows if row['lf_luma_mse_delta'] > 0),
        'wrong_direction_count': sum(1 for row in rows if row['lf_residual_cosine'] < 0),
        'possible_overshoot_count': sum(
            1 for row in rows
            if row['lf_residual_cosine'] > 0.5
            and row['lf_residual_norm_ratio'] != ''
            and row['lf_residual_norm_ratio'] > 1.5
        ),
        'possible_under_correction_count': sum(
            1 for row in rows
            if row['lf_residual_cosine'] > 0.5
            and row['lf_residual_norm_ratio'] != ''
            and row['lf_residual_norm_ratio'] < 0.5
        ),
        'weak_baseline_mean_delta_psnr': mean([row['delta_psnr'] for row in weak_rows]),
        'strong_baseline_mean_delta_psnr': mean([row['delta_psnr'] for row in strong_rows]),
        'weak_baseline_mean_lf_residual_cosine': mean([row['lf_residual_cosine'] for row in weak_rows]),
        'strong_baseline_mean_lf_residual_cosine': mean([row['lf_residual_cosine'] for row in strong_rows]),
        'weak_baseline_mean_lf_norm_ratio': mean([row['lf_residual_norm_ratio'] for row in weak_rows]),
        'strong_baseline_mean_lf_norm_ratio': mean([row['lf_residual_norm_ratio'] for row in strong_rows]),
        'weak_baseline_mean_lf_mse_delta': mean([row['lf_mse_delta'] for row in weak_rows]),
        'strong_baseline_mean_lf_mse_delta': mean([row['lf_mse_delta'] for row in strong_rows]),
        'corr_delta_psnr_lf_residual_cosine': pearson(
            [row['delta_psnr'] for row in rows],
            [row['lf_residual_cosine'] for row in rows]
        ),
        'corr_delta_psnr_lf_residual_norm_ratio': pearson(
            [row['delta_psnr'] for row in rows],
            [row['lf_residual_norm_ratio'] for row in rows]
        ),
        'corr_delta_psnr_lf_mse_delta': pearson(
            [row['delta_psnr'] for row in rows],
            [row['lf_mse_delta'] for row in rows]
        ),
    }

    write_csv(output_dir / 'per_image_residual_metrics.csv', rows)
    write_csv(output_dir / 'group_summary.csv', group_rows)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / 'hard_cases.json', 'w', encoding='utf-8') as f:
        json.dump(hard_cases, f, indent=2)
    write_report(output_dir / 'analysis_report.md', summary, group_rows, hard_cases)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
