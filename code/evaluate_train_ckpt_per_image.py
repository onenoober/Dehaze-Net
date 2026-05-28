import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs
from metric import psnr, ssim
from model import DEANet


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
    parser.add_argument('--baseline_lf_multiscale_refiner', action='store_true')
    parser.add_argument('--lf_multiscale_refiner', action='store_true')
    parser.add_argument('--lf_mbr_channels', type=int, default=8)
    parser.add_argument('--lf_mbr_pool_sizes', type=str, default='4,8,16')
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


def load_model(checkpoint_path, use_lf_prior, args, conditional_mask=False, haze_aware_mask=False, residual_calibration=False, residual_selector=False, multiscale_refiner=False):
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
        lf_selector_init_bias=args.lf_selector_init_bias,
        lf_multiscale_refiner=multiscale_refiner,
        lf_mbr_channels=args.lf_mbr_channels,
        lf_mbr_pool_sizes=args.lf_mbr_pool_sizes
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    return model, checkpoint


def lf_gate_value(model):
    module = getattr(model, 'module', model)
    lf_prior = getattr(module, 'lf_prior', None)
    if lf_prior is None:
        return None
    return float(lf_prior.gate.detach().cpu().item())


def infer_one(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    pred = model(padded).clamp(0, 1)
    return pred[:, :, :h, :w]


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


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x != '' and y != '']
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


def numeric(values):
    return [float(v) for v in values if v != '']


def percentile(values, q):
    values = numeric(values)
    if not values:
        return ''
    return float(np.percentile(np.array(values, dtype=np.float64), q))


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


def make_bin(value, edges, labels):
    if value == '':
        return 'unknown'
    value = float(value)
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def group_stats(group_name, group_value, rows):
    deltas = [row['delta_psnr'] for row in rows]
    delta_ssims = [row['delta_ssim'] for row in rows]
    baseline_psnr = [row['baseline_psnr'] for row in rows]
    current_psnr = [row['current_psnr'] for row in rows]
    return {
        'group_name': group_name,
        'group_value': group_value,
        'num_images': len(rows),
        'better_psnr_count': sum(1 for row in rows if row['delta_psnr'] > 0),
        'worse_psnr_count': sum(1 for row in rows if row['delta_psnr'] < 0),
        'better_030db_count': sum(1 for row in rows if row['delta_psnr'] >= 0.30),
        'worse_030db_count': sum(1 for row in rows if row['delta_psnr'] <= -0.30),
        'mean_baseline_psnr': mean(baseline_psnr),
        'mean_current_psnr': mean(current_psnr),
        'mean_delta_psnr': mean(deltas),
        'median_delta_psnr': median(deltas),
        'p10_delta_psnr': percentile(deltas, 10),
        'p90_delta_psnr': percentile(deltas, 90),
        'min_delta_psnr': min(deltas),
        'max_delta_psnr': max(deltas),
        'mean_delta_ssim': mean(delta_ssims),
    }


def build_group_rows(rows):
    a_edges = [0.65, 0.75, 0.85, 0.95]
    a_labels = ['A<0.65', '0.65<=A<0.75', '0.75<=A<0.85', '0.85<=A<0.95', 'A>=0.95']
    beta_edges = [0.8, 1.1, 1.4, 1.7]
    beta_labels = ['beta<0.8', '0.8<=beta<1.1', '1.1<=beta<1.4', '1.4<=beta<1.7', 'beta>=1.7']

    for row in rows:
        row['airlight_bin'] = make_bin(row['airlight'], a_edges, a_labels)
        row['beta_bin'] = make_bin(row['beta'], beta_edges, beta_labels)
        row['airlight_beta_bin'] = row['airlight_bin'] + ' | ' + row['beta_bin']

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
        ('beta_bin', 'beta_bin'),
        ('airlight_beta_bin', 'airlight_beta_bin')
    ]:
        values = sorted(set(row[key] for row in rows))
        for value in values:
            subset = [row for row in rows if row[key] == value]
            group_rows.append(group_stats(name, value, subset))
    return group_rows


def row_subset(rows, predicate, top_k, reverse=False):
    subset = [row for row in rows if predicate(row)]
    subset = sorted(subset, key=lambda row: row['delta_psnr'], reverse=reverse)
    return subset[:top_k]


def compact_rows(rows):
    keep = [
        'filename', 'image_id', 'airlight', 'beta', 'baseline_psnr',
        'current_psnr', 'delta_psnr', 'baseline_ssim', 'current_ssim',
        'delta_ssim', 'airlight_bin', 'beta_bin', 'baseline_strength_bin'
    ]
    return [{key: row[key] for key in keep} for row in rows]


def build_hard_cases(rows, top_k):
    sorted_rows = sorted(rows, key=lambda row: row['delta_psnr'])
    baseline_sorted = sorted(rows, key=lambda row: row['baseline_psnr'])
    baseline_weak_cutoff = baseline_sorted[max(0, len(rows) // 4 - 1)]['baseline_psnr']
    baseline_strong_cutoff = baseline_sorted[min(len(rows) - 1, (len(rows) * 3) // 4)]['baseline_psnr']
    return {
        'worst_delta_psnr': compact_rows(sorted_rows[:top_k]),
        'best_delta_psnr': compact_rows(list(reversed(sorted_rows[-top_k:]))),
        'baseline_weak_current_helps': compact_rows(row_subset(
            rows,
            lambda row: row['baseline_psnr'] <= baseline_weak_cutoff and row['delta_psnr'] >= 0.30,
            top_k,
            reverse=True
        )),
        'baseline_strong_current_hurts': compact_rows(row_subset(
            rows,
            lambda row: row['baseline_psnr'] >= baseline_strong_cutoff and row['delta_psnr'] <= -0.30,
            top_k,
            reverse=False
        )),
        'high_airlight_failures': compact_rows(row_subset(
            rows,
            lambda row: row['airlight'] != '' and row['airlight'] >= 0.85 and row['delta_psnr'] <= -0.30,
            top_k,
            reverse=False
        )),
        'low_airlight_gains': compact_rows(row_subset(
            rows,
            lambda row: row['airlight'] != '' and row['airlight'] < 0.75 and row['delta_psnr'] >= 0.30,
            top_k,
            reverse=True
        )),
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary, group_rows, hard_cases):
    lines = [
        '# Train Checkpoint Per-Image Comparison',
        '',
        '## Summary',
        '',
        f"- Dataset: `{summary['dataset']}` / `{summary['split']}`",
        f"- Images: `{summary['num_images']}`",
        f"- Baseline: `{summary['baseline_label']}` step `{summary['baseline_checkpoint_step']}`",
        f"- Current: `{summary['current_label']}` step `{summary['current_checkpoint_step']}`",
        f"- Mean baseline: PSNR `{summary['mean_baseline_psnr']:.4f}`, SSIM `{summary['mean_baseline_ssim']:.4f}`",
        f"- Mean current: PSNR `{summary['mean_current_psnr']:.4f}`, SSIM `{summary['mean_current_ssim']:.4f}`",
        f"- Mean delta: PSNR `{summary['mean_delta_psnr']:.4f}`, SSIM `{summary['mean_delta_ssim']:.6f}`",
        f"- Better / worse by PSNR: `{summary['better_psnr_count']}` / `{summary['worse_psnr_count']}`",
        f"- Meaningful better / worse at 0.30 dB: `{summary['better_030db_count']}` / `{summary['worse_030db_count']}`",
        f"- Baseline strongest 25% mean delta: `{summary['strong_baseline_mean_delta_psnr']:.4f}`; regressions <= -0.30 dB: `{summary['strong_baseline_worse_030db_count']}`",
        f"- Baseline weakest 25% mean delta: `{summary['weak_baseline_mean_delta_psnr']:.4f}`; gains >= +0.30 dB: `{summary['weak_baseline_better_030db_count']}`",
        f"- Pearson corr(A, delta PSNR): `{summary['corr_airlight_delta_psnr']}`",
        f"- Pearson corr(beta, delta PSNR): `{summary['corr_beta_delta_psnr']}`",
        '',
        '## Group Highlights',
        '',
    ]

    for group in group_rows:
        if group['group_name'] in ('baseline_strength_bin', 'airlight_bin', 'beta_bin'):
            lines.append(
                f"- `{group['group_value']}` n=`{group['num_images']}` "
                f"mean delta PSNR `{group['mean_delta_psnr']:.4f}`, "
                f"better/worse `{group['better_psnr_count']}/{group['worse_psnr_count']}`"
            )

    lines.extend([
        '',
        '## Hard Cases',
        '',
        '### Worst Delta PSNR',
        ''
    ])
    for row in hard_cases['worst_delta_psnr'][:10]:
        lines.append(
            f"- `{row['filename']}` A=`{row['airlight']}` beta=`{row['beta']}` "
            f"delta `{row['delta_psnr']:.4f}`"
        )
    lines.extend(['', '### Best Delta PSNR', ''])
    for row in hard_cases['best_delta_psnr'][:10]:
        lines.append(
            f"- `{row['filename']}` A=`{row['airlight']}` beta=`{row['beta']}` "
            f"delta `{row['delta_psnr']:.4f}`"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_root = Path('../dataset') / args.dataset / args.split
    hazy_dir, clear_dir = resolve_pair_dirs(split_root)
    hazy_names = list_image_files(hazy_dir)
    if args.max_images > 0:
        hazy_names = hazy_names[:args.max_images]

    baseline_model, baseline_ckpt = load_model(
        args.baseline_checkpoint,
        args.baseline_use_lf_prior,
        args,
        conditional_mask=args.baseline_lf_conditional_mask,
        haze_aware_mask=args.baseline_lf_haze_aware_mask,
        residual_calibration=args.baseline_lf_residual_calibration,
        residual_selector=args.baseline_lf_residual_selector,
        multiscale_refiner=args.baseline_lf_multiscale_refiner
    )
    current_model, current_ckpt = load_model(
        args.current_checkpoint,
        args.current_use_lf_prior,
        args,
        conditional_mask=args.lf_conditional_mask,
        haze_aware_mask=args.lf_haze_aware_mask,
        residual_calibration=args.lf_residual_calibration,
        residual_selector=args.lf_residual_selector,
        multiscale_refiner=args.lf_multiscale_refiner
    )

    to_tensor = ToTensor()
    rows = []
    with torch.no_grad():
        for idx, name in enumerate(hazy_names, start=1):
            hazy_path = Path(hazy_dir) / name
            clear_path = Path(find_clear_image(clear_dir, name))
            hazy_img = Image.open(hazy_path).convert('RGB')
            clear_img = Image.open(clear_path).convert('RGB')

            hazy = to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = to_tensor(clear_img).unsqueeze(0).to(args.device)

            baseline = infer_one(baseline_model, hazy, args.pad_size)
            current = infer_one(current_model, hazy, args.pad_size)

            input_psnr = metric_value(psnr(hazy, clear))
            input_ssim = metric_value(ssim(hazy, clear))
            baseline_psnr = metric_value(psnr(baseline, clear))
            baseline_ssim = metric_value(ssim(baseline, clear))
            current_psnr = metric_value(psnr(current, clear))
            current_ssim = metric_value(ssim(current, clear))
            meta = parse_haze4k_name(name)

            rows.append({
                'index': idx,
                'filename': name,
                'image_id': meta['image_id'],
                'airlight': meta['airlight'],
                'beta': meta['beta'],
                'height': hazy.shape[-2],
                'width': hazy.shape[-1],
                'input_psnr': input_psnr,
                'input_ssim': input_ssim,
                'baseline_psnr': baseline_psnr,
                'baseline_ssim': baseline_ssim,
                'current_psnr': current_psnr,
                'current_ssim': current_ssim,
                'delta_psnr': current_psnr - baseline_psnr,
                'delta_ssim': current_ssim - baseline_ssim,
            })

            if idx % 50 == 0 or idx == len(hazy_names):
                print('evaluated {}/{}'.format(idx, len(hazy_names)), flush=True)

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
        'baseline_lf_residual_calibration': args.baseline_lf_residual_calibration,
        'baseline_lf_residual_selector': args.baseline_lf_residual_selector,
        'lf_conditional_mask': args.lf_conditional_mask,
        'lf_mask_hidden_channels': args.lf_mask_hidden_channels,
        'lf_mask_init_bias': args.lf_mask_init_bias,
        'lf_haze_aware_mask': args.lf_haze_aware_mask,
        'lf_haze_mask_strength': args.lf_haze_mask_strength,
        'lf_residual_calibration': args.lf_residual_calibration,
        'lf_calib_hidden_channels': args.lf_calib_hidden_channels,
        'lf_calib_alpha_max': args.lf_calib_alpha_max,
        'lf_residual_selector': args.lf_residual_selector,
        'lf_selector_hidden_channels': args.lf_selector_hidden_channels,
        'lf_selector_init_bias': args.lf_selector_init_bias,
        'mean_input_psnr': mean([row['input_psnr'] for row in rows]),
        'mean_input_ssim': mean([row['input_ssim'] for row in rows]),
        'mean_baseline_psnr': mean([row['baseline_psnr'] for row in rows]),
        'mean_baseline_ssim': mean([row['baseline_ssim'] for row in rows]),
        'mean_current_psnr': mean([row['current_psnr'] for row in rows]),
        'mean_current_ssim': mean([row['current_ssim'] for row in rows]),
        'mean_delta_psnr': mean([row['delta_psnr'] for row in rows]),
        'median_delta_psnr': median([row['delta_psnr'] for row in rows]),
        'mean_delta_ssim': mean([row['delta_ssim'] for row in rows]),
        'min_delta_psnr': min(row['delta_psnr'] for row in rows),
        'max_delta_psnr': max(row['delta_psnr'] for row in rows),
        'better_psnr_count': sum(1 for row in rows if row['delta_psnr'] > 0),
        'worse_psnr_count': sum(1 for row in rows if row['delta_psnr'] < 0),
        'better_010db_count': sum(1 for row in rows if row['delta_psnr'] >= 0.10),
        'worse_010db_count': sum(1 for row in rows if row['delta_psnr'] <= -0.10),
        'better_030db_count': sum(1 for row in rows if row['delta_psnr'] >= 0.30),
        'worse_030db_count': sum(1 for row in rows if row['delta_psnr'] <= -0.30),
        'better_100db_count': sum(1 for row in rows if row['delta_psnr'] >= 1.00),
        'worse_100db_count': sum(1 for row in rows if row['delta_psnr'] <= -1.00),
        'weak_baseline_cutoff_psnr': max(row['baseline_psnr'] for row in weak_rows) if weak_rows else None,
        'strong_baseline_cutoff_psnr': min(row['baseline_psnr'] for row in strong_rows) if strong_rows else None,
        'weak_baseline_mean_delta_psnr': mean([row['delta_psnr'] for row in weak_rows]),
        'strong_baseline_mean_delta_psnr': mean([row['delta_psnr'] for row in strong_rows]),
        'weak_baseline_better_030db_count': sum(1 for row in weak_rows if row['delta_psnr'] >= 0.30),
        'weak_baseline_worse_030db_count': sum(1 for row in weak_rows if row['delta_psnr'] <= -0.30),
        'strong_baseline_better_030db_count': sum(1 for row in strong_rows if row['delta_psnr'] >= 0.30),
        'strong_baseline_worse_030db_count': sum(1 for row in strong_rows if row['delta_psnr'] <= -0.30),
        'corr_airlight_delta_psnr': pearson([row['airlight'] for row in rows], [row['delta_psnr'] for row in rows]),
        'corr_beta_delta_psnr': pearson([row['beta'] for row in rows], [row['delta_psnr'] for row in rows]),
    }

    write_csv(output_dir / 'per_image_metrics.csv', rows)
    write_csv(output_dir / 'group_summary.csv', group_rows)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / 'hard_cases.json', 'w', encoding='utf-8') as f:
        json.dump(hard_cases, f, indent=2)
    write_report(output_dir / 'analysis_report.md', summary, group_rows, hard_cases)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
