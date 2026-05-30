import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs
from metric import psnr, ssim
from model import BaselineRelativeFrequencyResidualCorrector, DEANet, DEANetCBRFRC


EPS = 1e-8


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1'):
        return True
    if value in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HAZE4K')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--cr_checkpoint', type=str, required=True)
    parser.add_argument('--lfv1_checkpoint', type=str, required=True)
    parser.add_argument('--candidate_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pad_size', type=int, default=4)
    parser.add_argument('--lowfreq_pool', type=int, default=8)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--lfv1_use_lf_prior', action='store_true')
    parser.add_argument('--lf_prior_channels', type=int, default=8)
    parser.add_argument('--lf_prior_pool', type=int, default=8)
    parser.add_argument('--lf_prior_gate_init', type=float, default=0.0)
    parser.add_argument('--lf_prior_residual_center', action='store_true')
    parser.add_argument('--lf_prior_gate_max', type=float, default=0.0)
    parser.add_argument('--lf_prior_injection', type=str, default='pre_mix', choices=['pre_mix', 'post_mix'])
    parser.add_argument('--brf_hidden_channels', type=int, default=16)
    parser.add_argument('--brf_wavelet_levels', type=int, default=2)
    parser.add_argument('--brf_pyramid_type', type=str, default='laplacian', choices=['laplacian', 'haar'])
    parser.add_argument('--brf_gate_init', type=float, default=-4.0)
    parser.add_argument('--brf_hf_gate_init', type=float, default=-5.0)
    parser.add_argument('--brf_max_residual', type=float, default=0.08)
    parser.add_argument('--brf_max_color_residual', type=float, default=0.04)
    parser.add_argument('--brf_max_hf_residual', type=float, default=0.03)
    parser.add_argument('--brf_hf_scale', type=float, default=0.1)
    parser.add_argument('--brf_use_haze_prior', action='store_true')
    parser.add_argument('--brf_preserve_highfreq', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--brf_lf_pool', type=int, default=8)
    parser.add_argument('--brf_mid_pool', type=int, default=4)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def strip_module_prefix(state_dict):
    if not any(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


def load_plain_deanet(checkpoint_path, args, use_lf_prior=False):
    model = DEANet(
        base_dim=32,
        use_lf_prior=use_lf_prior,
        lf_prior_channels=args.lf_prior_channels,
        lf_prior_pool=args.lf_prior_pool,
        lf_prior_gate_init=args.lf_prior_gate_init,
        lf_prior_residual_center=args.lf_prior_residual_center,
        lf_prior_gate_max=args.lf_prior_gate_max,
        lf_prior_injection=args.lf_prior_injection,
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(strip_module_prefix(state_dict_from_checkpoint(checkpoint)))
    model.to(args.device)
    model.eval()
    return model, checkpoint


def load_brf_model(checkpoint_path, args):
    baseline = DEANet(base_dim=32)
    corrector = BaselineRelativeFrequencyResidualCorrector(
        hidden_channels=args.brf_hidden_channels,
        wavelet_levels=args.brf_wavelet_levels,
        gate_init=args.brf_gate_init,
        hf_gate_init=args.brf_hf_gate_init,
        max_residual=args.brf_max_residual,
        max_color_residual=args.brf_max_color_residual,
        max_hf_residual=args.brf_max_hf_residual,
        hf_scale=args.brf_hf_scale,
        use_haze_prior=args.brf_use_haze_prior,
        preserve_highfreq=args.brf_preserve_highfreq,
        pyramid_type=args.brf_pyramid_type,
        lf_pool=args.brf_lf_pool,
        mid_pool=args.brf_mid_pool,
    )
    model = DEANetCBRFRC(baseline, corrector, freeze_baseline=True, use_baseline_detach=True)
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(strip_module_prefix(state_dict_from_checkpoint(checkpoint)))
    model.to(args.device)
    model.eval()
    return model, checkpoint


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')


def pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def infer_plain(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    pred = model(padded).clamp(0, 1)
    return pred[:, :, :h, :w]


def infer_brf(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    out_dict = model(padded, return_aux=True)
    cropped = {}
    for key, value in out_dict.items():
        if torch.is_tensor(value) and value.dim() == 4:
            cropped[key] = value[:, :, :h, :w]
        else:
            cropped[key] = value
    cropped['out'] = cropped['out'].clamp(0, 1)
    cropped['j0'] = cropped['j0'].clamp(0, 1)
    return cropped


def lowpass(x, pool_size):
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def l2_norm(x):
    return scalar(x.reshape(x.shape[0], -1).norm(dim=1).mean())


def mse(a, b):
    return scalar(F.mse_loss(a, b))


def cosine(a, b):
    a_vec = a.reshape(a.shape[0], -1)
    b_vec = b.reshape(b.shape[0], -1)
    value = (a_vec * b_vec).sum(dim=1) / (a_vec.norm(dim=1) * b_vec.norm(dim=1) + EPS)
    return scalar(value.mean())


def tensor_stat(prefix, value):
    value = value.detach()
    return {
        prefix + '_mean': scalar(value.mean()),
        prefix + '_std': scalar(value.std(unbiased=False)),
        prefix + '_min': scalar(value.min()),
        prefix + '_max': scalar(value.max()),
    }


def parse_haze4k_name(filename):
    stem = Path(filename).stem
    parts = stem.split('_')
    meta = {'image_id': parts[0] if parts else stem, 'airlight': '', 'beta': ''}
    if len(parts) >= 3:
        try:
            meta['airlight'] = float(parts[1])
            meta['beta'] = float(parts[2])
        except ValueError:
            pass
    return meta


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


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_groups(rows):
    cr_sorted = sorted(rows, key=lambda row: row['psnr_cr'])
    weak_cutoff = cr_sorted[max(0, len(rows) // 4 - 1)]['psnr_cr']
    strong_cutoff = cr_sorted[min(len(rows) - 1, (len(rows) * 3) // 4)]['psnr_cr']
    for row in rows:
        row['weak_cr_group'] = row['psnr_cr'] <= weak_cutoff
        row['strong_cr_group'] = row['psnr_cr'] >= strong_cutoff
        row['lfv1_gain_group'] = (row['psnr_lfv1'] - row['psnr_cr']) >= 0.30
        row['lfv1_regression_group'] = (row['psnr_lfv1'] - row['psnr_cr']) <= -0.30


def group_delta(rows, predicate, key='delta_brf_vs_cr'):
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return ''
    return mean([row[key] for row in subset])


def build_summary(rows, args, cr_ckpt, lfv1_ckpt, brf_ckpt):
    weak_rows = [row for row in rows if row['weak_cr_group']]
    strong_rows = [row for row in rows if row['strong_cr_group']]
    gain_rows = [row for row in rows if row['lfv1_gain_group']]
    reg_rows = [row for row in rows if row['lfv1_regression_group']]
    preservation_count = sum(1 for row in gain_rows if row['delta_brf_vs_cr'] >= 0)
    rescue_count = sum(1 for row in reg_rows if row['delta_brf_vs_lfv1'] > 0)
    return {
        'dataset': args.dataset,
        'split': args.split,
        'num_images': len(rows),
        'cr_checkpoint': args.cr_checkpoint,
        'lfv1_checkpoint': args.lfv1_checkpoint,
        'candidate_checkpoint': args.candidate_checkpoint,
        'cr_checkpoint_step': cr_ckpt.get('step') if isinstance(cr_ckpt, dict) else None,
        'lfv1_checkpoint_step': lfv1_ckpt.get('step') if isinstance(lfv1_ckpt, dict) else None,
        'candidate_checkpoint_step': brf_ckpt.get('step') if isinstance(brf_ckpt, dict) else None,
        'mean_psnr_cr': mean([row['psnr_cr'] for row in rows]),
        'mean_psnr_lfv1': mean([row['psnr_lfv1'] for row in rows]),
        'mean_psnr_brf': mean([row['psnr_brf'] for row in rows]),
        'mean_ssim_cr': mean([row['ssim_cr'] for row in rows]),
        'mean_ssim_lfv1': mean([row['ssim_lfv1'] for row in rows]),
        'mean_ssim_brf': mean([row['ssim_brf'] for row in rows]),
        'delta_brf_vs_cr': mean([row['delta_brf_vs_cr'] for row in rows]),
        'delta_brf_vs_lfv1': mean([row['delta_brf_vs_lfv1'] for row in rows]),
        'wrong_direction_count': sum(1 for row in rows if row['wrong_direction']),
        'mean_residual_cosine': mean([row['residual_cosine'] for row in rows]),
        'median_residual_cosine': median([row['residual_cosine'] for row in rows]),
        'p10_residual_cosine': percentile([row['residual_cosine'] for row in rows], 10),
        'lf_mse_improved_count': sum(1 for row in rows if row['lf_mse_delta'] < 0),
        'lf_mse_regressed_count': sum(1 for row in rows if row['lf_mse_delta'] > 0),
        'weak_cr_delta': mean([row['delta_brf_vs_cr'] for row in weak_rows]),
        'strong_cr_delta': mean([row['delta_brf_vs_cr'] for row in strong_rows]),
        'lfv1_gain_count': len(gain_rows),
        'lfv1_regression_count': len(reg_rows),
        'lfv1_gain_preservation_count': preservation_count,
        'lfv1_gain_preservation_rate': preservation_count / len(gain_rows) if gain_rows else '',
        'lfv1_regression_rescue_count': rescue_count,
        'lfv1_regression_rescue_rate': rescue_count / len(reg_rows) if reg_rows else '',
        'gate_lf_mean_global': mean([row['gate_lf_mean'] for row in rows]),
        'gate_lf_std_global': mean([row['gate_lf_std'] for row in rows]),
        'gate_hf_mean_global': mean([row['gate_hf_mean'] for row in rows]),
    }


def build_group_summary(rows):
    groups = [
        ('all', lambda row: True),
        ('weak_cr', lambda row: row['weak_cr_group']),
        ('strong_cr', lambda row: row['strong_cr_group']),
        ('lfv1_gain', lambda row: row['lfv1_gain_group']),
        ('lfv1_regression', lambda row: row['lfv1_regression_group']),
    ]
    out = []
    for name, predicate in groups:
        subset = [row for row in rows if predicate(row)]
        if not subset:
            continue
        out.append({
            'group': name,
            'num_images': len(subset),
            'mean_delta_brf_vs_cr': mean([row['delta_brf_vs_cr'] for row in subset]),
            'mean_delta_brf_vs_lfv1': mean([row['delta_brf_vs_lfv1'] for row in subset]),
            'wrong_direction_count': sum(1 for row in subset if row['wrong_direction']),
            'lf_mse_improved_count': sum(1 for row in subset if row['lf_mse_delta'] < 0),
            'lf_mse_regressed_count': sum(1 for row in subset if row['lf_mse_delta'] > 0),
            'mean_residual_cosine': mean([row['residual_cosine'] for row in subset]),
            'mean_gate_lf': mean([row['gate_lf_mean'] for row in subset]),
        })
    return out


def main():
    args = parse_args()
    if args.lowfreq_pool <= 0:
        raise ValueError('--lowfreq_pool must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hazy_dir, clear_dir = resolve_pair_dirs(Path('../dataset') / args.dataset / args.split)
    hazy_names = list_image_files(hazy_dir)
    if args.max_images > 0:
        hazy_names = hazy_names[:args.max_images]

    cr_model, cr_ckpt = load_plain_deanet(args.cr_checkpoint, args, use_lf_prior=False)
    lfv1_model, lfv1_ckpt = load_plain_deanet(args.lfv1_checkpoint, args, use_lf_prior=args.lfv1_use_lf_prior)
    brf_model, brf_ckpt = load_brf_model(args.candidate_checkpoint, args)

    rows = []
    with torch.no_grad():
        for idx, name in enumerate(hazy_names, start=1):
            hazy_img = Image.open(Path(hazy_dir) / name).convert('RGB')
            clear_img = Image.open(find_clear_image(clear_dir, name)).convert('RGB')
            hazy = pil_to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = pil_to_tensor(clear_img).unsqueeze(0).to(args.device)

            cr = infer_plain(cr_model, hazy, args.pad_size)
            lfv1 = infer_plain(lfv1_model, hazy, args.pad_size)
            brf = infer_brf(brf_model, hazy, args.pad_size)
            j0 = brf['j0']
            j = brf['out']

            target_residual = lowpass(clear, args.lowfreq_pool) - lowpass(j0, args.lowfreq_pool)
            pred_residual = lowpass(j, args.lowfreq_pool) - lowpass(j0, args.lowfreq_pool)
            residual_error = pred_residual - target_residual
            lf_mse_j0 = mse(lowpass(j0, args.lowfreq_pool), lowpass(clear, args.lowfreq_pool))
            lf_mse_j = mse(lowpass(j, args.lowfreq_pool), lowpass(clear, args.lowfreq_pool))
            residual_cos = cosine(pred_residual, target_residual)
            psnr_cr = scalar(psnr(cr, clear))
            psnr_lfv1 = scalar(psnr(lfv1, clear))
            psnr_brf = scalar(psnr(j, clear))
            meta = parse_haze4k_name(name)
            row = {
                'index': idx,
                'image': name,
                'image_id': meta['image_id'],
                'airlight': meta['airlight'],
                'beta': meta['beta'],
                'psnr_cr': psnr_cr,
                'psnr_lfv1': psnr_lfv1,
                'psnr_brf': psnr_brf,
                'ssim_cr': scalar(ssim(cr, clear)),
                'ssim_lfv1': scalar(ssim(lfv1, clear)),
                'ssim_brf': scalar(ssim(j, clear)),
                'delta_brf_vs_cr': psnr_brf - psnr_cr,
                'delta_brf_vs_lfv1': psnr_brf - psnr_lfv1,
                'target_residual_norm': l2_norm(target_residual),
                'pred_residual_norm': l2_norm(pred_residual),
                'residual_cosine': residual_cos,
                'residual_norm_ratio': l2_norm(pred_residual) / (l2_norm(target_residual) + EPS),
                'residual_error_ratio': l2_norm(residual_error) / (l2_norm(target_residual) + EPS),
                'lf_mse_cr': mse(lowpass(cr, args.lowfreq_pool), lowpass(clear, args.lowfreq_pool)),
                'lf_mse_brf': lf_mse_j,
                'lf_mse_j0': lf_mse_j0,
                'lf_mse_j': lf_mse_j,
                'lf_mse_delta': lf_mse_j - lf_mse_j0,
                'wrong_direction': residual_cos < 0,
                'c_lf_norm': l2_norm(brf['c_lf']),
                'c_color_norm': l2_norm(brf['c_color']),
                'c_hf_norm': l2_norm(brf['c_hf']),
            }
            row.update(tensor_stat('gate_lf', brf['gate_lf']))
            row.update(tensor_stat('gate_hf', brf['gate_hf']))
            rows.append(row)
            if idx % 50 == 0 or idx == len(hazy_names):
                print('diagnosed {}/{}'.format(idx, len(hazy_names)), flush=True)

    add_groups(rows)
    summary = build_summary(rows, args, cr_ckpt, lfv1_ckpt, brf_ckpt)
    group_summary = build_group_summary(rows)
    write_csv(output_dir / 'per_image_metrics.csv', rows)
    write_csv(output_dir / 'group_summary.csv', group_summary)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
