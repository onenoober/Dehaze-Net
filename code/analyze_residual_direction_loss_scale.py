import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from metric import psnr
from model import DEANet


EPS = 1e-8
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_float_list(value):
    values = []
    for item in value.split(','):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError('expected at least one float')
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HAZE4K')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_label', type=str, default='current')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pad_size', type=int, default=4)
    parser.add_argument('--lowfreq_pool', type=int, default=8)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--target_norm_floor', type=float, default=0.0)
    parser.add_argument(
        '--candidate_weights',
        type=parse_float_list,
        default=parse_float_list('0.001,0.003,0.005,0.01')
    )
    parser.add_argument('--use_lf_prior', action='store_true')
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
    parser.add_argument('--lf_residual_selector', action='store_true')
    parser.add_argument('--lf_selector_hidden_channels', type=int, default=8)
    parser.add_argument('--lf_selector_init_bias', type=float, default=2.0)
    return parser.parse_args()


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
    return hazy_path, clear_path


def find_clear_image(clear_path, hazy_image_name):
    hazy_stem = Path(hazy_image_name).stem
    stems = [hazy_stem]
    prefix_stem = hazy_stem.split('_')[0]
    if prefix_stem not in stems:
        stems.append(prefix_stem)
    for stem in stems:
        for suffix in IMAGE_SUFFIXES + tuple(s.upper() for s in IMAGE_SUFFIXES):
            candidate = Path(clear_path) / '{}{}'.format(stem, suffix)
            if candidate.exists():
                return candidate
    raise FileNotFoundError('No clear image found for {} in {}'.format(hazy_image_name, clear_path))


def pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def center_crop_pair(hazy_img, clear_img, patch_size):
    if patch_size <= 0:
        return hazy_img, clear_img
    width, height = hazy_img.size
    if width < patch_size or height < patch_size:
        raise ValueError(
            'patch_size {} is larger than image {}x{}'.format(patch_size, width, height)
        )
    left = (width - patch_size) // 2
    top = (height - patch_size) // 2
    box = (left, top, left + patch_size, top + patch_size)
    return hazy_img.crop(box), clear_img.crop(box)


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


def strip_module_prefix(state_dict):
    if not any(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


def load_model(args):
    model = DEANet(
        base_dim=32,
        use_lf_prior=args.use_lf_prior,
        lf_prior_channels=args.lf_prior_channels,
        lf_prior_pool=args.lf_prior_pool,
        lf_prior_gate_init=args.lf_prior_gate_init,
        lf_prior_residual_center=args.lf_prior_residual_center,
        lf_prior_gate_max=args.lf_prior_gate_max,
        lf_prior_injection=args.lf_prior_injection,
        lf_conditional_mask=args.lf_conditional_mask,
        lf_mask_hidden_channels=args.lf_mask_hidden_channels,
        lf_mask_init_bias=args.lf_mask_init_bias,
        lf_haze_aware_mask=args.lf_haze_aware_mask,
        lf_haze_mask_strength=args.lf_haze_mask_strength,
        lf_residual_calibration=args.lf_residual_calibration,
        lf_calib_hidden_channels=args.lf_calib_hidden_channels,
        lf_calib_alpha_max=args.lf_calib_alpha_max,
        lf_residual_selector=args.lf_residual_selector,
        lf_selector_hidden_channels=args.lf_selector_hidden_channels,
        lf_selector_init_bias=args.lf_selector_init_bias
    )
    checkpoint = load_checkpoint(args.checkpoint)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(args.device)
    model.eval()
    return model, checkpoint


def infer_one(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    pred = model(padded).clamp(0, 1)
    return pred[:, :, :h, :w]


def lowpass(x, pool_size):
    return F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)


def residual_direction_stats(out, hazy, target, pool_size, target_norm_floor):
    low_out = lowpass(out, pool_size)
    low_hazy = lowpass(hazy, pool_size)
    low_target = lowpass(target, pool_size)
    pred_residual = low_out - low_hazy
    target_residual = low_target - low_hazy
    pred_vec = pred_residual.reshape(pred_residual.shape[0], -1)
    target_vec = target_residual.reshape(target_residual.shape[0], -1)
    pred_norm = pred_vec.norm(dim=1)
    target_norm = target_vec.norm(dim=1)
    cosine = (pred_vec * target_vec).sum(dim=1) / (pred_norm * target_norm + EPS)
    valid = target_norm > target_norm_floor
    if valid.any():
        valid_cosine = cosine[valid]
        direction_loss = (1.0 - valid_cosine).mean()
        valid_fraction = valid.float().mean()
    else:
        valid_cosine = cosine
        direction_loss = torch.zeros((), device=out.device, dtype=out.dtype)
        valid_fraction = torch.zeros((), device=out.device, dtype=out.dtype)
    return {
        'direction_loss': float(direction_loss.detach().cpu().item()),
        'residual_cosine': float(cosine.mean().detach().cpu().item()),
        'valid_residual_cosine': float(valid_cosine.mean().detach().cpu().item()),
        'valid_fraction': float(valid_fraction.detach().cpu().item()),
        'pred_residual_norm': float(pred_norm.mean().detach().cpu().item()),
        'target_residual_norm': float(target_norm.mean().detach().cpu().item()),
        'lowfreq_l1': float(F.l1_loss(low_out, low_target).detach().cpu().item()),
    }


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def numeric(values):
    return [float(v) for v in values if v != '' and v is not None and not math.isnan(float(v))]


def mean(values):
    values = numeric(values)
    return float(np.mean(values)) if values else ''


def median(values):
    values = numeric(values)
    return float(np.median(values)) if values else ''


def percentile(values, q):
    values = numeric(values)
    return float(np.percentile(values, q)) if values else ''


def safe_ratio(numerator, denominator):
    denominator = float(denominator)
    if abs(denominator) <= EPS:
        return ''
    return float(numerator) / denominator


def summarize(rows, args, checkpoint):
    summary = {
        'dataset': args.dataset,
        'split': args.split,
        'model_label': args.model_label,
        'checkpoint': args.checkpoint,
        'checkpoint_step': checkpoint.get('step') if isinstance(checkpoint, dict) else None,
        'checkpoint_max_psnr': checkpoint.get('max_psnr') if isinstance(checkpoint, dict) else None,
        'num_images': len(rows),
        'lowfreq_pool': args.lowfreq_pool,
        'patch_size': args.patch_size,
        'target_norm_floor': args.target_norm_floor,
        'mean_psnr': mean([row['psnr'] for row in rows]),
        'mean_l1': mean([row['l1'] for row in rows]),
        'median_l1': median([row['l1'] for row in rows]),
        'mean_lowfreq_l1': mean([row['lowfreq_l1'] for row in rows]),
        'mean_direction_loss': mean([row['direction_loss'] for row in rows]),
        'median_direction_loss': median([row['direction_loss'] for row in rows]),
        'p90_direction_loss': percentile([row['direction_loss'] for row in rows], 90),
        'mean_residual_cosine': mean([row['residual_cosine'] for row in rows]),
        'mean_valid_residual_cosine': mean([row['valid_residual_cosine'] for row in rows]),
        'mean_valid_fraction': mean([row['valid_fraction'] for row in rows]),
        'mean_pred_residual_norm': mean([row['pred_residual_norm'] for row in rows]),
        'mean_target_residual_norm': mean([row['target_residual_norm'] for row in rows]),
        'direction_loss_to_l1_ratio': safe_ratio(
            mean([row['direction_loss'] for row in rows]),
            mean([row['l1'] for row in rows])
        ),
        'lowfreq_l1_to_l1_ratio': safe_ratio(
            mean([row['lowfreq_l1'] for row in rows]),
            mean([row['l1'] for row in rows])
        ),
        'candidate_weights': args.candidate_weights,
        'candidate_weighted_ratios': []
    }
    mean_l1 = summary['mean_l1']
    mean_dir = summary['mean_direction_loss']
    for weight in args.candidate_weights:
        weighted = weight * mean_dir
        summary['candidate_weighted_ratios'].append({
            'weight': weight,
            'weighted_direction_loss': weighted,
            'ratio_to_l1': safe_ratio(weighted, mean_l1)
        })
    return summary


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=6):
    if value == '' or value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}'


def write_report(path, summary):
    lines = [
        '# Residual Direction Loss Scale Diagnostic',
        '',
        '## Summary',
        '',
        f"- Dataset: `{summary['dataset']}` / `{summary['split']}`",
        f"- Model: `{summary['model_label']}`",
        f"- Images: `{summary['num_images']}`",
        f"- Checkpoint step: `{summary['checkpoint_step']}`",
        f"- Low-frequency pool: `{summary['lowfreq_pool']}`",
        f"- Patch size: `{summary['patch_size']}`",
        f"- Mean PSNR: `{fmt(summary['mean_psnr'], 4)}`",
        f"- Mean L1: `{fmt(summary['mean_l1'])}`",
        f"- Mean low-frequency L1: `{fmt(summary['mean_lowfreq_l1'])}`",
        f"- Mean residual-direction loss: `{fmt(summary['mean_direction_loss'])}`",
        f"- Mean residual cosine: `{fmt(summary['mean_residual_cosine'])}`",
        f"- Direction-loss / L1 ratio: `{fmt(summary['direction_loss_to_l1_ratio'])}`",
        f"- Low-frequency-L1 / L1 ratio: `{fmt(summary['lowfreq_l1_to_l1_ratio'])}`",
        '',
        '## Candidate Weights',
        '',
    ]
    for item in summary['candidate_weighted_ratios']:
        lines.append(
            f"- `w={item['weight']}`: weighted direction loss `{fmt(item['weighted_direction_loss'])}`, "
            f"ratio to L1 `{fmt(item['ratio_to_l1'])}`"
        )
    lines.extend([
        '',
        '## Reading Guide',
        '',
        '- This is a scale check only; it does not prove the loss will improve training.',
        '- A first scout weight should keep the weighted direction loss small relative to L1.',
        '- If even tiny weights dominate L1, do not launch the training run without changing normalization.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.lowfreq_pool <= 0:
        raise ValueError('--lowfreq_pool must be positive')
    if args.pad_size <= 0:
        raise ValueError('--pad_size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_root = Path('../dataset') / args.dataset / args.split
    hazy_dir, clear_dir = resolve_pair_dirs(split_root)
    hazy_names = [path.name for path in list_image_files(hazy_dir)]
    if args.max_images > 0:
        hazy_names = hazy_names[:args.max_images]

    model, checkpoint = load_model(args)
    rows = []
    with torch.no_grad():
        for idx, name in enumerate(hazy_names, start=1):
            hazy_img = Image.open(hazy_dir / name).convert('RGB')
            clear_img = Image.open(find_clear_image(clear_dir, name)).convert('RGB')
            hazy_img, clear_img = center_crop_pair(hazy_img, clear_img, args.patch_size)
            hazy = pil_to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = pil_to_tensor(clear_img).unsqueeze(0).to(args.device)
            out = infer_one(model, hazy, args.pad_size)
            row = {
                'index': idx,
                'filename': name,
                'height': hazy.shape[-2],
                'width': hazy.shape[-1],
                'psnr': scalar(psnr(out, clear)),
                'l1': float(F.l1_loss(out, clear).detach().cpu().item()),
            }
            row.update(residual_direction_stats(
                out,
                hazy,
                clear,
                args.lowfreq_pool,
                args.target_norm_floor
            ))
            rows.append(row)
            if idx % 50 == 0 or idx == len(hazy_names):
                print('analyzed {}/{}'.format(idx, len(hazy_names)), flush=True)

    summary = summarize(rows, args, checkpoint)
    write_csv(output_dir / 'per_image_loss_scale.csv', rows)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    write_report(output_dir / 'analysis_report.md', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
