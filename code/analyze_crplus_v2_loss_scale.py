import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from metric import psnr
from model import DEANet


EPS = 1e-8
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
VGG_WEIGHTS = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
NEGATIVE_MODES = ('hazy', 'hazy_lowpass', 'output_lowpass', 'under_dehazed_mix')


def parse_float_list(value):
    values = []
    for item in value.split(','):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError('expected at least one float')
    return values


def parse_str_list(value):
    values = []
    for item in value.split(','):
        item = item.strip()
        if item:
            values.append(item)
    if not values:
        raise argparse.ArgumentTypeError('expected at least one value')
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
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--lowpass_pool', type=int, default=8)
    parser.add_argument(
        '--negative_modes',
        type=parse_str_list,
        default=parse_str_list(','.join(NEGATIVE_MODES))
    )
    parser.add_argument(
        '--selected_negative_modes',
        type=parse_str_list,
        default=parse_str_list('hazy,output_lowpass,under_dehazed_mix')
    )
    parser.add_argument('--under_dehazed_mix', type=float, default=0.5)
    parser.add_argument('--frequency_weight', type=float, default=0.1)
    parser.add_argument('--lowfreq_weight', type=float, default=0.1)
    parser.add_argument('--candidate_objective', type=str, default='ratio', choices=['ratio', 'margin'])
    parser.add_argument('--training_margin', type=float, default=0.02)
    parser.add_argument('--margins', type=parse_float_list, default=parse_float_list('0.0,0.01,0.02,0.05'))
    parser.add_argument(
        '--candidate_weights',
        type=parse_float_list,
        default=parse_float_list('0.005,0.01,0.03,0.05,0.1')
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


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            features = models.vgg19(weights=weights).features
        except AttributeError:
            features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


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


def lowpass_image(x, pool_size):
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def lowfreq_l1(a, b, pool_size):
    low_a = F.avg_pool2d(a, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    low_b = F.avg_pool2d(b, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.l1_loss(low_a, low_b)


def frequency_amplitude_l1(a, b):
    amp_a = torch.log1p(torch.abs(torch.fft.rfft2(a, norm='ortho')))
    amp_b = torch.log1p(torch.abs(torch.fft.rfft2(b, norm='ortho')))
    return F.l1_loss(amp_a, amp_b)


def vgg_distance(features_a, features_b):
    total = features_a[0].new_zeros(())
    for weight, feat_a, feat_b in zip(VGG_WEIGHTS, features_a, features_b):
        total = total + weight * F.l1_loss(feat_a, feat_b)
    return total


def make_negative_tensors(out, hazy, args):
    mix = args.under_dehazed_mix
    return {
        'hazy': hazy,
        'hazy_lowpass': lowpass_image(hazy, args.lowpass_pool),
        'output_lowpass': lowpass_image(out, args.lowpass_pool),
        'under_dehazed_mix': (mix * out + (1.0 - mix) * hazy).clamp(0, 1)
    }


def margin_loss(d_pos, d_neg, margin):
    return max(0.0, float(margin) + float(d_pos) - float(d_neg))


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def metric_prefix(metric_name, negative_name):
    return '{}_{}'.format(metric_name, negative_name)


def margin_key(margin):
    return str(margin).replace('-', 'm').replace('.', 'p')


def compute_row(vgg, out, hazy, clear, name, index, args):
    negative_tensors = make_negative_tensors(out, hazy, args)
    invalid = [mode for mode in args.negative_modes if mode not in negative_tensors]
    if invalid:
        raise ValueError('Unsupported negative_modes: {}'.format(', '.join(invalid)))

    out_features = vgg(out)
    clear_features = vgg(clear)
    d_pos_vgg = scalar(vgg_distance(out_features, clear_features))
    d_pos_freq = scalar(frequency_amplitude_l1(out, clear))
    d_pos_lowfreq = scalar(lowfreq_l1(out, clear, args.lowpass_pool))
    d_pos_combined = d_pos_vgg + args.frequency_weight * d_pos_freq + args.lowfreq_weight * d_pos_lowfreq

    row = {
        'index': index,
        'filename': name,
        'height': hazy.shape[-2],
        'width': hazy.shape[-1],
        'psnr': scalar(psnr(out, clear)),
        'l1': scalar(F.l1_loss(out, clear)),
        'd_pos_vgg': d_pos_vgg,
        'd_pos_freq': d_pos_freq,
        'd_pos_lowfreq': d_pos_lowfreq,
        'd_pos_combined': d_pos_combined,
    }

    for mode in args.negative_modes:
        negative = negative_tensors[mode]
        neg_features = vgg(negative)
        d_neg_vgg = scalar(vgg_distance(out_features, neg_features))
        d_neg_freq = scalar(frequency_amplitude_l1(out, negative))
        d_neg_lowfreq = scalar(lowfreq_l1(out, negative, args.lowpass_pool))
        d_neg_combined = d_neg_vgg + args.frequency_weight * d_neg_freq + args.lowfreq_weight * d_neg_lowfreq
        values = {
            'vgg': (d_pos_vgg, d_neg_vgg),
            'freq': (d_pos_freq, d_neg_freq),
            'lowfreq': (d_pos_lowfreq, d_neg_lowfreq),
            'combined': (d_pos_combined, d_neg_combined),
        }
        for metric_name, (d_pos, d_neg) in values.items():
            prefix = metric_prefix(metric_name, mode)
            row[prefix + '_d_neg'] = d_neg
            row[prefix + '_ratio'] = float(d_pos) / (float(d_neg) + EPS)
            row[prefix + '_gap'] = float(d_neg) - float(d_pos)
            for margin in args.margins:
                row[prefix + '_margin_' + margin_key(margin)] = margin_loss(d_pos, d_neg, margin)

    return row


def numeric(values):
    result = []
    for value in values:
        if value == '' or value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        result.append(value)
    return result


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
        'patch_size': args.patch_size,
        'lowpass_pool': args.lowpass_pool,
        'frequency_weight': args.frequency_weight,
        'lowfreq_weight': args.lowfreq_weight,
        'candidate_objective': args.candidate_objective,
        'training_margin': args.training_margin,
        'negative_modes': args.negative_modes,
        'selected_negative_modes': args.selected_negative_modes,
        'mean_psnr': mean([row['psnr'] for row in rows]),
        'mean_l1': mean([row['l1'] for row in rows]),
        'median_l1': median([row['l1'] for row in rows]),
        'mean_d_pos_vgg': mean([row['d_pos_vgg'] for row in rows]),
        'mean_d_pos_freq': mean([row['d_pos_freq'] for row in rows]),
        'mean_d_pos_lowfreq': mean([row['d_pos_lowfreq'] for row in rows]),
        'mean_d_pos_combined': mean([row['d_pos_combined'] for row in rows]),
        'negative_summary': {},
        'candidate_weights': args.candidate_weights,
        'candidate_weighted_ratios': [],
    }

    selected_margin_key = margin_key(args.training_margin)
    selected_losses = []
    selected_margin_losses = []
    selected_ratio_losses = []
    for mode in args.negative_modes:
        mode_summary = {}
        for metric_name in ('vgg', 'freq', 'lowfreq', 'combined'):
            prefix = metric_prefix(metric_name, mode)
            mode_summary[metric_name] = {
                'mean_d_neg': mean([row[prefix + '_d_neg'] for row in rows]),
                'mean_ratio': mean([row[prefix + '_ratio'] for row in rows]),
                'median_ratio': median([row[prefix + '_ratio'] for row in rows]),
                'mean_gap': mean([row[prefix + '_gap'] for row in rows]),
                'p10_gap': percentile([row[prefix + '_gap'] for row in rows], 10),
                'margins': {}
            }
            for margin in args.margins:
                key = prefix + '_margin_' + margin_key(margin)
                mode_summary[metric_name]['margins'][str(margin)] = {
                    'mean_loss': mean([row[key] for row in rows]),
                    'active_fraction': mean([1.0 if row[key] > 0 else 0.0 for row in rows]),
                }
            if metric_name == 'combined' and mode in args.selected_negative_modes:
                selected_key = prefix + '_margin_' + selected_margin_key
                selected_margin_losses.extend([row[selected_key] for row in rows])
                selected_ratio_losses.extend([row[prefix + '_ratio'] for row in rows])
        summary['negative_summary'][mode] = mode_summary

    selected_margin_mean = mean(selected_margin_losses)
    selected_ratio_mean = mean(selected_ratio_losses)
    selected_objective_mean = selected_ratio_mean if args.candidate_objective == 'ratio' else selected_margin_mean
    summary['selected_combined_margin_loss'] = selected_margin_mean
    summary['selected_combined_ratio_loss'] = selected_ratio_mean
    summary['selected_objective_loss'] = selected_objective_mean
    for weight in args.candidate_weights:
        weighted = weight * selected_objective_mean
        summary['candidate_weighted_ratios'].append({
            'weight': weight,
            'objective': args.candidate_objective,
            'weighted_crplus_v2_loss': weighted,
            'ratio_to_l1': safe_ratio(weighted, summary['mean_l1'])
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
        '# CRPlus-v2 Loss Scale Diagnostic',
        '',
        '## Summary',
        '',
        f"- Dataset: `{summary['dataset']}` / `{summary['split']}`",
        f"- Model: `{summary['model_label']}`",
        f"- Images: `{summary['num_images']}`",
        f"- Checkpoint step: `{summary['checkpoint_step']}`",
        f"- Patch size: `{summary['patch_size']}`",
        f"- Low-pass pool: `{summary['lowpass_pool']}`",
        f"- Mean PSNR: `{fmt(summary['mean_psnr'], 4)}`",
        f"- Mean L1: `{fmt(summary['mean_l1'])}`",
        f"- Mean VGG positive distance: `{fmt(summary['mean_d_pos_vgg'])}`",
        f"- Mean frequency positive distance: `{fmt(summary['mean_d_pos_freq'])}`",
        f"- Mean low-frequency positive distance: `{fmt(summary['mean_d_pos_lowfreq'])}`",
        f"- Selected combined margin loss: `{fmt(summary['selected_combined_margin_loss'])}`",
        f"- Selected combined ratio loss: `{fmt(summary['selected_combined_ratio_loss'])}`",
        f"- Candidate objective: `{summary['candidate_objective']}`",
        '',
        '## Negative Candidates',
        '',
    ]
    for mode, mode_summary in summary['negative_summary'].items():
        combined = mode_summary['combined']
        margin_stats = combined['margins'].get(str(summary['training_margin']), {})
        lines.extend([
            f"### {mode}",
            '',
            f"- Combined mean negative distance: `{fmt(combined['mean_d_neg'])}`",
            f"- Combined mean ratio: `{fmt(combined['mean_ratio'])}`",
            f"- Combined mean gap: `{fmt(combined['mean_gap'])}`",
            f"- Combined p10 gap: `{fmt(combined['p10_gap'])}`",
            f"- Margin `{summary['training_margin']}` mean loss: `{fmt(margin_stats.get('mean_loss'))}`",
            f"- Margin `{summary['training_margin']}` active fraction: `{fmt(margin_stats.get('active_fraction'))}`",
            ''
        ])
    lines.extend([
        '## Candidate Weights',
        '',
    ])
    for item in summary['candidate_weighted_ratios']:
        lines.append(
            f"- `w={item['weight']}`: weighted CRPlus-v2 loss `{fmt(item['weighted_crplus_v2_loss'])}`, "
            f"ratio to L1 `{fmt(item['ratio_to_l1'])}`"
        )
    lines.extend([
        '',
        '## Reading Guide',
        '',
        '- This is a read-only scale check; it does not prove a training run will improve PSNR.',
        '- Good candidates have a usable ratio signal without the weighted loss dominating L1.',
        '- Margin values are diagnostic; if the combined margin loss is zero, use bounded ratio or harder staged negatives.',
        '- If a negative has tiny or negative gaps on most samples, do not use it as an early hard negative.',
        '- If the selected weighted loss exceeds the predeclared ratio limit, change margins or curriculum before training.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.lowpass_pool <= 0:
        raise ValueError('--lowpass_pool must be positive')
    if args.pad_size <= 0:
        raise ValueError('--pad_size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    invalid_selected = [mode for mode in args.selected_negative_modes if mode not in args.negative_modes]
    if invalid_selected:
        raise ValueError('selected_negative_modes must be in negative_modes: {}'.format(', '.join(invalid_selected)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_root = Path('../dataset') / args.dataset / args.split
    hazy_dir, clear_dir = resolve_pair_dirs(split_root)
    hazy_names = [path.name for path in list_image_files(hazy_dir)]
    if args.max_images > 0:
        hazy_names = hazy_names[:args.max_images]

    model, checkpoint = load_model(args)
    vgg = Vgg19().to(args.device).eval()
    rows = []
    with torch.no_grad():
        for idx, name in enumerate(hazy_names, start=1):
            hazy_img = Image.open(hazy_dir / name).convert('RGB')
            clear_img = Image.open(find_clear_image(clear_dir, name)).convert('RGB')
            hazy_img, clear_img = center_crop_pair(hazy_img, clear_img, args.patch_size)
            hazy = pil_to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = pil_to_tensor(clear_img).unsqueeze(0).to(args.device)
            out = infer_one(model, hazy, args.pad_size)
            rows.append(compute_row(vgg, out, hazy, clear, name, idx, args))
            if idx % 25 == 0 or idx == len(hazy_names):
                print('analyzed {}/{}'.format(idx, len(hazy_names)), flush=True)

    summary = summarize(rows, args, checkpoint)
    write_csv(output_dir / 'per_image_crplus_v2_scale.csv', rows)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    write_report(output_dir / 'analysis_report.md', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
