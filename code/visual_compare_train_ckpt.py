import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
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
    parser.add_argument('--lf_checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--sample_list', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pad_size', type=int, default=4)
    parser.add_argument('--lf_prior_channels', type=int, default=8)
    parser.add_argument('--lf_prior_pool', type=int, default=8)
    parser.add_argument('--lf_prior_gate_init', type=float, default=0.0)
    parser.add_argument('--lf_gate_scale', type=float, default=1.0)
    parser.add_argument('--lf_label', type=str, default='DEA-Net-LF')
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


def load_model(checkpoint_path, use_lf_prior, args):
    model = DEANet(
        base_dim=32,
        use_lf_prior=use_lf_prior,
        lf_prior_channels=args.lf_prior_channels,
        lf_prior_pool=args.lf_prior_pool,
        lf_prior_gate_init=args.lf_prior_gate_init
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    return model, checkpoint


def apply_lf_gate_scale(model, scale):
    if model.lf_prior is None:
        return None, None
    with torch.no_grad():
        original_gate = float(model.lf_prior.gate.detach().cpu().item())
        model.lf_prior.gate.mul_(float(scale))
        scaled_gate = float(model.lf_prior.gate.detach().cpu().item())
    return original_gate, scaled_gate


def choose_samples(hazy_names, num_samples, sample_list_path):
    if sample_list_path:
        with open(sample_list_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        missing = [name for name in names if name not in hazy_names]
        if missing:
            raise FileNotFoundError('Samples not found in hazy dir: {}'.format(', '.join(missing)))
        return names

    if num_samples <= 0 or num_samples >= len(hazy_names):
        return hazy_names
    indices = np.linspace(0, len(hazy_names) - 1, num_samples, dtype=int)
    return [hazy_names[int(i)] for i in indices]


def tensor_to_pil(x):
    x = x.detach().clamp(0, 1).squeeze(0).cpu()
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    return Image.fromarray(x)


def draw_panel(images, labels, metrics):
    widths, heights = zip(*(img.size for img in images))
    label_h = 46
    gap = 8
    total_w = sum(widths) + gap * (len(images) - 1)
    total_h = max(heights) + label_h
    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)

    x = 0
    for img, label, metric in zip(images, labels, metrics):
        canvas.paste(img, (x, label_h))
        draw.text((x + 4, 4), label, fill=(0, 0, 0))
        if metric:
            draw.text((x + 4, 22), metric, fill=(0, 0, 0))
        x += img.size[0] + gap
    return canvas


def infer_one(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    pred = model(padded).clamp(0, 1)
    return pred[:, :, :h, :w]


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')

    output_dir = Path(args.output_dir)
    panels_dir = output_dir / 'panels'
    baseline_dir = output_dir / 'baseline'
    lf_dir = output_dir / 'lf'
    input_dir = output_dir / 'input'
    gt_dir = output_dir / 'gt'
    for path in (panels_dir, baseline_dir, lf_dir, input_dir, gt_dir):
        path.mkdir(parents=True, exist_ok=True)

    split_root = Path('../dataset') / args.dataset / args.split
    hazy_dir, clear_dir = resolve_pair_dirs(split_root)
    hazy_names = list_image_files(hazy_dir)
    sample_names = choose_samples(hazy_names, args.num_samples, args.sample_list)

    baseline_model, baseline_ckpt = load_model(args.baseline_checkpoint, False, args)
    lf_model, lf_ckpt = load_model(args.lf_checkpoint, True, args)
    lf_original_gate, lf_effective_gate = apply_lf_gate_scale(lf_model, args.lf_gate_scale)

    to_tensor = ToTensor()
    rows = []
    with torch.no_grad():
        for name in sample_names:
            hazy_path = Path(hazy_dir) / name
            clear_path = Path(find_clear_image(clear_dir, name))
            hazy_img = Image.open(hazy_path).convert('RGB')
            clear_img = Image.open(clear_path).convert('RGB')

            hazy = to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = to_tensor(clear_img).unsqueeze(0).to(args.device)

            baseline = infer_one(baseline_model, hazy, args.pad_size)
            lf = infer_one(lf_model, hazy, args.pad_size)

            baseline_psnr = psnr(baseline, clear)
            baseline_ssim = ssim(baseline, clear).item()
            lf_psnr = psnr(lf, clear)
            lf_ssim = ssim(lf, clear).item()

            stem = Path(name).stem
            input_img = tensor_to_pil(hazy)
            baseline_img = tensor_to_pil(baseline)
            lf_img = tensor_to_pil(lf)
            gt_img = tensor_to_pil(clear)

            input_img.save(input_dir / f'{stem}.png')
            baseline_img.save(baseline_dir / f'{stem}.png')
            lf_img.save(lf_dir / f'{stem}.png')
            gt_img.save(gt_dir / f'{stem}.png')

            panel = draw_panel(
                [input_img, baseline_img, lf_img, gt_img],
                ['hazy input', 'DEA-Net-CR baseline', args.lf_label, 'clear GT'],
                [
                    '',
                    'PSNR {:.4f} SSIM {:.4f}'.format(baseline_psnr, baseline_ssim),
                    'PSNR {:.4f} SSIM {:.4f}'.format(lf_psnr, lf_ssim),
                    ''
                ]
            )
            panel.save(panels_dir / f'{stem}.png')

            rows.append({
                'filename': name,
                'baseline_psnr': baseline_psnr,
                'baseline_ssim': baseline_ssim,
                'lf_psnr': lf_psnr,
                'lf_ssim': lf_ssim,
                'delta_psnr': lf_psnr - baseline_psnr,
                'delta_ssim': lf_ssim - baseline_ssim
            })

    with open(output_dir / 'samples.txt', 'w', encoding='utf-8') as f:
        for name in sample_names:
            f.write(name + '\n')

    with open(output_dir / 'metrics.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'dataset': args.dataset,
        'split': args.split,
        'num_samples': len(sample_names),
        'baseline_checkpoint': args.baseline_checkpoint,
        'lf_checkpoint': args.lf_checkpoint,
        'baseline_checkpoint_step': baseline_ckpt.get('step'),
        'lf_checkpoint_step': lf_ckpt.get('step'),
        'lf_gate_scale': args.lf_gate_scale,
        'lf_original_gate': lf_original_gate,
        'lf_effective_gate': lf_effective_gate,
        'mean_baseline_psnr': float(np.mean([row['baseline_psnr'] for row in rows])),
        'mean_baseline_ssim': float(np.mean([row['baseline_ssim'] for row in rows])),
        'mean_lf_psnr': float(np.mean([row['lf_psnr'] for row in rows])),
        'mean_lf_ssim': float(np.mean([row['lf_ssim'] for row in rows])),
        'mean_delta_psnr': float(np.mean([row['delta_psnr'] for row in rows])),
        'mean_delta_ssim': float(np.mean([row['delta_ssim'] for row in rows])),
        'min_delta_psnr': float(np.min([row['delta_psnr'] for row in rows])),
        'max_delta_psnr': float(np.max([row['delta_psnr'] for row in rows]))
    }
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
