import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--input_dir_name', type=str, default='input')
    parser.add_argument('--baseline_dir_name', type=str, default='baseline')
    parser.add_argument('--current_dir_name', type=str, default='lf')
    parser.add_argument('--gt_dir_name', type=str, default='gt')
    parser.add_argument('--baseline_label', type=str, default='baseline')
    parser.add_argument('--current_label', type=str, default='current')
    parser.add_argument('--save_heatmaps', action='store_true')
    parser.add_argument('--save_diagnostic_panels', action='store_true')
    parser.add_argument('--top_k', type=int, default=8)
    return parser.parse_args()


def list_images(path):
    return sorted(p.name for p in Path(path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_rgb(path):
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def resize_like(img, ref):
    if img.shape[:2] == ref.shape[:2]:
        return img
    h, w = ref.shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def psnr_np(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_np(img1, img2):
    img1 = np.clip(img1, 0, 1).astype(np.float32)
    img2 = np.clip(img2, 0, 1).astype(np.float32)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    vals = []
    for ch in range(3):
        x = img1[:, :, ch]
        y = img2[:, :, ch]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        vals.append(float(np.mean(ssim_map)))
    return float(np.mean(vals))


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


def edge_features(img):
    y = luminance(img).astype(np.float32)
    sobel_x = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    lap = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
    blur = cv2.GaussianBlur(y, (0, 0), 3.0)
    high = y - blur
    return {
        'edge_mean': float(np.mean(mag)),
        'edge_p90': float(np.percentile(mag, 90)),
        'lap_var': float(np.var(lap)),
        'highfreq_abs_mean': float(np.mean(np.abs(high))),
        'sobel_map': mag,
    }


def image_features(img):
    y = luminance(img)
    hsv = hsv_image(img)
    dc = dark_channel(img)
    edges = edge_features(img)
    return {
        'luma_mean': float(np.mean(y)),
        'luma_std': float(np.std(y)),
        'saturation_mean': float(np.mean(hsv[:, :, 1])),
        'dark_channel_mean': float(np.mean(dc)),
        'dark_channel_p90': float(np.percentile(dc, 90)),
        'edge_mean': edges['edge_mean'],
        'edge_p90': edges['edge_p90'],
        'lap_var': edges['lap_var'],
        'highfreq_abs_mean': edges['highfreq_abs_mean'],
        'sobel_map': edges['sobel_map'],
    }


def compare_to_gt(img, gt):
    diff = img - gt
    abs_diff = np.abs(diff)
    img_lab = lab_image(img)
    gt_lab = lab_image(gt)
    delta_e = np.linalg.norm(img_lab - gt_lab, axis=2)
    f_img = image_features(img)
    f_gt = image_features(gt)
    edge_error = np.mean(np.abs(f_img['sobel_map'] - f_gt['sobel_map']))
    return {
        'psnr': psnr_np(img, gt),
        'ssim': ssim_np(img, gt),
        'mae': float(np.mean(abs_diff)),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'delta_e_mean': float(np.mean(delta_e)),
        'delta_e_p95': float(np.percentile(delta_e, 95)),
        'luma_mean': f_img['luma_mean'],
        'luma_std': f_img['luma_std'],
        'luma_bias': f_img['luma_mean'] - f_gt['luma_mean'],
        'luma_std_delta': f_img['luma_std'] - f_gt['luma_std'],
        'saturation_mean': f_img['saturation_mean'],
        'saturation_bias': f_img['saturation_mean'] - f_gt['saturation_mean'],
        'dark_channel_mean': f_img['dark_channel_mean'],
        'dark_channel_bias': f_img['dark_channel_mean'] - f_gt['dark_channel_mean'],
        'dark_channel_abs_bias': abs(f_img['dark_channel_mean'] - f_gt['dark_channel_mean']),
        'edge_mean': f_img['edge_mean'],
        'edge_mean_bias': f_img['edge_mean'] - f_gt['edge_mean'],
        'edge_error': float(edge_error),
        'lap_var': f_img['lap_var'],
        'lap_var_ratio_gt': f_img['lap_var'] / max(f_gt['lap_var'], 1e-8),
        'highfreq_abs_mean': f_img['highfreq_abs_mean'],
        'highfreq_bias': f_img['highfreq_abs_mean'] - f_gt['highfreq_abs_mean'],
    }


def prefix_metrics(prefix, metrics):
    return {f'{prefix}_{k}': v for k, v in metrics.items() if not isinstance(v, np.ndarray)}


def risk_tags(baseline, current):
    tags = []
    if current['psnr'] - baseline['psnr'] > 0.30 and current['mae'] < baseline['mae']:
        tags.append('current_better_objective')
    elif current['psnr'] - baseline['psnr'] < -0.30 and current['mae'] > baseline['mae']:
        tags.append('current_worse_objective')
    else:
        tags.append('mixed_or_neutral')

    if current['delta_e_mean'] - baseline['delta_e_mean'] > 1.0:
        tags.append('color_shift_risk')
    if abs(current['luma_bias']) - abs(baseline['luma_bias']) > 0.02:
        tags.append('tone_shift_risk')
    if abs(current['saturation_bias']) - abs(baseline['saturation_bias']) > 0.03:
        tags.append('saturation_shift_risk')
    if current['dark_channel_abs_bias'] - baseline['dark_channel_abs_bias'] > 0.015 and current['luma_std'] < baseline['luma_std']:
        tags.append('residual_haze_or_low_contrast_risk')
    if current['edge_error'] - baseline['edge_error'] > 0.02 and current['lap_var_ratio_gt'] > 1.25:
        tags.append('oversharpen_or_halo_risk')
    if current['edge_mean'] < baseline['edge_mean'] - 0.02 and current['highfreq_abs_mean'] < baseline['highfreq_abs_mean']:
        tags.append('oversmooth_or_texture_loss_risk')
    return tags


def normalize_map(values, percentile=98.0):
    vmax = float(np.percentile(values, percentile))
    if vmax <= 1e-8:
        vmax = float(np.max(values))
    if vmax <= 1e-8:
        return np.zeros_like(values, dtype=np.uint8)
    return np.clip(values / vmax * 255.0, 0, 255).astype(np.uint8)


def error_heatmap(img, gt):
    err = np.mean(np.abs(img - gt), axis=2)
    norm = normalize_map(err)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def improvement_heatmap(baseline, current, gt):
    baseline_err = np.mean(np.abs(baseline - gt), axis=2)
    current_err = np.mean(np.abs(current - gt), axis=2)
    delta = baseline_err - current_err
    scale = float(np.percentile(np.abs(delta), 98))
    if scale <= 1e-8:
        scale = float(np.max(np.abs(delta)))
    if scale <= 1e-8:
        return np.zeros((*delta.shape, 3), dtype=np.uint8)
    mag = np.clip(np.abs(delta) / scale, 0, 1)
    rgb = np.zeros((*delta.shape, 3), dtype=np.float32)
    rgb[:, :, 1] = mag * (delta > 0)
    rgb[:, :, 0] = mag * (delta < 0)
    rgb[:, :, 2] = 0.15 * mag
    return (rgb * 255.0).astype(np.uint8)


def pil_from_array(img):
    return Image.fromarray(to_uint8(img) if img.dtype != np.uint8 else img)


def draw_labeled_panel(items):
    label_h = 44
    gap = 8
    pil_images = [(label, pil_from_array(img).convert('RGB')) for label, img in items]
    widths, heights = zip(*(im.size for _, im in pil_images))
    total_w = sum(widths) + gap * (len(pil_images) - 1)
    total_h = max(heights) + label_h
    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, im in pil_images:
        canvas.paste(im, (x, label_h))
        draw.text((x + 4, 6), label, fill=(0, 0, 0))
        x += im.size[0] + gap
    return canvas


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_value(rows, key):
    vals = [float(row[key]) for row in rows]
    return float(np.mean(vals)) if vals else 0.0


def top_rows(rows, key, top_k, reverse=True):
    return sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:top_k]


def markdown_table(rows, columns):
    out = []
    out.append('| ' + ' | '.join(columns) + ' |')
    out.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f'{val:.4f}')
            else:
                vals.append(str(val))
        out.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)


def write_report(path, rows, summary, args):
    top_k = args.top_k
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Objective Visual Comparison Analysis\n\n')
        f.write(f"- Compare dir: `{args.compare_dir}`\n")
        f.write(f"- Baseline: `{args.baseline_label}`\n")
        f.write(f"- Current: `{args.current_label}`\n")
        f.write(f"- Samples: `{len(rows)}`\n\n")

        f.write('## Summary\n\n')
        for key, val in summary.items():
            if key == 'tag_counts':
                continue
            if isinstance(val, float):
                f.write(f"- `{key}`: `{val:.6f}`\n")
            else:
                f.write(f"- `{key}`: `{val}`\n")
        f.write('\n')
        f.write('Tag counts:\n\n')
        for tag, count in summary['tag_counts'].items():
            f.write(f"- `{tag}`: `{count}`\n")
        f.write('\n')

        cols = ['filename', 'delta_psnr', 'delta_ssim', 'delta_mae_improvement', 'delta_e_improvement', 'tags']
        f.write(f'## Top {top_k} Current Improvements By PSNR\n\n')
        f.write(markdown_table(top_rows(rows, 'delta_psnr', top_k, True), cols))
        f.write('\n\n')
        f.write(f'## Top {top_k} Current Regressions By PSNR\n\n')
        f.write(markdown_table(top_rows(rows, 'delta_psnr', top_k, False), cols))
        f.write('\n\n')
        f.write(f'## Top {top_k} Color Regressions\n\n')
        f.write(markdown_table(top_rows(rows, 'delta_e_regression', top_k, True), cols))
        f.write('\n\n')
        f.write('## Reading Notes\n\n')
        f.write('- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.\n')
        f.write('- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.\n')
        f.write('- Risk tags are objective hints for triage, not final visual judgments.\n')


def main():
    args = parse_args()
    compare_dir = Path(args.compare_dir)
    output_dir = Path(args.output_dir) if args.output_dir else compare_dir / 'analysis'
    heatmap_dir = output_dir / 'heatmaps'
    panel_dir = output_dir / 'diagnostic_panels'
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    if args.save_diagnostic_panels:
        panel_dir.mkdir(parents=True, exist_ok=True)

    input_dir = compare_dir / args.input_dir_name
    baseline_dir = compare_dir / args.baseline_dir_name
    current_dir = compare_dir / args.current_dir_name
    gt_dir = compare_dir / args.gt_dir_name
    for path in (input_dir, baseline_dir, current_dir, gt_dir):
        if not path.is_dir():
            raise FileNotFoundError(f'Missing image directory: {path}')

    names = sorted(set(list_images(gt_dir)) & set(list_images(input_dir)) & set(list_images(baseline_dir)) & set(list_images(current_dir)))
    if not names:
        raise RuntimeError('No common image names found across input/baseline/current/gt directories')

    rows = []
    for name in names:
        input_img = read_rgb(input_dir / name)
        gt = read_rgb(gt_dir / name)
        baseline = resize_like(read_rgb(baseline_dir / name), gt)
        current = resize_like(read_rgb(current_dir / name), gt)
        input_img = resize_like(input_img, gt)

        input_metrics = compare_to_gt(input_img, gt)
        baseline_metrics = compare_to_gt(baseline, gt)
        current_metrics = compare_to_gt(current, gt)
        tags = risk_tags(baseline_metrics, current_metrics)

        row = {'filename': name}
        row.update(prefix_metrics('input', input_metrics))
        row.update(prefix_metrics('baseline', baseline_metrics))
        row.update(prefix_metrics('current', current_metrics))
        row.update({
            'delta_psnr': current_metrics['psnr'] - baseline_metrics['psnr'],
            'delta_ssim': current_metrics['ssim'] - baseline_metrics['ssim'],
            'delta_mae_improvement': baseline_metrics['mae'] - current_metrics['mae'],
            'delta_rmse_improvement': baseline_metrics['rmse'] - current_metrics['rmse'],
            'delta_e_improvement': baseline_metrics['delta_e_mean'] - current_metrics['delta_e_mean'],
            'delta_e_regression': current_metrics['delta_e_mean'] - baseline_metrics['delta_e_mean'],
            'luma_abs_bias_improvement': abs(baseline_metrics['luma_bias']) - abs(current_metrics['luma_bias']),
            'saturation_abs_bias_improvement': abs(baseline_metrics['saturation_bias']) - abs(current_metrics['saturation_bias']),
            'dark_channel_abs_bias_improvement': baseline_metrics['dark_channel_abs_bias'] - current_metrics['dark_channel_abs_bias'],
            'edge_error_improvement': baseline_metrics['edge_error'] - current_metrics['edge_error'],
            'tags': ';'.join(tags),
        })
        rows.append(row)

        if args.save_heatmaps or args.save_diagnostic_panels:
            b_err = error_heatmap(baseline, gt)
            c_err = error_heatmap(current, gt)
            imp = improvement_heatmap(baseline, current, gt)
            stem = Path(name).stem
            if args.save_heatmaps:
                Image.fromarray(b_err).save(heatmap_dir / f'{stem}_baseline_error.png')
                Image.fromarray(c_err).save(heatmap_dir / f'{stem}_current_error.png')
                Image.fromarray(imp).save(heatmap_dir / f'{stem}_current_vs_baseline_improvement.png')
            if args.save_diagnostic_panels:
                panel = draw_labeled_panel([
                    ('input', input_img),
                    (args.baseline_label, baseline),
                    (args.current_label, current),
                    ('GT', gt),
                    ('baseline error', b_err),
                    ('current error', c_err),
                    ('green better / red worse', imp),
                ])
                panel.save(panel_dir / f'{stem}.png')

    tag_counts = Counter()
    for row in rows:
        tag_counts.update(tag for tag in row['tags'].split(';') if tag)

    summary = {
        'num_samples': len(rows),
        'mean_input_psnr': mean_value(rows, 'input_psnr'),
        'mean_baseline_psnr': mean_value(rows, 'baseline_psnr'),
        'mean_current_psnr': mean_value(rows, 'current_psnr'),
        'mean_delta_psnr': mean_value(rows, 'delta_psnr'),
        'mean_input_ssim': mean_value(rows, 'input_ssim'),
        'mean_baseline_ssim': mean_value(rows, 'baseline_ssim'),
        'mean_current_ssim': mean_value(rows, 'current_ssim'),
        'mean_delta_ssim': mean_value(rows, 'delta_ssim'),
        'mean_delta_mae_improvement': mean_value(rows, 'delta_mae_improvement'),
        'mean_delta_e_improvement': mean_value(rows, 'delta_e_improvement'),
        'mean_luma_abs_bias_improvement': mean_value(rows, 'luma_abs_bias_improvement'),
        'mean_saturation_abs_bias_improvement': mean_value(rows, 'saturation_abs_bias_improvement'),
        'mean_dark_channel_abs_bias_improvement': mean_value(rows, 'dark_channel_abs_bias_improvement'),
        'mean_edge_error_improvement': mean_value(rows, 'edge_error_improvement'),
        'tag_counts': dict(sorted(tag_counts.items())),
    }

    write_csv(output_dir / 'analysis_metrics.csv', rows)
    with open(output_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_report(output_dir / 'analysis_report.md', rows, summary, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
