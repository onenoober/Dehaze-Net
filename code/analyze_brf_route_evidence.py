import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_image_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()


def coerce(value):
    if value in ('', 'True', 'False'):
        if value == 'True':
            return True
        if value == 'False':
            return False
        return ''
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return [{key: coerce(value) for key, value in row.items()} for row in reader]


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


def summarize(rows):
    gain_rows = [row for row in rows if row.get('lfv1_gain_group') is True]
    regression_rows = [row for row in rows if row.get('lfv1_regression_group') is True]
    preservation_count = sum(1 for row in gain_rows if row['delta_brf_vs_cr'] >= 0)
    rescue_count = sum(1 for row in regression_rows if row['delta_brf_vs_lfv1'] > 0)
    return {
        'num_images': len(rows),
        'mean_psnr_cr': mean([row['psnr_cr'] for row in rows]),
        'mean_psnr_lfv1': mean([row['psnr_lfv1'] for row in rows]),
        'mean_psnr_brf': mean([row['psnr_brf'] for row in rows]),
        'delta_brf_vs_cr': mean([row['delta_brf_vs_cr'] for row in rows]),
        'delta_brf_vs_lfv1': mean([row['delta_brf_vs_lfv1'] for row in rows]),
        'wrong_direction_count': sum(1 for row in rows if row['wrong_direction'] is True),
        'mean_residual_cosine': mean([row['residual_cosine'] for row in rows]),
        'median_residual_cosine': median([row['residual_cosine'] for row in rows]),
        'p10_residual_cosine': percentile([row['residual_cosine'] for row in rows], 10),
        'lf_mse_improved_count': sum(1 for row in rows if row['lf_mse_delta'] < 0),
        'lf_mse_regressed_count': sum(1 for row in rows if row['lf_mse_delta'] > 0),
        'weak_cr_delta': mean([row['delta_brf_vs_cr'] for row in rows if row.get('weak_cr_group') is True]),
        'strong_cr_delta': mean([row['delta_brf_vs_cr'] for row in rows if row.get('strong_cr_group') is True]),
        'lfv1_gain_preservation_count': preservation_count,
        'lfv1_gain_preservation_rate': preservation_count / len(gain_rows) if gain_rows else '',
        'lfv1_regression_rescue_count': rescue_count,
        'lfv1_regression_rescue_rate': rescue_count / len(regression_rows) if regression_rows else '',
        'gate_lf_mean_global': mean([row['gate_lf_mean'] for row in rows]),
        'gate_lf_std_global': mean([row['gate_lf_std'] for row in rows]),
        'gate_hf_mean_global': mean([row['gate_hf_mean'] for row in rows]),
    }


def group_rows(rows):
    groups = [
        ('all', lambda row: True),
        ('weak_cr', lambda row: row.get('weak_cr_group') is True),
        ('strong_cr', lambda row: row.get('strong_cr_group') is True),
        ('lfv1_gain', lambda row: row.get('lfv1_gain_group') is True),
        ('lfv1_regression', lambda row: row.get('lfv1_regression_group') is True),
    ]
    output = []
    for name, predicate in groups:
        subset = [row for row in rows if predicate(row)]
        if not subset:
            continue
        output.append({
            'group': name,
            'num_images': len(subset),
            'mean_delta_brf_vs_cr': mean([row['delta_brf_vs_cr'] for row in subset]),
            'mean_delta_brf_vs_lfv1': mean([row['delta_brf_vs_lfv1'] for row in subset]),
            'wrong_direction_count': sum(1 for row in subset if row['wrong_direction'] is True),
            'lf_mse_improved_count': sum(1 for row in subset if row['lf_mse_delta'] < 0),
            'lf_mse_regressed_count': sum(1 for row in subset if row['lf_mse_delta'] > 0),
            'mean_residual_cosine': mean([row['residual_cosine'] for row in subset]),
            'mean_gate_lf': mean([row['gate_lf_mean'] for row in subset]),
            'mean_gate_hf': mean([row['gate_hf_mean'] for row in subset]),
        })
    return output


def write_report(path, summary, groups):
    lines = [
        '# CBRFRC Route Evidence',
        '',
        '## Summary',
        '',
        f"- Images: `{summary['num_images']}`",
        f"- Mean PSNR CR / LF-v1 / CBRFRC: `{summary['mean_psnr_cr']:.4f}` / `{summary['mean_psnr_lfv1']:.4f}` / `{summary['mean_psnr_brf']:.4f}`",
        f"- Delta CBRFRC vs CR / LF-v1: `{summary['delta_brf_vs_cr']:.4f}` / `{summary['delta_brf_vs_lfv1']:.4f}`",
        f"- Wrong-direction count: `{summary['wrong_direction_count']}`",
        f"- Residual cosine mean / median / p10: `{summary['mean_residual_cosine']:.4f}` / `{summary['median_residual_cosine']:.4f}` / `{summary['p10_residual_cosine']:.4f}`",
        f"- LF MSE improved / regressed: `{summary['lf_mse_improved_count']}` / `{summary['lf_mse_regressed_count']}`",
        f"- Weak / strong CR delta: `{summary['weak_cr_delta']:.4f}` / `{summary['strong_cr_delta']:.4f}`",
        f"- LF-v1 gain preservation: `{summary['lfv1_gain_preservation_count']}` / rate `{summary['lfv1_gain_preservation_rate']}`",
        f"- LF-v1 regression rescue: `{summary['lfv1_regression_rescue_count']}` / rate `{summary['lfv1_regression_rescue_rate']}`",
        f"- Gate LF mean / HF mean: `{summary['gate_lf_mean_global']:.6f}` / `{summary['gate_hf_mean_global']:.6f}`",
        '',
        '## Groups',
        '',
    ]
    for group in groups:
        lines.append(
            f"- `{group['group']}` n=`{group['num_images']}` "
            f"delta_cr `{group['mean_delta_brf_vs_cr']:.4f}`, "
            f"cos `{group['mean_residual_cosine']:.4f}`, "
            f"LF MSE improved/regressed `{group['lf_mse_improved_count']}/{group['lf_mse_regressed_count']}`"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.per_image_csv)
    summary = summarize(rows)
    groups = group_rows(rows)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    write_csv(output_dir / 'group_summary.csv', groups)
    write_report(output_dir / 'analysis_report.md', summary, groups)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
