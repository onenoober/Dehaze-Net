import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


EPS = 1e-12
MODEL_NAMES = ('lfv1', 'residual')
SAFE_OUTPUT_STATS = (
    'luma_mean',
    'luma_std',
    'saturation_mean',
    'dark_channel_mean',
    'edge_mean',
    'lap_var',
    'highfreq_abs_mean',
)
METADATA_FEATURES = ('airlight', 'beta', 'height', 'width')
GT_DIAGNOSTIC_FEATURES = (
    'lfv1_from_baseline_residual_cosine',
    'lfv1_from_baseline_luma_residual_cosine',
    'lfv1_from_baseline_residual_norm_ratio',
    'lfv1_from_baseline_residual_error_ratio',
    'lfv1_from_baseline_lf_mse_delta',
    'rescalib_from_baseline_residual_cosine',
    'rescalib_from_baseline_luma_residual_cosine',
    'rescalib_from_baseline_residual_norm_ratio',
    'rescalib_from_baseline_residual_error_ratio',
    'rescalib_from_baseline_lf_mse_delta',
    'rescalib_from_lfv1_residual_cosine',
    'rescalib_from_lfv1_luma_residual_cosine',
    'rescalib_from_lfv1_residual_norm_ratio',
    'rescalib_from_lfv1_residual_error_ratio',
    'rescalib_from_lfv1_lf_mse_delta',
    'residual_color_regression_vs_lfv1',
    'residual_luma_abs_bias_regression_vs_lfv1',
    'residual_dark_channel_abs_bias_regression_vs_lfv1',
    'residual_edge_error_regression_vs_lfv1',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Audit whether GT-free proxy features can learn the LF-v1 vs '
            'ResidualCalib selector target before launching another scout.'
        )
    )
    parser.add_argument('--three_way_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sample_list', type=str, default='')
    parser.add_argument('--seed', type=int, default=20260526)
    parser.add_argument('--valid_fraction', type=float, default=0.30)
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--threshold_steps', type=int, default=9)
    parser.add_argument('--min_select_fraction', type=float, default=0.05)
    parser.add_argument('--logistic_steps', type=int, default=250)
    parser.add_argument('--logistic_lr', type=float, default=0.10)
    parser.add_argument('--logistic_l2', type=float, default=0.10)
    parser.add_argument('--min_gain', type=float, default=0.12)
    parser.add_argument('--min_oracle_recovery', type=float, default=0.20)
    parser.add_argument('--min_precision', type=float, default=0.65)
    return parser.parse_args()


def parse_value(value):
    if value is None:
        return ''
    value = value.strip()
    if value == '':
        return ''
    try:
        return float(value)
    except ValueError:
        return value


def is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def read_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = [{key: parse_value(value) for key, value in raw.items()} for raw in reader]
    if not rows:
        raise ValueError('No rows found in {}'.format(path))
    return rows


def filter_rows(rows, sample_list):
    if not sample_list:
        return rows
    with open(sample_list, 'r', encoding='utf-8') as f:
        wanted = [line.strip() for line in f if line.strip()]
    by_name = {row['filename']: row for row in rows}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise FileNotFoundError('Missing samples: {}'.format(', '.join(missing[:20])))
    return [by_name[name] for name in wanted]


def add_proxy_deltas(rows):
    pairs = (
        ('residual', 'lfv1'),
        ('residual', 'baseline'),
        ('lfv1', 'baseline'),
    )
    for row in rows:
        for left, right in pairs:
            for stat in SAFE_OUTPUT_STATS:
                left_key = '{}_{}'.format(left, stat)
                right_key = '{}_{}'.format(right, stat)
                out_key = '{}_{}_{}_delta'.format(left, right, stat)
                left_value = row.get(left_key, '')
                right_value = row.get(right_key, '')
                if is_number(left_value) and is_number(right_value):
                    delta = left_value - right_value
                    row[out_key] = delta
                    row[out_key + '_abs'] = abs(delta)
                else:
                    row[out_key] = ''
                    row[out_key + '_abs'] = ''


def numeric_feature_names(rows, names):
    out = []
    for name in names:
        values = [row.get(name, '') for row in rows]
        if sum(1 for value in values if is_number(value)) >= max(5, len(rows) // 10):
            out.append(name)
    return out


def output_proxy_features(rows):
    names = []
    for prefix in ('baseline', 'lfv1', 'residual'):
        for stat in SAFE_OUTPUT_STATS:
            names.append('{}_{}'.format(prefix, stat))
    for row_key in rows[0].keys():
        if row_key.endswith('_delta') or row_key.endswith('_delta_abs'):
            if any(stat in row_key for stat in SAFE_OUTPUT_STATS):
                names.append(row_key)
    return numeric_feature_names(rows, sorted(set(names)))


def metadata_features(rows):
    return numeric_feature_names(rows, METADATA_FEATURES)


def gt_diagnostic_features(rows):
    return numeric_feature_names(rows, GT_DIAGNOSTIC_FEATURES)


def build_feature_sets(rows):
    output = output_proxy_features(rows)
    metadata = metadata_features(rows)
    gt_diag = gt_diagnostic_features(rows)
    return {
        'output_proxy': output,
        'metadata_proxy': metadata,
        'output_plus_metadata_proxy': sorted(set(output + metadata)),
        'gt_diagnostic_leakage_check': gt_diag,
    }


def label(row):
    return 1 if row['residual_psnr'] >= row['lfv1_psnr'] else 0


def selected_model(row, residual_selected):
    return 'residual' if residual_selected else 'lfv1'


def mean(values):
    values = [float(v) for v in values if is_number(v)]
    return float(np.mean(values)) if values else ''


def std(values):
    values = [float(v) for v in values if is_number(v)]
    return float(np.std(values)) if values else ''


def selection_metrics(rows, indices, residual_flags, name, feature_set, model_type):
    if len(indices) != len(residual_flags):
        raise ValueError('indices and residual_flags length mismatch')
    records = []
    for idx, choose_residual in zip(indices, residual_flags):
        row = rows[idx]
        model = selected_model(row, choose_residual)
        records.append({
            'selected_model': model,
            'selected_psnr': row['{}_psnr'.format(model)],
            'selected_ssim': row['{}_ssim'.format(model)],
            'delta_lfv1_psnr': row['{}_psnr'.format(model)] - row['lfv1_psnr'],
            'delta_baseline_psnr': row['{}_psnr'.format(model)] - row['baseline_psnr'],
            'label': label(row),
            'pattern': row.get('pattern', ''),
        })

    residual_selected = [record for record in records if record['selected_model'] == 'residual']
    residual_win_total = sum(1 for record in records if record['label'] == 1)
    true_residual = sum(1 for record in residual_selected if record['label'] == 1)
    precision = true_residual / len(residual_selected) if residual_selected else ''
    recall = true_residual / residual_win_total if residual_win_total else ''
    accuracy = sum(
        1 for record in records
        if (record['selected_model'] == 'residual') == bool(record['label'])
    ) / len(records)

    return {
        'name': name,
        'feature_set': feature_set,
        'model_type': model_type,
        'num_images': len(records),
        'mean_psnr': mean([record['selected_psnr'] for record in records]),
        'mean_ssim': mean([record['selected_ssim'] for record in records]),
        'gain_vs_lfv1_psnr': mean([record['delta_lfv1_psnr'] for record in records]),
        'gain_vs_baseline_psnr': mean([record['delta_baseline_psnr'] for record in records]),
        'residual_selection_count': sum(1 for record in records if record['selected_model'] == 'residual'),
        'lfv1_selection_count': sum(1 for record in records if record['selected_model'] == 'lfv1'),
        'residual_precision': precision,
        'residual_recall': recall,
        'two_way_accuracy': accuracy,
        'lost_lfv1_gain_selected_count': sum(
            1 for record in records
            if record['selected_model'] == 'residual' and record['pattern'] == 'lost_lfv1_gain'
        ),
        'residual_worst_selected_count': sum(
            1 for record in records
            if record['selected_model'] == 'residual' and record['pattern'] == 'residual_worst'
        ),
    }


def stratified_split(rows, seed, valid_fraction):
    rng = random.Random(seed)
    positives = [idx for idx, row in enumerate(rows) if label(row) == 1]
    negatives = [idx for idx, row in enumerate(rows) if label(row) == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_valid = max(1, int(round(len(positives) * valid_fraction)))
    neg_valid = max(1, int(round(len(negatives) * valid_fraction)))
    valid = positives[:pos_valid] + negatives[:neg_valid]
    train = positives[pos_valid:] + negatives[neg_valid:]
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def matrix(rows, indices, features):
    data = []
    for idx in indices:
        row = rows[idx]
        data.append([
            float(row[name]) if is_number(row.get(name, '')) else np.nan
            for name in features
        ])
    return np.array(data, dtype=np.float64)


def standardize(train_x, valid_x):
    means = np.nanmean(train_x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train_filled = np.where(np.isnan(train_x), means, train_x)
    valid_filled = np.where(np.isnan(valid_x), means, valid_x)
    stds = np.nanstd(train_filled, axis=0)
    stds = np.where(stds > EPS, stds, 1.0)
    return (train_filled - means) / stds, (valid_filled - means) / stds, means, stds


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_logistic(train_x, train_y, steps, lr, l2):
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    positive_rate = min(max(float(np.mean(train_y)), 1e-4), 1.0 - 1e-4)
    bias = math.log(positive_rate / (1.0 - positive_rate))
    for _ in range(steps):
        probs = sigmoid(train_x.dot(weights) + bias)
        error = probs - train_y
        grad_w = train_x.T.dot(error) / len(train_y) + l2 * weights
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def choose_threshold(rows, train_idx, probs, min_count):
    thresholds = sorted(set(float(v) for v in np.percentile(probs, np.linspace(0, 100, 41))))
    thresholds.append(0.5)
    best = None
    for threshold in thresholds:
        flags = [prob >= threshold for prob in probs]
        residual_count = sum(flags)
        lfv1_count = len(flags) - residual_count
        if residual_count < min_count or lfv1_count < min_count:
            continue
        metrics = selection_metrics(rows, train_idx, flags, 'logistic_train_threshold', '', 'logistic')
        if best is None or metrics['mean_psnr'] > best['metrics']['mean_psnr']:
            best = {'threshold': threshold, 'metrics': metrics}
    if best is None:
        return 0.5
    return best['threshold']


def thresholds_for(values, steps):
    clean = np.array([float(v) for v in values if is_number(v)], dtype=np.float64)
    if clean.size <= 1:
        return []
    quantiles = np.linspace(0, 100, steps + 2)[1:-1]
    return sorted(set(float(v) for v in np.percentile(clean, quantiles)))


def train_stump(rows, train_idx, features, min_count, steps):
    best = None
    for feature in features:
        values = [rows[idx].get(feature, '') for idx in train_idx]
        if any(not is_number(value) for value in values):
            continue
        for threshold in thresholds_for(values, steps):
            for op in ('>=', '<='):
                if op == '>=':
                    flags = [rows[idx][feature] >= threshold for idx in train_idx]
                else:
                    flags = [rows[idx][feature] <= threshold for idx in train_idx]
                residual_count = sum(flags)
                lfv1_count = len(flags) - residual_count
                if residual_count < min_count or lfv1_count < min_count:
                    continue
                metrics = selection_metrics(rows, train_idx, flags, 'stump_train', '', 'stump')
                candidate = {
                    'feature': feature,
                    'op': op,
                    'threshold': threshold,
                    'train_mean_psnr': metrics['mean_psnr'],
                    'train_gain_vs_lfv1_psnr': metrics['gain_vs_lfv1_psnr'],
                    'train_residual_precision': metrics['residual_precision'],
                    'train_residual_recall': metrics['residual_recall'],
                }
                if best is None or candidate['train_mean_psnr'] > best['train_mean_psnr']:
                    best = candidate
    return best


def predict_stump(rows, indices, stump):
    if stump is None:
        return [False for _ in indices]
    feature = stump['feature']
    threshold = stump['threshold']
    if stump['op'] == '>=':
        return [rows[idx][feature] >= threshold for idx in indices]
    return [rows[idx][feature] <= threshold for idx in indices]


def safe_ratio(numerator, denominator):
    if not is_number(numerator) or not is_number(denominator) or abs(denominator) <= EPS:
        return ''
    return numerator / denominator


def aggregate_metrics(metric_rows):
    groups = defaultdict(list)
    for row in metric_rows:
        key = (row['feature_set'], row['model_type'], row['name'])
        groups[key].append(row)
    out = []
    for (feature_set, model_type, name), rows in groups.items():
        item = {
            'feature_set': feature_set,
            'model_type': model_type,
            'name': name,
            'splits': len(rows),
        }
        for metric in (
            'mean_psnr',
            'mean_ssim',
            'gain_vs_lfv1_psnr',
            'gain_vs_baseline_psnr',
            'residual_precision',
            'residual_recall',
            'two_way_accuracy',
            'residual_selection_count',
            'lost_lfv1_gain_selected_count',
            'residual_worst_selected_count',
        ):
            values = [row[metric] for row in rows]
            item[metric + '_mean'] = mean(values)
            item[metric + '_std'] = std(values)
        item['oracle_recovery_mean'] = mean([row.get('oracle_recovery', '') for row in rows])
        item['oracle_recovery_std'] = std([row.get('oracle_recovery', '') for row in rows])
        out.append(item)
    out.sort(key=lambda row: (
        row['gain_vs_lfv1_psnr_mean'] if is_number(row['gain_vs_lfv1_psnr_mean']) else -999,
        row['oracle_recovery_mean'] if is_number(row['oracle_recovery_mean']) else -999,
    ), reverse=True)
    return out


def top_coefficients(features, weights, top_k=20):
    pairs = sorted(
        [{'feature': name, 'weight': float(weight), 'abs_weight': abs(float(weight))}
         for name, weight in zip(features, weights)],
        key=lambda item: item['abs_weight'],
        reverse=True,
    )
    return pairs[:top_k]


def evaluate_feature_set(rows, feature_set_name, features, args):
    split_rows = []
    stump_rows = []
    coefficient_rows = []
    if not features:
        return split_rows, stump_rows, coefficient_rows

    for split_id in range(args.splits):
        train_idx, valid_idx = stratified_split(rows, args.seed + split_id, args.valid_fraction)
        train_y = np.array([label(rows[idx]) for idx in train_idx], dtype=np.float64)
        min_count = max(1, int(round(len(train_idx) * args.min_select_fraction)))

        lfv1_metrics = selection_metrics(
            rows, valid_idx, [False] * len(valid_idx),
            'always_lfv1', feature_set_name, 'baseline'
        )
        residual_metrics = selection_metrics(
            rows, valid_idx, [True] * len(valid_idx),
            'always_residual', feature_set_name, 'baseline'
        )
        oracle_flags = [label(rows[idx]) == 1 for idx in valid_idx]
        oracle_metrics = selection_metrics(
            rows, valid_idx, oracle_flags,
            'two_way_oracle', feature_set_name, 'oracle'
        )
        oracle_gain = oracle_metrics['gain_vs_lfv1_psnr']
        for metrics in (lfv1_metrics, residual_metrics, oracle_metrics):
            metrics['split'] = split_id
            metrics['oracle_recovery'] = (
                1.0 if metrics['name'] == 'two_way_oracle'
                else safe_ratio(metrics['gain_vs_lfv1_psnr'], oracle_gain)
            )
            split_rows.append(metrics)

        stump = train_stump(rows, train_idx, features, min_count, args.threshold_steps)
        if stump:
            flags = predict_stump(rows, valid_idx, stump)
            metrics = selection_metrics(rows, valid_idx, flags, 'decision_stump', feature_set_name, 'stump')
            metrics['split'] = split_id
            metrics['oracle_recovery'] = safe_ratio(metrics['gain_vs_lfv1_psnr'], oracle_gain)
            split_rows.append(metrics)
            stump_row = dict(stump)
            stump_row.update({
                'split': split_id,
                'feature_set': feature_set_name,
                'valid_mean_psnr': metrics['mean_psnr'],
                'valid_gain_vs_lfv1_psnr': metrics['gain_vs_lfv1_psnr'],
                'valid_oracle_recovery': metrics['oracle_recovery'],
                'valid_residual_precision': metrics['residual_precision'],
                'valid_residual_recall': metrics['residual_recall'],
            })
            stump_rows.append(stump_row)

        train_x = matrix(rows, train_idx, features)
        valid_x = matrix(rows, valid_idx, features)
        train_x, valid_x, _, _ = standardize(train_x, valid_x)
        weights, bias = fit_logistic(
            train_x,
            train_y,
            steps=args.logistic_steps,
            lr=args.logistic_lr,
            l2=args.logistic_l2,
        )
        train_probs = sigmoid(train_x.dot(weights) + bias)
        threshold = choose_threshold(rows, train_idx, train_probs, min_count)
        valid_probs = sigmoid(valid_x.dot(weights) + bias)
        flags = [prob >= threshold for prob in valid_probs]
        metrics = selection_metrics(rows, valid_idx, flags, 'ridge_logistic', feature_set_name, 'logistic')
        metrics['split'] = split_id
        metrics['oracle_recovery'] = safe_ratio(metrics['gain_vs_lfv1_psnr'], oracle_gain)
        metrics['prob_threshold'] = threshold
        split_rows.append(metrics)
        for coef in top_coefficients(features, weights, top_k=12):
            coef.update({
                'split': split_id,
                'feature_set': feature_set_name,
                'bias': bias,
                'prob_threshold': threshold,
            })
            coefficient_rows.append(coef)

    return split_rows, stump_rows, coefficient_rows


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, keys):
    if not rows:
        return ''
    lines = [
        '| ' + ' | '.join(keys) + ' |',
        '| ' + ' | '.join('---' for _ in keys) + ' |',
    ]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, '')
            if isinstance(value, float):
                values.append('{:.6f}'.format(value))
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def recommendation(aggregate_rows, args):
    safe_rows = [
        row for row in aggregate_rows
        if row['feature_set'] in ('output_proxy', 'metadata_proxy', 'output_plus_metadata_proxy')
        and row['model_type'] in ('stump', 'logistic')
    ]
    if not safe_rows:
        return {
            'recommendation': 'do_not_train_selector_v2_yet',
            'reason': 'No safe proxy feature set was available.',
            'best_safe_row': None,
        }
    best = max(safe_rows, key=lambda row: row['gain_vs_lfv1_psnr_mean'])
    pass_gain = best['gain_vs_lfv1_psnr_mean'] >= args.min_gain
    pass_recovery = best['oracle_recovery_mean'] >= args.min_oracle_recovery
    pass_precision = best['residual_precision_mean'] >= args.min_precision
    if pass_gain and pass_recovery and pass_precision:
        rec = 'proceed_to_selector_v2_experiment_card'
        reason = (
            'A GT-free proxy recovered enough oracle headroom on held-out splits '
            'to justify a selector-v2 route card.'
        )
    else:
        rec = 'do_not_train_selector_v2_yet'
        reason = (
            'Safe proxies did not recover enough oracle headroom. Improve proxy '
            'features or add an explicit supervised/distilled selector target before training.'
        )
    return {
        'recommendation': rec,
        'reason': reason,
        'best_safe_row': best,
        'passes': {
            'gain': pass_gain,
            'oracle_recovery': pass_recovery,
            'precision': pass_precision,
        },
    }


def write_report(path, summary, aggregate_rows, stump_rows, coefficient_rows):
    top_agg = aggregate_rows[:12]
    top_stumps = sorted(
        stump_rows,
        key=lambda row: row.get('valid_gain_vs_lfv1_psnr', -999),
        reverse=True,
    )[:12]
    top_coef = sorted(
        coefficient_rows,
        key=lambda row: row['abs_weight'],
        reverse=True,
    )[:20]
    rec = summary['decision']
    lines = [
        '# Selector Proxy Learnability Audit',
        '',
        '## Verdict',
        '',
        '- Recommendation: `{}`'.format(rec['recommendation']),
        '- Reason: {}'.format(rec['reason']),
        '- Images: `{}`'.format(summary['num_images']),
        '- Splits: `{}`'.format(summary['splits']),
        '- Validation fraction: `{}`'.format(summary['valid_fraction']),
        '- Minimum pass line: gain `>= {:.3f}` dB, oracle recovery `>= {:.2f}`, residual precision `>= {:.2f}`'.format(
            summary['min_gain'],
            summary['min_oracle_recovery'],
            summary['min_precision'],
        ),
        '',
        '## Feature Sets',
        '',
    ]
    for name, count in summary['feature_counts'].items():
        lines.append('- `{}`: `{}` features'.format(name, count))
    lines.extend([
        '',
        '## Cross-Validated Results',
        '',
        markdown_table(top_agg, [
            'feature_set',
            'model_type',
            'name',
            'gain_vs_lfv1_psnr_mean',
            'gain_vs_lfv1_psnr_std',
            'oracle_recovery_mean',
            'residual_precision_mean',
            'residual_recall_mean',
            'two_way_accuracy_mean',
            'lost_lfv1_gain_selected_count_mean',
            'residual_worst_selected_count_mean',
        ]),
        '',
        '## Top Held-Out Stumps',
        '',
        markdown_table(top_stumps, [
            'feature_set',
            'feature',
            'op',
            'threshold',
            'train_gain_vs_lfv1_psnr',
            'valid_gain_vs_lfv1_psnr',
            'valid_oracle_recovery',
            'valid_residual_precision',
            'valid_residual_recall',
        ]),
        '',
        '## Top Logistic Coefficients',
        '',
        markdown_table(top_coef, [
            'feature_set',
            'feature',
            'weight',
            'abs_weight',
            'prob_threshold',
        ]),
        '',
        '## Interpretation',
        '',
        '- `gt_diagnostic_leakage_check` is a leakage ceiling. It is useful for explaining headroom, but it is not an inference-time selector input.',
        '- `output_proxy` and `output_plus_metadata_proxy` are the scientifically relevant sets for deciding whether another selector run is justified.',
        '- A selector-v2 training run should start only if safe proxy results pass the written line; otherwise the next useful step is better proxy extraction or thesis documentation, not another 100k scout.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.splits <= 0:
        raise ValueError('--splits must be positive')
    if not 0.0 < args.valid_fraction < 1.0:
        raise ValueError('--valid_fraction must be in (0, 1)')

    rows = filter_rows(read_rows(args.three_way_csv), args.sample_list)
    add_proxy_deltas(rows)
    feature_sets = build_feature_sets(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = []
    stump_rows = []
    coefficient_rows = []
    for name, features in feature_sets.items():
        rows_for_set, stumps_for_set, coefs_for_set = evaluate_feature_set(rows, name, features, args)
        split_rows.extend(rows_for_set)
        stump_rows.extend(stumps_for_set)
        coefficient_rows.extend(coefs_for_set)

    aggregate_rows = aggregate_metrics(split_rows)
    decision = recommendation(aggregate_rows, args)
    summary = {
        'input_csv': args.three_way_csv,
        'sample_list': args.sample_list,
        'num_images': len(rows),
        'seed': args.seed,
        'splits': args.splits,
        'valid_fraction': args.valid_fraction,
        'min_gain': args.min_gain,
        'min_oracle_recovery': args.min_oracle_recovery,
        'min_precision': args.min_precision,
        'feature_counts': {name: len(features) for name, features in feature_sets.items()},
        'decision': decision,
    }

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / 'split_metrics.csv', split_rows)
    write_csv(output_dir / 'aggregate_summary.csv', aggregate_rows)
    write_csv(output_dir / 'stump_rules.csv', stump_rows)
    write_csv(output_dir / 'logistic_coefficients.csv', coefficient_rows)
    write_report(output_dir / 'analysis_report.md', summary, aggregate_rows, stump_rows, coefficient_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
