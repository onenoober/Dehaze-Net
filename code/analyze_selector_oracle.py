import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


EPS = 1e-12
MODEL_NAMES = ('baseline', 'lfv1', 'residual')
PAIR_MODEL_NAMES = ('lfv1', 'residual')
GT_AWARE_FEATURES = {
    'baseline_psnr',
    'baseline_ssim',
    'lfv1_psnr',
    'lfv1_ssim',
    'residual_psnr',
    'residual_ssim',
    'input_psnr',
    'input_ssim',
    'lfv1_delta_baseline_psnr',
    'lfv1_delta_baseline_ssim',
    'residual_delta_baseline_psnr',
    'residual_delta_baseline_ssim',
    'residual_delta_lfv1_psnr',
    'residual_delta_lfv1_ssim',
    'baseline_np_psnr',
    'lfv1_np_psnr',
    'residual_np_psnr',
    'baseline_mae',
    'lfv1_mae',
    'residual_mae',
    'baseline_rmse',
    'lfv1_rmse',
    'residual_rmse',
    'baseline_delta_e_mean',
    'lfv1_delta_e_mean',
    'residual_delta_e_mean',
    'baseline_delta_e_p95',
    'lfv1_delta_e_p95',
    'residual_delta_e_p95',
    'baseline_luma_bias',
    'lfv1_luma_bias',
    'residual_luma_bias',
    'baseline_luma_abs_bias',
    'lfv1_luma_abs_bias',
    'residual_luma_abs_bias',
    'baseline_luma_std_delta',
    'lfv1_luma_std_delta',
    'residual_luma_std_delta',
    'baseline_saturation_bias',
    'lfv1_saturation_bias',
    'residual_saturation_bias',
    'baseline_saturation_abs_bias',
    'lfv1_saturation_abs_bias',
    'residual_saturation_abs_bias',
    'baseline_dark_channel_abs_bias',
    'lfv1_dark_channel_abs_bias',
    'residual_dark_channel_abs_bias',
    'baseline_edge_error',
    'lfv1_edge_error',
    'residual_edge_error',
    'baseline_lap_var_ratio_gt',
    'lfv1_lap_var_ratio_gt',
    'residual_lap_var_ratio_gt',
    'baseline_highfreq_abs_error',
    'lfv1_highfreq_abs_error',
    'residual_highfreq_abs_error',
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
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--three_way_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sample_list', type=str, default='')
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--min_select_count', type=int, default=30)
    parser.add_argument('--threshold_steps', type=int, default=19)
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


def read_rows(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            row = {key: parse_value(value) for key, value in raw.items()}
            rows.append(row)
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


def numeric(values):
    out = []
    for value in values:
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def mean(values):
    values = numeric(values)
    if not values:
        return ''
    return sum(values) / len(values)


def median(values):
    values = sorted(numeric(values))
    if not values:
        return ''
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def percentile(values, q):
    values = sorted(numeric(values))
    if not values:
        return ''
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if is_number(x) and is_number(y)]
    if len(pairs) < 2:
        return ''
    x_vals = [x for x, _ in pairs]
    y_vals = [y for _, y in pairs]
    x_mean = mean(x_vals)
    y_mean = mean(y_vals)
    x_var = sum((x - x_mean) ** 2 for x in x_vals)
    y_var = sum((y - y_mean) ** 2 for y in y_vals)
    if x_var <= EPS or y_var <= EPS:
        return ''
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return cov / math.sqrt(x_var * y_var)


def is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def best_model(row, models):
    return max(models, key=lambda model: row[f'{model}_psnr'])


def selected_metrics(rows, selector):
    selected = []
    for row in rows:
        model = selector(row)
        selected.append({
            'filename': row['filename'],
            'selected_model': model,
            'selected_psnr': row[f'{model}_psnr'],
            'selected_ssim': row[f'{model}_ssim'],
            'two_way_winner': best_model(row, PAIR_MODEL_NAMES),
            'three_way_winner': best_model(row, MODEL_NAMES),
            'pattern': row.get('pattern', ''),
            'baseline_strength_bin': row.get('baseline_strength_bin', ''),
            'lfv1_psnr': row['lfv1_psnr'],
            'residual_psnr': row['residual_psnr'],
            'baseline_psnr': row['baseline_psnr'],
            'selected_delta_lfv1_psnr': row[f'{model}_psnr'] - row['lfv1_psnr'],
            'selected_delta_baseline_psnr': row[f'{model}_psnr'] - row['baseline_psnr'],
        })
    return selected


def summarize_selection(name, rows, selected):
    model_counts = Counter(row['selected_model'] for row in selected)
    three_correct = sum(1 for row in selected if row['selected_model'] == row['three_way_winner'])
    two_correct = sum(
        1 for row in selected
        if row['selected_model'] in PAIR_MODEL_NAMES and row['selected_model'] == row['two_way_winner']
    )
    residual_selected = [row for row in selected if row['selected_model'] == 'residual']
    residual_precision = ''
    if residual_selected:
        residual_precision = sum(1 for row in residual_selected if row['two_way_winner'] == 'residual') / len(residual_selected)
    residual_win_total = sum(1 for row in selected if row['two_way_winner'] == 'residual')
    residual_recall = ''
    if residual_win_total:
        residual_recall = sum(1 for row in residual_selected if row['two_way_winner'] == 'residual') / residual_win_total
    return {
        'name': name,
        'num_images': len(selected),
        'mean_psnr': mean([row['selected_psnr'] for row in selected]),
        'mean_ssim': mean([row['selected_ssim'] for row in selected]),
        'gain_vs_lfv1_psnr': mean([row['selected_delta_lfv1_psnr'] for row in selected]),
        'gain_vs_baseline_psnr': mean([row['selected_delta_baseline_psnr'] for row in selected]),
        'median_delta_lfv1_psnr': median([row['selected_delta_lfv1_psnr'] for row in selected]),
        'p10_delta_lfv1_psnr': percentile([row['selected_delta_lfv1_psnr'] for row in selected], 10),
        'p90_delta_lfv1_psnr': percentile([row['selected_delta_lfv1_psnr'] for row in selected], 90),
        'three_way_accuracy': three_correct / len(selected),
        'two_way_accuracy_on_pair_rows': two_correct / len(selected),
        'residual_selection_count': model_counts.get('residual', 0),
        'lfv1_selection_count': model_counts.get('lfv1', 0),
        'baseline_selection_count': model_counts.get('baseline', 0),
        'residual_precision_vs_lfv1': residual_precision,
        'residual_recall_vs_lfv1': residual_recall,
        'lost_lfv1_gain_selected_count': sum(
            1 for row in selected if row['selected_model'] == 'residual' and row['pattern'] == 'lost_lfv1_gain'
        ),
        'residual_worst_selected_count': sum(
            1 for row in selected if row['selected_model'] == 'residual' and row['pattern'] == 'residual_worst'
        ),
        'residual_beats_both_selected_count': sum(
            1 for row in selected if row['selected_model'] == 'residual' and row['pattern'] == 'residual_beats_both'
        ),
        'mitigates_lfv1_regression_selected_count': sum(
            1 for row in selected if row['selected_model'] == 'residual' and row['pattern'] == 'mitigates_lfv1_regression'
        ),
    }


def model_summary(rows):
    summaries = []
    for model in MODEL_NAMES:
        summaries.append({
            'name': model,
            'num_images': len(rows),
            'mean_psnr': mean([row[f'{model}_psnr'] for row in rows]),
            'mean_ssim': mean([row[f'{model}_ssim'] for row in rows]),
            'gain_vs_lfv1_psnr': mean([row[f'{model}_psnr'] - row['lfv1_psnr'] for row in rows]),
            'gain_vs_baseline_psnr': mean([row[f'{model}_psnr'] - row['baseline_psnr'] for row in rows]),
            'winner_count': sum(1 for row in rows if best_model(row, MODEL_NAMES) == model),
            'two_way_winner_count': sum(1 for row in rows if model in PAIR_MODEL_NAMES and best_model(row, PAIR_MODEL_NAMES) == model),
        })
    return summaries


def add_derived_features(rows):
    pairs = [
        ('luma_mean', 'residual_lfv1_luma_mean_delta'),
        ('luma_std', 'residual_lfv1_luma_std_delta'),
        ('saturation_mean', 'residual_lfv1_saturation_mean_delta'),
        ('dark_channel_mean', 'residual_lfv1_dark_channel_mean_delta'),
        ('edge_mean', 'residual_lfv1_edge_mean_delta'),
        ('lap_var', 'residual_lfv1_lap_var_delta'),
        ('highfreq_abs_mean', 'residual_lfv1_highfreq_abs_mean_delta'),
    ]
    for row in rows:
        for suffix, name in pairs:
            left = row.get(f'residual_{suffix}', '')
            right = row.get(f'lfv1_{suffix}', '')
            if is_number(left) and is_number(right):
                row[name] = left - right
                row[name + '_abs'] = abs(left - right)
            else:
                row[name] = ''
                row[name + '_abs'] = ''


def feature_columns(rows):
    columns = []
    for key in rows[0].keys():
        values = [row.get(key, '') for row in rows]
        if sum(1 for value in values if is_number(value)) >= max(5, len(rows) // 10):
            columns.append(key)
    outcome_prefixes = ('selected_',)
    excluded = {
        'index',
        'height',
        'width',
    }
    return [
        key for key in columns
        if key not in excluded
        and not key.startswith(outcome_prefixes)
        and not key.endswith('_psnr')
        and not key.endswith('_ssim')
        and not key.endswith('_delta_psnr')
        and not key.endswith('_delta_ssim')
    ]


def thresholds_for(values, steps):
    values = sorted(set(numeric(values)))
    if len(values) <= 1:
        return []
    if len(values) <= steps:
        return values[:-1]
    quantiles = [100.0 * i / (steps + 1) for i in range(1, steps + 1)]
    return sorted(set(percentile(values, q) for q in quantiles))


def evaluate_rule(rows, feature, threshold, op):
    if op == '>=':
        selector = lambda row: 'residual' if row[feature] >= threshold else 'lfv1'
    else:
        selector = lambda row: 'residual' if row[feature] <= threshold else 'lfv1'
    selected = selected_metrics(rows, selector)
    summary = summarize_selection('rule', rows, selected)
    summary.update({
        'feature': feature,
        'threshold': threshold,
        'op': op,
        'feature_class': 'gt_aware_diagnostic' if feature in GT_AWARE_FEATURES else 'proxy_or_output_feature',
        'selected_fraction': summary['residual_selection_count'] / len(rows),
    })
    return summary, selected


def sweep_rules(rows, min_select_count, threshold_steps):
    rule_rows = []
    best_selected = None
    best_rule = None
    features = feature_columns(rows)
    for feature in features:
        values = [row.get(feature, '') for row in rows]
        for threshold in thresholds_for(values, threshold_steps):
            for op in ('>=', '<='):
                valid_rows = [row for row in rows if is_number(row.get(feature, ''))]
                if len(valid_rows) != len(rows):
                    continue
                summary, selected = evaluate_rule(rows, feature, threshold, op)
                if summary['residual_selection_count'] < min_select_count:
                    continue
                if summary['lfv1_selection_count'] < min_select_count:
                    continue
                rule_rows.append(summary)
                if best_rule is None or summary['mean_psnr'] > best_rule['mean_psnr']:
                    best_rule = summary
                    best_selected = selected
    rule_rows.sort(key=lambda row: (row['mean_psnr'], row['residual_precision_vs_lfv1'] or 0), reverse=True)
    return rule_rows, best_rule, best_selected


def feature_correlations(rows):
    target = [row['residual_delta_lfv1_psnr'] for row in rows]
    corr_rows = []
    for feature in feature_columns(rows):
        value = pearson([row.get(feature, '') for row in rows], target)
        if value == '':
            continue
        corr_rows.append({
            'feature': feature,
            'feature_class': 'gt_aware_diagnostic' if feature in GT_AWARE_FEATURES else 'proxy_or_output_feature',
            'corr_with_residual_delta_lfv1_psnr': value,
            'abs_corr': abs(value),
        })
    corr_rows.sort(key=lambda row: row['abs_corr'], reverse=True)
    return corr_rows


def group_summary(rows):
    groups = []
    keys = ['baseline_strength_bin', 'pattern', 'winner_by_psnr', 'airlight_bin', 'beta_bin']
    for key in keys:
        values = sorted(set(str(row.get(key, '')) for row in rows))
        for value in values:
            subset = [row for row in rows if str(row.get(key, '')) == value]
            if not subset:
                continue
            two_oracle = selected_metrics(subset, lambda row: best_model(row, PAIR_MODEL_NAMES))
            three_oracle = selected_metrics(subset, lambda row: best_model(row, MODEL_NAMES))
            groups.append({
                'group_name': key,
                'group_value': value,
                'num_images': len(subset),
                'lfv1_mean_psnr': mean([row['lfv1_psnr'] for row in subset]),
                'residual_mean_psnr': mean([row['residual_psnr'] for row in subset]),
                'baseline_mean_psnr': mean([row['baseline_psnr'] for row in subset]),
                'two_way_oracle_mean_psnr': mean([row['selected_psnr'] for row in two_oracle]),
                'three_way_oracle_mean_psnr': mean([row['selected_psnr'] for row in three_oracle]),
                'two_way_oracle_gain_vs_lfv1': mean([row['selected_delta_lfv1_psnr'] for row in two_oracle]),
                'three_way_oracle_gain_vs_lfv1': mean([row['selected_delta_lfv1_psnr'] for row in three_oracle]),
                'residual_winner_count': sum(1 for row in subset if best_model(row, PAIR_MODEL_NAMES) == 'residual'),
                'lfv1_winner_count': sum(1 for row in subset if best_model(row, PAIR_MODEL_NAMES) == 'lfv1'),
                'baseline_winner_count': sum(1 for row in subset if best_model(row, MODEL_NAMES) == 'baseline'),
            })
    return groups


def compact_case(row):
    keys = [
        'filename',
        'pattern',
        'baseline_strength_bin',
        'baseline_psnr',
        'lfv1_psnr',
        'residual_psnr',
        'lfv1_delta_baseline_psnr',
        'residual_delta_baseline_psnr',
        'residual_delta_lfv1_psnr',
        'lfv1_from_baseline_residual_cosine',
        'rescalib_from_baseline_residual_cosine',
        'rescalib_from_lfv1_residual_cosine',
        'rescalib_from_baseline_residual_norm_ratio',
        'rescalib_from_baseline_lf_mse_delta',
    ]
    return {key: row.get(key, '') for key in keys}


def top_cases(rows, key, reverse, top_k, predicate=None):
    subset = [row for row in rows if predicate is None or predicate(row)]
    subset = sorted(subset, key=lambda row: row[key], reverse=reverse)
    return [compact_case(row) for row in subset[:top_k]]


def build_hard_cases(rows, best_selected, top_k):
    cases = {
        'largest_two_way_oracle_gains': top_cases(rows, 'residual_delta_lfv1_psnr', True, top_k),
        'largest_two_way_oracle_losses_if_residual': top_cases(rows, 'residual_delta_lfv1_psnr', False, top_k),
        'baseline_beats_both': top_cases(
            rows,
            'baseline_psnr',
            True,
            top_k,
            lambda row: row['baseline_psnr'] > row['lfv1_psnr'] and row['baseline_psnr'] > row['residual_psnr'],
        ),
        'residual_beats_both': top_cases(rows, 'residual_delta_lfv1_psnr', True, top_k, lambda row: row.get('pattern') == 'residual_beats_both'),
        'lost_lfv1_gain': top_cases(rows, 'residual_delta_lfv1_psnr', False, top_k, lambda row: row.get('pattern') == 'lost_lfv1_gain'),
        'residual_worst': top_cases(rows, 'residual_delta_baseline_psnr', False, top_k, lambda row: row.get('pattern') == 'residual_worst'),
    }
    if best_selected:
        selected_by_name = {row['filename']: row for row in best_selected}
        false_positive_names = {
            row['filename'] for row in best_selected
            if row['selected_model'] == 'residual' and row['two_way_winner'] == 'lfv1'
        }
        false_negative_names = {
            row['filename'] for row in best_selected
            if row['selected_model'] == 'lfv1' and row['two_way_winner'] == 'residual'
        }
        cases['best_rule_false_positives'] = top_cases(
            [row for row in rows if row['filename'] in false_positive_names],
            'residual_delta_lfv1_psnr',
            False,
            top_k,
        )
        cases['best_rule_false_negatives'] = top_cases(
            [row for row in rows if row['filename'] in false_negative_names],
            'residual_delta_lfv1_psnr',
            True,
            top_k,
        )
        for row in cases['best_rule_false_positives'] + cases['best_rule_false_negatives']:
            selected = selected_by_name.get(row['filename'])
            if selected:
                row['best_rule_selected_model'] = selected['selected_model']
                row['best_rule_selected_delta_lfv1_psnr'] = selected['selected_delta_lfv1_psnr']
    return cases


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
    lines = []
    lines.append('| ' + ' | '.join(keys) + ' |')
    lines.append('| ' + ' | '.join('---' for _ in keys) + ' |')
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


def write_report(path, summary, model_rows, selection_rows, rule_rows, corr_rows, group_rows):
    top_rules = rule_rows[:10]
    top_corr = corr_rows[:12]
    top_groups = sorted(group_rows, key=lambda row: row['two_way_oracle_gain_vs_lfv1'], reverse=True)[:12]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# HAZE4K Selector Oracle Analysis\n\n')
        f.write('## Summary\n\n')
        f.write('- Input CSV: `{}`\n'.format(summary['input_csv']))
        f.write('- Images: `{}`\n'.format(summary['num_images']))
        f.write('- LF-v1 mean PSNR: `{:.4f}`\n'.format(summary['lfv1_mean_psnr']))
        f.write('- ResidualCalib mean PSNR: `{:.4f}`\n'.format(summary['residual_mean_psnr']))
        f.write('- Two-way oracle mean PSNR: `{:.4f}`; gain vs LF-v1 `{:+.4f}` dB\n'.format(
            summary['two_way_oracle_mean_psnr'],
            summary['two_way_oracle_gain_vs_lfv1_psnr'],
        ))
        f.write('- Three-way oracle mean PSNR: `{:.4f}`; gain vs LF-v1 `{:+.4f}` dB\n'.format(
            summary['three_way_oracle_mean_psnr'],
            summary['three_way_oracle_gain_vs_lfv1_psnr'],
        ))
        f.write('- Best one-rule selector mean PSNR: `{:.4f}`; gain vs LF-v1 `{:+.4f}` dB\n'.format(
            summary.get('best_rule_mean_psnr', 0.0),
            summary.get('best_rule_gain_vs_lfv1_psnr', 0.0),
        ))
        f.write('- Recommendation: **{}**\n\n'.format(summary['recommendation']))
        f.write('## Model And Oracle Rows\n\n')
        f.write(markdown_table(model_rows + selection_rows, [
            'name', 'num_images', 'mean_psnr', 'mean_ssim', 'gain_vs_lfv1_psnr',
            'gain_vs_baseline_psnr', 'residual_selection_count', 'lfv1_selection_count',
            'baseline_selection_count',
        ]))
        f.write('\n\n## Top Rule Selectors\n\n')
        f.write(markdown_table(top_rules, [
            'feature', 'op', 'threshold', 'feature_class', 'mean_psnr',
            'gain_vs_lfv1_psnr', 'residual_selection_count',
            'residual_precision_vs_lfv1', 'residual_recall_vs_lfv1',
            'lost_lfv1_gain_selected_count', 'residual_worst_selected_count',
        ]))
        f.write('\n\n## Top Correlations With ResidualCalib Minus LF-v1\n\n')
        f.write(markdown_table(top_corr, [
            'feature', 'feature_class', 'corr_with_residual_delta_lfv1_psnr',
            'abs_corr',
        ]))
        f.write('\n\n## Groups With Largest Two-Way Oracle Gain\n\n')
        f.write(markdown_table(top_groups, [
            'group_name', 'group_value', 'num_images', 'lfv1_mean_psnr',
            'residual_mean_psnr', 'two_way_oracle_mean_psnr',
            'two_way_oracle_gain_vs_lfv1', 'residual_winner_count',
            'lfv1_winner_count',
        ]))
        f.write('\n\n## Interpretation\n\n')
        f.write(summary['interpretation'] + '\n')


def recommendation(summary):
    two_gain = summary['two_way_oracle_gain_vs_lfv1_psnr']
    rule_gain = summary.get('best_rule_gain_vs_lfv1_psnr', 0.0)
    if two_gain >= 0.20 and rule_gain >= 0.05:
        return 'continue_to_lf_residual_selector'
    if two_gain >= 0.20:
        return 'selector_has_oracle_value_but_proxy_is_weak'
    return 'do_not_train_selector_yet'


def interpretation(summary):
    rec = summary['recommendation']
    if rec == 'continue_to_lf_residual_selector':
        return (
            'The two-way oracle has enough headroom and a simple rule recovers '
            'some of it, so a bounded LFResidualSelector is justified. Keep the '
            'first implementation close to LF-v1 and use gate checks to avoid '
            'large ResidualCalib false positives.'
        )
    if rec == 'selector_has_oracle_value_but_proxy_is_weak':
        return (
            'The oracle headroom is real, but the simple CSV-level proxies do '
            'not recover much of it. Before a long training run, improve the '
            'selector signal or add a small supervised confidence objective.'
        )
    return (
        'The oracle headroom is too small for a selector to be the next best '
        'training run. Prefer residual-direction loss scale diagnosis before '
        'spending another 100k scout.'
    )


def main():
    args = parse_args()
    rows = filter_rows(read_rows(args.three_way_csv), args.sample_list)
    add_derived_features(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_rows = model_summary(rows)
    lfv1_selected = selected_metrics(rows, lambda row: 'lfv1')
    residual_selected = selected_metrics(rows, lambda row: 'residual')
    two_way_selected = selected_metrics(rows, lambda row: best_model(row, PAIR_MODEL_NAMES))
    three_way_selected = selected_metrics(rows, lambda row: best_model(row, MODEL_NAMES))
    selection_rows = [
        summarize_selection('always_lfv1', rows, lfv1_selected),
        summarize_selection('always_residual', rows, residual_selected),
        summarize_selection('two_way_oracle_lfv1_residual', rows, two_way_selected),
        summarize_selection('three_way_oracle_baseline_lfv1_residual', rows, three_way_selected),
    ]

    rule_rows, best_rule, best_selected = sweep_rules(rows, args.min_select_count, args.threshold_steps)
    corr_rows = feature_correlations(rows)
    group_rows = group_summary(rows)
    hard_cases = build_hard_cases(rows, best_selected, args.top_k)

    summary = {
        'input_csv': str(args.three_way_csv),
        'sample_list': args.sample_list,
        'num_images': len(rows),
        'min_select_count': args.min_select_count,
        'threshold_steps': args.threshold_steps,
        'baseline_mean_psnr': mean([row['baseline_psnr'] for row in rows]),
        'lfv1_mean_psnr': mean([row['lfv1_psnr'] for row in rows]),
        'residual_mean_psnr': mean([row['residual_psnr'] for row in rows]),
        'two_way_oracle_mean_psnr': mean([row['selected_psnr'] for row in two_way_selected]),
        'two_way_oracle_gain_vs_lfv1_psnr': mean([row['selected_delta_lfv1_psnr'] for row in two_way_selected]),
        'three_way_oracle_mean_psnr': mean([row['selected_psnr'] for row in three_way_selected]),
        'three_way_oracle_gain_vs_lfv1_psnr': mean([row['selected_delta_lfv1_psnr'] for row in three_way_selected]),
        'two_way_oracle_residual_selection_count': sum(1 for row in two_way_selected if row['selected_model'] == 'residual'),
        'three_way_oracle_model_counts': dict(Counter(row['selected_model'] for row in three_way_selected)),
        'pattern_counts': dict(Counter(str(row.get('pattern', '')) for row in rows)),
        'best_rule': best_rule,
        'best_rule_mean_psnr': best_rule['mean_psnr'] if best_rule else 0.0,
        'best_rule_gain_vs_lfv1_psnr': best_rule['gain_vs_lfv1_psnr'] if best_rule else 0.0,
        'best_rule_feature': best_rule['feature'] if best_rule else '',
        'best_rule_feature_class': best_rule['feature_class'] if best_rule else '',
    }
    summary['recommendation'] = recommendation(summary)
    summary['interpretation'] = interpretation(summary)

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / 'model_and_oracle_summary.csv', model_rows + selection_rows)
    write_csv(output_dir / 'rule_sweep.csv', rule_rows)
    write_csv(output_dir / 'feature_correlations.csv', corr_rows)
    write_csv(output_dir / 'group_summary.csv', group_rows)
    if best_selected:
        write_csv(output_dir / 'best_rule_selections.csv', best_selected)
    with open(output_dir / 'hard_cases.json', 'w', encoding='utf-8') as f:
        json.dump(hard_cases, f, indent=2, ensure_ascii=False)
    write_report(output_dir / 'analysis_report.md', summary, model_rows, selection_rows, rule_rows, corr_rows, group_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
