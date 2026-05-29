import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import analyze_selector_proxy_learning as base


EPS = 1e-12
MODEL_PREFIXES = ('baseline', 'lfv1', 'residual')
PAIR_PREFIXES = (
    ('residual', 'lfv1'),
    ('residual', 'baseline'),
    ('lfv1', 'baseline'),
)
SAFE_PRIMARY_FEATURE_SETS = (
    'strict_output_proxy',
    'agreement_proxy',
    'rich_output_proxy',
)
DIAGNOSTIC_FEATURE_SETS = (
    'metadata_diagnostic',
    'gt_diagnostic_leakage_check',
)
STUMP_FEATURE_SETS = (
    'strict_output_proxy',
    'agreement_proxy',
    'metadata_diagnostic',
    'gt_diagnostic_leakage_check',
)
REQUIRED_PASS_FAMILIES = (
    'random_stratified',
    'airlight_leave_one',
    'beta_leave_one',
    'degradation_combo_group5',
)
INTERACTION_CONTEXTS = (
    'baseline_luma_mean',
    'baseline_luma_std',
    'baseline_saturation_mean',
    'baseline_dark_channel_mean',
    'baseline_edge_mean',
    'baseline_lap_var',
    'baseline_highfreq_abs_mean',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Audit richer GT-free proxy features for LF-v1 vs ResidualCalib '
            'selection under random and degradation-held-out splits.'
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
    parser.add_argument('--logistic_l2', type=float, default=0.15)
    parser.add_argument('--min_gain', type=float, default=0.12)
    parser.add_argument('--min_oracle_recovery', type=float, default=0.20)
    parser.add_argument('--min_precision', type=float, default=0.65)
    return parser.parse_args()


def as_float(value):
    if base.is_number(value):
        return float(value)
    return None


def set_value(row, name, value):
    if value is None or not math.isfinite(float(value)):
        row[name] = ''
    else:
        row[name] = float(value)


def ratio(left, right):
    if left is None or right is None:
        return None
    return (left + EPS) / (right + EPS)


def norm_delta(left, right):
    if left is None or right is None:
        return None
    denom = (abs(left) + abs(right)) * 0.5 + EPS
    return (left - right) / denom


def rank_value(values, prefix):
    clean = [(name, value) for name, value in values.items() if value is not None]
    if len(clean) != len(MODEL_PREFIXES):
        return None
    ordered = sorted(clean, key=lambda item: item[1])
    for rank, (name, _) in enumerate(ordered):
        if name == prefix:
            return float(rank)
    return None


def generated_feature_names():
    strict = set(base.allowed_output_proxy_feature_names())
    agreement = set()
    rich_extra = set()

    for left, right in PAIR_PREFIXES:
        for stat in base.SAFE_OUTPUT_STATS:
            agreement.update({
                '{}_{}_{}_ratio'.format(left, right, stat),
                '{}_{}_{}_signed_norm_delta'.format(left, right, stat),
                '{}_{}_{}_higher_flag'.format(left, right, stat),
            })

    for stat in base.SAFE_OUTPUT_STATS:
        rich_extra.update({
            'models_{}_mean'.format(stat),
            'models_{}_std'.format(stat),
            'models_{}_range'.format(stat),
            'models_{}_max_minus_mean'.format(stat),
            'models_{}_mean_minus_min'.format(stat),
        })
        for prefix in MODEL_PREFIXES:
            rich_extra.add('{}_{}_rank'.format(prefix, stat))

    for stat in base.SAFE_OUTPUT_STATS:
        for context in INTERACTION_CONTEXTS:
            rich_extra.add('residual_lfv1_{}_delta_x_{}'.format(stat, context))
            rich_extra.add('residual_lfv1_{}_abs_delta_x_{}'.format(stat, context))

    rich = strict | agreement | rich_extra
    return strict, agreement, rich


def add_rich_proxy_features(rows):
    strict_names, agreement_names, rich_names = generated_feature_names()
    for row in rows:
        for left, right in PAIR_PREFIXES:
            for stat in base.SAFE_OUTPUT_STATS:
                left_value = as_float(row.get('{}_{}'.format(left, stat), ''))
                right_value = as_float(row.get('{}_{}'.format(right, stat), ''))
                delta_value = None
                if left_value is not None and right_value is not None:
                    delta_value = left_value - right_value
                set_value(
                    row,
                    '{}_{}_{}_ratio'.format(left, right, stat),
                    ratio(left_value, right_value),
                )
                set_value(
                    row,
                    '{}_{}_{}_signed_norm_delta'.format(left, right, stat),
                    norm_delta(left_value, right_value),
                )
                set_value(
                    row,
                    '{}_{}_{}_higher_flag'.format(left, right, stat),
                    1.0 if delta_value is not None and delta_value >= 0.0 else 0.0
                    if delta_value is not None else None,
                )

        for stat in base.SAFE_OUTPUT_STATS:
            values = {
                prefix: as_float(row.get('{}_{}'.format(prefix, stat), ''))
                for prefix in MODEL_PREFIXES
            }
            clean_values = [value for value in values.values() if value is not None]
            if len(clean_values) == len(MODEL_PREFIXES):
                mean_value = float(np.mean(clean_values))
                std_value = float(np.std(clean_values))
                max_value = max(clean_values)
                min_value = min(clean_values)
                set_value(row, 'models_{}_mean'.format(stat), mean_value)
                set_value(row, 'models_{}_std'.format(stat), std_value)
                set_value(row, 'models_{}_range'.format(stat), max_value - min_value)
                set_value(row, 'models_{}_max_minus_mean'.format(stat), max_value - mean_value)
                set_value(row, 'models_{}_mean_minus_min'.format(stat), mean_value - min_value)
            else:
                for suffix in ('mean', 'std', 'range', 'max_minus_mean', 'mean_minus_min'):
                    row['models_{}_{}'.format(stat, suffix)] = ''

            for prefix in MODEL_PREFIXES:
                set_value(row, '{}_{}_rank'.format(prefix, stat), rank_value(values, prefix))

            delta = as_float(row.get('residual_lfv1_{}_delta'.format(stat), ''))
            abs_delta = as_float(row.get('residual_lfv1_{}_delta_abs'.format(stat), ''))
            for context in INTERACTION_CONTEXTS:
                context_value = as_float(row.get(context, ''))
                if delta is None or context_value is None:
                    row['residual_lfv1_{}_delta_x_{}'.format(stat, context)] = ''
                else:
                    row['residual_lfv1_{}_delta_x_{}'.format(stat, context)] = delta * context_value
                if abs_delta is None or context_value is None:
                    row['residual_lfv1_{}_abs_delta_x_{}'.format(stat, context)] = ''
                else:
                    row['residual_lfv1_{}_abs_delta_x_{}'.format(stat, context)] = (
                        abs_delta * context_value
                    )
    return strict_names, agreement_names, rich_names


def numeric_names(rows, names):
    return base.numeric_feature_names(rows, sorted(names))


def build_feature_sets(rows):
    strict_names, agreement_names, rich_names = add_rich_proxy_features(rows)
    metadata = set(base.metadata_features(rows))
    gt_diag = set(base.gt_diagnostic_features(rows))
    return {
        'strict_output_proxy': numeric_names(rows, strict_names),
        'agreement_proxy': numeric_names(rows, agreement_names),
        'rich_output_proxy': numeric_names(rows, rich_names),
        'metadata_diagnostic': numeric_names(rows, metadata),
        'gt_diagnostic_leakage_check': numeric_names(rows, gt_diag),
    }, {
        'safe_primary_allowed': sorted(rich_names),
        'metadata_allowed': sorted(metadata),
        'gt_diagnostic_allowed': sorted(gt_diag),
    }


def validate_feature_sets(feature_sets, allowed):
    safe_allowed = set(allowed['safe_primary_allowed'])
    for name in SAFE_PRIMARY_FEATURE_SETS:
        unexpected = sorted(set(feature_sets.get(name, [])) - safe_allowed)
        if unexpected:
            raise ValueError(
                'Unsafe feature(s) found in {}: {}'.format(name, ', '.join(unexpected))
            )


def label_counts(rows, indices):
    positives = sum(1 for idx in indices if base.label(rows[idx]) == 1)
    negatives = len(indices) - positives
    return positives, negatives


def usable_split(rows, train_idx, valid_idx):
    if len(train_idx) < 20 or len(valid_idx) < 20:
        return False
    train_pos, train_neg = label_counts(rows, train_idx)
    valid_pos, valid_neg = label_counts(rows, valid_idx)
    return min(train_pos, train_neg, valid_pos, valid_neg) > 0


def add_split(split_specs, rows, family, name, train_idx, valid_idx):
    if usable_split(rows, train_idx, valid_idx):
        split_specs.append({
            'split_family': family,
            'split': name,
            'train_idx': train_idx,
            'valid_idx': valid_idx,
        })


def degradation_combo_kfold(rows, seed, folds):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for idx, row in enumerate(rows):
        key = '{}|{}'.format(row.get('airlight_bin', ''), row.get('beta_bin', ''))
        grouped[key].append(idx)
    items = list(grouped.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    fold_indices = [[] for _ in range(folds)]
    for _, indices in items:
        fold_id = min(range(folds), key=lambda item: len(fold_indices[item]))
        fold_indices[fold_id].extend(indices)
    return fold_indices


def split_specs(rows, args):
    specs = []
    all_idx = list(range(len(rows)))

    for split_id in range(args.splits):
        train_idx, valid_idx = base.stratified_split(
            rows,
            args.seed + split_id,
            args.valid_fraction,
        )
        add_split(specs, rows, 'random_stratified', str(split_id), train_idx, valid_idx)

    for column, family in (
        ('airlight_bin', 'airlight_leave_one'),
        ('beta_bin', 'beta_leave_one'),
    ):
        values = sorted(set(row.get(column, '') for row in rows if row.get(column, '') != ''))
        for value in values:
            valid = [idx for idx, row in enumerate(rows) if row.get(column, '') == value]
            train = [idx for idx in all_idx if idx not in set(valid)]
            add_split(specs, rows, family, value, train, valid)

    for split_id, valid in enumerate(degradation_combo_kfold(rows, args.seed, args.splits)):
        valid_set = set(valid)
        train = [idx for idx in all_idx if idx not in valid_set]
        add_split(specs, rows, 'degradation_combo_group5', str(split_id), train, valid)

    return specs


def evaluate_feature_set(rows, feature_set_name, features, specs, args):
    split_rows = []
    stump_rows = []
    coefficient_rows = []
    if not features:
        return split_rows, stump_rows, coefficient_rows

    for spec in specs:
        train_idx = spec['train_idx']
        valid_idx = spec['valid_idx']
        train_y = np.array([base.label(rows[idx]) for idx in train_idx], dtype=np.float64)
        min_count = max(1, int(round(len(train_idx) * args.min_select_fraction)))

        lfv1_metrics = base.selection_metrics(
            rows, valid_idx, [False] * len(valid_idx),
            'always_lfv1', feature_set_name, 'baseline'
        )
        residual_metrics = base.selection_metrics(
            rows, valid_idx, [True] * len(valid_idx),
            'always_residual', feature_set_name, 'baseline'
        )
        oracle_flags = [base.label(rows[idx]) == 1 for idx in valid_idx]
        oracle_metrics = base.selection_metrics(
            rows, valid_idx, oracle_flags,
            'two_way_oracle', feature_set_name, 'oracle'
        )
        oracle_gain = oracle_metrics['gain_vs_lfv1_psnr']

        for metrics in (lfv1_metrics, residual_metrics, oracle_metrics):
            metrics['split_family'] = spec['split_family']
            metrics['split'] = spec['split']
            metrics['oracle_recovery'] = (
                1.0 if metrics['name'] == 'two_way_oracle'
                else base.safe_ratio(metrics['gain_vs_lfv1_psnr'], oracle_gain)
            )
            split_rows.append(metrics)

        stump = None
        if feature_set_name in STUMP_FEATURE_SETS:
            stump = base.train_stump(rows, train_idx, features, min_count, args.threshold_steps)
        if stump:
            flags = base.predict_stump(rows, valid_idx, stump)
            metrics = base.selection_metrics(
                rows, valid_idx, flags, 'decision_stump', feature_set_name, 'stump'
            )
            metrics['split_family'] = spec['split_family']
            metrics['split'] = spec['split']
            metrics['oracle_recovery'] = base.safe_ratio(metrics['gain_vs_lfv1_psnr'], oracle_gain)
            split_rows.append(metrics)
            stump_row = dict(stump)
            stump_row.update({
                'split_family': spec['split_family'],
                'split': spec['split'],
                'feature_set': feature_set_name,
                'valid_mean_psnr': metrics['mean_psnr'],
                'valid_gain_vs_lfv1_psnr': metrics['gain_vs_lfv1_psnr'],
                'valid_oracle_recovery': metrics['oracle_recovery'],
                'valid_residual_precision': metrics['residual_precision'],
                'valid_residual_recall': metrics['residual_recall'],
            })
            stump_rows.append(stump_row)

        train_x = base.matrix(rows, train_idx, features)
        valid_x = base.matrix(rows, valid_idx, features)
        train_x, valid_x, _, _ = base.standardize(train_x, valid_x)
        weights, bias = base.fit_logistic(
            train_x,
            train_y,
            steps=args.logistic_steps,
            lr=args.logistic_lr,
            l2=args.logistic_l2,
        )
        train_probs = base.sigmoid(train_x.dot(weights) + bias)
        threshold = base.choose_threshold(rows, train_idx, train_probs, min_count)
        valid_probs = base.sigmoid(valid_x.dot(weights) + bias)
        flags = [prob >= threshold for prob in valid_probs]
        metrics = base.selection_metrics(
            rows, valid_idx, flags, 'ridge_logistic', feature_set_name, 'logistic'
        )
        metrics['split_family'] = spec['split_family']
        metrics['split'] = spec['split']
        metrics['oracle_recovery'] = base.safe_ratio(metrics['gain_vs_lfv1_psnr'], oracle_gain)
        metrics['prob_threshold'] = threshold
        split_rows.append(metrics)
        for coef in base.top_coefficients(features, weights, top_k=12):
            coef.update({
                'split_family': spec['split_family'],
                'split': spec['split'],
                'feature_set': feature_set_name,
                'bias': bias,
                'prob_threshold': threshold,
            })
            coefficient_rows.append(coef)

    return split_rows, stump_rows, coefficient_rows


def aggregate_metrics(metric_rows):
    groups = defaultdict(list)
    for row in metric_rows:
        key = (row['split_family'], row['feature_set'], row['model_type'], row['name'])
        groups[key].append(row)
    out = []
    for (split_family, feature_set, model_type, name), rows in groups.items():
        item = {
            'split_family': split_family,
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
            item[metric + '_mean'] = base.mean(values)
            item[metric + '_std'] = base.std(values)
        item['oracle_recovery_mean'] = base.mean([row.get('oracle_recovery', '') for row in rows])
        item['oracle_recovery_std'] = base.std([row.get('oracle_recovery', '') for row in rows])
        out.append(item)
    out.sort(key=lambda row: (
        row['split_family'],
        row['gain_vs_lfv1_psnr_mean'] if base.is_number(row['gain_vs_lfv1_psnr_mean']) else -999,
        row['oracle_recovery_mean'] if base.is_number(row['oracle_recovery_mean']) else -999,
    ), reverse=True)
    return out


def passes(row, args):
    return (
        row['gain_vs_lfv1_psnr_mean'] >= args.min_gain
        and row['oracle_recovery_mean'] >= args.min_oracle_recovery
        and row['residual_precision_mean'] >= args.min_precision
    )


def recommendation(aggregate_rows, args):
    primary_rows = [
        row for row in aggregate_rows
        if row['feature_set'] in SAFE_PRIMARY_FEATURE_SETS
        and row['model_type'] in ('stump', 'logistic')
    ]
    best_by_family = {}
    for family in sorted(set(row['split_family'] for row in primary_rows)):
        family_rows = [row for row in primary_rows if row['split_family'] == family]
        best_by_family[family] = max(
            family_rows,
            key=lambda row: row['gain_vs_lfv1_psnr_mean'],
        )

    grouped = defaultdict(dict)
    for row in primary_rows:
        key = (row['feature_set'], row['model_type'], row['name'])
        grouped[key][row['split_family']] = row

    passing_candidates = []
    for key, by_family in grouped.items():
        if all(family in by_family for family in REQUIRED_PASS_FAMILIES):
            if all(passes(by_family[family], args) for family in REQUIRED_PASS_FAMILIES):
                passing_candidates.append({
                    'feature_set': key[0],
                    'model_type': key[1],
                    'name': key[2],
                    'rows': [by_family[family] for family in REQUIRED_PASS_FAMILIES],
                })

    if passing_candidates:
        return {
            'recommendation': 'proceed_to_selector_v2_experiment_card',
            'reason': (
                'A metadata-free rich proxy passed the gain, oracle-recovery, '
                'and precision lines under random and degradation-held-out splits.'
            ),
            'best_by_family': best_by_family,
            'passing_candidates': passing_candidates,
        }

    return {
        'recommendation': 'do_not_train_selector_v2_yet',
        'reason': (
            'No metadata-free rich proxy passed the required random and '
            'degradation-held-out audit lines. Keep improving proxy evidence or '
            'stop selector-v2 training plans.'
        ),
        'best_by_family': best_by_family,
        'passing_candidates': [],
    }


def markdown_table(rows, keys):
    return base.markdown_table(rows, keys)


def write_report(path, summary, aggregate_rows, stump_rows, coefficient_rows):
    rec = summary['decision']
    primary_best = list(rec['best_by_family'].values())
    primary_best.sort(key=lambda row: row['split_family'])
    top_agg = sorted(
        aggregate_rows,
        key=lambda row: (
            row['split_family'],
            row['gain_vs_lfv1_psnr_mean']
            if base.is_number(row['gain_vs_lfv1_psnr_mean']) else -999,
        ),
        reverse=True,
    )[:24]
    top_stumps = sorted(
        stump_rows,
        key=lambda row: row.get('valid_gain_vs_lfv1_psnr', -999),
        reverse=True,
    )[:20]
    top_coef = sorted(
        coefficient_rows,
        key=lambda row: row['abs_weight'],
        reverse=True,
    )[:24]
    lines = [
        '# Rich Selector Proxy Learnability Audit',
        '',
        '## Verdict',
        '',
        '- Recommendation: `{}`'.format(rec['recommendation']),
        '- Reason: {}'.format(rec['reason']),
        '- Images: `{}`'.format(summary['num_images']),
        '- Split families: `{}`'.format(', '.join(summary['split_families'])),
        '- Minimum pass line: gain `>= {:.3f}` dB, oracle recovery `>= {:.2f}`, residual precision `>= {:.2f}`'.format(
            summary['min_gain'],
            summary['min_oracle_recovery'],
            summary['min_precision'],
        ),
        '- Proceed requires a metadata-free proxy to pass all required families: `{}`'.format(
            ', '.join(REQUIRED_PASS_FAMILIES)
        ),
        '',
        '## Feature Sets',
        '',
    ]
    for name, count in summary['feature_counts'].items():
        lines.append('- `{}`: `{}` features'.format(name, count))
    lines.extend([
        '',
        '## Best Metadata-Free Row By Split Family',
        '',
        markdown_table(primary_best, [
            'split_family',
            'feature_set',
            'model_type',
            'name',
            'gain_vs_lfv1_psnr_mean',
            'oracle_recovery_mean',
            'residual_precision_mean',
            'residual_recall_mean',
            'two_way_accuracy_mean',
        ]),
        '',
        '## Top Aggregate Rows',
        '',
        markdown_table(top_agg, [
            'split_family',
            'feature_set',
            'model_type',
            'name',
            'gain_vs_lfv1_psnr_mean',
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
            'split_family',
            'split',
            'feature_set',
            'feature',
            'op',
            'threshold',
            'valid_gain_vs_lfv1_psnr',
            'valid_oracle_recovery',
            'valid_residual_precision',
        ]),
        '',
        '## Top Logistic Coefficients',
        '',
        markdown_table(top_coef, [
            'split_family',
            'split',
            'feature_set',
            'feature',
            'weight',
            'abs_weight',
            'prob_threshold',
        ]),
        '',
        '## Interpretation',
        '',
        '- Metadata-derived feature sets are diagnostic only and cannot justify selector-v2.',
        '- `gt_diagnostic_leakage_check` is a leakage ceiling, not an inference-time selector input.',
        '- A selector-v2 route card should be written only after a metadata-free proxy passes both random and degradation-held-out split families.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.splits <= 1:
        raise ValueError('--splits must be greater than 1')
    if not 0.0 < args.valid_fraction < 1.0:
        raise ValueError('--valid_fraction must be in (0, 1)')

    rows = base.filter_rows(base.read_rows(args.three_way_csv), args.sample_list)
    base.add_proxy_deltas(rows)
    feature_sets, allowed = build_feature_sets(rows)
    validate_feature_sets(feature_sets, allowed)
    specs = split_specs(rows, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = []
    stump_rows = []
    coefficient_rows = []
    for name, features in feature_sets.items():
        rows_for_set, stumps_for_set, coefs_for_set = evaluate_feature_set(
            rows, name, features, specs, args
        )
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
        'split_families': sorted(set(spec['split_family'] for spec in specs)),
        'split_count': len(specs),
        'min_gain': args.min_gain,
        'min_oracle_recovery': args.min_oracle_recovery,
        'min_precision': args.min_precision,
        'feature_counts': {name: len(features) for name, features in feature_sets.items()},
        'decision': decision,
    }

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'feature_lists.json', 'w', encoding='utf-8') as f:
        json.dump(feature_sets, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'allowed_feature_sets.json', 'w', encoding='utf-8') as f:
        json.dump(allowed, f, indent=2, ensure_ascii=False)
    base.write_csv(output_dir / 'split_metrics.csv', split_rows)
    base.write_csv(output_dir / 'aggregate_summary.csv', aggregate_rows)
    base.write_csv(output_dir / 'stump_rules.csv', stump_rows)
    base.write_csv(output_dir / 'logistic_coefficients.csv', coefficient_rows)
    write_report(output_dir / 'analysis_report.md', summary, aggregate_rows, stump_rows, coefficient_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
