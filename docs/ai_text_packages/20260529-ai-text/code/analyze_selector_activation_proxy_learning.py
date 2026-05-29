import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

import analyze_selector_proxy_learning as base
import analyze_selector_rich_proxy_learning as rich
from data.data_loader import list_image_files, resolve_pair_dirs
from model import DEANet


EPS = 1e-12
MODEL_PREFIXES = ('baseline', 'lfv1', 'residual')
COMMON_HOOKS = (
    'down1',
    'down_level1_block4',
    'down2',
    'down_level2_block4',
    'down3',
    'level3_block4',
    'level3_block8',
    'mix1',
    'mix2',
)
PAIR_PREFIXES = (
    ('residual', 'lfv1'),
    ('residual', 'baseline'),
    ('lfv1', 'baseline'),
)
SAFE_PRIMARY_FEATURE_SETS = (
    'activation_only_proxy',
    'activation_plus_strict_output_proxy',
    'activation_plus_rich_output_proxy',
)
SAFE_REFERENCE_FEATURE_SETS = (
    'rich_output_reference',
)
DIAGNOSTIC_FEATURE_SETS = (
    'metadata_diagnostic',
    'gt_diagnostic_leakage_check',
)
STUMP_FEATURE_SETS = (
    'rich_output_reference',
    'metadata_diagnostic',
    'gt_diagnostic_leakage_check',
)
REQUIRED_PASS_FAMILIES = rich.REQUIRED_PASS_FAMILIES
UNSAFE_SAFE_FEATURE_FRAGMENTS = (
    'psnr',
    'ssim',
    'gt_',
    '_gt',
    'clear',
    'mae',
    'rmse',
    'delta_e',
    '_error',
    '_bias',
    'residual_cosine',
    'residual_norm_ratio',
    'lf_mse',
)
SUMMARY_STATS = (
    'mean',
    'std',
    'abs_mean',
    'rms',
    'min',
    'max',
    'channel_mean_std',
    'spatial_std_mean',
    'active_frac_001',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Forward frozen checkpoints and audit whether inference-time '
            'activation features can learn the LF-v1 vs ResidualCalib selector target.'
        )
    )
    parser.add_argument('--three_way_csv', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default='dataset/HAZE4K/test')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--lfv1_checkpoint', type=str, required=True)
    parser.add_argument('--residual_checkpoint', type=str, required=True)
    parser.add_argument('--sample_list', type=str, default='')
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pad_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260526)
    parser.add_argument('--valid_fraction', type=float, default=0.30)
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--threshold_steps', type=int, default=7)
    parser.add_argument('--min_select_fraction', type=float, default=0.05)
    parser.add_argument('--logistic_steps', type=int, default=220)
    parser.add_argument('--logistic_lr', type=float, default=0.08)
    parser.add_argument('--logistic_l2', type=float, default=0.20)
    parser.add_argument('--min_gain', type=float, default=0.12)
    parser.add_argument('--min_oracle_recovery', type=float, default=0.20)
    parser.add_argument('--min_precision', type=float, default=0.65)
    parser.add_argument('--min_conclusive_images', type=int, default=800)
    parser.add_argument('--min_class_count', type=int, default=120)
    parser.add_argument('--lf_prior_channels', type=int, default=8)
    parser.add_argument('--lf_prior_pool', type=int, default=8)
    parser.add_argument('--lf_prior_gate_init', type=float, default=0.0)
    parser.add_argument('--lf_prior_residual_center', action='store_true')
    parser.add_argument('--lf_prior_gate_max', type=float, default=0.0)
    parser.add_argument('--lf_prior_injection', type=str, default='pre_mix', choices=['pre_mix', 'post_mix'])
    parser.add_argument('--lf_calib_hidden_channels', type=int, default=8)
    parser.add_argument('--lf_calib_alpha_max', type=float, default=1.0)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_model(checkpoint_path, args, use_lf_prior=False, residual_calibration=False):
    model = DEANet(
        base_dim=32,
        use_lf_prior=use_lf_prior,
        lf_prior_channels=args.lf_prior_channels,
        lf_prior_pool=args.lf_prior_pool,
        lf_prior_gate_init=args.lf_prior_gate_init,
        lf_prior_residual_center=args.lf_prior_residual_center,
        lf_prior_gate_max=args.lf_prior_gate_max,
        lf_prior_injection=args.lf_prior_injection,
        lf_residual_calibration=residual_calibration,
        lf_calib_hidden_channels=args.lf_calib_hidden_channels,
        lf_calib_alpha_max=args.lf_calib_alpha_max,
    )
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def tensor_stats(x):
    x = x.detach().float()
    flat = x.reshape(x.shape[0], -1)
    channel_mean = x.mean(dim=(2, 3))
    spatial_std = x.flatten(2).std(dim=2, unbiased=False)
    return {
        'mean': scalar(x.mean()),
        'std': scalar(x.std(unbiased=False)),
        'abs_mean': scalar(x.abs().mean()),
        'rms': scalar(torch.sqrt(torch.mean(x * x) + EPS)),
        'min': scalar(x.min()),
        'max': scalar(x.max()),
        'channel_mean_std': scalar(channel_mean.std(dim=1, unbiased=False).mean()),
        'spatial_std_mean': scalar(spatial_std.mean()),
        'active_frac_001': scalar((x.abs() > 0.01).float().mean()),
    }


def add_stats(row, prefix, x):
    for key, value in tensor_stats(x).items():
        row['{}_{}'.format(prefix, key)] = value


def lowpass(x, pool_size):
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def rgb_to_luma(x):
    return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114


def input_features(hazy, pool_size):
    row = {}
    luma = rgb_to_luma(hazy)
    dark = torch.min(hazy, dim=1, keepdim=True)[0]
    low = lowpass(hazy, pool_size)
    low_luma = rgb_to_luma(low)
    high_luma = luma - low_luma
    grad_x = luma[:, :, :, 1:] - luma[:, :, :, :-1]
    grad_y = luma[:, :, 1:, :] - luma[:, :, :-1, :]
    row['input_hazy_luma_mean'] = scalar(luma.mean())
    row['input_hazy_luma_std'] = scalar(luma.std(unbiased=False))
    row['input_hazy_dark_mean'] = scalar(dark.mean())
    row['input_hazy_channel_mean_std'] = scalar(hazy.mean(dim=(2, 3)).std(dim=1, unbiased=False).mean())
    row['input_hazy_low_luma_mean'] = scalar(low_luma.mean())
    row['input_hazy_low_luma_std'] = scalar(low_luma.std(unbiased=False))
    row['input_hazy_high_luma_abs_mean'] = scalar(high_luma.abs().mean())
    row['input_hazy_grad_abs_mean'] = scalar((grad_x.abs().mean() + grad_y.abs().mean()) * 0.5)
    return row


class ActivationProbe:
    def __init__(self, model, prefix):
        self.model = model
        self.prefix = prefix
        self.row = {}
        self.handles = []
        modules = dict(model.named_modules())
        for name in COMMON_HOOKS:
            if name in modules:
                self.handles.append(
                    modules[name].register_forward_hook(self._simple_hook('{}_{}'.format(prefix, name)))
                )
        lf_prior = getattr(model, 'lf_prior', None)
        if lf_prior is not None:
            self.handles.append(lf_prior.register_forward_hook(self._lf_prior_hook(prefix)))
            self.handles.append(lf_prior.adapter.register_forward_hook(self._simple_hook('{}_lf_adapter'.format(prefix))))
            if getattr(lf_prior, 'residual_calibration', False):
                self.handles.append(
                    lf_prior.calib_shared.register_forward_hook(
                        self._simple_hook('{}_lf_calib_shared'.format(prefix))
                    )
                )
                self.handles.append(
                    lf_prior.calib_direction_head.register_forward_hook(
                        self._transform_hook(
                            '{}_lf_calib_direction'.format(prefix),
                            lambda value: torch.tanh(value),
                        )
                    )
                )
                self.handles.append(
                    lf_prior.calib_alpha_head.register_forward_hook(
                        self._transform_hook(
                            '{}_lf_calib_alpha'.format(prefix),
                            lambda value, module=lf_prior: torch.sigmoid(value) * module.calib_alpha_max,
                        )
                    )
                )

    def reset(self):
        self.row = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _simple_hook(self, prefix):
        def hook(_module, _inputs, output):
            add_stats(self.row, prefix, output)
        return hook

    def _transform_hook(self, prefix, transform):
        def hook(_module, _inputs, output):
            add_stats(self.row, prefix, transform(output))
        return hook

    def _lf_prior_hook(self, prefix):
        def hook(module, inputs, output):
            target = inputs[1]
            injected = output - target
            self.row['{}_lf_gate'.format(prefix)] = scalar(module.gate.detach())
            add_stats(self.row, '{}_lf_target'.format(prefix), target)
            add_stats(self.row, '{}_lf_injected_delta'.format(prefix), injected)
        return hook


def choose_names(hazy_names, sample_list, max_images):
    if sample_list:
        with open(sample_list, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        missing = [name for name in names if name not in hazy_names]
        if missing:
            raise FileNotFoundError('Samples not found: {}'.format(', '.join(missing[:20])))
    else:
        names = list(hazy_names)
    if max_images > 0:
        names = names[:max_images]
    return names


def read_image(path, device):
    image = Image.open(path).convert('RGB')
    return ToTensor()(image).unsqueeze(0).to(device)


def infer_with_probe(model, probe, hazy, pad_size):
    _, _, h, w = hazy.shape
    probe.reset()
    padded = pad_img(hazy, pad_size)
    _ = model(padded)
    return dict(probe.row), h, w


def activation_rows(args):
    hazy_dir, _clear_dir = resolve_pair_dirs(args.dataset_root)
    hazy_names = list_image_files(hazy_dir)
    names = choose_names(hazy_names, args.sample_list, args.max_images)

    models = {
        'baseline': load_model(args.baseline_checkpoint, args, use_lf_prior=False)[0],
        'lfv1': load_model(args.lfv1_checkpoint, args, use_lf_prior=True)[0],
        'residual': load_model(
            args.residual_checkpoint,
            args,
            use_lf_prior=True,
            residual_calibration=True,
        )[0],
    }
    probes = {name: ActivationProbe(model, name) for name, model in models.items()}
    rows = []
    try:
        iterator = tqdm(names, desc='activation-forward', dynamic_ncols=True)
        with torch.no_grad():
            for index, filename in enumerate(iterator, start=1):
                hazy = read_image(Path(hazy_dir) / filename, args.device)
                row = {
                    'index': index,
                    'filename': filename,
                }
                row.update(input_features(hazy, args.lf_prior_pool))
                height = ''
                width = ''
                for model_name in MODEL_PREFIXES:
                    probe_row, height, width = infer_with_probe(
                        models[model_name], probes[model_name], hazy, args.pad_size
                    )
                    row.update(probe_row)
                row['height'] = height
                row['width'] = width
                add_pair_features(row)
                rows.append(row)
    finally:
        for probe in probes.values():
            probe.close()
    return rows


def numeric_value(value):
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


def suffixes_for_prefix(row, prefix):
    prefix_text = prefix + '_'
    return {
        key[len(prefix_text):]
        for key, value in row.items()
        if key.startswith(prefix_text) and base.is_number(value)
    }


def add_pair_features(row):
    suffixes = {prefix: suffixes_for_prefix(row, prefix) for prefix in MODEL_PREFIXES}
    for left, right in PAIR_PREFIXES:
        for suffix in sorted(suffixes[left] & suffixes[right]):
            left_value = numeric_value(row.get('{}_{}'.format(left, suffix), ''))
            right_value = numeric_value(row.get('{}_{}'.format(right, suffix), ''))
            if left_value is None or right_value is None:
                continue
            base_name = '{}_{}_{}'.format(left, right, suffix)
            delta = left_value - right_value
            row[base_name + '_delta'] = delta
            row[base_name + '_abs_delta'] = abs(delta)
            set_value(row, base_name + '_ratio', ratio(left_value, right_value))
            set_value(row, base_name + '_signed_norm_delta', norm_delta(left_value, right_value))


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


def read_activation_csv(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return [{key: base.parse_value(value) for key, value in raw.items()} for raw in reader]


def join_activation_features(metric_rows, activation_feature_rows):
    by_name = {row['filename']: row for row in activation_feature_rows}
    missing = [row['filename'] for row in metric_rows if row['filename'] not in by_name]
    if missing:
        raise FileNotFoundError('Missing activation features for: {}'.format(', '.join(missing[:20])))
    for row in metric_rows:
        features = by_name[row['filename']]
        for key, value in features.items():
            if key in ('index', 'filename', 'height', 'width'):
                continue
            row[key] = value


def is_activation_feature_name(name):
    if name.startswith('input_hazy_'):
        return True
    for prefix in MODEL_PREFIXES:
        if name.startswith(prefix + '_lf_'):
            return True
        for hook in COMMON_HOOKS:
            if name.startswith('{}_{}_'.format(prefix, hook)):
                return True
    for left, right in PAIR_PREFIXES:
        pair_prefix = '{}_{}_'.format(left, right)
        if not name.startswith(pair_prefix):
            continue
        suffix = name[len(pair_prefix):]
        if suffix.startswith('lf_'):
            return True
        if any(suffix.startswith(hook + '_') for hook in COMMON_HOOKS):
            return True
    return False


def activation_feature_names(rows):
    blocked = set()
    metric_or_meta = {
        'index',
        'filename',
        'image_id',
        'airlight',
        'beta',
        'height',
        'width',
        'airlight_bin',
        'beta_bin',
        'baseline_strength_bin',
        'lfv1_relation',
        'residual_relation',
        'residual_vs_lfv1_relation',
        'winner_by_psnr',
        'pattern',
    }
    for row in rows:
        blocked.update(metric_or_meta)
        for key in row.keys():
            if any(fragment in key for fragment in UNSAFE_SAFE_FEATURE_FRAGMENTS):
                blocked.add(key)
            if key.endswith('_psnr') or key.endswith('_ssim'):
                blocked.add(key)
    names = [
        key for key in rows[0].keys()
        if key not in blocked
        and base.is_number(rows[0].get(key, ''))
        and is_activation_feature_name(key)
    ]
    return base.numeric_feature_names(rows, sorted(set(names)))


def validate_safe_feature_names(feature_sets, allowed):
    allowed_safe = set(allowed['safe_activation_or_output'])
    for feature_set in SAFE_PRIMARY_FEATURE_SETS + SAFE_REFERENCE_FEATURE_SETS:
        unexpected = sorted(set(feature_sets.get(feature_set, [])) - allowed_safe)
        if unexpected:
            raise ValueError(
                'Unsafe feature(s) found in {}: {}'.format(
                    feature_set, ', '.join(unexpected[:50])
                )
            )
        unsafe = [
            name for name in feature_sets.get(feature_set, [])
            if any(fragment in name for fragment in UNSAFE_SAFE_FEATURE_FRAGMENTS)
        ]
        if unsafe:
            raise ValueError(
                'Leakage-like feature name(s) found in {}: {}'.format(
                    feature_set, ', '.join(unsafe[:50])
                )
            )


def build_feature_sets(rows):
    base.add_proxy_deltas(rows)
    strict_names, _agreement_names, rich_names = rich.add_rich_proxy_features(rows)
    activation_names = set(activation_feature_names(rows))
    strict_names = set(base.numeric_feature_names(rows, strict_names))
    rich_names = set(base.numeric_feature_names(rows, rich_names))
    metadata = set(base.metadata_features(rows))
    gt_diag = set(base.gt_diagnostic_features(rows))
    feature_sets = {
        'activation_only_proxy': sorted(activation_names),
        'activation_plus_strict_output_proxy': sorted(activation_names | strict_names),
        'activation_plus_rich_output_proxy': sorted(activation_names | rich_names),
        'rich_output_reference': sorted(rich_names),
        'metadata_diagnostic': sorted(metadata),
        'gt_diagnostic_leakage_check': sorted(gt_diag),
    }
    allowed = {
        'safe_activation_or_output': sorted(activation_names | strict_names | rich_names),
        'activation_allowed': sorted(activation_names),
        'strict_output_allowed': sorted(strict_names),
        'rich_output_allowed': sorted(rich_names),
        'metadata_allowed': sorted(metadata),
        'gt_diagnostic_allowed': sorted(gt_diag),
    }
    validate_safe_feature_names(feature_sets, allowed)
    return feature_sets, allowed


def label_counts(rows, indices):
    positives = sum(1 for idx in indices if base.label(rows[idx]) == 1)
    negatives = len(indices) - positives
    return positives, negatives


def sample_size_verdict(rows, specs, args):
    positives = sum(1 for row in rows if base.label(row) == 1)
    negatives = len(rows) - positives
    min_split_class = None
    for spec in specs:
        counts = list(label_counts(rows, spec['train_idx'])) + list(label_counts(rows, spec['valid_idx']))
        current = min(counts)
        min_split_class = current if min_split_class is None else min(min_split_class, current)
    powered = (
        len(rows) >= args.min_conclusive_images
        and positives >= args.min_class_count
        and negatives >= args.min_class_count
        and min_split_class is not None
        and min_split_class >= 20
    )
    reason = 'conclusive_sample_size' if powered else 'underpowered_smoke_only'
    return {
        'verdict': reason,
        'powered': powered,
        'num_images': len(rows),
        'positive_count': positives,
        'negative_count': negatives,
        'min_conclusive_images': args.min_conclusive_images,
        'min_class_count': args.min_class_count,
        'min_split_class_count': min_split_class if min_split_class is not None else '',
    }


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


def recommendation(aggregate_rows, sample_verdict, args):
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

    if not sample_verdict['powered']:
        return {
            'recommendation': 'do_not_train_selector_v2_underpowered_sample',
            'reason': (
                'The activation audit ran on too few or too imbalanced images for a '
                'conclusive selector decision. Treat it as smoke only.'
            ),
            'best_by_family': best_by_family,
            'passing_candidates': [],
        }

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
                'A frozen-activation, GT-free proxy passed the gain, oracle-recovery, '
                'and precision lines under random and degradation-held-out splits.'
            ),
            'best_by_family': best_by_family,
            'passing_candidates': passing_candidates,
        }

    return {
        'recommendation': 'do_not_train_selector_v2_yet',
        'reason': (
            'No frozen-activation, GT-free proxy passed the required random and '
            'degradation-held-out audit lines. Stop selector-v2 training plans or '
            'move to explicitly supervised/distilled selector targets.'
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
    )[:30]
    top_stumps = sorted(
        stump_rows,
        key=lambda row: row.get('valid_gain_vs_lfv1_psnr', -999),
        reverse=True,
    )[:20]
    top_coef = sorted(
        coefficient_rows,
        key=lambda row: row['abs_weight'],
        reverse=True,
    )[:30]
    lines = [
        '# Activation Selector Proxy Learnability Audit',
        '',
        '## Verdict',
        '',
        '- Recommendation: `{}`'.format(rec['recommendation']),
        '- Reason: {}'.format(rec['reason']),
        '- Sample-size verdict: `{}`'.format(summary['sample_size']['verdict']),
        '- Images: `{}`; positive/negative target counts: `{}` / `{}`'.format(
            summary['num_images'],
            summary['sample_size']['positive_count'],
            summary['sample_size']['negative_count'],
        ),
        '- Minimum conclusive sample rule: images `>= {}`, class counts `>= {}`, min split class count `>= 20`'.format(
            summary['sample_size']['min_conclusive_images'],
            summary['sample_size']['min_class_count'],
        ),
        '- Split families: `{}`'.format(', '.join(summary['split_families'])),
        '- Minimum pass line: gain `>= {:.3f}` dB, oracle recovery `>= {:.2f}`, residual precision `>= {:.2f}`'.format(
            summary['min_gain'],
            summary['min_oracle_recovery'],
            summary['min_precision'],
        ),
        '- Proceed requires a metadata-free activation proxy to pass all required families: `{}`'.format(
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
        '## Best Metadata-Free Activation Row By Split Family',
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
        '## Interpretation Rules',
        '',
        '- Safe activation feature sets are inference-time only: hazy input summaries, frozen internal activations, LF prior statistics, and output/agreement proxies.',
        '- Metadata and GT-aware feature sets are diagnostic only and cannot justify selector-v2.',
        '- Any run below the sample-size rule is smoke-only even if a metric looks favorable.',
        '- A selector-v2 route card should be written only after a metadata-free activation proxy passes random and degradation-held-out split families.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.splits <= 1:
        raise ValueError('--splits must be greater than 1')
    if not 0.0 < args.valid_fraction < 1.0:
        raise ValueError('--valid_fraction must be in (0, 1)')
    if args.min_conclusive_images <= 0:
        raise ValueError('--min_conclusive_images must be positive')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activation_feature_rows = activation_rows(args)
    write_csv(output_dir / 'activation_features.csv', activation_feature_rows)

    rows = base.filter_rows(base.read_rows(args.three_way_csv), args.sample_list)
    if args.max_images > 0:
        chosen = [row['filename'] for row in activation_feature_rows]
        chosen_set = set(chosen)
        rows = [row for row in rows if row['filename'] in chosen_set]
        rows.sort(key=lambda row: chosen.index(row['filename']))
    join_activation_features(rows, activation_feature_rows)
    feature_sets, allowed = build_feature_sets(rows)
    specs = rich.split_specs(rows, args)
    sample_verdict = sample_size_verdict(rows, specs, args)

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
    decision = recommendation(aggregate_rows, sample_verdict, args)
    summary = {
        'input_csv': args.three_way_csv,
        'dataset_root': args.dataset_root,
        'baseline_checkpoint': args.baseline_checkpoint,
        'lfv1_checkpoint': args.lfv1_checkpoint,
        'residual_checkpoint': args.residual_checkpoint,
        'sample_list': args.sample_list,
        'max_images': args.max_images,
        'num_images': len(rows),
        'seed': args.seed,
        'splits': args.splits,
        'valid_fraction': args.valid_fraction,
        'split_families': sorted(set(spec['split_family'] for spec in specs)),
        'split_count': len(specs),
        'min_gain': args.min_gain,
        'min_oracle_recovery': args.min_oracle_recovery,
        'min_precision': args.min_precision,
        'sample_size': sample_verdict,
        'feature_counts': {name: len(features) for name, features in feature_sets.items()},
        'decision': decision,
    }

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'feature_lists.json', 'w', encoding='utf-8') as f:
        json.dump(feature_sets, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'allowed_feature_sets.json', 'w', encoding='utf-8') as f:
        json.dump(allowed, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / 'joined_features.csv', rows)
    write_csv(output_dir / 'split_metrics.csv', split_rows)
    write_csv(output_dir / 'aggregate_summary.csv', aggregate_rows)
    write_csv(output_dir / 'stump_rules.csv', stump_rows)
    write_csv(output_dir / 'logistic_coefficients.csv', coefficient_rows)
    write_report(output_dir / 'analysis_report.md', summary, aggregate_rows, stump_rows, coefficient_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
