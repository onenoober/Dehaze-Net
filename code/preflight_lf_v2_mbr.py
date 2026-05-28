import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import DEANet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preflight cost, neutral-init, and branch-activity checks for LF-v2 MBR.'
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=20260528)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--latency_warmup', type=int, default=20)
    parser.add_argument('--latency_iters', type=int, default=80)
    parser.add_argument('--lf_prior_channels', type=int, default=8)
    parser.add_argument('--lf_prior_pool', type=int, default=8)
    parser.add_argument('--lf_prior_gate_init', type=float, default=0.0)
    parser.add_argument('--lf_mbr_channels', type=int, default=8)
    parser.add_argument('--lf_mbr_pool_sizes', type=str, default='4,8,16')
    parser.add_argument('--param_budget_pct', type=float, default=3.0)
    parser.add_argument('--latency_budget_pct', type=float, default=8.0)
    parser.add_argument('--neutral_tolerance', type=float, default=1e-7)
    parser.add_argument('--branch_std_floor', type=float, default=1e-8)
    parser.add_argument('--strict_exit', action='store_true')
    return parser.parse_args()


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def make_model(args, multiscale_refiner):
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = DEANet(
        base_dim=32,
        use_lf_prior=True,
        lf_prior_channels=args.lf_prior_channels,
        lf_prior_pool=args.lf_prior_pool,
        lf_prior_gate_init=args.lf_prior_gate_init,
        lf_prior_injection='pre_mix',
        lf_multiscale_refiner=multiscale_refiner,
        lf_mbr_channels=args.lf_mbr_channels,
        lf_mbr_pool_sizes=args.lf_mbr_pool_sizes
    )
    return model.to(args.device)


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def measure_latency(model, x, args):
    model.eval()
    with torch.no_grad():
        for _ in range(args.latency_warmup):
            model(x)
        synchronize(args.device)
        start = time.perf_counter()
        for _ in range(args.latency_iters):
            model(x)
        synchronize(args.device)
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / float(args.latency_iters)


def tensor_stats(stats):
    if stats is None:
        return None
    return {key: float(value.detach().cpu().item()) for key, value in stats.items()}


def random_backward_check(model, x):
    model.train()
    target = torch.rand_like(model(x).detach())
    out = model(x)
    loss = F.l1_loss(out, target)
    loss.backward()
    lf_prior = getattr(model, 'lf_prior', None)
    gate_grad = None
    if lf_prior is not None and lf_prior.gate.grad is not None:
        gate_grad = float(lf_prior.gate.grad.detach().abs().cpu().item())
    finite_grads = True
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            finite_grads = False
            break
    return {
        'loss': float(loss.detach().cpu().item()),
        'loss_is_finite': bool(torch.isfinite(loss).item()),
        'gate_grad_abs': gate_grad,
        'gate_grad_is_finite': gate_grad is not None and math.isfinite(gate_grad),
        'all_grads_finite': finite_grads
    }


def write_report(path, summary):
    lines = [
        '# LF-v2 MBR Preflight',
        '',
        '- Recommendation: `{}`'.format(summary['recommendation']),
        '- Device: `{}`'.format(summary['device']),
        '- LF-v1 params: `{}`'.format(summary['lfv1_params']),
        '- LF-v2 MBR params: `{}`'.format(summary['mbr_params']),
        '- Param overhead: `{:.4f}%`'.format(summary['param_overhead_pct']),
        '- LF-v1 latency: `{:.4f} ms`'.format(summary['lfv1_latency_ms']),
        '- LF-v2 MBR latency: `{:.4f} ms`'.format(summary['mbr_latency_ms']),
        '- Latency overhead: `{:.4f}%`'.format(summary['latency_overhead_pct']),
        '- Neutral max abs diff: `{:.10f}`'.format(summary['neutral_max_abs_diff']),
        '- Branch std: `{:.10f}`'.format(summary['branch_stats']['std'] if summary['branch_stats'] else float('nan')),
        '- Random backward loss: `{:.8f}`'.format(summary['random_backward']['loss']),
        '',
        '## Pass Checks',
        ''
    ]
    for key, value in summary['pass_checks'].items():
        lines.append('- `{}`: `{}`'.format(key, value))
    path.write_text('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = args.device == 'cuda'
    x = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=args.device)

    lfv1 = make_model(args, multiscale_refiner=False)
    mbr = make_model(args, multiscale_refiner=True)
    lfv1.eval()
    mbr.eval()

    lfv1_params = count_parameters(lfv1)
    mbr_params = count_parameters(mbr)
    param_overhead_pct = (float(mbr_params - lfv1_params) / float(lfv1_params)) * 100.0

    with torch.no_grad():
        lfv1_out = lfv1(x)
        mbr_out = mbr(x)
        neutral_max_abs_diff = float((lfv1_out - mbr_out).abs().max().detach().cpu().item())
        branch_stats = tensor_stats(getattr(mbr.lf_prior, 'last_multiscale_stats', None))

    lfv1_latency_ms = measure_latency(lfv1, x, args)
    mbr_latency_ms = measure_latency(mbr, x, args)
    latency_overhead_pct = ((mbr_latency_ms - lfv1_latency_ms) / max(lfv1_latency_ms, 1e-12)) * 100.0

    random_backward = random_backward_check(mbr, x.detach().clone())

    pass_checks = {
        'param_budget': param_overhead_pct <= args.param_budget_pct,
        'latency_budget': latency_overhead_pct <= args.latency_budget_pct,
        'neutral_init': neutral_max_abs_diff <= args.neutral_tolerance,
        'branch_stats_present': branch_stats is not None,
        'branch_non_degenerate': branch_stats is not None and branch_stats['std'] > args.branch_std_floor,
        'random_backward_finite': (
            random_backward['loss_is_finite']
            and random_backward['gate_grad_is_finite']
            and random_backward['all_grads_finite']
        )
    }
    recommendation = 'proceed_to_training_smoke' if all(pass_checks.values()) else 'do_not_train_lf_v2_mbr_yet'

    summary = {
        'recommendation': recommendation,
        'device': args.device,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'lf_prior_channels': args.lf_prior_channels,
        'lf_prior_pool': args.lf_prior_pool,
        'lf_prior_gate_init': args.lf_prior_gate_init,
        'lf_mbr_channels': args.lf_mbr_channels,
        'lf_mbr_pool_sizes': args.lf_mbr_pool_sizes,
        'lfv1_params': lfv1_params,
        'mbr_params': mbr_params,
        'param_overhead_pct': param_overhead_pct,
        'param_budget_pct': args.param_budget_pct,
        'lfv1_latency_ms': lfv1_latency_ms,
        'mbr_latency_ms': mbr_latency_ms,
        'latency_overhead_pct': latency_overhead_pct,
        'latency_budget_pct': args.latency_budget_pct,
        'neutral_max_abs_diff': neutral_max_abs_diff,
        'neutral_tolerance': args.neutral_tolerance,
        'branch_stats': branch_stats,
        'branch_std_floor': args.branch_std_floor,
        'random_backward': random_backward,
        'pass_checks': pass_checks
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    write_report(output_dir / 'analysis_report.md', summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict_exit and recommendation != 'proceed_to_training_smoke':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
