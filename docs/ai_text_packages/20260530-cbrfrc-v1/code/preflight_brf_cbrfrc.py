import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from data.data_loader import find_clear_image, list_image_files, resolve_pair_dirs
from metric import psnr, ssim
from model import BaselineRelativeFrequencyResidualCorrector, DEANet, DEANetCBRFRC


EPS = 1e-8


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1'):
        return True
    if value in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HAZE4K')
    parser.add_argument('--cr_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pad_size', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--oracle_bound', type=float, default=0.08)
    parser.add_argument('--micro_steps', type=int, default=0)
    parser.add_argument('--micro_subset_size', type=int, default=64)
    parser.add_argument('--micro_batch_size', type=int, default=8)
    parser.add_argument('--micro_lr', type=float, default=0.001)
    parser.add_argument('--micro_log_interval', type=int, default=100)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--brf_hidden_channels', type=int, default=16)
    parser.add_argument('--brf_wavelet_levels', type=int, default=2)
    parser.add_argument('--brf_pyramid_type', type=str, default='laplacian', choices=['laplacian', 'haar'])
    parser.add_argument('--brf_gate_init', type=float, default=-4.0)
    parser.add_argument('--brf_hf_gate_init', type=float, default=-5.0)
    parser.add_argument('--brf_max_residual', type=float, default=0.08)
    parser.add_argument('--brf_max_color_residual', type=float, default=0.04)
    parser.add_argument('--brf_max_hf_residual', type=float, default=0.03)
    parser.add_argument('--brf_hf_scale', type=float, default=0.1)
    parser.add_argument('--brf_use_haze_prior', action='store_true')
    parser.add_argument('--brf_preserve_highfreq', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--brf_lf_pool', type=int, default=8)
    parser.add_argument('--brf_mid_pool', type=int, default=4)
    parser.add_argument('--brf_dir_norm_floor', type=float, default=0.01)
    parser.add_argument('--brf_preserve_target_thr', type=float, default=0.015)
    parser.add_argument('--brf_preserve_warmup_steps', type=int, default=0)
    parser.add_argument('--brf_preserve_gate_weight', type=float, default=1.0)
    parser.add_argument('--w_loss_L1', type=float, default=1.0)
    parser.add_argument('--w_loss_brf_res_lf', type=float, default=0.10)
    parser.add_argument('--w_loss_brf_dir', type=float, default=0.02)
    parser.add_argument('--w_loss_brf_preserve', type=float, default=0.05)
    parser.add_argument('--w_loss_brf_bound', type=float, default=0.01)
    parser.add_argument('--w_loss_brf_color', type=float, default=0.02)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_module_prefix(state_dict):
    if not any(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def build_corrector(args):
    return BaselineRelativeFrequencyResidualCorrector(
        hidden_channels=args.brf_hidden_channels,
        wavelet_levels=args.brf_wavelet_levels,
        gate_init=args.brf_gate_init,
        hf_gate_init=args.brf_hf_gate_init,
        max_residual=args.brf_max_residual,
        max_color_residual=args.brf_max_color_residual,
        max_hf_residual=args.brf_max_hf_residual,
        hf_scale=args.brf_hf_scale,
        use_haze_prior=args.brf_use_haze_prior,
        preserve_highfreq=args.brf_preserve_highfreq,
        pyramid_type=args.brf_pyramid_type,
        lf_pool=args.brf_lf_pool,
        mid_pool=args.brf_mid_pool,
    )


def load_baseline(args):
    checkpoint = load_checkpoint(args.cr_checkpoint, args.device)
    model = DEANet(base_dim=32)
    model.load_state_dict(strip_module_prefix(checkpoint_state_dict(checkpoint)))
    model.to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint


def build_wrapper(args, baseline):
    wrapper = DEANetCBRFRC(
        baseline=baseline,
        corrector=build_corrector(args),
        freeze_baseline=True,
        use_baseline_detach=True,
    )
    wrapper.to(args.device)
    return wrapper


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')


def pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def center_crop(image, patch_size):
    width, height = image.size
    if width < patch_size or height < patch_size:
        raise ValueError('Image {}x{} is smaller than patch size {}'.format(width, height, patch_size))
    left = (width - patch_size) // 2
    top = (height - patch_size) // 2
    return image.crop((left, top, left + patch_size, top + patch_size))


def lowpass(x, pool_size):
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def mean_l2_norm(x):
    return x.reshape(x.shape[0], -1).norm(dim=1).mean()


def infer_baseline(model, hazy, pad_size):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    out = model(padded).clamp(0, 1)
    return out[:, :, :h, :w]


def infer_wrapper(wrapper, hazy, pad_size, target=None):
    _, _, h, w = hazy.shape
    padded = pad_img(hazy, pad_size)
    target_padded = None
    if target is not None:
        target_padded = pad_img(target, pad_size)
    out_dict = wrapper(padded, target=target_padded, return_aux=True)
    cropped = {}
    for key, value in out_dict.items():
        if torch.is_tensor(value) and value.dim() == 4:
            cropped[key] = value[:, :, :h, :w]
        else:
            cropped[key] = value
    cropped['out'] = cropped['out'].clamp(0, 1)
    cropped['j0'] = cropped['j0'].clamp(0, 1)
    return cropped


def cosine(pred, target, floor):
    pred_vec = pred.reshape(pred.shape[0], -1)
    target_vec = target.reshape(target.shape[0], -1)
    pred_norm = pred_vec.norm(dim=1)
    target_norm = target_vec.norm(dim=1)
    valid = target_norm > floor
    if not valid.any():
        return pred.new_zeros(())
    value = (pred_vec[valid] * target_vec[valid]).sum(dim=1)
    value = value / (pred_norm[valid] * target_norm[valid] + EPS)
    return value.mean()


def brf_losses(out_dict, target, args, step=0):
    out = out_dict['out']
    j0 = out_dict['j0'].detach()
    pred_lf = lowpass(out, args.brf_lf_pool) - lowpass(j0, args.brf_lf_pool)
    target_lf = lowpass(target, args.brf_lf_pool) - lowpass(j0, args.brf_lf_pool)
    loss_res_lf = F.l1_loss(pred_lf, target_lf)
    residual_cosine = cosine(pred_lf, target_lf, args.brf_dir_norm_floor)
    loss_dir = 1.0 - residual_cosine
    preserve_mask = (
        target_lf.abs().mean(dim=(1, 2, 3), keepdim=True) < args.brf_preserve_target_thr
    ).to(out.dtype)
    if args.brf_preserve_warmup_steps > 0 and step < args.brf_preserve_warmup_steps:
        loss_preserve = out.new_zeros(())
    else:
        loss_preserve = torch.mean(
            preserve_mask * (out_dict['c_lf'].abs() + args.brf_preserve_gate_weight * out_dict['gate_lf'])
        )
    loss_bound = (out_dict['c_lf'] + out_dict['c_color'] + out_dict['c_hf']).abs().mean()
    loss_color = F.l1_loss(out.mean(dim=(2, 3)), target.mean(dim=(2, 3)))
    loss_l1 = F.l1_loss(out, target)
    loss = (
        args.w_loss_L1 * loss_l1
        + args.w_loss_brf_res_lf * loss_res_lf
        + args.w_loss_brf_dir * loss_dir
        + args.w_loss_brf_preserve * loss_preserve
        + args.w_loss_brf_bound * loss_bound
        + args.w_loss_brf_color * loss_color
    )
    pred_norm = mean_l2_norm(pred_lf)
    target_norm = mean_l2_norm(target_lf)
    return loss, {
        'loss': loss,
        'l1': loss_l1,
        'res_lf': loss_res_lf,
        'dir': loss_dir,
        'preserve': loss_preserve,
        'bound': loss_bound,
        'color': loss_color,
        'residual_cosine': residual_cosine,
        'pred_lf_norm': pred_norm,
        'target_lf_norm': target_norm,
        'residual_norm_ratio': pred_norm / (target_norm + EPS),
        'gate_lf_mean': out_dict['gate_lf'].mean(),
        'gate_hf_mean': out_dict['gate_hf'].mean(),
        'c_lf_norm': mean_l2_norm(out_dict['c_lf']),
    }


def tensor_float_dict(values):
    return {
        key: scalar(value)
        for key, value in values.items()
    }


def run_shape_smoke(wrapper, args):
    wrapper.eval()
    with torch.no_grad():
        hazy = torch.rand(2, 3, 256, 256, device=args.device)
        target = torch.rand(2, 3, 256, 256, device=args.device)
        out_dict = wrapper(hazy, target=target, return_aux=True)
    return {
        'out_shape': list(out_dict['out'].shape),
        'j0_shape': list(out_dict['j0'].shape),
        'c_lf_shape': list(out_dict['c_lf'].shape),
        'gate_lf_shape': list(out_dict['gate_lf'].shape),
        'out_min': scalar(out_dict['out'].min()),
        'out_max': scalar(out_dict['out'].max()),
        'gate_lf_mean': scalar(out_dict['gate_lf'].mean()),
        'gate_hf_mean': scalar(out_dict['gate_hf'].mean()),
        'c_lf_norm': scalar(mean_l2_norm(out_dict['c_lf'])),
        'max_abs_out_minus_j0': scalar((out_dict['out'] - out_dict['j0']).abs().max()),
    }


def set_gate_biases(wrapper, value):
    for head in (
        wrapper.corrector.gate_lf_head,
        wrapper.corrector.gate_color_head,
        wrapper.corrector.gate_hf_head,
    ):
        torch.nn.init.constant_(head.bias, float(value))


def evaluate_identity_and_oracle(wrapper, baseline, args):
    test_root = Path('../dataset') / args.dataset / 'test'
    hazy_dir, clear_dir = resolve_pair_dirs(test_root)
    names = list_image_files(hazy_dir)
    if args.max_images > 0:
        names = names[:args.max_images]
    rows = []
    with torch.no_grad():
        for idx, name in enumerate(names, start=1):
            hazy_img = Image.open(Path(hazy_dir) / name).convert('RGB')
            clear_img = Image.open(find_clear_image(clear_dir, name)).convert('RGB')
            hazy = pil_to_tensor(hazy_img).unsqueeze(0).to(args.device)
            clear = pil_to_tensor(clear_img).unsqueeze(0).to(args.device)
            j0 = infer_baseline(baseline, hazy, args.pad_size)
            out_dict = infer_wrapper(wrapper, hazy, args.pad_size)
            j = out_dict['out']
            identity_abs = (j - j0).abs().max()
            residual = lowpass(clear, args.brf_lf_pool) - lowpass(j0, args.brf_lf_pool)
            oracle = torch.clamp(j0 + residual, 0, 1)
            oracle_bound = torch.clamp(j0 + residual.clamp(-args.oracle_bound, args.oracle_bound), 0, 1)
            psnr_j0 = scalar(psnr(j0, clear))
            psnr_j = scalar(psnr(j, clear))
            psnr_oracle = scalar(psnr(oracle, clear))
            psnr_oracle_bound = scalar(psnr(oracle_bound, clear))
            rows.append({
                'index': idx,
                'image': name,
                'psnr_j0': psnr_j0,
                'ssim_j0': scalar(ssim(j0, clear)),
                'psnr_j': psnr_j,
                'ssim_j': scalar(ssim(j, clear)),
                'identity_abs_max': scalar(identity_abs),
                'psnr_oracle_lf': psnr_oracle,
                'ssim_oracle_lf': scalar(ssim(oracle, clear)),
                'psnr_oracle_lf_bound': psnr_oracle_bound,
                'ssim_oracle_lf_bound': scalar(ssim(oracle_bound, clear)),
                'oracle_lf_vs_j0_delta': psnr_oracle - psnr_j0,
                'oracle_lf_bound_vs_j0_delta': psnr_oracle_bound - psnr_j0,
            })
            if idx % 50 == 0 or idx == len(names):
                print('identity/oracle {}/{}'.format(idx, len(names)), flush=True)
    psnr_j0_values = [row['psnr_j0'] for row in rows]
    sorted_rows = sorted(rows, key=lambda row: row['psnr_j0'])
    strong_cutoff = sorted_rows[min(len(rows) - 1, (len(rows) * 3) // 4)]['psnr_j0']
    strong_rows = [row for row in rows if row['psnr_j0'] >= strong_cutoff]
    summary = {
        'num_images': len(rows),
        'mean_psnr_j0': float(np.mean(psnr_j0_values)),
        'mean_ssim_j0': float(np.mean([row['ssim_j0'] for row in rows])),
        'mean_psnr_j': float(np.mean([row['psnr_j'] for row in rows])),
        'mean_ssim_j': float(np.mean([row['ssim_j'] for row in rows])),
        'identity_mean_psnr_delta': float(np.mean([row['psnr_j'] - row['psnr_j0'] for row in rows])),
        'identity_max_abs_psnr_delta': float(max(abs(row['psnr_j'] - row['psnr_j0']) for row in rows)),
        'identity_max_abs_pixel_delta': float(max(row['identity_abs_max'] for row in rows)),
        'mean_psnr_oracle_lf': float(np.mean([row['psnr_oracle_lf'] for row in rows])),
        'mean_psnr_oracle_lf_bound': float(np.mean([row['psnr_oracle_lf_bound'] for row in rows])),
        'oracle_lf_vs_j0_delta': float(np.mean([row['oracle_lf_vs_j0_delta'] for row in rows])),
        'oracle_lf_bound_vs_j0_delta': float(np.mean([row['oracle_lf_bound_vs_j0_delta'] for row in rows])),
        'oracle_lf_bound_strong_regression_count': sum(
            1 for row in strong_rows if row['oracle_lf_bound_vs_j0_delta'] < 0
        ),
        'strong_baseline_count': len(strong_rows),
        'strong_baseline_cutoff_psnr': float(strong_cutoff),
    }
    return rows, summary


def run_micro_overfit(args):
    if args.micro_steps <= 0:
        return None, []
    train_root = Path('../dataset') / args.dataset / 'train'
    hazy_dir, clear_dir = resolve_pair_dirs(train_root)
    names = list_image_files(hazy_dir)[:args.micro_subset_size]
    hazy_tensors = []
    clear_tensors = []
    for name in names:
        hazy_img = Image.open(Path(hazy_dir) / name).convert('RGB')
        clear_img = Image.open(find_clear_image(clear_dir, name)).convert('RGB')
        hazy_tensors.append(pil_to_tensor(center_crop(hazy_img, args.patch_size)))
        clear_tensors.append(pil_to_tensor(center_crop(clear_img, args.patch_size)))
    hazy_tensor = torch.stack(hazy_tensors, dim=0)
    clear_tensor = torch.stack(clear_tensors, dim=0)
    subset_size = hazy_tensor.shape[0]
    dataset = TensorDataset(hazy_tensor, clear_tensor)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    probe_hazy = hazy_tensor[:args.micro_batch_size].to(args.device)
    probe_clear = clear_tensor[:args.micro_batch_size].to(args.device)
    baseline, _ = load_baseline(args)
    wrapper = build_wrapper(args, baseline)
    wrapper.train()
    wrapper.baseline.eval()
    optimizer = torch.optim.Adam(wrapper.corrector.parameters(), lr=args.micro_lr, betas=(0.9, 0.999))
    loader_iter = iter(loader)
    rows = []
    def probe_metrics(step):
        was_training = wrapper.training
        wrapper.eval()
        with torch.no_grad():
            probe_dict = wrapper(probe_hazy, target=probe_clear, return_aux=True)
            _, metrics = brf_losses(probe_dict, probe_clear, args, step=step)
        if was_training:
            wrapper.train()
            wrapper.baseline.eval()
        return tensor_float_dict(metrics)

    first_metrics = probe_metrics(0)
    last_metrics = first_metrics
    for step in range(1, args.micro_steps + 1):
        try:
            hazy, clear = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            hazy, clear = next(loader_iter)
        hazy = hazy.to(args.device, non_blocking=True)
        clear = clear.to(args.device, non_blocking=True)
        out_dict = wrapper(hazy, target=clear, return_aux=True)
        loss, metrics = brf_losses(out_dict, clear, args, step=step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.micro_log_interval == 0 or step == args.micro_steps:
            last_metrics = probe_metrics(step)
            row = dict(last_metrics)
            row['step'] = step
            rows.append(row)
            print('micro step {} loss {:.6f} res_lf {:.6f} cosine {:.6f} gate_lf {:.6f}'.format(
                step,
                row['loss'],
                row['res_lf'],
                row['residual_cosine'],
                row['gate_lf_mean'],
            ), flush=True)
    summary = {
        'steps': args.micro_steps,
        'subset_size': subset_size,
        'batch_size': args.micro_batch_size,
        'lr': args.micro_lr,
        'initial_loss': first_metrics['loss'],
        'final_loss': last_metrics['loss'],
        'initial_res_lf': first_metrics['res_lf'],
        'final_res_lf': last_metrics['res_lf'],
        'initial_residual_cosine': first_metrics['residual_cosine'],
        'final_residual_cosine': last_metrics['residual_cosine'],
        'initial_gate_lf_mean': first_metrics['gate_lf_mean'],
        'final_gate_lf_mean': last_metrics['gate_lf_mean'],
        'initial_c_lf_norm': first_metrics['c_lf_norm'],
        'final_c_lf_norm': last_metrics['c_lf_norm'],
    }
    return summary, rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline, baseline_ckpt = load_baseline(args)
    wrapper = build_wrapper(args, baseline)
    shape_smoke = run_shape_smoke(wrapper, args)
    set_gate_biases(wrapper, -20.0)
    identity_rows, identity_summary = evaluate_identity_and_oracle(wrapper, baseline, args)
    micro_summary, micro_rows = run_micro_overfit(args)

    summary = {
        'dataset': args.dataset,
        'cr_checkpoint': args.cr_checkpoint,
        'cr_checkpoint_step': baseline_ckpt.get('step') if isinstance(baseline_ckpt, dict) else None,
        'shape_smoke': shape_smoke,
        'identity_oracle': identity_summary,
        'micro_overfit': micro_summary,
    }
    write_csv(output_dir / 'identity_oracle_per_image.csv', identity_rows)
    write_csv(output_dir / 'micro_overfit_log.csv', micro_rows)
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
