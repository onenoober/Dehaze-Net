import os
import csv
import json
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils import AverageMeter, pad_img, val_psnr, val_ssim
from data import ValDataset
from data.data_loader import resolve_pair_dirs
from option import opt
from model import Backbone, BaselineRelativeFrequencyResidualCorrector, DEANet, DEANetCBRFRC


EPS = 1e-8


def load_checkpoint(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    if not any(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


def resolve_checkpoint_path(checkpoint_name, default_root='../trained_models'):
    if os.path.isabs(checkpoint_name):
        candidates = [checkpoint_name]
    else:
        candidates = [
            checkpoint_name,
            os.path.join(default_root, opt.dataset, checkpoint_name),
            os.path.join(opt.exp_dir, opt.dataset, opt.model_name, 'saved_model', checkpoint_name),
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError('No checkpoint found. Tried: {}'.format(', '.join(candidates)))


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def create_brf_network(candidate_checkpoint_path):
    baseline = DEANet(base_dim=32)
    corrector = BaselineRelativeFrequencyResidualCorrector(
        hidden_channels=opt.brf_hidden_channels,
        wavelet_levels=opt.brf_wavelet_levels,
        gate_init=opt.brf_gate_init,
        hf_gate_init=opt.brf_hf_gate_init,
        max_residual=opt.brf_max_residual,
        max_color_residual=opt.brf_max_color_residual,
        max_hf_residual=opt.brf_max_hf_residual,
        hf_scale=opt.brf_hf_scale,
        use_haze_prior=opt.brf_use_haze_prior,
        preserve_highfreq=opt.brf_preserve_highfreq,
        pyramid_type=opt.brf_pyramid_type,
        lf_pool=opt.brf_lf_pool,
        mid_pool=opt.brf_mid_pool,
    )
    network = DEANetCBRFRC(
        baseline=baseline,
        corrector=corrector,
        freeze_baseline=opt.brf_freeze_baseline,
        use_baseline_detach=opt.brf_use_baseline_detach,
    )
    candidate = load_checkpoint(candidate_checkpoint_path)
    candidate_state = strip_module_prefix(checkpoint_state_dict(candidate))
    is_full_wrapper = any(key.startswith('baseline.') for key in candidate_state) and any(
        key.startswith('corrector.') for key in candidate_state
    )
    if is_full_wrapper:
        network.load_state_dict(candidate_state)
    else:
        if opt.brf_baseline_checkpoint == 'null':
            raise ValueError('--brf_baseline_checkpoint is required unless the candidate checkpoint stores the full wrapper')
        baseline_path = resolve_checkpoint_path(opt.brf_baseline_checkpoint)
        baseline_ckpt = load_checkpoint(baseline_path)
        baseline.load_state_dict(strip_module_prefix(checkpoint_state_dict(baseline_ckpt)))
        if all(key.startswith('corrector.') for key in candidate_state):
            network.load_state_dict(candidate_state, strict=False)
        else:
            corrector.load_state_dict(candidate_state)
    return network.cuda()


def lowpass(x, pool_size):
    low = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def metric_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def mean_l2_norm(x):
    return metric_float(x.reshape(x.shape[0], -1).norm(dim=1).mean())


def residual_cosine(pred_lf, target_lf):
    pred_vec = pred_lf.reshape(pred_lf.shape[0], -1)
    target_vec = target_lf.reshape(target_lf.shape[0], -1)
    cosine = (pred_vec * target_vec).sum(dim=1)
    cosine = cosine / (pred_vec.norm(dim=1) * target_vec.norm(dim=1) + EPS)
    return metric_float(cosine.mean())


def tensor_stats(prefix, value):
    detached = value.detach()
    return {
        prefix + '_mean': metric_float(detached.mean()),
        prefix + '_std': metric_float(detached.std(unbiased=False)),
        prefix + '_min': metric_float(detached.min()),
        prefix + '_max': metric_float(detached.max()),
    }


def add_brf_groups(rows):
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda row: row['psnr_j0'])
    weak_cutoff = sorted_rows[max(0, len(rows) // 4 - 1)]['psnr_j0']
    strong_cutoff = sorted_rows[min(len(rows) - 1, (len(rows) * 3) // 4)]['psnr_j0']
    for row in rows:
        row['weak_baseline_group'] = row['psnr_j0'] <= weak_cutoff
        row['strong_baseline_group'] = row['psnr_j0'] >= strong_cutoff


def write_per_image_outputs(rows):
    if not rows:
        return
    if 'psnr_j0' in rows[0]:
        add_brf_groups(rows)
    csv_path = os.path.join(opt.saved_infer_dir, 'per_image_metrics.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if 'psnr_j0' not in rows[0]:
        return
    summary = {
        'num_images': len(rows),
        'mean_psnr_j0': sum(row['psnr_j0'] for row in rows) / len(rows),
        'mean_ssim_j0': sum(row['ssim_j0'] for row in rows) / len(rows),
        'mean_psnr_j': sum(row['psnr_j'] for row in rows) / len(rows),
        'mean_ssim_j': sum(row['ssim_j'] for row in rows) / len(rows),
        'mean_delta_vs_j0': sum(row['delta_vs_j0'] for row in rows) / len(rows),
        'wrong_direction_count': sum(1 for row in rows if row['wrong_direction']),
        'lf_mse_improved_count': sum(1 for row in rows if row['lf_mse_delta'] < 0),
        'lf_mse_regressed_count': sum(1 for row in rows if row['lf_mse_delta'] > 0),
        'gate_lf_mean_global': sum(row['gate_lf_mean'] for row in rows) / len(rows),
        'gate_hf_mean_global': sum(row['gate_hf_mean'] for row in rows) / len(rows),
    }
    with open(os.path.join(opt.saved_infer_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def eval(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    per_image_rows = []

    torch.cuda.empty_cache()

    network.eval()

    for i, batch in enumerate(tqdm(val_loader, desc='evaluation')):
        max_batches = opt.max_eval_batches if opt.max_eval_batches > 0 else opt.max_test_batches
        if max_batches > 0 and i >= max_batches:
            break
        hazy_img = batch['hazy'].cuda()
        clear_img = batch['clear'].cuda()

        with torch.no_grad():
            H, W = hazy_img.shape[2:]
            hazy_img = pad_img(hazy_img, 4)
            if opt.use_brf_frequency_corrector:
                out_dict = network(hazy_img, return_aux=True)
                output = out_dict['out'].clamp(0, 1)
                j0 = out_dict['j0'].clamp(0, 1)
            else:
                out_dict = None
                output = network(hazy_img)
                output = output.clamp(0, 1)
            output = output[:, :, :H, :W]
            if out_dict is not None:
                j0 = j0[:, :, :H, :W]
            if opt.save_infer_results:
                save_image(output, os.path.join(opt.saved_infer_dir, batch['filename'][0]))

        psnr_tmp = val_psnr(output, clear_img)
        ssim_tmp = val_ssim(output, clear_img).item()
        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)
        if out_dict is not None or opt.save_per_image_csv:
            row = {
                'image': batch['filename'][0],
                'psnr_j': metric_float(psnr_tmp),
                'ssim_j': metric_float(ssim_tmp),
            }
            if out_dict is not None:
                psnr_j0 = val_psnr(j0, clear_img)
                ssim_j0 = val_ssim(j0, clear_img).item()
                target_lf = lowpass(clear_img, opt.brf_lf_pool) - lowpass(j0, opt.brf_lf_pool)
                pred_lf = lowpass(output, opt.brf_lf_pool) - lowpass(j0, opt.brf_lf_pool)
                lf_mse_j0 = metric_float(F.mse_loss(lowpass(j0, opt.brf_lf_pool), lowpass(clear_img, opt.brf_lf_pool)))
                lf_mse_j = metric_float(F.mse_loss(lowpass(output, opt.brf_lf_pool), lowpass(clear_img, opt.brf_lf_pool)))
                cosine = residual_cosine(pred_lf, target_lf)
                row.update({
                    'psnr_j0': metric_float(psnr_j0),
                    'ssim_j0': metric_float(ssim_j0),
                    'delta_vs_j0': metric_float(psnr_tmp) - metric_float(psnr_j0),
                    'delta_vs_lfv1_optional': '',
                    'target_lf_norm': mean_l2_norm(target_lf),
                    'pred_lf_norm': mean_l2_norm(pred_lf),
                    'residual_cosine': cosine,
                    'lf_mse_j0': lf_mse_j0,
                    'lf_mse_j': lf_mse_j,
                    'lf_mse_delta': lf_mse_j - lf_mse_j0,
                    'wrong_direction': cosine < 0,
                    'c_lf_norm': mean_l2_norm(out_dict['c_lf'][:, :, :H, :W]),
                    'c_color_norm': mean_l2_norm(out_dict['c_color'][:, :, :H, :W]),
                    'c_hf_norm': mean_l2_norm(out_dict['c_hf'][:, :, :H, :W]),
                })
                row.update(tensor_stats('gate_lf', out_dict['gate_lf'][:, :, :H, :W]))
                row.update(tensor_stats('gate_hf', out_dict['gate_hf'][:, :, :H, :W]))
            per_image_rows.append(row)

    if per_image_rows:
        write_per_image_outputs(per_image_rows)
    return PSNR.avg, SSIM.avg


if __name__ == '__main__':
    checkpoint_path = resolve_checkpoint_path(opt.pre_trained_model)
    if opt.use_brf_frequency_corrector:
        network = create_brf_network(checkpoint_path)
    else:
        network = Backbone().cuda()

    hazy_dir, clear_dir = resolve_pair_dirs(opt.val_dataset_dir)
    print('val_hazy_dir:', hazy_dir)
    print('val_clear_dir:', clear_dir)
    val_dataset = ValDataset(hazy_dir, clear_dir)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=False)

    # load pre-trained model
    if not opt.use_brf_frequency_corrector:
        ckpt = load_checkpoint(checkpoint_path)
        network.load_state_dict(checkpoint_state_dict(ckpt))

    # start evaluation
    avg_psnr, avg_ssim = eval(val_loader, network) 
    print('Evaluation on {}\nPSNR:{}\nSSIM:{}'.format(opt.dataset, avg_psnr, avg_ssim))
