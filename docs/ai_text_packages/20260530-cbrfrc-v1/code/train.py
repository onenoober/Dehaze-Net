import os, time, math
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from logger import plot_loss_log, plot_psnr_log
from metric import psnr, ssim
from model import BaselineRelativeFrequencyResidualCorrector, DEANet, DEANetCBRFRC
from loss import CRPlusV2Loss, ContrastLoss
from option_train import opt
from data.data_loader import TrainDataset, TestDataset, resolve_pair_dirs
from warmstart_freeze import TrainableSchedule


start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps

BRF_LOG_KEYS = (
    'BRF_res_lf', 'BRF_dir', 'BRF_preserve', 'BRF_bound', 'BRF_color',
    'BRF_gate_lf_mean', 'BRF_gate_lf_std', 'BRF_gate_lf_min', 'BRF_gate_lf_max',
    'BRF_gate_color_mean', 'BRF_gate_color_std', 'BRF_gate_color_min', 'BRF_gate_color_max',
    'BRF_gate_hf_mean', 'BRF_gate_hf_std', 'BRF_gate_hf_min', 'BRF_gate_hf_max',
    'BRF_c_lf_norm', 'BRF_c_color_norm', 'BRF_c_hf_norm',
    'BRF_target_lf_norm', 'BRF_residual_norm_ratio', 'BRF_train_residual_cosine',
)


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def create_summary_writer(start_step=0):
    if opt.no_tensorboard:
        return None
    if SummaryWriter is None:
        print('TensorBoard is not installed; continuing without TensorBoard logging.')
        return None
    purge_step = start_step if start_step > 0 else None
    writer = SummaryWriter(log_dir=opt.tensorboard_log_dir, purge_step=purge_step)
    print('TensorBoard log_dir:', opt.tensorboard_log_dir)
    return writer


def save_checkpoint(path, checkpoint):
    tmp_path = path + '.tmp'
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def resolve_resume_checkpoint_path():
    if not opt.resume:
        return None

    checkpoint_name = opt.pre_trained_model
    if checkpoint_name == 'null':
        candidates = [
            os.path.join(opt.saved_model_dir, 'latest.pk'),
            os.path.join(opt.saved_model_dir, 'best.pk')
        ]
    elif os.path.isabs(checkpoint_name):
        candidates = [checkpoint_name]
    else:
        candidates = [
            checkpoint_name,
            os.path.join(opt.saved_model_dir, checkpoint_name),
            os.path.join(opt.model_dir, checkpoint_name)
        ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError('No resume checkpoint found. Tried: {}'.format(', '.join(candidates)))


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint_file(path):
    try:
        return torch.load(path, map_location=opt.device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=opt.device)


def load_training_state(net, optimizer):
    checkpoint_path = resolve_resume_checkpoint_path()
    if checkpoint_path is None:
        return {}

    print('Resuming training from:', checkpoint_path)
    checkpoint = load_checkpoint_file(checkpoint_path)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    move_optimizer_state_to_device(optimizer, opt.device)

    start_step = int(checkpoint.get('step', 0))
    if start_step >= steps:
        raise ValueError(
            'Checkpoint step {} is already >= target steps {}. '
            'Increase --epochs/--iters_per_epoch or choose a new run.'.format(start_step, steps)
        )

    print('Resume step: {} / {}'.format(start_step, steps))
    return checkpoint


def resolve_checkpoint_path(checkpoint_name, base_dir=None):
    if checkpoint_name == 'null':
        return None
    if os.path.isabs(checkpoint_name):
        candidates = [checkpoint_name]
    else:
        candidates = [checkpoint_name]
        if base_dir is not None:
            candidates.append(os.path.join(base_dir, checkpoint_name))
        candidates.append(os.path.join(opt.model_dir, checkpoint_name))
        candidates.append(os.path.join(opt.saved_model_dir, checkpoint_name))

    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError('No checkpoint found. Tried: {}'.format(', '.join(candidates)))


def strip_module_prefix(state_dict):
    if not any(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {key.replace('module.', '', 1): value for key, value in state_dict.items()}


def create_teacher_model():
    if opt.w_loss_teacher_guard <= 0:
        return None
    checkpoint_path = resolve_checkpoint_path(opt.teacher_checkpoint)
    if checkpoint_path is None:
        raise ValueError('--teacher_checkpoint is required when --w_loss_teacher_guard > 0')

    teacher = DEANet(
        base_dim=32,
        use_lf_prior=opt.teacher_use_lf_prior,
        lf_prior_channels=opt.lf_prior_channels,
        lf_prior_pool=opt.lf_prior_pool,
        lf_prior_gate_init=opt.lf_prior_gate_init,
        lf_prior_residual_center=opt.lf_prior_residual_center,
        lf_prior_train_dropout=0.0,
        lf_prior_gate_max=opt.lf_prior_gate_max,
        lf_prior_injection=opt.lf_prior_injection,
        lf_conditional_mask=opt.lf_conditional_mask if opt.teacher_use_lf_prior else False,
        lf_mask_hidden_channels=opt.lf_mask_hidden_channels,
        lf_mask_init_bias=opt.lf_mask_init_bias,
        lf_haze_aware_mask=opt.lf_haze_aware_mask if opt.teacher_use_lf_prior else False,
        lf_haze_mask_strength=opt.lf_haze_mask_strength,
        lf_residual_calibration=opt.lf_residual_calibration if opt.teacher_use_lf_prior else False,
        lf_calib_hidden_channels=opt.lf_calib_hidden_channels,
        lf_calib_alpha_max=opt.lf_calib_alpha_max,
        lf_residual_selector=opt.lf_residual_selector if opt.teacher_use_lf_prior else False,
        lf_selector_hidden_channels=opt.lf_selector_hidden_channels,
        lf_selector_init_bias=opt.lf_selector_init_bias,
        lf_multiscale_refiner=opt.lf_multiscale_refiner if opt.teacher_use_lf_prior else False,
        lf_mbr_channels=opt.lf_mbr_channels,
        lf_mbr_pool_sizes=opt.lf_mbr_pool_sizes
    )
    checkpoint = load_checkpoint_file(checkpoint_path)
    teacher.load_state_dict(strip_module_prefix(checkpoint['model']))
    teacher.to(opt.device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print('Using teacher guard checkpoint:', checkpoint_path)
    print(
        'Teacher guard: weight={} margin={} warmup_steps={} max_weight={} patch_pool={}'.format(
            opt.w_loss_teacher_guard,
            opt.teacher_guard_margin,
            opt.teacher_guard_warmup_steps,
            opt.teacher_guard_max_weight,
            opt.teacher_guard_patch_pool
        )
    )
    return teacher


def create_cr_ref_model():
    if opt.w_loss_cr_ref_residual <= 0:
        return None
    checkpoint_path = resolve_checkpoint_path(opt.cr_ref_checkpoint)
    if checkpoint_path is None:
        raise ValueError('--cr_ref_checkpoint is required when --w_loss_cr_ref_residual > 0')

    teacher = DEANet(base_dim=32)
    checkpoint = load_checkpoint_file(checkpoint_path)
    teacher.load_state_dict(strip_module_prefix(checkpoint['model']))
    teacher.to(opt.device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(
        'Using CR reference residual-field checkpoint: {} | weight={} pool={} warmup={} magnitude_weight={}'.format(
            checkpoint_path,
            opt.w_loss_cr_ref_residual,
            opt.cr_ref_residual_pool,
            opt.cr_ref_residual_warmup_steps,
            opt.cr_ref_residual_magnitude_weight
        )
    )
    return teacher


def should_evaluate(step, loader_train_len):
    if opt.eval_interval_steps > 0:
        return step % opt.eval_interval_steps == 0 or step == steps
    return (
        (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step)
        or (step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * loader_train_len) == 0)
    )


def estimate_epoch(step, loader_train_len):
    if opt.eval_interval_steps > 0:
        return int(math.ceil(step / opt.iters_per_epoch))
    if step > opt.finer_eval_step:
        return opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (5 * loader_train_len)
    return int(step / opt.iters_per_epoch)


def create_early_stop_state(training_state, max_psnr, max_ssim):
    state = dict(training_state.get('early_stop_state', {}))
    if state:
        return state
    if opt.early_stop_metric == 'ssim':
        best_score = max_ssim
    else:
        best_score = max_psnr
    return {
        'best_score': best_score,
        'bad_evals': 0,
        'best_step': int(training_state.get('step', 0)),
        'stopped': False,
        'stop_step': 0,
        'stop_reason': ''
    }


def update_early_stop_state(state, step, psnr_eval, ssim_eval):
    if opt.early_stop_patience_evals <= 0:
        return False
    if step < opt.early_stop_after_step:
        return False

    score = ssim_eval if opt.early_stop_metric == 'ssim' else psnr_eval
    best_score = state.get('best_score', 0)
    if score > best_score + opt.early_stop_min_delta:
        state['best_score'] = score
        state['bad_evals'] = 0
        state['best_step'] = step
        return False

    state['bad_evals'] = int(state.get('bad_evals', 0)) + 1
    if state['bad_evals'] >= opt.early_stop_patience_evals:
        state['stopped'] = True
        state['stop_step'] = step
        state['stop_reason'] = (
            '{} did not improve by more than {} for {} evaluations after step {}.'
            .format(opt.early_stop_metric, opt.early_stop_min_delta, opt.early_stop_patience_evals, opt.early_stop_after_step)
        )
        return True
    return False


def low_frequency_loss(out, target):
    if opt.w_loss_lowfreq <= 0:
        return None
    if opt.lowfreq_pool <= 0:
        raise ValueError('lowfreq_pool must be positive')
    low_out = F.avg_pool2d(
        out,
        kernel_size=opt.lowfreq_pool,
        stride=opt.lowfreq_pool,
        ceil_mode=True
    )
    low_target = F.avg_pool2d(
        target,
        kernel_size=opt.lowfreq_pool,
        stride=opt.lowfreq_pool,
        ceil_mode=True
    )
    return F.l1_loss(low_out, low_target)


def residual_direction_loss(out, hazy, target, step):
    if opt.w_loss_residual_dir <= 0:
        return None
    if step < opt.residual_dir_warmup_steps:
        return None
    if opt.residual_dir_pool <= 0:
        raise ValueError('residual_dir_pool must be positive')
    low_out = F.avg_pool2d(
        out,
        kernel_size=opt.residual_dir_pool,
        stride=opt.residual_dir_pool,
        ceil_mode=True
    )
    low_hazy = F.avg_pool2d(
        hazy,
        kernel_size=opt.residual_dir_pool,
        stride=opt.residual_dir_pool,
        ceil_mode=True
    )
    low_target = F.avg_pool2d(
        target,
        kernel_size=opt.residual_dir_pool,
        stride=opt.residual_dir_pool,
        ceil_mode=True
    )
    pred_residual = low_out - low_hazy
    target_residual = low_target - low_hazy
    pred_vec = pred_residual.reshape(pred_residual.shape[0], -1)
    target_vec = target_residual.reshape(target_residual.shape[0], -1)
    pred_norm = pred_vec.norm(dim=1)
    target_norm = target_vec.norm(dim=1)
    valid = target_norm > opt.residual_dir_target_norm_floor
    if not valid.any():
        return out.new_zeros(())
    cosine = (pred_vec[valid] * target_vec[valid]).sum(dim=1)
    cosine = cosine / (pred_norm[valid] * target_norm[valid] + 1e-8)
    return (1.0 - cosine).mean()


def cr_ref_residual_field_loss(out, hazy, target, step, cr_ref_net):
    if cr_ref_net is None or opt.w_loss_cr_ref_residual <= 0:
        return None
    if step < opt.cr_ref_residual_warmup_steps:
        return None
    if opt.cr_ref_residual_pool <= 0:
        raise ValueError('cr_ref_residual_pool must be positive')
    with torch.no_grad():
        ref = cr_ref_net(hazy).clamp(0, 1)
    low_out = F.avg_pool2d(
        out,
        kernel_size=opt.cr_ref_residual_pool,
        stride=opt.cr_ref_residual_pool,
        ceil_mode=True
    )
    low_ref = F.avg_pool2d(
        ref,
        kernel_size=opt.cr_ref_residual_pool,
        stride=opt.cr_ref_residual_pool,
        ceil_mode=True
    )
    low_target = F.avg_pool2d(
        target,
        kernel_size=opt.cr_ref_residual_pool,
        stride=opt.cr_ref_residual_pool,
        ceil_mode=True
    )
    pred_residual = low_out - low_ref.detach()
    target_residual = low_target - low_ref.detach()
    pred_vec = pred_residual.reshape(pred_residual.shape[0], -1)
    target_vec = target_residual.reshape(target_residual.shape[0], -1)
    pred_norm = pred_vec.norm(dim=1)
    target_norm = target_vec.norm(dim=1)
    valid = target_norm > opt.cr_ref_residual_target_norm_floor
    if not valid.any():
        return out.new_zeros(())
    cosine = (pred_vec[valid] * target_vec[valid]).sum(dim=1)
    cosine = cosine / (pred_norm[valid] * target_norm[valid] + 1e-8)
    loss = (1.0 - cosine).mean()
    if opt.cr_ref_residual_magnitude_weight > 0:
        ratio = pred_norm[valid] / (target_norm[valid] + 1e-8)
        if opt.cr_ref_residual_magnitude_cap > 0:
            ratio = ratio.clamp(max=opt.cr_ref_residual_magnitude_cap)
        magnitude_target = torch.ones_like(ratio)
        loss = loss + opt.cr_ref_residual_magnitude_weight * F.smooth_l1_loss(ratio, magnitude_target)
    return loss


def brf_lowpass(x):
    if opt.brf_lf_pool <= 0:
        raise ValueError('brf_lf_pool must be positive')
    low = F.avg_pool2d(
        x,
        kernel_size=opt.brf_lf_pool,
        stride=opt.brf_lf_pool,
        ceil_mode=True
    )
    return F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)


def mean_l2_norm(x):
    return x.reshape(x.shape[0], -1).norm(dim=1).mean()


def brf_loss_terms(out_dict, target, step):
    if out_dict is None:
        return None, {}
    out = out_dict['out']
    j0 = out_dict['j0'].detach()
    pred_lf = brf_lowpass(out) - brf_lowpass(j0)
    target_lf = out_dict.get('target_lf')
    if target_lf is None:
        target_lf = brf_lowpass(target) - brf_lowpass(j0)
    target_lf = target_lf.detach()

    loss_res_lf = F.l1_loss(pred_lf, target_lf)
    pred_vec = pred_lf.reshape(pred_lf.shape[0], -1)
    target_vec = target_lf.reshape(target_lf.shape[0], -1)
    pred_norm = pred_vec.norm(dim=1)
    target_norm = target_vec.norm(dim=1)
    cosine = (pred_vec * target_vec).sum(dim=1) / (pred_norm * target_norm + 1e-8)
    valid = target_norm > opt.brf_dir_norm_floor
    if valid.any():
        loss_dir = (1.0 - cosine[valid]).mean()
        train_cosine = cosine[valid].mean()
        residual_norm_ratio = (pred_norm[valid] / (target_norm[valid] + 1e-8)).mean()
    else:
        loss_dir = out.new_zeros(())
        train_cosine = out.new_zeros(())
        residual_norm_ratio = out.new_zeros(())

    preserve_mask = (
        target_lf.abs().mean(dim=(1, 2, 3), keepdim=True) < opt.brf_preserve_target_thr
    ).to(out.dtype)
    if opt.brf_preserve_warmup_steps > 0 and step < opt.brf_preserve_warmup_steps:
        loss_preserve = out.new_zeros(())
    else:
        loss_preserve = torch.mean(
            preserve_mask * (out_dict['c_lf'].abs() + opt.brf_preserve_gate_weight * out_dict['gate_lf'])
        )
    total_correction = out_dict['c_lf'] + out_dict['c_color'] + out_dict['c_hf']
    loss_bound = total_correction.abs().mean()
    loss_color = F.l1_loss(out.mean(dim=(2, 3)), target.mean(dim=(2, 3)))

    losses = {
        'res_lf': loss_res_lf,
        'dir': loss_dir,
        'preserve': loss_preserve,
        'bound': loss_bound,
        'color': loss_color,
    }
    metrics = {
        'BRF_res_lf': loss_res_lf,
        'BRF_dir': loss_dir,
        'BRF_preserve': loss_preserve,
        'BRF_bound': loss_bound,
        'BRF_color': loss_color,
        'BRF_target_lf_norm': mean_l2_norm(target_lf),
        'BRF_residual_norm_ratio': residual_norm_ratio,
        'BRF_train_residual_cosine': train_cosine,
    }
    return losses, metrics


def brf_stats(out_dict, brf_metrics):
    if out_dict is None:
        return None
    stats = {}
    for name in ('lf', 'color', 'hf'):
        gate = out_dict['gate_' + name].detach()
        stats['BRF_gate_' + name + '_mean'] = gate.mean()
        stats['BRF_gate_' + name + '_std'] = gate.std(unbiased=False)
        stats['BRF_gate_' + name + '_min'] = gate.min()
        stats['BRF_gate_' + name + '_max'] = gate.max()
    stats['BRF_c_lf_norm'] = mean_l2_norm(out_dict['c_lf'].detach())
    stats['BRF_c_color_norm'] = mean_l2_norm(out_dict['c_color'].detach())
    stats['BRF_c_hf_norm'] = mean_l2_norm(out_dict['c_hf'].detach())
    stats.update(brf_metrics)
    return {
        key: float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
        for key, value in stats.items()
    }


def per_sample_l1(a, b, pool_size=0):
    if pool_size < 0:
        raise ValueError('teacher_guard_patch_pool must be non-negative')
    diff = torch.abs(a - b)
    if pool_size > 0:
        diff = F.avg_pool2d(
            diff,
            kernel_size=pool_size,
            stride=pool_size,
            ceil_mode=True
        )
    return diff.mean(dim=(1, 2, 3))


def teacher_guard_loss(out, hazy, target, step, teacher_net):
    if teacher_net is None or opt.w_loss_teacher_guard <= 0:
        return None
    if opt.teacher_guard_warmup_steps > 0 and step < opt.teacher_guard_warmup_steps:
        return None
    with torch.no_grad():
        teacher_out = teacher_net(hazy).clamp(0, 1)
        if opt.teacher_guard_patch_pool > 0:
            teacher_err = torch.abs(teacher_out - target).mean(dim=1, keepdim=True)
            current_err = torch.abs(out - target).mean(dim=1, keepdim=True)
            teacher_err = F.avg_pool2d(
                teacher_err,
                kernel_size=opt.teacher_guard_patch_pool,
                stride=opt.teacher_guard_patch_pool,
                ceil_mode=True
            )
            current_err = F.avg_pool2d(
                current_err,
                kernel_size=opt.teacher_guard_patch_pool,
                stride=opt.teacher_guard_patch_pool,
                ceil_mode=True
            )
            weight = (current_err - teacher_err - opt.teacher_guard_margin).clamp(min=0)
            weight = F.interpolate(weight, size=out.shape[-2:], mode='nearest')
        else:
            teacher_l1 = per_sample_l1(teacher_out, target)
            current_l1 = per_sample_l1(out, target)
            weight = (current_l1 - teacher_l1 - opt.teacher_guard_margin).clamp(min=0)
            weight = weight.view(-1, 1, 1, 1)
        weight = weight / (weight.detach().mean() + 1e-6)
        if opt.teacher_guard_max_weight > 0:
            weight = weight.clamp(max=opt.teacher_guard_max_weight)
    return torch.mean(weight * torch.abs(out - teacher_out.detach()))


def build_checkpoint(epoch, step, max_psnr, max_ssim, ssims, psnrs, losses, loss_log, psnr_log, net, optimizer, early_stop_state=None):
    return {
        'epoch': epoch,
        'step': step,
        'max_psnr': max_psnr,
        'max_ssim': max_ssim,
        'ssims': ssims,
        'psnrs': psnrs,
        'losses': losses,
        'loss_log': loss_log,
        'psnr_log': psnr_log,
        'early_stop_state': early_stop_state or {},
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }


def log_trainable_stage(step, stats):
    message = (
        'Trainable stage at step {}: {} | params {}/{} | tensors {}/{}'.format(
            step,
            stats['stage_label'],
            stats['trainable_params'],
            stats['total_params'],
            stats['trainable_tensors'],
            stats['total_tensors']
        )
    )
    print(message)
    if opt.dry_run:
        return
    os.makedirs(opt.saved_data_dir, exist_ok=True)
    payload = dict(stats)
    payload['step'] = step
    with open(os.path.join(opt.saved_data_dir, 'trainable_schedule.jsonl'), 'a') as f:
        f.write(json.dumps(payload) + '\n')


def train(net, loader_train, loader_test, optim, criterion, writer=None, training_state=None, teacher_net=None, cr_ref_net=None, trainable_schedule=None):
    training_state = training_state or {}
    losses = list(training_state.get('losses', []))

    loss_log = {'L1': [], 'CR': [], 'total': []}
    loss_log_tmp = {'L1': [], 'CR': [], 'total': []}
    if 'loss_log' in training_state:
        loss_log = training_state['loss_log']
    for key in (
        'CRPlusV2', 'CRPlusV2_weight', 'LF_gate', 'LowFreq', 'ResidualDir', 'CRRefResidual', 'TeacherGuard',
        'LF_mask_mean', 'LF_mask_std', 'LF_mask_min', 'LF_mask_max',
        'LF_alpha_mean', 'LF_alpha_std', 'LF_alpha_min', 'LF_alpha_max',
        'LF_selector_mean', 'LF_selector_std', 'LF_selector_min', 'LF_selector_max',
        'LF_mbr_mean', 'LF_mbr_std', 'LF_mbr_min', 'LF_mbr_max',
        'TrainableParamCount', 'TrainableStage'
    ) + BRF_LOG_KEYS:
        loss_log.setdefault(key, [])
        loss_log_tmp.setdefault(key, [])
    psnr_log = list(training_state.get('psnr_log', training_state.get('psnrs', [])))

    start_step = int(training_state.get('step', 0))
    max_ssim = training_state.get('max_ssim', 0)
    max_psnr = training_state.get('max_psnr', 0)
    ssims = list(training_state.get('ssims', []))
    psnrs = list(training_state.get('psnrs', []))
    early_stop_state = create_early_stop_state(training_state, max_psnr, max_ssim)

    loader_train_iter = iter(loader_train)
    progress_bar = tqdm(
        range(start_step + 1, steps + 1),
        total=steps,
        initial=start_step,
        desc='training',
        dynamic_ncols=True,
        mininterval=1.0,
        disable=opt.no_tqdm
    )

    try:
        for step in progress_bar:
            net.train()
            trainable_stats = None
            if trainable_schedule is not None:
                trainable_stats = trainable_schedule.apply(net, step)
                if trainable_stats.get('changed'):
                    log_trainable_stage(step, trainable_stats)
            lr = opt.start_lr
            if not opt.no_lr_sche:
                lr = lr_schedule_cosdecay(step, T)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr

            x, y = next(loader_train_iter)
            x = x.to(opt.device)
            y = y.to(opt.device)

            out_dict = None
            if opt.use_brf_frequency_corrector:
                out_dict = net(x, target=y, return_aux=True)
                out = out_dict['out']
            else:
                out = net(x)
            if opt.w_loss_L1 > 0:
                loss_L1 = criterion[0](out, y)
            if opt.w_loss_CR > 0:
                loss_CR = criterion[1](out, y, x)
            loss_lf_gate = lf_gate_regularization(net)
            loss_crplus_v2_weight = crplus_v2_weight(step)
            loss_crplus_v2 = crplus_v2_loss(out, y, x, step, criterion[2])
            loss_lowfreq = low_frequency_loss(out, y)
            loss_residual_dir = residual_direction_loss(out, x, y, step)
            loss_cr_ref_residual = cr_ref_residual_field_loss(out, x, y, step, cr_ref_net)
            loss_teacher_guard = teacher_guard_loss(out, x, y, step, teacher_net)
            loss_brf, brf_metric_tensors = brf_loss_terms(out_dict, y, step)
            loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_CR * loss_CR
            if loss_crplus_v2 is not None:
                loss = loss + loss_crplus_v2_weight * loss_crplus_v2
            if loss_lf_gate is not None:
                loss = loss + opt.w_loss_lf_gate * loss_lf_gate
            if loss_lowfreq is not None:
                loss = loss + opt.w_loss_lowfreq * loss_lowfreq
            if loss_residual_dir is not None:
                loss = loss + opt.w_loss_residual_dir * loss_residual_dir
            if loss_cr_ref_residual is not None:
                loss = loss + opt.w_loss_cr_ref_residual * loss_cr_ref_residual
            if loss_teacher_guard is not None:
                loss = loss + opt.w_loss_teacher_guard * loss_teacher_guard
            if loss_brf is not None:
                loss = loss + opt.w_loss_brf_res_lf * loss_brf['res_lf']
                loss = loss + opt.w_loss_brf_dir * loss_brf['dir']
                loss = loss + opt.w_loss_brf_preserve * loss_brf['preserve']
                loss = loss + opt.w_loss_brf_bound * loss_brf['bound']
                loss = loss + opt.w_loss_brf_color * loss_brf['color']
            loss.backward()
            optim.step()
            optim.zero_grad()
            lf_mask_stats = lf_prior_mask_stats(net)
            lf_alpha_stats = lf_prior_alpha_stats(net)
            lf_selector_stats = lf_prior_selector_stats(net)
            lf_mbr_stats = lf_prior_multiscale_stats(net)
            brf_metric_values = brf_stats(out_dict, brf_metric_tensors)
            losses.append(loss.item())
            loss_log_tmp['L1'].append(loss_L1.item())
            loss_log_tmp['CR'].append(loss_CR.item())
            loss_log_tmp['total'].append(loss.item())
            if loss_crplus_v2 is not None:
                loss_log_tmp['CRPlusV2'].append(loss_crplus_v2.item())
                loss_log_tmp['CRPlusV2_weight'].append(loss_crplus_v2_weight)
            if loss_lf_gate is not None:
                loss_log_tmp['LF_gate'].append(loss_lf_gate.item())
            if loss_lowfreq is not None:
                loss_log_tmp['LowFreq'].append(loss_lowfreq.item())
            if loss_residual_dir is not None:
                loss_log_tmp['ResidualDir'].append(loss_residual_dir.item())
            if loss_cr_ref_residual is not None:
                loss_log_tmp['CRRefResidual'].append(loss_cr_ref_residual.item())
            if loss_teacher_guard is not None:
                loss_log_tmp['TeacherGuard'].append(loss_teacher_guard.item())
            if brf_metric_values is not None:
                for key, value in brf_metric_values.items():
                    loss_log_tmp[key].append(value)
            if lf_mask_stats is not None:
                for key, value in lf_mask_stats.items():
                    loss_log_tmp['LF_mask_' + key].append(value)
            if lf_alpha_stats is not None:
                for key, value in lf_alpha_stats.items():
                    loss_log_tmp['LF_alpha_' + key].append(value)
            if lf_selector_stats is not None:
                for key, value in lf_selector_stats.items():
                    loss_log_tmp['LF_selector_' + key].append(value)
            if lf_mbr_stats is not None:
                for key, value in lf_mbr_stats.items():
                    loss_log_tmp['LF_mbr_' + key].append(value)
            if trainable_stats is not None:
                loss_log_tmp['TrainableParamCount'].append(trainable_stats['trainable_params'])
                loss_log_tmp['TrainableStage'].append(trainable_stats['stage_index'])

            if writer is not None and opt.tb_log_interval > 0 and (step == 1 or step % opt.tb_log_interval == 0):
                writer.add_scalar('train/loss_total', loss.item(), step)
                writer.add_scalar('train/loss_L1', loss_L1.item(), step)
                writer.add_scalar('train/loss_CR', loss_CR.item(), step)
                writer.add_scalar('train/loss_CR_weighted', opt.w_loss_CR * loss_CR.item(), step)
                if loss_crplus_v2 is not None:
                    writer.add_scalar('train/loss_crplus_v2', loss_crplus_v2.item(), step)
                    writer.add_scalar('train/loss_crplus_v2_weight', loss_crplus_v2_weight, step)
                    writer.add_scalar('train/loss_crplus_v2_weighted', loss_crplus_v2_weight * loss_crplus_v2.item(), step)
                if loss_lf_gate is not None:
                    writer.add_scalar('train/loss_lf_gate', loss_lf_gate.item(), step)
                    writer.add_scalar('train/loss_lf_gate_weighted', opt.w_loss_lf_gate * loss_lf_gate.item(), step)
                if loss_lowfreq is not None:
                    writer.add_scalar('train/loss_lowfreq', loss_lowfreq.item(), step)
                    writer.add_scalar('train/loss_lowfreq_weighted', opt.w_loss_lowfreq * loss_lowfreq.item(), step)
                if loss_residual_dir is not None:
                    writer.add_scalar('train/loss_residual_dir', loss_residual_dir.item(), step)
                    writer.add_scalar('train/loss_residual_dir_weighted', opt.w_loss_residual_dir * loss_residual_dir.item(), step)
                if loss_cr_ref_residual is not None:
                    writer.add_scalar('train/loss_cr_ref_residual', loss_cr_ref_residual.item(), step)
                    writer.add_scalar('train/loss_cr_ref_residual_weighted', opt.w_loss_cr_ref_residual * loss_cr_ref_residual.item(), step)
                if loss_teacher_guard is not None:
                    writer.add_scalar('train/loss_teacher_guard', loss_teacher_guard.item(), step)
                    writer.add_scalar('train/loss_teacher_guard_weighted', opt.w_loss_teacher_guard * loss_teacher_guard.item(), step)
                if brf_metric_values is not None:
                    for key, value in brf_metric_values.items():
                        writer.add_scalar('train/' + key, value, step)
                    writer.add_scalar('train/loss_brf_res_lf_weighted', opt.w_loss_brf_res_lf * brf_metric_values['BRF_res_lf'], step)
                    writer.add_scalar('train/loss_brf_dir_weighted', opt.w_loss_brf_dir * brf_metric_values['BRF_dir'], step)
                    writer.add_scalar('train/loss_brf_preserve_weighted', opt.w_loss_brf_preserve * brf_metric_values['BRF_preserve'], step)
                    writer.add_scalar('train/loss_brf_bound_weighted', opt.w_loss_brf_bound * brf_metric_values['BRF_bound'], step)
                    writer.add_scalar('train/loss_brf_color_weighted', opt.w_loss_brf_color * brf_metric_values['BRF_color'], step)
                if lf_mask_stats is not None:
                    for key, value in lf_mask_stats.items():
                        writer.add_scalar('train/lf_mask_' + key, value, step)
                if lf_alpha_stats is not None:
                    for key, value in lf_alpha_stats.items():
                        writer.add_scalar('train/lf_alpha_' + key, value, step)
                if lf_selector_stats is not None:
                    for key, value in lf_selector_stats.items():
                        writer.add_scalar('train/lf_selector_' + key, value, step)
                if lf_mbr_stats is not None:
                    for key, value in lf_mbr_stats.items():
                        writer.add_scalar('train/lf_mbr_' + key, value, step)
                if trainable_stats is not None:
                    writer.add_scalar('train/trainable_param_count', trainable_stats['trainable_params'], step)
                    writer.add_scalar('train/trainable_stage', trainable_stats['stage_index'], step)
                writer.add_scalar('train/lr', lr, step)

            if opt.no_tqdm:
                print(
                    f'\rloss:{loss.item():.5f} | L1:{loss_L1.item():.5f} | CR:{opt.w_loss_CR * loss_CR.item():.5f} | step :{step}/{steps} | lr :{lr :.7f} | time_used :{(time.time() - start_time) / 60 :.1f}',
                    end='', flush=True)
            else:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.5f}',
                    'L1': f'{loss_L1.item():.5f}',
                    'CR': f'{opt.w_loss_CR * loss_CR.item():.5f}',
                    'lr': f'{lr:.7f}',
                    'min': f'{(time.time() - start_time) / 60:.1f}'
                }, refresh=False)

            if step % len(loader_train) == 0:
                loader_train_iter = iter(loader_train)
                for key in loss_log.keys():
                    if loss_log_tmp[key]:
                        loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                    loss_log_tmp[key] = []
                if not opt.no_pdf_plots:
                    plot_loss_log(loss_log, int(step / len(loader_train)), opt.saved_plot_dir)
                np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
            if should_evaluate(step, len(loader_train)):
                epoch = estimate_epoch(step, len(loader_train))
                with torch.no_grad():
                    ssim_eval, psnr_eval = test(net, loader_test)

                log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'
                if opt.no_tqdm:
                    print(log)
                else:
                    progress_bar.write(log)
                with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                    f.write(log + '\n')

                if writer is not None:
                    writer.add_scalar('eval/ssim', ssim_eval, step)
                    writer.add_scalar('eval/psnr', psnr_eval, step)

                ssims.append(ssim_eval)
                psnrs.append(psnr_eval)
                psnr_log.append(psnr_eval)
                if not opt.no_pdf_plots:
                    plot_psnr_log(psnr_log, epoch, opt.saved_plot_dir)

                improved = psnr_eval > max_psnr
                if improved:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                    save_log = f'\n model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}'
                    if opt.no_tqdm:
                        print(save_log)
                    else:
                        progress_bar.write(save_log)
                if writer is not None:
                    writer.add_scalar('eval/max_psnr', max_psnr, step)
                    writer.add_scalar('eval/max_ssim', max_ssim, step)

                should_stop = update_early_stop_state(early_stop_state, step, psnr_eval, ssim_eval)
                if writer is not None and opt.early_stop_patience_evals > 0:
                    writer.add_scalar('early_stop/bad_evals', early_stop_state.get('bad_evals', 0), step)
                    writer.add_scalar('early_stop/best_score', early_stop_state.get('best_score', 0), step)

                checkpoint = build_checkpoint(
                    epoch, step, max_psnr, max_ssim, ssims, psnrs, losses,
                    loss_log, psnr_log, net, optim, early_stop_state
                )
                if improved:
                    save_checkpoint(os.path.join(opt.saved_model_dir, 'best.pk'), checkpoint)
                if opt.save_epoch_checkpoints:
                    saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch) + '.pk')
                    save_checkpoint(saved_single_model_path, checkpoint)
                save_checkpoint(os.path.join(opt.saved_model_dir, 'latest.pk'), checkpoint)
                loader_train_iter = iter(loader_train)
                np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
                np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)
                if should_stop:
                    stop_log = '\nEarly stopping at step {}: {}'.format(step, early_stop_state['stop_reason'])
                    if opt.no_tqdm:
                        print(stop_log)
                    else:
                        progress_bar.write(stop_log)
                    with open(os.path.join(opt.saved_data_dir, 'early_stop.txt'), 'w') as f:
                        f.write(stop_log.strip() + '\n')
                    with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                        f.write(stop_log.strip() + '\n')
                    break
            elif opt.checkpoint_interval_steps > 0 and step % opt.checkpoint_interval_steps == 0:
                checkpoint = build_checkpoint(
                    estimate_epoch(step, len(loader_train)), step, max_psnr, max_ssim,
                    ssims, psnrs, losses, loss_log, psnr_log, net, optim, early_stop_state
                )
                save_checkpoint(os.path.join(opt.saved_model_dir, 'latest.pk'), checkpoint)
                np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
    finally:
        progress_bar.close()


def resolve_lf_prior_module(net):
    module = net.module if hasattr(net, 'module') else net
    return getattr(module, 'lf_prior', None)


def lf_gate_regularization(net):
    lf_prior = resolve_lf_prior_module(net)
    if lf_prior is None or opt.w_loss_lf_gate <= 0:
        return None
    return lf_prior.gate.pow(2)


def crplus_v2_loss(out, target, hazy, step, criterion):
    if crplus_v2_weight(step) <= 0 or criterion is None:
        return None
    return criterion(out, target, hazy, step)


def crplus_v2_weight(step):
    if opt.w_loss_crplus_v2 <= 0:
        return 0.0
    if opt.crplus_v2_weight_schedule == 'constant':
        return opt.w_loss_crplus_v2
    if opt.crplus_v2_weight_schedule == 'linear_decay':
        start = opt.crplus_v2_weight_decay_start_step
        end = opt.crplus_v2_weight_decay_end_step
        min_weight = opt.crplus_v2_min_weight
        if start < 0 or end < 0:
            raise ValueError('CRPlus-v2 weight decay steps must be non-negative')
        if min_weight < 0:
            raise ValueError('crplus_v2_min_weight must be non-negative')
        if end <= start:
            return min_weight if step >= start else opt.w_loss_crplus_v2
        if step <= start:
            return opt.w_loss_crplus_v2
        if step >= end:
            return min_weight
        ratio = float(step - start) / float(end - start)
        return opt.w_loss_crplus_v2 + ratio * (min_weight - opt.w_loss_crplus_v2)
    raise ValueError('Unsupported CRPlus-v2 weight schedule: {}'.format(opt.crplus_v2_weight_schedule))


def lf_prior_mask_stats(net):
    lf_prior = resolve_lf_prior_module(net)
    if lf_prior is None:
        return None
    stats = getattr(lf_prior, 'last_mask_stats', None)
    if stats is None:
        return None
    return {key: float(value.detach().cpu().item()) for key, value in stats.items()}


def lf_prior_alpha_stats(net):
    lf_prior = resolve_lf_prior_module(net)
    if lf_prior is None:
        return None
    stats = getattr(lf_prior, 'last_alpha_stats', None)
    if stats is None:
        return None
    return {key: float(value.detach().cpu().item()) for key, value in stats.items()}


def lf_prior_selector_stats(net):
    lf_prior = resolve_lf_prior_module(net)
    if lf_prior is None:
        return None
    stats = getattr(lf_prior, 'last_selector_stats', None)
    if stats is None:
        return None
    return {key: float(value.detach().cpu().item()) for key, value in stats.items()}


def lf_prior_multiscale_stats(net):
    lf_prior = resolve_lf_prior_module(net)
    if lf_prior is None:
        return None
    stats = getattr(lf_prior, 'last_multiscale_stats', None)
    if stats is None:
        return None
    return {key: float(value.detach().cpu().item()) for key, value in stats.items()}


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        if opt.max_test_batches > 0 and i >= opt.max_test_batches:
            break
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            pred = net(inputs).clamp(0, 1)
            pred = pred[:, :, :H, :W]
            # save_path = os.path.join(opt.saved_infer_dir, hazy_name[0])
            # save_image(pred, save_path)
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def resolve_dataset_root(dataset):
    dataset_root = os.path.join('../dataset', dataset)
    if os.path.isdir(dataset_root):
        return dataset_root

    reside_dataset_root = os.path.join('../dataset/RESIDE', dataset)
    if os.path.isdir(reside_dataset_root):
        return reside_dataset_root

    raise FileNotFoundError(f'No dataset directory found for {dataset}')


def create_data_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': opt.pin_memory
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = opt.persistent_workers
        kwargs['prefetch_factor'] = opt.prefetch_factor
    return DataLoader(**kwargs)


def limit_dataset_for_smoke(dataset, max_batches, batch_size):
    if max_batches <= 0:
        return dataset
    max_items = min(len(dataset), max_batches * batch_size)
    print('Limiting dataset to {} samples for smoke test.'.format(max_items))
    return torch.utils.data.Subset(dataset, range(max_items))


def create_deanet_from_options():
    return DEANet(
        base_dim=32,
        use_lf_prior=opt.use_lf_prior,
        lf_prior_channels=opt.lf_prior_channels,
        lf_prior_pool=opt.lf_prior_pool,
        lf_prior_gate_init=opt.lf_prior_gate_init,
        lf_prior_residual_center=opt.lf_prior_residual_center,
        lf_prior_train_dropout=opt.lf_prior_train_dropout,
        lf_prior_gate_max=opt.lf_prior_gate_max,
        lf_prior_injection=opt.lf_prior_injection,
        lf_conditional_mask=opt.lf_conditional_mask,
        lf_mask_hidden_channels=opt.lf_mask_hidden_channels,
        lf_mask_init_bias=opt.lf_mask_init_bias,
        lf_haze_aware_mask=opt.lf_haze_aware_mask,
        lf_haze_mask_strength=opt.lf_haze_mask_strength,
        lf_residual_calibration=opt.lf_residual_calibration,
        lf_calib_hidden_channels=opt.lf_calib_hidden_channels,
        lf_calib_alpha_max=opt.lf_calib_alpha_max,
        lf_residual_selector=opt.lf_residual_selector,
        lf_selector_hidden_channels=opt.lf_selector_hidden_channels,
        lf_selector_init_bias=opt.lf_selector_init_bias,
        lf_multiscale_refiner=opt.lf_multiscale_refiner,
        lf_mbr_channels=opt.lf_mbr_channels,
        lf_mbr_pool_sizes=opt.lf_mbr_pool_sizes
    )


def create_brf_model():
    checkpoint_path = resolve_checkpoint_path(opt.brf_baseline_checkpoint)
    if checkpoint_path is None and not opt.resume:
        raise ValueError('--brf_baseline_checkpoint is required when --use_brf_frequency_corrector is set')
    baseline = DEANet(base_dim=32)
    if checkpoint_path is not None:
        checkpoint = load_checkpoint_file(checkpoint_path)
        baseline.load_state_dict(strip_module_prefix(checkpoint['model']))
    else:
        print('No CBRFRC baseline checkpoint provided; expecting resume checkpoint to restore wrapper state.')
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
    net = DEANetCBRFRC(
        baseline=baseline,
        corrector=corrector,
        freeze_baseline=opt.brf_freeze_baseline,
        use_baseline_detach=opt.brf_use_baseline_detach,
    )
    print(
        'Using CBRFRC: baseline={} freeze_baseline={} detach={} hidden={} pyramid={} lf_pool={} mid_pool={} gate_init={} hf_gate_init={} max_residual={} max_color={} max_hf={} hf_scale={} haze_prior={} preserve_highfreq={}'.format(
            checkpoint_path,
            opt.brf_freeze_baseline,
            opt.brf_use_baseline_detach,
            opt.brf_hidden_channels,
            opt.brf_pyramid_type,
            opt.brf_lf_pool,
            opt.brf_mid_pool,
            opt.brf_gate_init,
            opt.brf_hf_gate_init,
            opt.brf_max_residual,
            opt.brf_max_color_residual,
            opt.brf_max_hf_residual,
            opt.brf_hf_scale,
            opt.brf_use_haze_prior,
            opt.brf_preserve_highfreq,
        )
    )
    return net


if __name__ == "__main__":

    set_seed_torch(666)

    dataset_root = resolve_dataset_root(opt.dataset)
    train_dir = os.path.join(dataset_root, 'train')
    test_dir = os.path.join(dataset_root, 'test')
    print('train_dir:', train_dir)
    print('test_dir:', test_dir)

    train_hazy_dir, train_clear_dir = resolve_pair_dirs(train_dir)
    test_hazy_dir, test_clear_dir = resolve_pair_dirs(test_dir)
    print('train_hazy_dir:', train_hazy_dir)
    print('train_clear_dir:', train_clear_dir)
    print('test_hazy_dir:', test_hazy_dir)
    print('test_clear_dir:', test_clear_dir)

    train_set = TrainDataset(train_hazy_dir, train_clear_dir, patch_size=opt.patch_size)
    test_set = TestDataset(test_hazy_dir, test_clear_dir)
    train_set = limit_dataset_for_smoke(train_set, opt.max_train_batches, opt.bs)
    loader_train = create_data_loader(train_set, opt.bs, True, opt.num_workers)
    loader_test = create_data_loader(test_set, 1, False, opt.test_num_workers)

    if opt.use_brf_frequency_corrector:
        net = create_brf_model()
    else:
        net = create_deanet_from_options()
    net = net.to(opt.device)
    if opt.use_lf_prior:
        print(
            'Using LF prior: channels={} pool={} gate_init={} residual_center={} train_dropout={} gate_max={} injection={} conditional_mask={} mask_hidden={} mask_init_bias={} haze_aware_mask={} haze_mask_strength={} residual_calibration={} calib_hidden={} calib_alpha_max={} residual_selector={} selector_hidden={} selector_init_bias={} multiscale_refiner={} mbr_channels={} mbr_pool_sizes={} gate_l2={}'.format(
                opt.lf_prior_channels,
                opt.lf_prior_pool,
                opt.lf_prior_gate_init,
                opt.lf_prior_residual_center,
                opt.lf_prior_train_dropout,
                opt.lf_prior_gate_max,
                opt.lf_prior_injection,
                opt.lf_conditional_mask,
                opt.lf_mask_hidden_channels,
                opt.lf_mask_init_bias,
                opt.lf_haze_aware_mask,
                opt.lf_haze_mask_strength,
                opt.lf_residual_calibration,
                opt.lf_calib_hidden_channels,
                opt.lf_calib_alpha_max,
                opt.lf_residual_selector,
                opt.lf_selector_hidden_channels,
                opt.lf_selector_init_bias,
                opt.lf_multiscale_refiner,
                opt.lf_mbr_channels,
                opt.lf_mbr_pool_sizes,
                opt.w_loss_lf_gate
            )
        )

    epoch_size = len(loader_train)
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        print('Using DataParallel with {} GPUs'.format(torch.cuda.device_count()))
    elif opt.device == 'cuda':
        print('Using single CUDA device:', torch.cuda.get_device_name(0))
    if opt.device == 'cuda':
        cudnn.benchmark = True

    trainable_schedule = TrainableSchedule(opt.trainable_schedule)
    if trainable_schedule.enabled:
        print('Using trainable schedule:', trainable_schedule.describe())

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(ContrastLoss(
        ablation=False,
        negative_mode=opt.cr_negative_mode,
        lowpass_pool=opt.cr_lowpass_pool,
        lowpass_weight=opt.cr_lowpass_weight
    ))
    if opt.w_loss_crplus_v2 > 0:
        criterion.append(CRPlusV2Loss(
            negative_modes=opt.crplus_v2_negative_modes,
            start_negative_modes=opt.crplus_v2_start_negative_modes,
            curriculum_steps=opt.crplus_v2_curriculum_steps,
            lowpass_pool=opt.crplus_v2_lowpass_pool,
            frequency_weight=opt.crplus_v2_frequency_weight,
            lowfreq_weight=opt.crplus_v2_lowfreq_weight,
            under_dehazed_mix=opt.crplus_v2_under_dehazed_mix,
            ratio_cap=opt.crplus_v2_ratio_cap
        ))
        print(
            'Using CRPlus-v2: weight={} schedule={} min_weight={} decay_start={} decay_end={} negatives={} start_negatives={} curriculum_steps={} pool={} freq_weight={} lowfreq_weight={} mix={} ratio_cap={}'.format(
                opt.w_loss_crplus_v2,
                opt.crplus_v2_weight_schedule,
                opt.crplus_v2_min_weight,
                opt.crplus_v2_weight_decay_start_step,
                opt.crplus_v2_weight_decay_end_step,
                opt.crplus_v2_negative_modes,
                opt.crplus_v2_start_negative_modes,
                opt.crplus_v2_curriculum_steps,
                opt.crplus_v2_lowpass_pool,
                opt.crplus_v2_frequency_weight,
                opt.crplus_v2_lowfreq_weight,
                opt.crplus_v2_under_dehazed_mix,
                opt.crplus_v2_ratio_cap
            )
        )
    else:
        criterion.append(None)

    optimizer = optim.Adam(params=net.parameters(), lr=opt.start_lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    training_state = load_training_state(net, optimizer)
    trainable_stats = trainable_schedule.apply(
        net,
        int(training_state.get('step', 0)),
        force=trainable_schedule.enabled
    )
    if trainable_schedule.enabled:
        log_trainable_stage(int(training_state.get('step', 0)), trainable_stats)
    print("Total_params: ==> {}".format(trainable_stats['total_params']))
    print("Trainable_params: ==> {}".format(trainable_stats['trainable_params']))
    teacher_net = create_teacher_model()
    cr_ref_net = create_cr_ref_model()
    if opt.dry_run:
        print('Dry run complete.')
        print('Training target steps: {}'.format(steps))
        print('Training will start from step: {}'.format(int(training_state.get('step', 0)) + 1))
        raise SystemExit(0)
    writer = create_summary_writer(int(training_state.get('step', 0)))
    try:
        trainable_schedule_for_loop = trainable_schedule if trainable_schedule.enabled else None
        train(net, loader_train, loader_test, optimizer, criterion, writer, training_state, teacher_net, cr_ref_net, trainable_schedule_for_loop)
    finally:
        if writer is not None:
            writer.close()
