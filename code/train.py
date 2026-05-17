import os, time, math
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from logger import plot_loss_log, plot_psnr_log
from metric import psnr, ssim
from model import DEANet
from loss import ContrastLoss
from option_train import opt
from data.data_loader import TrainDataset, TestDataset


start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps


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


def train(net, loader_train, loader_test, optim, criterion, writer=None, training_state=None):
    training_state = training_state or {}
    losses = list(training_state.get('losses', []))

    loss_log = {'L1': [], 'CR': [], 'total': []}
    loss_log_tmp = {'L1': [], 'CR': [], 'total': []}
    if 'loss_log' in training_state:
        loss_log = training_state['loss_log']
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
            lr = opt.start_lr
            if not opt.no_lr_sche:
                lr = lr_schedule_cosdecay(step, T)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr

            x, y = next(loader_train_iter)
            x = x.to(opt.device)
            y = y.to(opt.device)

            out = net(x)
            if opt.w_loss_L1 > 0:
                loss_L1 = criterion[0](out, y)
            if opt.w_loss_CR > 0:
                loss_CR = criterion[1](out, y, x)
            loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_CR * loss_CR
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            loss_log_tmp['L1'].append(loss_L1.item())
            loss_log_tmp['CR'].append(loss_CR.item())
            loss_log_tmp['total'].append(loss.item())

            if writer is not None and opt.tb_log_interval > 0 and (step == 1 or step % opt.tb_log_interval == 0):
                writer.add_scalar('train/loss_total', loss.item(), step)
                writer.add_scalar('train/loss_L1', loss_L1.item(), step)
                writer.add_scalar('train/loss_CR', loss_CR.item(), step)
                writer.add_scalar('train/loss_CR_weighted', opt.w_loss_CR * loss_CR.item(), step)
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


if __name__ == "__main__":

    set_seed_torch(666)

    dataset_root = resolve_dataset_root(opt.dataset)
    train_dir = os.path.join(dataset_root, 'train')
    test_dir = os.path.join(dataset_root, 'test')
    print('train_dir:', train_dir)
    print('test_dir:', test_dir)

    train_set = TrainDataset(os.path.join(train_dir, 'hazy'), os.path.join(train_dir, 'clear'))
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'))
    loader_train = create_data_loader(train_set, opt.bs, True, opt.num_workers)
    loader_test = create_data_loader(test_set, 1, False, opt.test_num_workers)

    net = DEANet(base_dim=32)
    net = net.to(opt.device)

    epoch_size = len(loader_train)
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(ContrastLoss(ablation=False))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.start_lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    training_state = load_training_state(net, optimizer)
    if opt.dry_run:
        print('Dry run complete.')
        print('Training target steps: {}'.format(steps))
        print('Training will start from step: {}'.format(int(training_state.get('step', 0)) + 1))
        raise SystemExit(0)
    writer = create_summary_writer(int(training_state.get('step', 0)))
    try:
        train(net, loader_train, loader_test, optimizer, criterion, writer, training_state)
    finally:
        if writer is not None:
            writer.close()
