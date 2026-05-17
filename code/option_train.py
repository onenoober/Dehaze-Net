import torch,os,sys,torchvision,argparse
import torch,warnings
import json

# warnings.filterwarnings('ignore')


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1'):
        return True
    if value in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str,default='Automatic detection')

parser.add_argument('--epochs', type=int,default=100)
parser.add_argument('--iters_per_epoch', type=int,default=5000)
parser.add_argument('--finer_eval_step', type=int,default=400000)
parser.add_argument('--bs', type=int,default=16,help='batch size')
parser.add_argument('--num_workers', type=int, default=12, help='training DataLoader worker count')
parser.add_argument('--test_num_workers', type=int, default=4, help='test DataLoader worker count')
parser.add_argument('--pin_memory', action='store_true', help='enable pinned CPU memory in DataLoaders')
parser.add_argument('--persistent_workers', action='store_true', help='keep DataLoader workers alive between iterator recreations')
parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor when num_workers > 0')
parser.add_argument('--start_lr', default=0.0004, type=float, help='start learning rate')
parser.add_argument('--end_lr', default=0.000001, type=float, help='end learning rate')
parser.add_argument('--no_lr_sche', action='store_true',help='no lr cos schedule')
parser.add_argument('--use_warm_up', type=bool, default=False, help='using warm up in learning rate')

parser.add_argument('--w_loss_L1', default=1., type=float, help='weight of loss L1')
parser.add_argument('--w_loss_CR', default=0.1, type=float, help='weight of loss CR')

parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--model_name', type=str, default='MDCTDN')
parser.add_argument('--saved_model_dir', type=str, default='saved_model')
parser.add_argument('--saved_data_dir', type=str, default='saved_data')
parser.add_argument('--saved_plot_dir', type=str, default='saved_plot')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')

parser.add_argument('--dataset', type=str, default='ITS')
parser.add_argument('--no_tqdm', action='store_true', help='disable tqdm training progress bar')
parser.add_argument('--no_tensorboard', action='store_true', help='disable TensorBoard scalar logging')
parser.add_argument('--tensorboard_log_dir', type=str, default='tensorboard', help='TensorBoard log directory; relative paths are created under model_dir')
parser.add_argument('--tb_log_interval', type=int, default=20, help='write TensorBoard train scalars every N steps')
parser.add_argument('--dry_run', action='store_true', help='validate setup and resume checkpoint, then exit before training')
parser.add_argument('--no_pdf_plots', action='store_true', help='skip PDF curve rendering during training')
parser.add_argument('--eval_interval_steps', type=int, default=0, help='override evaluation interval in steps; 0 keeps upstream schedule')
parser.add_argument('--checkpoint_interval_steps', type=int, default=0, help='save latest checkpoint without evaluation every N steps; 0 disables extra saves')
parser.add_argument('--save_epoch_checkpoints', type=str2bool, nargs='?', const=True, default=True, help='save numbered epoch checkpoints in addition to best/latest')

# only need for resume
parser.add_argument('--resume', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--pre_trained_model', type=str,default='null')

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'

dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
model_dir = os.path.join(dataset_dir, opt.model_name)
opt.model_dir = model_dir
opt.saved_model_dir = os.path.join(model_dir, 'saved_model')
opt.saved_data_dir = os.path.join(model_dir, 'saved_data')
opt.saved_plot_dir = os.path.join(model_dir, 'saved_plot')
opt.saved_infer_dir = os.path.join(model_dir, 'saved_infer')

if os.path.exists(model_dir) and not opt.resume and not opt.dry_run:
    print(f'{model_dir} has already existed!')
    print('Use --resume to continue an existing training run, or choose a new --model_name.')
    exit()

if not opt.dry_run:
    os.makedirs(opt.exp_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(opt.saved_model_dir, exist_ok=True)
    os.makedirs(opt.saved_data_dir, exist_ok=True)
    os.makedirs(opt.saved_plot_dir, exist_ok=True)
    os.makedirs(opt.saved_infer_dir, exist_ok=True)

if not os.path.isabs(opt.tensorboard_log_dir):
    opt.tensorboard_log_dir = os.path.join(model_dir, opt.tensorboard_log_dir)
if not opt.dry_run:
    os.makedirs(opt.tensorboard_log_dir, exist_ok=True)

print(opt)
print('model_dir:', model_dir)
print('tensorboard_log_dir:', opt.tensorboard_log_dir)

if not opt.dry_run:
    with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
