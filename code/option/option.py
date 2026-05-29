import os,argparse
import json


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

parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--dataset', type=str, default='ITS')
parser.add_argument('--val_dataset_dir', type=str)
parser.add_argument('--model_name', type=str, default='DEA-Net', help='experiment name')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')
parser.add_argument('--num_workers', type=int, default=4, help='evaluation DataLoader worker count')
parser.add_argument('--max_test_batches', type=int, default=0, help='limit evaluation batches for smoke tests; 0 disables')

# only need for evaluation
parser.add_argument('--pre_trained_model', type=str, default='null', help='path of pre trained model for resume training')
parser.add_argument('--save_infer_results', action='store_true', default=False, help='save the infer results during validation')
parser.add_argument('--max_eval_batches', type=int, default=0, help='limit evaluation batches; 0 evaluates the full dataset')
parser.add_argument('--save_per_image_csv', action='store_true', help='save per-image evaluation metrics')
parser.add_argument('--use_brf_frequency_corrector', action='store_true', help='evaluate a CBRFRC wrapper checkpoint')
parser.add_argument('--brf_baseline_checkpoint', type=str, default='null', help='DEA-Net-CR checkpoint for CBRFRC J0 if candidate checkpoint is corrector-only')
parser.add_argument('--brf_freeze_baseline', type=str2bool, nargs='?', const=True, default=True, help='freeze the J0 baseline inside CBRFRC')
parser.add_argument('--brf_use_baseline_detach', type=str2bool, nargs='?', const=True, default=True, help='detach J0 inside CBRFRC')
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
opt=parser.parse_args()

if opt.val_dataset_dir is None:
    opt.val_dataset_dir = os.path.join('../dataset/', opt.dataset, 'test')
exp_dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
exp_model_dir = os.path.join(exp_dataset_dir, opt.model_name)

if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)

if not os.path.exists(exp_dataset_dir):
    os.mkdir(exp_dataset_dir)

opt.saved_infer_dir = os.path.join(exp_model_dir, opt.pre_trained_model.split('.pth')[0])
if not os.path.exists(exp_model_dir):
    os.mkdir(exp_model_dir)
    os.mkdir(opt.saved_infer_dir)
if not os.path.exists(opt.saved_infer_dir):
    os.mkdir(opt.saved_infer_dir)

with open(os.path.join(exp_model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
