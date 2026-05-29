import torch,os,sys,torchvision,argparse
import torch,warnings
import json
from datetime import datetime, timezone

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
parser.add_argument('--patch_size', type=int, default=256, help='training crop size')
parser.add_argument('--num_workers', type=int, default=12, help='training DataLoader worker count')
parser.add_argument('--test_num_workers', type=int, default=4, help='test DataLoader worker count')
parser.add_argument('--pin_memory', action='store_true', help='enable pinned CPU memory in DataLoaders')
parser.add_argument('--persistent_workers', action='store_true', help='keep DataLoader workers alive between iterator recreations')
parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor when num_workers > 0')
parser.add_argument('--max_train_batches', type=int, default=0, help='limit train dataset batches for smoke tests; 0 disables')
parser.add_argument('--max_test_batches', type=int, default=0, help='limit validation batches for smoke tests; 0 disables')
parser.add_argument('--start_lr', default=0.0004, type=float, help='start learning rate')
parser.add_argument('--end_lr', default=0.000001, type=float, help='end learning rate')
parser.add_argument('--no_lr_sche', action='store_true',help='no lr cos schedule')
parser.add_argument('--use_warm_up', type=bool, default=False, help='using warm up in learning rate')

parser.add_argument('--w_loss_L1', default=1., type=float, help='weight of loss L1')
parser.add_argument('--w_loss_CR', default=0.1, type=float, help='weight of loss CR')
parser.add_argument('--cr_negative_mode', type=str, default='hazy', choices=['hazy', 'hazy_lowpass'], help='negative source used by VGG contrastive regularization')
parser.add_argument('--cr_lowpass_pool', type=int, default=8, help='average-pooling size for CR low-pass negative')
parser.add_argument('--cr_lowpass_weight', type=float, default=1.0, help='relative weight of low-pass negative when cr_negative_mode=hazy_lowpass')
parser.add_argument('--w_loss_crplus_v2', type=float, default=0.0, help='weight of CRPlus-v2 frequency curriculum loss; 0 disables')
parser.add_argument('--crplus_v2_weight_schedule', type=str, default='constant', choices=['constant', 'linear_decay'], help='schedule for CRPlus-v2 loss weight')
parser.add_argument('--crplus_v2_weight_decay_start_step', type=int, default=0, help='start step for linear CRPlus-v2 weight decay; 0 starts immediately')
parser.add_argument('--crplus_v2_weight_decay_end_step', type=int, default=0, help='end step for linear CRPlus-v2 weight decay; 0 disables decay range')
parser.add_argument('--crplus_v2_min_weight', type=float, default=0.0, help='final/minimum CRPlus-v2 loss weight for scheduled decay')
parser.add_argument('--crplus_v2_negative_modes', type=str, default='hazy,output_lowpass,under_dehazed_mix', help='comma-separated CRPlus-v2 negatives after curriculum')
parser.add_argument('--crplus_v2_start_negative_modes', type=str, default='hazy,under_dehazed_mix', help='comma-separated CRPlus-v2 negatives before curriculum_steps')
parser.add_argument('--crplus_v2_curriculum_steps', type=int, default=20000, help='steps before switching to full CRPlus-v2 negative set')
parser.add_argument('--crplus_v2_lowpass_pool', type=int, default=8, help='average-pooling size for CRPlus-v2 low-frequency terms')
parser.add_argument('--crplus_v2_frequency_weight', type=float, default=0.1, help='weight of Fourier amplitude distance inside CRPlus-v2 distance')
parser.add_argument('--crplus_v2_lowfreq_weight', type=float, default=0.1, help='weight of low-frequency L1 inside CRPlus-v2 distance')
parser.add_argument('--crplus_v2_under_dehazed_mix', type=float, default=0.5, help='output weight for under-dehazed mix negative')
parser.add_argument('--crplus_v2_ratio_cap', type=float, default=2.0, help='cap for each CRPlus-v2 ratio component')
parser.add_argument('--w_loss_lowfreq', type=float, default=0.0, help='weight of low-frequency reconstruction consistency loss; 0 disables')
parser.add_argument('--lowfreq_pool', type=int, default=8, help='average-pooling size for low-frequency reconstruction consistency loss')
parser.add_argument('--w_loss_residual_dir', type=float, default=0.0, help='weight of low-frequency residual direction cosine loss; 0 disables')
parser.add_argument('--residual_dir_pool', type=int, default=8, help='average-pooling size for residual direction cosine loss')
parser.add_argument('--residual_dir_warmup_steps', type=int, default=0, help='do not apply residual direction loss before this step')
parser.add_argument('--residual_dir_target_norm_floor', type=float, default=0.0, help='ignore samples whose target low-frequency residual norm is at or below this value')
parser.add_argument('--w_loss_cr_ref_residual', type=float, default=0.0, help='weight of CR-reference residual-field cosine/magnitude loss; 0 disables')
parser.add_argument('--cr_ref_checkpoint', type=str, default='null', help='CR baseline training checkpoint used as the residual-field reference')
parser.add_argument('--cr_ref_residual_pool', type=int, default=8, help='average-pooling size for CR-reference residual-field loss')
parser.add_argument('--cr_ref_residual_warmup_steps', type=int, default=0, help='do not apply CR-reference residual-field loss before this step')
parser.add_argument('--cr_ref_residual_target_norm_floor', type=float, default=0.0, help='ignore samples whose CR-reference target residual norm is at or below this value')
parser.add_argument('--cr_ref_residual_magnitude_weight', type=float, default=0.25, help='relative weight for CR-reference residual magnitude-ratio loss')
parser.add_argument('--cr_ref_residual_magnitude_cap', type=float, default=2.0, help='cap predicted/target residual norm ratio before magnitude loss; 0 disables cap')
parser.add_argument('--use_lf_prior', action='store_true', help='enable lightweight low-frequency prior at the training bottleneck')
parser.add_argument('--lf_prior_channels', type=int, default=8, help='hidden channels in the LF prior adapter')
parser.add_argument('--lf_prior_pool', type=int, default=8, help='average-pooling size for fixed low-pass input')
parser.add_argument('--lf_prior_gate_init', type=float, default=0.0, help='initial scalar gate for LF prior residual fusion')
parser.add_argument('--lf_prior_residual_center', action='store_true', help='remove spatial mean from LF residual before fusion')
parser.add_argument('--lf_prior_train_dropout', type=float, default=0.0, help='sample-wise LF branch dropout during training; 0 disables')
parser.add_argument('--lf_prior_gate_max', type=float, default=0.0, help='clamp effective LF gate magnitude during forward; 0 disables')
parser.add_argument('--lf_prior_injection', type=str, default='pre_mix', choices=['pre_mix', 'post_mix'], help='where to inject LF prior relative to the first CGA fusion')
parser.add_argument('--lf_conditional_mask', action='store_true', help='enable content-aware spatial mask for LF prior fusion')
parser.add_argument('--lf_mask_hidden_channels', type=int, default=8, help='hidden channels in the conditional LF mask branch')
parser.add_argument('--lf_mask_init_bias', type=float, default=2.0, help='initial bias for conditional LF mask logits')
parser.add_argument('--lf_haze_aware_mask', action='store_true', help='add dark-channel and luma low-frequency cues to the conditional LF mask')
parser.add_argument('--lf_haze_mask_strength', type=float, default=1.0, help='scale applied to haze-aware LF mask cues')
parser.add_argument('--lf_residual_calibration', action='store_true', help='enable bounded direction/amplitude calibration for the LF residual')
parser.add_argument('--lf_calib_hidden_channels', type=int, default=8, help='hidden channels in the LF residual calibration branch')
parser.add_argument('--lf_calib_alpha_max', type=float, default=1.0, help='maximum local amplitude multiplier for LF residual calibration')
parser.add_argument('--lf_residual_selector', action='store_true', help='learn a bounded selector between LF-v1 residual and calibrated LF residual')
parser.add_argument('--lf_selector_hidden_channels', type=int, default=8, help='hidden channels in the LF residual selector branch')
parser.add_argument('--lf_selector_init_bias', type=float, default=2.0, help='initial selector logit bias; positive values start closer to LF-v1')
parser.add_argument('--lf_multiscale_refiner', action='store_true', help='replace LF adapter with a compact multiscale bottleneck refiner')
parser.add_argument('--lf_mbr_channels', type=int, default=8, help='hidden channels in the LF-v2 multiscale bottleneck refiner')
parser.add_argument('--lf_mbr_pool_sizes', type=str, default='4,8,16', help='comma-separated low-pass pool sizes for LF-v2 multiscale bottleneck refiner')
parser.add_argument('--w_loss_lf_gate', type=float, default=0.0, help='L2 penalty weight for the LF scalar gate; 0 disables')
parser.add_argument('--w_loss_teacher_guard', type=float, default=0.0, help='weight of frozen-teacher no-regression guard loss; 0 disables')
parser.add_argument('--teacher_checkpoint', type=str, default='null', help='training checkpoint used by teacher guard')
parser.add_argument('--teacher_use_lf_prior', action='store_true', help='build the teacher checkpoint with LF prior enabled')
parser.add_argument('--teacher_guard_margin', type=float, default=0.0, help='minimum teacher advantage in L1 before the guard is active')
parser.add_argument('--teacher_guard_warmup_steps', type=int, default=0, help='do not apply teacher guard before this step')
parser.add_argument('--teacher_guard_max_weight', type=float, default=1.0, help='maximum per-sample guard weight')
parser.add_argument('--teacher_guard_patch_pool', type=int, default=0, help='optional pooling size for local teacher guard weighting; 0 uses image-level weights')
parser.add_argument('--use_brf_frequency_corrector', action='store_true', help='wrap frozen DEA-Net-CR J0 with baseline-relative frequency residual corrector')
parser.add_argument('--brf_baseline_checkpoint', type=str, default='null', help='DEA-Net-CR training checkpoint used as frozen J0 baseline')
parser.add_argument('--brf_freeze_baseline', type=str2bool, nargs='?', const=True, default=True, help='freeze the J0 baseline inside CBRFRC')
parser.add_argument('--brf_use_baseline_detach', type=str2bool, nargs='?', const=True, default=True, help='detach J0 before passing it into the CBRFRC corrector')
parser.add_argument('--brf_hidden_channels', type=int, default=16, help='hidden channels in CBRFRC')
parser.add_argument('--brf_wavelet_levels', type=int, default=2, help='reserved wavelet level count for future non-laplacian CBRFRC variants')
parser.add_argument('--brf_pyramid_type', type=str, default='laplacian', choices=['laplacian', 'haar'], help='frequency decomposition used by CBRFRC')
parser.add_argument('--brf_gate_init', type=float, default=-4.0, help='initial LF/color gate logit for CBRFRC')
parser.add_argument('--brf_hf_gate_init', type=float, default=-5.0, help='initial HF gate logit for CBRFRC')
parser.add_argument('--brf_max_residual', type=float, default=0.08, help='maximum applied LF residual magnitude in normalized RGB')
parser.add_argument('--brf_max_color_residual', type=float, default=0.04, help='maximum applied color residual magnitude in normalized RGB')
parser.add_argument('--brf_max_hf_residual', type=float, default=0.03, help='maximum applied HF residual magnitude in normalized RGB before hf_scale')
parser.add_argument('--brf_hf_scale', type=float, default=0.1, help='extra scale on the CBRFRC HF residual when preserving high frequency')
parser.add_argument('--brf_use_haze_prior', action='store_true', help='append dark/luma haze cues to CBRFRC inputs')
parser.add_argument('--brf_preserve_highfreq', type=str2bool, nargs='?', const=True, default=True, help='keep CBRFRC HF residual under brf_hf_scale')
parser.add_argument('--w_loss_brf_img', type=float, default=1.0, help='reserved image loss multiplier for CBRFRC reports')
parser.add_argument('--w_loss_brf_res_lf', type=float, default=0.1, help='weight for applied baseline-relative LF residual loss')
parser.add_argument('--w_loss_brf_dir', type=float, default=0.02, help='weight for applied baseline-relative residual direction loss')
parser.add_argument('--w_loss_brf_preserve', type=float, default=0.05, help='weight for CBRFRC preserve loss when target residual is tiny')
parser.add_argument('--w_loss_brf_bound', type=float, default=0.01, help='weight for CBRFRC applied residual magnitude bound')
parser.add_argument('--w_loss_brf_color', type=float, default=0.02, help='weight for CBRFRC image mean-color consistency')
parser.add_argument('--brf_lf_pool', type=int, default=8, help='low-frequency pool for CBRFRC residual target')
parser.add_argument('--brf_mid_pool', type=int, default=4, help='mid-frequency pool for CBRFRC inputs')
parser.add_argument('--brf_dir_norm_floor', type=float, default=0.01, help='ignore CBRFRC direction samples below this target residual norm')
parser.add_argument('--brf_preserve_target_thr', type=float, default=0.015, help='preserve samples whose mean abs target LF residual is below this threshold')

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
parser.add_argument('--early_stop_patience_evals', type=int, default=0, help='stop after N evaluations without enough improvement; 0 disables early stopping')
parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='minimum metric improvement required to reset early-stop patience')
parser.add_argument('--early_stop_after_step', type=int, default=0, help='do not count early-stop patience before this training step')
parser.add_argument('--early_stop_metric', type=str, default='psnr', choices=['psnr', 'ssim'], help='metric used for early stopping')
parser.add_argument('--trainable_schedule', type=str, default='none', help="optional staged fine-tune schedule, e.g. '0:lf_prior;10001:lf_prior,bottleneck,fusion;30001:all'; 'none' keeps all params trainable")

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
    args_payload = dict(opt.__dict__)
    args_payload['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
    args_payload['argv'] = sys.argv

    args_initial_path = os.path.join(model_dir, 'args_initial.txt')
    if not os.path.exists(args_initial_path):
        with open(args_initial_path, 'w') as f:
            json.dump(args_payload, f, indent=2)
    with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
        json.dump(args_payload, f, indent=2)
    with open(os.path.join(model_dir, 'args_history.jsonl'), 'a') as f:
        f.write(json.dumps(args_payload) + '\n')
