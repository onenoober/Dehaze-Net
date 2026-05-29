# DEA-Net Reproduction Guide

## Environment
The upstream project was tested with:
- Python 3.8
- PyTorch 1.10.0
- torchvision 0.11.0
- torchaudio 0.10.0
- CUDA 11.3

For the current HAZE4K workflow, local Windows is for coding/docs/Git/static
checks only. Training, smoke, benchmark, and evaluation run on the cloud Ubuntu
CUDA server; use `docs/WORKFLOW.md` for the current PowerShell-to-SSH command
templates and `docs/CORE_SERVER_RUNBOOK.md` for the current server-side Python
path.

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

## Dataset layout
Prepare the datasets under `dataset/` as described in `dataset/README.md`.

Expected roots:
- `dataset/RESIDE/ITS`
- `dataset/RESIDE/OTS`
- `dataset/HAZE4K`

## Pretrained weights
Place checkpoints under `trained_models/` following `trained_models/README.md`.
The HAZE4K checkpoint name is inconsistent in the upstream docs, so match the actual downloaded filename when you run evaluation.

## Train
Upstream-style local command, run from the `code/` directory only when the local
environment and dataset are deliberately prepared:

```powershell
python train.py --epochs 300 --iters_per_epoch 5000 --finer_eval_step 1400000 --w_loss_L1 1.0 --w_loss_CR 0.1 --start_lr 0.0001 --end_lr 0.000001 --exp_dir ../experiment/ --model_name DEA-Net-CR --dataset ITS
```

## Evaluate
Upstream-style local command, run from the `code/` directory only when the local
environment, checkpoints, and dataset are deliberately prepared. For current
server evaluation, use the templates in `docs/WORKFLOW.md`.

```powershell
python eval.py --dataset HAZE4K --model_name DEA-Net-CR --pre_trained_model <HAZE4K-checkpoint-name>
python eval.py --dataset ITS --model_name DEA-Net-CR --pre_trained_model PSNR4131_SSIM9945.pth
python eval.py --dataset OTS --model_name DEA-Net-CR --pre_trained_model PSNR3659_SSIM9897.pth
```

## Common pitfalls
- Keep the working directory at `code/` when launching training or evaluation.
- For OTS JPEG evaluation, use Pillow 8.3.2 to keep decoding consistent with the official setup.
- Do not place checkpoints or logs inside Git-tracked folders.
