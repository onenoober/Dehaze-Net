# Core Training Server Runbook

Last updated: 2026-05-24

This document records the current core DEA-Net training and evaluation server.
Read it before changing dataset links, environment assumptions, or training and
evaluation commands.

## Server Role

- Role: core training and evaluation server.
- Project root: `/root/workspace/Dehaze-Net`
- Preferred access path from Windows: `ssh runyun-ts`
- Previous root on the old server: `/root/Dehaze-Net`
- Current status: all smoke tests passed after relinking datasets under the new
  project root.

The project was copied from the old cloud server. Absolute symlinks pointing to
`/root/Dehaze-Net` were removed and recreated to point to
`/root/workspace/Dehaze-Net`.

Use the Tailscale-backed `runyun-ts` SSH alias for routine command execution,
VS Code Remote SSH, TensorBoard tunnels, and log checks. The public cloud SSH
host should be treated as a fallback path for repairing Tailscale access.

Windows examples:

```powershell
ssh runyun-ts
code --remote ssh-remote+runyun-ts /root/workspace/Dehaze-Net
ssh -N -L 6006:127.0.0.1:6006 runyun-ts
```

Local Windows commands are PowerShell commands. The `bash` blocks in this file
are for the Ubuntu server shell, or for the PowerShell here-string-over-SSH
pattern documented in `docs/WORKFLOW.md`.

This server environment does not expose `/dev/net/tun` and does not use
systemd, so Tailscale is currently run in userspace mode. If the server
restarts, restore `runyun-ts` from the public SSH fallback:

```powershell
ssh runyun "bash /root/workspace/tailscale-ssh/start.sh"
ssh runyun-ts "hostname && pwd"
```

The recovery script is installed at `/root/workspace/tailscale-ssh/start.sh`.
Its repository copy is `scripts/runyun-tailscale-ssh-start.sh`.

For non-interactive `ssh runyun-ts "<command>"` calls, source conda explicitly
before `conda activate`:

```bash
source /opt/anaconda/etc/profile.d/conda.sh && conda activate py310
```

## Expected Environment

Use a conda environment such as `py310`. The current verified interpreter path
is `/opt/anaconda/envs/py310/bin/python`; prefer that absolute path in
non-interactive commands.

```bash
conda create -n py310 python=3.10 -y
conda activate py310

python -m pip install -U pip setuptools wheel

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install numpy==1.26.4 matplotlib tqdm einops
pip install opencv-python-headless==4.6.0.66
```

Optional, only when strict JPEG decoding compatibility is needed:

```bash
pip install pillow==8.3.2
```

Environment verification:

```bash
cd /root/workspace/Dehaze-Net

/opt/anaconda/envs/py310/bin/python - <<'PY'
import torch, torchvision, cv2, numpy
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("cv2:", cv2.__version__)
print("numpy:", numpy.__version__)
PY
```

## Dataset Layout

The code expects these dataset entry points:

```text
/root/workspace/Dehaze-Net/dataset/ITS
/root/workspace/Dehaze-Net/dataset/OTS
/root/workspace/Dehaze-Net/dataset/HAZE4K
```

The active layout uses symlinks:

```text
dataset/ITS -> dataset/RESIDE/ITS
dataset/OTS -> dataset/RESIDE/OTS
```

Each dataset should expose:

```text
train/hazy
train/clear
test/hazy
test/clear
```

Important mapping decisions:

- `ITS/train/hazy` points to `RESIDE/ITS/train/ITS_haze`.
- `ITS/train/clear` points to `RESIDE/ITS/train/ITS_clear`.
- `ITS/test` points to `SOTS_raw/SOTS/indoor`.
- `OTS/test` points to `SOTS_raw/SOTS/outdoor`.
- `HAZE4K/*/hazy` points to `haze`.
- `HAZE4K/*/clear` points to `gt`.

## Recreate Symlinks

Run this after copying the project to a new absolute path.
This block deletes and recreates symlinks, so treat it as a migration/recovery
operation rather than a routine check.

```bash
ROOT=/root/workspace/Dehaze-Net
cd "$ROOT"

find "$ROOT/dataset" -type l -print -delete
find "$ROOT/trained_models" -type l -print -delete

ln -sfnT "$ROOT/dataset/RESIDE/ITS" "$ROOT/dataset/ITS"
ln -sfnT "$ROOT/dataset/RESIDE/OTS" "$ROOT/dataset/OTS"

ln -sfnT "$ROOT/dataset/RESIDE/ITS/train/ITS_haze" \
  "$ROOT/dataset/RESIDE/ITS/train/hazy"

ln -sfnT "$ROOT/dataset/RESIDE/ITS/train/ITS_clear" \
  "$ROOT/dataset/RESIDE/ITS/train/clear"

ln -sfnT "$ROOT/dataset/SOTS_raw/SOTS/indoor" \
  "$ROOT/dataset/RESIDE/ITS/test"

ln -sfnT "$ROOT/dataset/SOTS_raw/SOTS/indoor/gt" \
  "$ROOT/dataset/SOTS_raw/SOTS/indoor/clear"

ln -sfnT "$ROOT/dataset/RESIDE/ITS/val/haze" \
  "$ROOT/dataset/RESIDE/ITS/val/hazy"

ln -sfnT "$ROOT/dataset/SOTS_raw/SOTS/outdoor" \
  "$ROOT/dataset/RESIDE/OTS/test"

ln -sfnT "$ROOT/dataset/SOTS_raw/SOTS/outdoor/gt" \
  "$ROOT/dataset/SOTS_raw/SOTS/outdoor/clear"

ln -sfnT "$ROOT/dataset/HAZE4K/train/haze" \
  "$ROOT/dataset/HAZE4K/train/hazy"

ln -sfnT "$ROOT/dataset/HAZE4K/train/gt" \
  "$ROOT/dataset/HAZE4K/train/clear"

ln -sfnT "$ROOT/dataset/HAZE4K/test/haze" \
  "$ROOT/dataset/HAZE4K/test/hazy"

ln -sfnT "$ROOT/dataset/HAZE4K/test/gt" \
  "$ROOT/dataset/HAZE4K/test/clear"
```

Check for stale links:

```bash
ROOT=/root/workspace/Dehaze-Net
find "$ROOT/dataset" "$ROOT/trained_models" -type l -exec ls -l {} \; \
  | grep '/root/Dehaze-Net' || echo "No stale old-server symlinks"
```

## Code Compatibility Notes

The repository code has been updated and pushed to GitHub.

Commit:

```text
1c7a18c Make DEA-Net data handling compatible
```

Important behavior in this code version:

- `train.py` resolves datasets from `../dataset/<dataset>` and falls back to
  `../dataset/RESIDE/<dataset>`.
- `train.py` uses `--bs` for the training batch size.
- `eval.py` creates the DataLoader with `num_workers=4` directly, avoiding the
  PyTorch 2.7 `prefetch_factor=None` error.
- `data_loader.py` matches clear images with `.png`, `.jpg`, or `.jpeg`, which
  supports OTS names such as `0001_0.85_0.04.jpg -> 0001.jpg`.

## Smoke Tests

All checks below passed on the core server.

Dataset and link count check:

```bash
ROOT=/root/workspace/Dehaze-Net

for DATA in ITS OTS HAZE4K; do
  echo "===== $DATA ====="
  readlink -f "$ROOT/dataset/$DATA"
  find -L "$ROOT/dataset/$DATA/test/hazy" -type f | wc -l
  find -L "$ROOT/dataset/$DATA/test/clear" -type f | wc -l
  find -L "$ROOT/dataset/$DATA/train/hazy" -type f | wc -l
  find -L "$ROOT/dataset/$DATA/train/clear" -type f | wc -l
done
```

Use `find -L` because several dataset entry points are symlinks. Without `-L`,
HAZE4K `hazy` and `clear` count as link files rather than directories and can
misleadingly report `0`.

Dataset loader check:

```bash
cd /root/workspace/Dehaze-Net/code

/opt/anaconda/envs/py310/bin/python - <<'PY'
from data.data_loader import TrainDataset, TestDataset, ValDataset
from pathlib import Path

root = Path("/root/workspace/Dehaze-Net")

for name in ["ITS", "OTS", "HAZE4K"]:
    ds_root = root / "dataset" / name
    print("====", name, "====")

    train = TrainDataset(ds_root / "train/hazy", ds_root / "train/clear")
    x, y = train[0]
    print("train:", len(train), x.shape, y.shape)

    test = TestDataset(ds_root / "test/hazy", ds_root / "test/clear")
    x, y, fn = test[0]
    print("test:", len(test), x.shape, y.shape, fn)

    val = ValDataset(ds_root / "test/hazy", ds_root / "test/clear")
    item = val[0]
    print("val:", len(val), item["hazy"].shape, item["clear"].shape, item["filename"])
PY
```

Model forward check:

```bash
cd /root/workspace/Dehaze-Net/code

/opt/anaconda/envs/py310/bin/python - <<'PY'
import torch
from model import Backbone, DEANet

x = torch.randn(1, 3, 64, 64).cuda()

for name, cls in [("Backbone", Backbone), ("DEANet", DEANet)]:
    net = cls().cuda().eval()
    with torch.no_grad():
        y = net(x)
    print(name, y.shape, y.min().item(), y.max().item())

print("model forward ok")
PY
```

Full pretrained evaluation checks:

```bash
cd /root/workspace/Dehaze-Net/code

/opt/anaconda/envs/py310/bin/python eval.py --dataset ITS --model_name eval-ITS-newserver --pre_trained_model PSNR4131_SSIM9945.pth
/opt/anaconda/envs/py310/bin/python eval.py --dataset OTS --model_name eval-OTS-newserver --pre_trained_model PSNR3659_SSIM9897.pth
/opt/anaconda/envs/py310/bin/python eval.py --dataset HAZE4K --model_name eval-HAZE4K-newserver --pre_trained_model PSNR3426_SSIM9885.pth
```

Training smoke checks:

```bash
cd /root/workspace/Dehaze-Net/code

for DATA in ITS OTS HAZE4K; do
  CUDA_VISIBLE_DEVICES=0 /opt/anaconda/envs/py310/bin/python train.py \
    --epochs 1 \
    --iters_per_epoch 2 \
    --finer_eval_step 2 \
    --w_loss_L1 1.0 \
    --w_loss_CR 0.1 \
    --start_lr 0.0001 \
    --end_lr 0.000001 \
    --bs 2 \
    --exp_dir ../experiment \
    --model_name "DEA-Net-CR-smoke-${DATA}-$(date +%Y%m%d-%H%M%S)" \
    --dataset "$DATA"
done
```

Success criteria:

```text
loss:... step :2/2
step :2 | epoch: 1 | ssim:... | psnr:...
model saved at step :2
```

## Standard Commands

Full evaluation:

```bash
cd /root/workspace/Dehaze-Net/code

/opt/anaconda/envs/py310/bin/python eval.py --dataset ITS --model_name eval-ITS --pre_trained_model PSNR4131_SSIM9945.pth
/opt/anaconda/envs/py310/bin/python eval.py --dataset OTS --model_name eval-OTS --pre_trained_model PSNR3659_SSIM9897.pth
/opt/anaconda/envs/py310/bin/python eval.py --dataset HAZE4K --model_name eval-HAZE4K --pre_trained_model PSNR3426_SSIM9885.pth
```

Full ITS training:

```bash
cd /root/workspace/Dehaze-Net/code

CUDA_VISIBLE_DEVICES=0 /opt/anaconda/envs/py310/bin/python train.py \
  --epochs 300 \
  --iters_per_epoch 5000 \
  --finer_eval_step 1400000 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --bs 16 \
  --exp_dir ../experiment \
  --model_name DEA-Net-CR-ITS \
  --dataset ITS
```

Full OTS training:

```bash
cd /root/workspace/Dehaze-Net/code

CUDA_VISIBLE_DEVICES=0 /opt/anaconda/envs/py310/bin/python train.py \
  --epochs 300 \
  --iters_per_epoch 5000 \
  --finer_eval_step 1400000 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --bs 16 \
  --exp_dir ../experiment \
  --model_name DEA-Net-CR-OTS \
  --dataset OTS
```

Full HAZE4K training:

```bash
cd /root/workspace/Dehaze-Net/code

CUDA_VISIBLE_DEVICES=0 /opt/anaconda/envs/py310/bin/python train.py \
  --epochs 300 \
  --iters_per_epoch 5000 \
  --finer_eval_step 1400000 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --bs 16 \
  --exp_dir ../experiment \
  --model_name DEA-Net-CR-HAZE4K \
  --dataset HAZE4K
```

Background training example:

```bash
cd /root/workspace/Dehaze-Net/code

nohup bash -c 'CUDA_VISIBLE_DEVICES=0 /opt/anaconda/envs/py310/bin/python train.py \
  --epochs 300 \
  --iters_per_epoch 5000 \
  --finer_eval_step 1400000 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --bs 16 \
  --exp_dir ../experiment \
  --model_name DEA-Net-CR-ITS \
  --dataset ITS' > /root/workspace/Dehaze-Net/experiment/train_ITS.log 2>&1 &

tail -f /root/workspace/Dehaze-Net/experiment/train_ITS.log
```
