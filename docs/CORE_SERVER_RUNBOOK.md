# Core Training Server Runbook

Last updated: 2026-05-28

Purpose: record the current server facts needed before changing environment,
dataset links, or training assumptions. Exact SSH, sync, smoke, evaluation, and
launch commands belong in `docs/WORKFLOW.md`.

## Authority

- Current executable status: `docs/CURRENT_CONTEXT.md`.
- Repeatable command templates: `docs/WORKFLOW.md`.
- Run metrics and stop/resume decisions: `docs/EXPERIMENT_LOG.md`.
- Artifact keep/delete policy: `docs/HAZE4K_RUN_MANIFEST.md`.

## Primary Server

- Role: first-choice cloud training and evaluation server.
- SSH alias from WSL: `autodl-dehaze`.
- SSH target: `root@connect.bjb1.seetacloud.com`, port `19285`.
- Local WSL key: `~/.ssh/autodl_dehaze_ed25519`.
- Project root: `/root/autodl-tmp/workspace/Dehaze-Net`.
- Python: `/root/miniconda3/envs/py310/bin/python`.
- Branch/commit at setup: `codex/haze4k-crplus-v2`,
  `ae9a70c` (`Record cloud CRPlus-v2 scout launch`).
- Data disk: `/root/autodl-tmp`, 200G, about 192G free after dataset,
  weights, environment, and experiment sync.

## Verified Environment

Validated on 2026-05-27:

```text
GPU: NVIDIA GeForce RTX 5090
Driver: 580.105.08
nvidia-smi CUDA API: 13.0
PyTorch CUDA runtime: 12.8
Visible device memory: 32607 MiB
Compute capability: 12.0
torch 2.11.0+cu128
torchvision 0.26.0+cu128
cv2 4.6.0
numpy 1.26.4
```

The system `/usr/local/cuda` symlink may point to CUDA 11.8. For this project,
prefer the PyTorch wheel runtime inside `py310`; do not use system CUDA as the
source of truth for training compatibility.

Environment creation command used on 2026-05-27:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n py310 python=3.10.13 pip -y
conda activate py310
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128
cd /root/autodl-tmp/workspace/Dehaze-Net
python -m pip install -r requirements.txt pillow
```

## Data and Weights

Validated on 2026-05-27:

```text
dataset/HAZE4K/train/hazy   3001
dataset/HAZE4K/train/clear  3000
dataset/HAZE4K/test/hazy    1000
dataset/HAZE4K/test/clear   1000

trained_models/HAZE4K/PSNR3426_SSIM9885.pth
trained_models/ITS/PSNR4131_SSIM9945.pth
trained_models/OTS/PSNR3659_SSIM9897.pth
```

Local WSL `experiment/` was synced to the default server on 2026-05-27:

```text
remote experiment size: about 6.3G
remote experiment/HAZE4K top-level entries: 41
remote experiment file count: 5377
```

## Smoke Evidence

Validated smoke run:

```text
Run ID: smoke-H4K-CRPlusV2-autodl-20260527-104052
Log: /root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/_run_logs/smoke-H4K-CRPlusV2-autodl-20260527-104052.log
Result: 2 training steps, CRPlusV2 loss, CUDA forward/backward, and HAZE4K eval passed.
Metric: step 2, SSIM 0.0651, PSNR 7.2473
```

## Code Compatibility Notes

- `train.py` resolves datasets from `../dataset/<dataset>` and falls back to
  `../dataset/RESIDE/<dataset>`.
- `train.py` uses `--bs` for the training batch size.
- `eval.py` creates the DataLoader with `num_workers=4` directly.
- `data_loader.py` accepts `.png`, `.jpg`, and `.jpeg` for clear images.

## Secondary Runyun Server

- Role: secondary active cloud training/evaluation server.
- SSH alias: `runyun-ts`.
- Main checkout: `/root/workspace/Dehaze-Net`.
- Clean Git-backed checkout: `/root/workspace/Dehaze-Net-audit-sync`.
- Python: `/opt/anaconda/envs/py310/bin/python`.
- Access is usually from Windows PowerShell via `ssh runyun-ts`.
- Additional runyun `py310` packages installed on 2026-05-28 for supervised
  proxy audits: `scikit-learn==1.7.2`, `scipy==1.15.3`, `joblib==1.5.3`,
  `threadpoolctl==3.6.0`.

Use AutoDL first for new work unless the user names runyun or AutoDL is
unavailable. Before using runyun, verify the intended checkout, branch, process
state, and GPU state. Older runyun paths remain useful reference points, but
they should not override the current branch/source truth in
`docs/CURRENT_CONTEXT.md`.
