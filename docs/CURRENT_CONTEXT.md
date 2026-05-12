# Current Project Context

This file is the handoff note for future Codex conversations. Read this first when continuing DEA-Net work.

## Project

- Research target: DEA-Net image dehazing model.
- Local workspace: `D:\Dehaze\Dehaze-Net`
- GitHub repository: `https://github.com/onenoober/Dehaze-Net`
- Repository visibility: private.
- Main branch: `main`
- Code should be kept in GitHub. Datasets, checkpoints, archives, and experiment outputs must stay out of Git.

## Cloud Server

- OS: Ubuntu 22.04.5 LTS
- Login user: `root`
- SSH host: `ssh.smoothcloud.com.cn`
- Server project path: `/root/workspace/Dehaze-Net`
- Server shell prompt seen before: `(base) root@tm5cq1-0:~/workspace/Dehaze-Net#`
- Recommended server working directory:

```bash
cd /root/workspace/Dehaze-Net
```

## Authentication

### Local Windows to GitHub

- A GitHub token is configured in the Windows user environment variable `GH_TOKEN`.
- Do not print, commit, or expose the token value.
- HTTPS push can use the token through a temporary Git extra header.

### Server to GitHub

- Preferred method: configure a GitHub SSH key on the server.
- Add only the public key `~/.ssh/id_ed25519.pub` to GitHub.
- Never store the private key in the repository.
- Test command:

```bash
ssh -T git@github.com
```

### Windows to Cloud Server

- Preferred method: SSH private key file, for example `D:\Dehaze\smoothcloud.pem`.
- Do not commit or paste the private key.
- Login example:

```powershell
ssh -i D:\Dehaze\smoothcloud.pem root@ssh.smoothcloud.com.cn
```

## Important Documentation

- New beginner Linux/server guide: `Linux.md`
- Download links and fallback plan: `docs/DOWNLOADS.md`
- Reproduction notes: `docs/REPRODUCTION.md`
- Collaboration workflow: `docs/WORKFLOW.md`
- Experiment log template: `docs/EXPERIMENT_LOG.md`
- Agent working notes: `AGENTS.md`

## Data And Weight Locations

Expected server layout:

```text
/root/workspace/Dehaze-Net/
  dataset/
    RESIDE/
      ITS/
      OTS/
    ITS -> RESIDE/ITS
    OTS -> RESIDE/OTS
    HAZE4K/
  trained_models/
    ITS/
    OTS/
    HAZE4K/
  downloads/
  experiment/
```

Previously known local dataset paths:

```text
D:\BaiduNetdiskDownload\dehaze\ITS\ITS.z01 ... ITS.z10
D:\BaiduNetdiskDownload\dehaze\ITS\ITS.zip
D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\clear
D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\depth
D:\BaiduNetdiskDownload\dehaze\OTS_ALPHA\haze
D:\BaiduNetdiskDownload\dehaze\Haze4K.zip
D:\BaiduNetdiskDownload\dehaze\trained_models\ITS
D:\BaiduNetdiskDownload\dehaze\trained_models\OTS
D:\BaiduNetdiskDownload\dehaze\trained_models\HAZE4K
```

Use Windows PowerShell `scp` to upload local files to the server. The server cannot directly read `D:\...` paths.

## Known Download Sources

See `docs/DOWNLOADS.md`. As of the previous check:

- DEA-Net pretrained weights: Baidu mirror reachable.
- RESIDE ITS / OTS: Baidu mirrors reachable.
- HAZE4K: Baidu mirror from the official DMT-Net repo reachable.
- Google Drive / Dropbox were not reliable from this workspace; use Baidu/local download plus upload first.

## Current Repository State

- `Linux.txt` was replaced by `Linux.md`.
- `README.md` links to `Linux.md`.
- Latest pushed commit for that rename: `eff6c04` on `main`.
- After cloning on the server, use:

```bash
cd /root/workspace/Dehaze-Net
git pull --ff-only
```

## DEA-Net Path Notes

- `code/train.py` reads `../dataset/RESIDE/ITS/train`.
- `code/eval.py` reads `../dataset/<dataset>/test`.
- Keep compatibility symlinks:

```bash
cd /root/workspace/Dehaze-Net
ln -sfn RESIDE/ITS dataset/ITS
ln -sfn RESIDE/OTS dataset/OTS
```

- If OTS data has `haze` but the code expects `hazy`, create a symlink after checking the actual layout:

```bash
cd /root/workspace/Dehaze-Net/dataset/RESIDE/OTS
ln -sfn haze hazy
```

## Environment Notes

- Server has Ubuntu 22.04.
- RTX 5090 was discussed as the target GPU.
- The old official DEA-Net environment, especially PyTorch 1.10 / CUDA 11.3, may not support RTX 5090.
- Prefer a modern PyTorch CUDA 12.x build when setting up the training environment.
- Example:

```bash
conda create -n deanet python=3.10 -y
conda activate deanet
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## Recommended First Run Order

1. Pull latest repo on the server.
2. Create dataset, checkpoint, downloads, and experiment directories.
3. Upload ITS and weights from Windows to server.
4. Extract data on the server.
5. Verify directory layout with `find`.
6. Create the Python environment.
7. Run official checkpoint evaluation first.
8. Run a small smoke test training.
9. Only then run full training in `tmux`.

Useful commands:

```bash
cd /root/workspace/Dehaze-Net
git status
ls -lah dataset
find dataset -maxdepth 4 -type d | sort | head -120
find trained_models -type f | sort
nvidia-smi
df -h
```

Evaluation example:

```bash
cd /root/workspace/Dehaze-Net/code
conda activate deanet
python eval.py --dataset ITS --model_name DEA-Net-CR --pre_trained_model PSNR4131_SSIM9945.pth
```

Smoke test example:

```bash
cd /root/workspace/Dehaze-Net/code
conda activate deanet
python train.py --epochs 1 --iters_per_epoch 10 --finer_eval_step 10 --w_loss_L1 1.0 --w_loss_CR 0.1 --start_lr 0.0001 --end_lr 0.000001 --exp_dir ../experiment/ --model_name smoke-test --dataset ITS
```

## How To Resume In A New Conversation

Tell Codex:

```text
请先阅读 AGENTS.md、docs/CURRENT_CONTEXT.md 和 Linux.md，然后继续 DEA-Net 云服务器复现工作。
```

Then provide the newest terminal output from the server if the task depends on current server state.
