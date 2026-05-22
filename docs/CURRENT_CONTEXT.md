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
- Preferred SSH alias: `runyun-ts`
- Tailscale address: `100.118.134.99`
- Tailscale SSH port: `2222`
- Public SSH host fallback: `ssh.smoothcloud.com.cn`
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

- Preferred method: `runyun-ts`, the local SSH alias that reaches the server
  through Tailscale Serve.
- The alias uses `C:\Users\Administrator\.ssh\key.pem`; do not commit or paste
  the private key.
- Login example:

```powershell
ssh runyun-ts
```

- Open the server project in VS Code:

```powershell
code --remote ssh-remote+runyun-ts /root/workspace/Dehaze-Net
```

- Keep the public `ssh.smoothcloud.com.cn` path only as a fallback for repairing
  Tailscale if `runyun-ts` stops working.
- If `runyun-ts` times out after a server restart, restore it through the
  public fallback:

```powershell
ssh runyun "bash /root/workspace/tailscale-ssh/start.sh"
```

- The recovery script starts userspace Tailscale, local SSHD
  `127.0.0.1:2223`, and Tailscale Serve `100.118.134.99:2222 -> 127.0.0.1:2223`.
- Because the container has no `systemd` and no `/dev/net/tun`, Tailscale is
  recovered through Supervisor plus userspace networking. The Supervisor program
  `runyun-tailscale-ssh` runs `/root/workspace/tailscale-ssh/supervisor-keepalive.sh`,
  which calls `/root/workspace/tailscale-ssh/start.sh` if tailscaled, Serve, or
  the local SSHD is missing.
- Tailscale state is persisted under
  `/root/workspace/tailscale-ssh/state/tailscaled.state`; logs are under
  `/root/workspace/tailscale-ssh/logs/`.

## Important Documentation

- New beginner Linux/server guide: `Linux.md`
- Core training server runbook: `docs/CORE_SERVER_RUNBOOK.md`
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
git -c http.version=HTTP/1.1 pull --ff-only
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
- The active `runyun-ts` training environment is `py310` under `/opt/anaconda`.
- One-shot SSH commands through `runyun-ts` should load conda explicitly:

```bash
source /opt/anaconda/etc/profile.d/conda.sh && conda activate py310
```

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

## Latest HAZE4K Training Reference

- Current short-run reference: `DEA-Net-CR-H4K-Baseline-scout-20260520-101334`.
- Remote artifact path:
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/`.
- Training config: HAZE4K, DEA-Net-CR, `bs=16`, `patch_size=256`,
  `epochs=20`, `iters_per_epoch=5000`, `100000` total steps,
  `w_loss_L1=1.0`, `w_loss_CR=0.1`, checkpoint/eval every `10000` steps.
- Best scout checkpoint: `saved_model/best.pk` at step `90000` / epoch `18`,
  PSNR `32.2255`, SSIM `0.9844`.
- Final scout checkpoint: `saved_model/latest.pk` at step `100000` / epoch `20`,
  PSNR `32.0952`, SSIM `0.9844`.
- Official HAZE4K `.pth` full eval reference:
  `eval-H4K-official-full-20260520-095415`, PSNR `34.2556`, SSIM `0.9885`,
  checkpoint `PSNR3426_SSIM9885.pth`.
- Batch-size benchmark selected `bs=16` for future HAZE4K scout/baseline runs:
  `bs=16` `4.2868` step/s / `68.59` img/s, `bs=24` `2.8391` step/s /
  `68.14` img/s, `bs=32` `2.1166` step/s / `67.73` img/s.
- Keep the checkpoint-format distinction: training outputs `best.pk` and
  `latest.pk` are not directly interchangeable with `eval.py` `.pth` weights.
  Final formal evaluation needs a train-checkpoint eval path or an explicit
  export/reparameterization path.
- Active LF prior scout:
  `DEA-Net-LF-H4K-scout-20260521-003100`, branch `codex/haze4k-lf-prior`,
  code commit `857661d`, tmux `h4k_lf_scout_20260521_003100`.
  It uses `bs=16`, `patch_size=256`, `epochs=20`, `iters_per_epoch=5000`,
  `use_lf_prior=true`, `lf_prior_channels=8`, `lf_prior_pool=8`,
  `lf_prior_gate_init=0.0`, and validates every `10000` steps.
  Launcher log:
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/_run_logs/DEA-Net-LF-H4K-scout-20260521-003100.log`.
  Training artifacts:
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/`.
  After the server restarted on 2026-05-21, `runyun-ts` was restored through
  `ssh runyun "bash /root/workspace/tailscale-ssh/start.sh"`. The LF scout had
  `best.pk` and `latest.pk` at step `80000` / epoch `16`, PSNR `32.1721`,
  SSIM `0.9841`; `losses.npy` contained about `86856` entries, so the
  uncheckpointed tail after step `80000` should be treated as lost. It was
  resumed from `latest.pk` in tmux `h4k_lf_resume_20260521_084700`; resume log:
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/_run_logs/DEA-Net-LF-H4K-scout-20260521-003100-resume-20260521-084700.log`.
  Final result: `best.pk` at step `90000` / epoch `18`, PSNR `32.4281`, SSIM
  `0.9845`; `latest.pk` at step `100000` / epoch `20`, PSNR `32.3857`, SSIM
  `0.9845`.
- Fixed-sample visual comparison completed after verifying `runyun-ts` on
  2026-05-22:
  `experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-20260522/`.
  It compares baseline
  `DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk` with LF
  `DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk`; both are train
  checkpoints at step `90000`, loaded through the train-checkpoint visual path,
  not through official `.pth` `eval.py`.
  On the 20 evenly spaced fixed test samples, baseline mean is PSNR `31.5373` /
  SSIM `0.9835`, LF mean is PSNR `31.2580` / SSIM `0.9829`, so the subset mean
  delta is `-0.2793` PSNR / `-0.0006` SSIM. Best LF cases include
  `195_0.61_1.47.png` (`+2.1630` PSNR) and `241_0.85_0.72.png` (`+1.1115`);
  worst cases include `384_0.97_0.82.png` (`-3.2380`) and
  `715_0.63_1.36.png` (`-1.9517`). This fixed subset is a qualitative
  diagnostic and should be read together with the full-test validation result,
  which still favors LF by about `+0.20` dB at `best.pk`.
  Selected panels were copied locally under
  `D:\Dehaze\Dehaze-Net\experiment\HAZE4K\visual_compare\DEA-Net-CR-vs-LF-20260522\panels\`
  for inspection; keep them out of Git.
- Objective visual analysis tooling was added on branch `codex/haze4k-lf-prior`
  at commit `a1424ea`: `code/analyze_visual_compare.py` plus
  `scripts/runyun-haze4k-objective-analysis.sh`. It reads an existing
  visual-compare directory with `input/`, `baseline/`, `lf` or other current
  output, and `gt/`; it writes `analysis_metrics.csv`,
  `analysis_summary.json`, `analysis_report.md`, per-image heatmaps, and
  diagnostic panels. The first run used the existing server `py310`
  environment through `/opt/anaconda/envs/py310/bin/python`; no extra package
  install was required because `numpy`, `PIL`, and `cv2` were already present.
  Output path:
  `experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-20260522/analysis/`.
  Objective triage result for the 20 fixed samples: LF better on `5`, worse on
  `9`, mixed/neutral on `6`; mean LF delta was `-0.2782` PSNR / `-0.0006`
  SSIM, mean delta-E improvement was `-0.0407`, and mean edge-error
  improvement was `+0.00015`. The tool is suitable for objective pre-screening,
  while final visual quality decisions still need subjective inspection of the
  diagnostic panels.
- Inference-time LF gate sweep was completed at commit `e46f8a4` with
  `scripts/runyun-haze4k-lf-gate-sweep.sh`. It uses the same fixed samples and
  compares baseline `best.pk` with the LF `best.pk` while multiplying
  `lf_prior.gate` by `0`, `0.25`, `0.5`, `0.75`, and `1.0`; this does not
  retrain. Output path:
  `experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-gate-sweep-20260522/`.
  The trained LF gate was `0.033890`. Mean delta PSNR over the 20 fixed samples
  was `-0.2694`, `-0.2683`, `-0.2695`, `-0.2732`, and `-0.2793` for scales
  `0`, `0.25`, `0.5`, `0.75`, and `1.0`; mean delta-E improvement stayed
  negative around `-0.04`. This means post-hoc gate weakening only mildly
  changes the subset result and does not fix the issue. Key subjective cases:
  `384`, `479`, and `952` get worse as gate increases, consistent with LF
  residual contributing to over-dehazing; `9` improves as gate increases, so it
  is a different failure mode; `80` improves in PSNR/color but still has lower
  SSIM, matching the backlight/brightness tradeoff.
- Conservative LF v2 scout completed on 2026-05-22:
  `DEA-Net-LF-Conservative-H4K-scout-20260522-145904`, code commit `242a7a6`,
  `bs=16`, `patch_size=256`, `epochs=20`, `iters_per_epoch=5000`,
  `lf_prior_channels=4`, `lf_prior_pool=8`,
  `lf_prior_residual_center=true`, `lf_prior_train_dropout=0.25`,
  `lf_prior_gate_max=0.02`, `w_loss_lf_gate=0.01`, `w_loss_CR=0.1`.
  It finished 100000 steps in tmux `h4k_lf_cons_20260522_145904`; final and
  best checkpoint are both at step `100000` / epoch `20`, PSNR `32.1083`,
  SSIM `0.9843`. This is lower than the baseline scout best `32.2255 / 0.9844`
  and lower than LF-v1 best `32.4281 / 0.9845`, so it should be treated as an
  over-constrained/failed LF ablation rather than a main candidate.
- Fixed-sample Conservative visual/objective comparison was generated at
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-Conservative-20260522/`
  and partially copied locally under
  `D:\Dehaze\Dehaze-Net\experiment\HAZE4K\visual_compare\DEA-Net-CR-vs-LF-Conservative-20260522\`.
  It uses the same 20 sample list as `DEA-Net-CR-vs-LF-20260522`, baseline
  `best.pk` at step `90000`, and Conservative `best.pk` at step `100000`.
  Subset visual summary: baseline `31.5373 / 0.9835`, Conservative
  `30.7523 / 0.9830`, mean delta `-0.7850` PSNR / `-0.0005` SSIM. Objective
  triage: Conservative better on `4`, worse on `14`, mixed on `2`; mean delta
  PSNR `-0.7825`, mean delta-E improvement `-0.1479`, dark-channel abs-bias
  improvement `-0.0026`, and edge-error improvement `-0.00063`. Compared with
  LF-v1 (`-0.2782` PSNR, better/worse/mixed `5/9/6`), the Conservative variant
  did not reduce fixed-sample risk and should not be promoted.

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
source /opt/anaconda/etc/profile.d/conda.sh
conda activate py310
python eval.py --dataset ITS --model_name DEA-Net-CR --pre_trained_model PSNR4131_SSIM9945.pth
```

Smoke test example:

```bash
cd /root/workspace/Dehaze-Net/code
source /opt/anaconda/etc/profile.d/conda.sh
conda activate py310
python train.py --epochs 1 --iters_per_epoch 10 --finer_eval_step 10 --w_loss_L1 1.0 --w_loss_CR 0.1 --start_lr 0.0001 --end_lr 0.000001 --exp_dir ../experiment/ --model_name smoke-test --dataset ITS
```

## How To Resume In A New Conversation

Tell Codex:

```text
请先阅读 AGENTS.md、docs/CURRENT_CONTEXT.md 和 Linux.md，然后继续 DEA-Net 云服务器复现工作。
```

Then provide the newest terminal output from the server if the task depends on current server state.
