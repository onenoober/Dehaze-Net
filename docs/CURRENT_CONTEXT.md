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

### HAZE4K Document Roles

Use these files in this order for the current HAZE4K thesis workflow:

1. `docs/CURRENT_CONTEXT.md`: handoff entry point, server state, active paths,
   and current documentation map.
2. `docs/DEA_NET_LFCR_HAZE4K_PLAN.md`: main HAZE4K training and testing plan;
   use this before starting or changing an experiment.
3. `docs/EXPERIMENT_LOG.md`: chronological run ledger with metrics, run IDs,
   and stop/continue decisions.
4. `docs/HAZE4K_RUN_MANIFEST.md`: current artifact index and keep/delete
   policy for remote `experiment/HAZE4K` outputs.
5. `docs/HAZE4K_FAILURE_ANALYSIS_20260523.md`: route-level failure analysis
   after LF-v1, Conservative LF, CRPlus, LowFreqLoss, TeacherGuard, and PostMix.
6. `docs/HAZE4K_OPTIMIZATION_WORKFLOW_REVIEW_20260523.md`: review of whether
   the current architecture optimization process is methodologically sound,
   mainstream, reliable, and sufficient for the next experiment.
7. `docs/HAZE4K_CONDITIONAL_LF_ROUTE_AUDIT_20260523.md`: pre-implementation
   audit for the next Conditional LF route, including reliability judgment,
   minimal design, stop gates, and experiment card.
8. `docs/HAZE4K_CLEANUP_PLAN_20260523.md`: cleanup audit trail; keep it as a
   dated maintenance record rather than a general experiment guide.

Do not delete HAZE4K documents just to reduce count. Prefer keeping
`CURRENT_CONTEXT.md` as the navigation layer, `EXPERIMENT_LOG.md` as the
chronological evidence layer, and `HAZE4K_RUN_MANIFEST.md` as the artifact
lookup layer. If documents are merged later, merge only after their unique
roles are already represented elsewhere.

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
- Full per-image train-checkpoint evaluation was completed on 2026-05-23 with
  `code/evaluate_train_ckpt_per_image.py`. Remote output:
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/per_image_eval/CR-vs-LF-v1-full-20260523/`.
  It compares baseline
  `DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk` with
  LF-v1 `DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk`, both at
  step `90000`. On all 1000 HAZE4K test images, baseline is
  `32.2253 / 0.9844`, LF-v1 is `32.4283 / 0.9845`, mean delta is
  `+0.2030` PSNR / `+0.000037` SSIM, and median delta is `+0.1567` PSNR.
  Better/worse by PSNR is `549/451`; `>=+0.3 dB` cases are `453`, and
  `<=-0.3 dB` cases are `351`. This confirms LF-v1 has a real full-test gain
  but high per-image variance, so it is the current positive candidate, not a
  finished robust final method.
- Three follow-up training routes were tested and stopped early on 2026-05-23
  because they did not beat the established baseline/LF-v1 curves:
  `DEA-Net-CRPlus-P1-w005-H4K-scout-20260523-011100` used a low-pass hazy
  negative (`w_loss_CR=0.05`, `cr_negative_mode=hazy_lowpass`) and reached only
  `24.9623 / 0.9504` at 10k, far below baseline 10k `27.1101 / 0.9615`.
  `DEA-Net-LowFreqLoss-w005-H4K-scout-20260523-015600` used only a
  low-frequency reconstruction loss (`w_loss_lowfreq=0.05`) and reached
  `27.8852 / 0.9716` at 20k, below baseline 20k `28.9030 / 0.9713`.
  `DEA-Net-LF-LowFreqLoss-w001-H4K-scout-20260523-031600` combined LF-v1 with
  weak low-frequency reconstruction (`w_loss_lowfreq=0.01`) and reached
  `31.0707 / 0.9811` at 50k, below baseline 50k `31.2384 / 0.9817` and LF-v1
  50k `31.3419 / 0.9817`. Do not continue these exact settings unless the goal
  is to document failed ablations.
- LF-v1 root-cause follow-up tested a frozen baseline no-regression guard.
  Rationale: full per-image eval showed LF-v1 improves weak-baseline samples
  but regresses many strong-baseline samples; a training-only baseline teacher
  can penalize LF outputs only when they are worse than the frozen DEA-Net-CR
  teacher on the same supervised crop, without adding inference cost. The
  implemented switches are `--w_loss_teacher_guard`, `--teacher_checkpoint`,
  `--teacher_guard_warmup_steps`, `--teacher_guard_max_weight`, and optional
  local weighting through `--teacher_guard_patch_pool`. `--lf_prior_injection`
  also supports `pre_mix` and `post_mix`.
- LF teacher-guard scout failed the 20k decision gate and was stopped on
  2026-05-23:
  `DEA-Net-LF-TeacherGuard-H4K-scout-20260523-112658`, tmux
  `h4k_lf_teacher_guard_20260523_112658`, log
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/_run_logs/DEA-Net-LF-TeacherGuard-H4K-scout-20260523-112658.log`,
  artifacts
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-TeacherGuard-H4K-scout-20260523-112658/`.
  Config: `bs=16`, `patch_size=256`, `epochs=20`, `iters_per_epoch=5000`,
  `use_lf_prior=true`, `lf_prior_channels=8`, `lf_prior_pool=8`,
  `lf_prior_injection=pre_mix`, `w_loss_CR=0.1`,
  `w_loss_teacher_guard=0.05`, teacher checkpoint
  `DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk`,
  guard warmup `20000` steps, max guard weight `2.0`. A dry-run and 2-step
  smoke test passed before launch. A malformed one-shot launch named
  `DEA-Net-LF-TeacherGuard-H4K-scout-20260523-` was stopped with
  `kill -TERM -- -2807`; do not use that partial artifact.
  The valid run reached step 10000 `24.8283 / 0.9425` and step 20000
  `27.7075 / 0.9704`, below baseline 20k `28.9030 / 0.9713` and LF-v1 20k
  `28.8563 / 0.9751`. It was stopped by process group after the failed 20k
  gate: confirmed train process PGID `3151`, ran `kill -TERM -- -3151`,
  then verified no matching run process, tmux session absent, and GPU
  `0 MiB / 0%`. Retained artifacts include `saved_data/log.txt`,
  `saved_model/best.pk`, and `saved_model/latest.pk`. Treat
  `w_loss_teacher_guard=0.05 + warmup=20000 + max_weight=2.0` as a failed
  over-constrained LF-v1 ablation. Do not continue this exact setting; if the
  teacher route is revisited, use a much weaker/later guard such as `0.01`
  with warmup around `50000`, or test the separate `post_mix` structural
  ablation first.
- LF `post_mix` structure ablation was launched and stopped on 2026-05-23:
  `DEA-Net-LF-PostMix-H4K-scout-20260523-133020`, tmux
  `h4k_lf_postmix_20260523_133020`, log
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/_run_logs/DEA-Net-LF-PostMix-H4K-scout-20260523-133020.log`,
  artifacts
  `/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-PostMix-H4K-scout-20260523-133020/`.
  Config: same as LF-v1 except `lf_prior_injection=post_mix`; no teacher
  guard, no low-frequency reconstruction loss, `w_loss_CR=0.1`, `bs=16`,
  `patch_size=256`, `epochs=20`, `iters_per_epoch=5000`, checkpoint/eval every
  `10000` steps. Rationale: LF-v1's fixed-sample failures and the gate-sweep
  result suggest the LF residual has co-adapted with the main trunk before
  `mix1`; moving it after `mix1` tests whether preserving CGA's original
  skip/deep fusion before adding low-frequency residual can keep LF-v1's full
  test gain while reducing low-frequency over-correction. A 2-step post-mix
  smoke test passed. During launch debugging, an unintended `MODEL_NAME=probe`
  process was started and then stopped with `kill -TERM -- -9491`; do not use
  the `probe` artifact. Valid PostMix curve: 10k `25.9303 / 0.9640`, 20k
  `29.0681 / 0.9740`, 30k `29.9618 / 0.9767`, 40k `30.1842 / 0.9790`, 50k
  `30.7103 / 0.9814`. Although 20k PSNR briefly exceeded baseline/LF-v1, the
  50k result was below baseline 50k `31.2384 / 0.9817` and LF-v1 50k
  `31.3419 / 0.9817`. It was stopped at the failed 50k gate on user request:
  confirmed train process PGID `9758`, ran `kill -TERM -- -9758`, then verified
  no matching process, tmux session absent, and GPU `0 MiB / 0%`. Retained
  artifacts include `saved_data/log.txt`, `saved_model/best.pk`, and
  `saved_model/latest.pk`. Treat post-mix as a negative structure ablation; it
  does not replace LF-v1 as the current positive candidate.

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
