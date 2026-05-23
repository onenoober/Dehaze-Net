# Current Project Context

This is the short handoff page for future DEA-Net work. Keep it compact. Use
`docs/README.md` to decide which deeper document to load next. Put detailed
metrics in `docs/EXPERIMENT_LOG.md`, artifact policy in
`docs/HAZE4K_RUN_MANIFEST.md`, and method reasoning in the dated analysis docs.

## Current State

- Local workspace: `D:\Dehaze\Dehaze-Net`
- Active branch: `codex/haze4k-conditional-lf`
- GitHub repo: `https://github.com/onenoober/Dehaze-Net` (private)
- Server SSH alias: `runyun-ts`
- Main server checkout: `/root/workspace/Dehaze-Net`
- Conditional LF checkout: `/root/workspace/Dehaze-Net-conditional-lf`
- Server env: `/opt/anaconda/envs/py310/bin/python`
- Local machine is for coding, docs, Git, and static checks only. Dry-run,
  smoke, training, benchmark, and evaluation run on the cloud CUDA server.

## Non-Negotiable Training Rule

Formal HAZE4K candidate runs must start with the same 100k target:

```text
epochs=20
iters_per_epoch=5000
total steps=100000
bs=16
patch_size=256
w_loss_L1=1.0
w_loss_CR=0.1
start_lr=0.0001
end_lr=0.000001
checkpoint_interval_steps=10000
eval_interval_steps=10000
save_epoch_checkpoints=false
```

`10k`, `20k`, and `50k` are only intermediate gates inside that same
100k-target run. Do not launch a formal comparison as `epochs=4`, `epochs=10`,
or a separate 50k target. If a shorter schedule is used for dry-run, smoke, or
diagnosis, label it invalid for fair comparison and keep it out of candidate
tables.

When resuming, keep `epochs * iters_per_epoch = 100000`. `train.py` uses this
as the cosine-LR horizon; changing it during resume changes the schedule.

## Current Conditional LF Run

- Run ID: `DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315`
- Status: paused on 2026-05-24 at about log step `21500/100000`
- Code used for the run: commit `09880be`
- Local/GitHub docs may be ahead of the server checkout. Do not sync the server
  until the user explicitly asks.
- Remote checkout: `/root/workspace/Dehaze-Net-conditional-lf`
- Log:
  `/root/workspace/Dehaze-Net-conditional-lf/experiment/HAZE4K/_run_logs/DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315.log`
- Artifact dir:
  `/root/workspace/Dehaze-Net-conditional-lf/experiment/HAZE4K/DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315/`
- Saved resume checkpoint: `saved_model/latest.pk` at step `20000`
- Current metric points:

| Step | PSNR | SSIM | Note |
| ---: | ---: | ---: | --- |
| 10000 | 27.1085 | 0.9638 | fair 100k run |
| 20000 | 28.8571 | 0.9724 | fair 100k run; latest/best saved |

The run was stopped by process group after confirming `MAIN_PID=39377` and
`PGID=39375`; follow-up checks showed no matching process/tmux and GPU memory
released.

## Baseline References

- Baseline 100k scout:
  `DEA-Net-CR-H4K-Baseline-scout-20260520-101334`
  - best step 90000: PSNR `32.2255`, SSIM `0.9844`
  - 20k: PSNR `28.9030`, SSIM `0.9713`
  - 50k: PSNR `31.2384`, SSIM `0.9817`
- LF-v1 positive candidate:
  `DEA-Net-LF-H4K-scout-20260521-003100`
  - best step 90000: PSNR `32.4281`, SSIM `0.9845`
  - 20k: PSNR `28.8563`, SSIM `0.9751`
  - 50k: PSNR `31.3419`, SSIM `0.9817`
- Invalid Conditional LF short schedule:
  `DEA-Net-LF-ConditionalMask-H4K-gate20k-20260523-205312`
  used `T=20000`; keep only as diagnostic evidence.

## Remote Command Rules

Use PowerShell here-strings for multi-line server commands:

```powershell
@'
set -euo pipefail
cd /root/workspace/Dehaze-Net-conditional-lf
git status -sb
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

Server GitHub access should use SSH for the private repo:

```bash
git remote set-url origin git@github.com:onenoober/Dehaze-Net.git
ssh -T git@github.com
git fetch origin
git pull --ff-only
```

For exact check/pause/resume templates, use `docs/WORKFLOW.md`.

## Documentation Map

For load-on-demand rules and future writing boundaries, read
`docs/README.md`.

Most common next files:

1. `docs/WORKFLOW.md`: exact local/server/GitHub/tmux/check/pause/resume
   command templates.
2. `docs/EXPERIMENT_LOG.md`: chronological metrics and decisions.
3. `docs/HAZE4K_RUN_MANIFEST.md`: artifact keep/delete policy.
4. `docs/DEA_NET_LFCR_HAZE4K_PLAN.md`: thesis experiment plan and method
   ladder.

Keep this file short; do not paste full experiment histories here.
