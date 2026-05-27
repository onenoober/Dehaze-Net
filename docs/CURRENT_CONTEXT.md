# Current Project Context

This is the compact handoff page for future DEA-Net work. Read this first, then
use `docs/README.md` to choose the one or two deeper documents needed for the
task. Do not paste full experiment histories here; put run facts in
`docs/EXPERIMENT_LOG.md`, artifact policy in `docs/HAZE4K_RUN_MANIFEST.md`, and
route reasoning in the dated analysis docs.

## Quick Handoff

- Active code branch/source truth: `codex/haze4k-crplus-v2`.
  The CRPlus-v2 scout was launched from commit `24085db` (`Record local
  CRPlus-v2 scout launch`). Later doc-only commits may exist on the same
  branch, so verify with `git log -1 --oneline` before syncing or launching.
- Primary cloud server for new operations: `autodl-dehaze` at
  `/root/autodl-tmp/workspace/Dehaze-Net`.
- Secondary cloud server: `runyun-ts`. Use it when the user asks for runyun,
  when AutoDL is unavailable, or when checking runyun artifacts.
- Latest runyun CRPlus-v2 status checked on 2026-05-27: run
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540` had no matching
  tmux/train process, latest log step about `47556/100000`, and synced compact
  local evidence through the 40k eval: 10k `26.7627/0.9629`,
  20k `29.2182/0.9714`, 30k `30.1416/0.9769`, 40k `30.6141/0.9795`.
  Large `best.pk`/`latest.pk` remain on runyun unless explicitly requested.
- Local WSL CRPlus-v2 scout
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-192721` was cancelled on
  2026-05-26 before the first 10k checkpoint. Do not resume it.
- Current positive model evidence remains LF-v1. ResidualCalib is a positive
  ablation, not a replacement for LF-v1.
- Selector evidence is closed for now: strict CSV, rich CSV, and
  activation-forward deployable proxies all failed the pass line. Do not launch
  another LFResidualSelector 100k scout from oracle evidence alone.
- Before any new model/loss/selector/mask/guard scout, write or update a route
  card using `docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md`.

## Storage And Server Roles

- Local WSL workspace: `/home/ubuntu/workspace/Dehaze-Net`.
- Local Windows workspace reference: `D:\Dehaze\Dehaze-Net`.
- GitHub repo: `https://github.com/onenoober/Dehaze-Net` (private). GitHub
  should carry code, docs, scripts, compact metrics, and conclusions only.
- Local WSL is the durable artifact home. Keep datasets, checkpoints, logs,
  plots, previews, and synced evidence under ignored paths such as
  `dataset/`, `trained_models/`, and `experiment/`.
- Cloud servers are temporary compute nodes. Sync compact evidence back to
  local WSL when it affects route decisions; sync large checkpoints, inference
  folders, image grids, arrays, datasets, and archives only when explicitly
  requested.

## Current Environments

Default AutoDL/SeetaCloud server:

- SSH alias: `autodl-dehaze`.
- SSH target: `root@connect.bjb1.seetacloud.com`, port `19285`.
- Local WSL key: `~/.ssh/autodl_dehaze_ed25519`.
- Project root: `/root/autodl-tmp/workspace/Dehaze-Net`.
- Python: `/root/miniconda3/envs/py310/bin/python`.
- Verified on 2026-05-27: RTX 5090, driver `580.105.08`,
  `torch 2.11.0+cu128`, CUDA runtime `12.8`, `torchvision 0.26.0+cu128`,
  OpenCV `4.6.0`, NumPy `1.26.4`.
- Data and smoke state on 2026-05-27: HAZE4K dataset, official weights, and
  local `experiment/` artifacts synced; smoke
  `smoke-H4K-CRPlusV2-autodl-20260527-104052` passed.

Secondary runyun server:

- SSH alias: `runyun-ts`.
- Main checkout: `/root/workspace/Dehaze-Net`.
- Clean Git-backed checkout for source sync / audit work:
  `/root/workspace/Dehaze-Net-audit-sync`.
- Python: `/opt/anaconda/envs/py310/bin/python`.
- Treat runyun as a secondary active compute node. Verify checkout, branch,
  process, GPU, and log state before any claim or launch.

Local WSL environment:

- Python env: `/home/ubuntu/miniconda3/envs/py310`.
- Validated on 2026-05-26 with PyTorch `2.11.0+cu128`, CUDA `12.8`,
  RTX 4080 SUPER, OpenCV `4.6.0`, NumPy `1.26.4`, and Matplotlib installed.
- Run local project commands in WSL/bash.

## Non-Negotiable Training Rule

Formal HAZE4K candidates must start with the same 100k target:

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

`10k`, `20k`, `30k`, and `50k` are internal gates inside that same run, not
separate short-horizon launches. When resuming, keep
`epochs * iters_per_epoch = 100000`; `train.py` uses this as the cosine-LR
horizon.

Use `docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md` for the full evidence chain and
gate policy. Use `docs/WORKFLOW.md` for exact launch/check/stop templates.

## Current Evidence Summary

| Topic | Current read | Load details |
| --- | --- | --- |
| Baseline CR | Strong reference: best 90k `32.2255 / 0.9844`. | `docs/EXPERIMENT_LOG.md`, `docs/HAZE4K_RUN_MANIFEST.md` |
| LF-v1 | Current positive model evidence: best 90k `32.4281 / 0.9845`, full-test mean delta about `+0.2030 dB` over CR. | `docs/HAZE4K_LF_RESIDUAL_DIRECTION_DIAGNOSIS_20260524.md` |
| ResidualCalib | Positive ablation but below LF-v1: best 90k `32.3936 / 0.9845`; useful for residual-direction evidence. | `docs/HAZE4K_LF_RESIDUAL_CALIBRATION_PLAN_20260525.md`, `docs/HAZE4K_THREE_WAY_OUTPUT_ANALYSIS_20260525.md` |
| Selector route | Closed for now. Oracle headroom is real, but deployable proxies failed; only reopen with a changed target and fresh full-sample proxy audit. | `docs/HAZE4K_SELECTOR_EVIDENCE_CLOSURE_20260526.md` |
| ResidualDirLoss | First `w_loss_residual_dir=0.005` fair scout failed the 30k hard gate and should not be resumed. | `docs/HAZE4K_RESIDUAL_DIRECTION_LOSS_SCALE_PLAN_20260526.md` |
| CRPlus-v2 | Current active code route, but first runyun scout is stopped/paused before 50k. It did not collapse and beat baseline at 20k/30k/40k, but remains below LF-v1's positive 30k/50k trajectory; decide next step from synced 40k evidence. | `docs/HAZE4K_CRPLUS_V2_FREQ_CURRICULUM_PLAN_20260526.md` |

## Do Not Do

- Do not resume cancelled local CRPlus-v2 or failed ResidualDirLoss,
  ResidualSelector, Conditional LF, or Haze-Aware Mask runs.
- Do not launch selector-v2 from oracle evidence alone.
- Do not treat short-horizon smoke, dry-run, or changed-LR-horizon resumes as
  fair candidate evidence.
- Do not edit source directly on cloud servers for experiment variants. Make
  source changes locally, then commit/push/sync intentionally.
- Do not add files under `experiment/`, `dataset/`, or `trained_models/` to Git.

## Load-On-Demand Map

Most common next reads:

1. `docs/README.md`: document map and writing boundaries.
2. `docs/WORKFLOW.md`: exact local/server/GitHub/tmux/check/pause/resume
   command templates.
3. `docs/ANALYSIS_COMMANDS.md`: reusable evaluation, visualization, and
   diagnostic script commands.
4. `docs/CORE_SERVER_RUNBOOK.md`: current server hardware, environment, data
   links, and recovery notes.
5. `docs/EXPERIMENT_LOG.md`: chronological run metrics and decisions.
6. `docs/HAZE4K_RUN_MANIFEST.md`: artifact keep/delete policy and path index.
7. `docs/DEA_NET_LFCR_HAZE4K_PLAN.md`: thesis route, stage plan, promotion
   rules, and reporting evidence chain.
