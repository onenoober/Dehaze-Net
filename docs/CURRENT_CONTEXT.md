# Current Project Context

This is the compact handoff page for future DEA-Net work. Read this first, then
use `docs/README.md` to choose the one or two deeper documents needed for the
task. Do not paste full experiment histories here; put run facts in
`docs/EXPERIMENT_LOG.md`, artifact policy in `docs/HAZE4K_RUN_MANIFEST.md`, and
route reasoning in the dated analysis docs.

## Quick Handoff

- Active LFCR-v1 implementation source: `codex/haze4k-lfcr-v1` commit
  `1a32a2a` (`Add LFCR v1 route and launch scripts`). The active runyun
  checkout was verified at `dbd1320`, which adds run-launch documentation on
  top of the same implementation.
- Primary cloud server for new operations: `autodl-dehaze` at
  `/root/autodl-tmp/workspace/Dehaze-Net`.
- Secondary cloud server: `runyun-ts`. Use it when the user asks for runyun,
  when AutoDL is unavailable, or when checking runyun artifacts.
- Current user instruction for model training: run on cloud servers by default,
  not local WSL, unless the user explicitly asks for local training. For
  runyun work, first connect to public `runyun` and run
  `bash /root/workspace/tailscale-ssh/start.sh`, then use the Tailscale SSH
  alias `runyun-ts` for all repo, training, and evaluation commands.
- Active LFCR-v2 decay scout on `runyun-ts`:
  `DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455` in
  `/root/workspace/Dehaze-Net-audit-sync`, log
  `experiment/HAZE4K/_run_logs/DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455.log`.
  Launched on 2026-05-28 09:14 CST from branch
  `codex/haze4k-lfcr-v2-decay` commit `9009515`. Tailscale was refreshed via
  public `runyun`, source was synced through GitHub, smoke
  `smoke-H4K-LFCR-v2-decay-runyun-20260528-091330` passed, and startup health
  showed tmux/process active with GPU about `17263 MiB / 88%`. The route keeps
  LF-v1 plus CRPlus-v2 at `w=0.005` through step `10000`, linearly decays to
  `0.0` by step `20000`, then disables CRPlus-v2. First hard mechanism check:
  20k confirms the schedule is off; 30k checks LF gate and trajectory recovery.
- Latest LFCR-v1 run on `runyun-ts`:
  `DEA-Net-LFCR-v1-w005-H4K-scout100k-20260527-231728` in
  `/root/workspace/Dehaze-Net-audit-sync`, log
  `experiment/HAZE4K/_run_logs/DEA-Net-LFCR-v1-w005-H4K-scout100k-20260527-231728.log`.
  Checked on 2026-05-28 08:14 CST: no tmux/train process remained and GPU was
  idle, so the fair 100k run is complete. `best.pk` and `latest.pk` are both
  step `100000`, PSNR `32.2098`, SSIM `0.9844`, LF gate `0.0201`. It improved
  early training at 10k, but the final result is below LF-v1 and CRPlus-v2.
  Full 1000-image diagnostics are synced locally under
  `experiment/HAZE4K/lfcr_v1_diagnostics/DEA-Net-LFCR-v1-w005-100k-20260528-084016`.
  Read this as a useful negative: constant high `w=0.005` rescues some LF-v1
  regression cases, but suppresses broader LF-v1 gains and lowers the LF gate.
  The best next LFCR attempt is likely an early-only or decayed CRPlus-v2
  schedule, not another constant high-weight run.
- Local WSL accidental LFCR launch
  `DEA-Net-LFCR-v1-w0.005-H4K-scout100k-20260527-224613` was stopped on
  2026-05-27 before any valid gate. Do not resume it or use it as evidence.
- Latest runyun CRPlus-v2 status checked on 2026-05-27: run
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540` completed the fair
  100k horizon on `runyun-ts`; no matching tmux/train process remained and GPU
  was idle. `best.pk` and `latest.pk` are both step `100000`, PSNR
  `32.3633`, SSIM `0.9847`. Compact local evidence and diagnostics are synced
  under ignored `experiment/HAZE4K/`; large checkpoints remain on runyun unless
  explicitly requested.
- Local WSL CRPlus-v2 scout
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-192721` was cancelled on
  2026-05-26 before the first 10k checkpoint. Do not resume it.
- Current best standalone model evidence remains LF-v1. ResidualCalib and
  CRPlus-v2 are positive ablations/components, not replacements for LF-v1.
- New isolated warm-start fine-tuning route:
  `docs/HAZE4K_OFFICIAL_WARMSTART_FINETUNE_PLAN_20260528.md` on branch
  `codex/haze4k-official-warmstart-finetune`. This route starts from the
  official HAZE4K `.pth` via a converted step-0 training checkpoint and must
  not be mixed into cold-start fair-candidate tables. Its default staged
  fine-tune schedule is LF-only through the 10k gate, then LF plus
  bottleneck/fusion through the 30k gate, then full-model tiny-LR unfreeze.
- Selector evidence is closed for now: strict CSV, rich CSV, and
  activation-forward deployable proxies all failed the pass line. Do not launch
  another LFResidualSelector 100k scout from oracle evidence alone.
- Before any new model/loss/selector/mask/guard scout, write or update a route
  card using `docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md`. The route must satisfy the
  most-valuable-attempt standard: highest route-decision value per training
  cost, with cheap preflight evidence, an earliest decisive gate, speed metrics,
  and mechanism metrics.

## Storage And Server Roles

- Local WSL workspace: `/home/ubuntu/workspace/Dehaze-Net`.
- Local Windows workspace reference: `D:\Dehaze\Dehaze-Net`.
- GitHub repo: `https://github.com/onenoober/Dehaze-Net` (private). GitHub
  should carry code, docs, scripts, compact metrics, and conclusions only.
- Local WSL is the durable artifact home. Keep datasets, checkpoints, logs,
  plots, previews, and synced evidence under ignored paths such as
  `dataset/`, `trained_models/`, and `experiment/`.
- Cloud servers are temporary compute nodes. Sync compact evidence back to
  local WSL when it affects route decisions, but do not let log sync delay or
  disturb active training. If syncing would contend for bandwidth, I/O, or
  wall time, leave logs in their initial cloud run path and fetch summaries at
  the next gate or final check. Sync large checkpoints, inference folders,
  image grids, arrays, datasets, and archives only when explicitly requested.

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
| CRPlus-v2 | Completed first fair scout at 100k: `32.3633 / 0.9847`. It is positive versus CR baseline (`+0.1396 dB` full-test mean delta), but below LF-v1 (`-0.0633 dB`) and ResidualCalib (`-0.0286 dB`) in PSNR while slightly higher in SSIM. Treat as a positive CR-only component candidate, not an LF-v1 replacement. | `docs/HAZE4K_CRPLUS_V2_FREQ_CURRICULUM_PLAN_20260526.md` |
| LFCR-v1 w0.005 | Completed first high-upside combination scout at 100k: `32.2098 / 0.9844`, LF gate `0.0201`. Full diagnostics show LFCR vs LF-v1 mean `-0.2178 dB`, better/worse `461/539`; it improves `182/351` LF-v1 regression cases by at least `0.30 dB`, but loses at least `0.30 dB` on `264/453` LF-v1 gain cases. Treat as proof that CRPlus helps early/rescue behavior but is harmful as a full-run constant high weight. | `docs/HAZE4K_LFCR_V1_COMBINATION_PLAN_20260527.md` |
| LFCR-v2 decay | Active runyun fair scout launched 2026-05-28. Tests whether LFCR-v1's useful early/rescue signal can be kept while removing late CRPlus-v2 pressure. Decisive checks are `CRPlusV2_weight` off by 20k, LF gate recovery by 30k, and matched PSNR/SSIM/time-to-quality versus LFCR-v1/LF-v1/CRPlus-v2. | `docs/HAZE4K_LFCR_V2_DECAY_PLAN_20260528.md` |
| Route evidence review | Local aggregation across CR, LF-v1, ResidualCalib, CRPlus-v2, and LFCR-v1 confirms LF-v1 remains the best standalone mean-PSNR route, but all-five GT oracle reaches `33.5098` mean PSNR (`+1.0815 dB` over LF-v1). Treat the headroom as real but not deployable under current selector/proxy evidence. | `docs/HAZE4K_ROUTE_EVIDENCE_REVIEW_20260528.md` |

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
8. `docs/HAZE4K_ROUTE_EVIDENCE_REVIEW_20260528.md`: current route evidence
   matrix, oracle headroom, failure modes, Pareto view, and next decision tree.
9. `docs/HAZE4K_OFFICIAL_WARMSTART_FINETUNE_PLAN_20260528.md`: isolated
   official-weight warm-start fine-tuning route, conversion script, gates, and
   DEA-Net reproduction pitfalls.
