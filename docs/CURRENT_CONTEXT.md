# Current Project Context

This is the short handoff page for future DEA-Net work. Keep it compact. Use
`docs/README.md` to decide which deeper document to load next. Put detailed
metrics in `docs/EXPERIMENT_LOG.md`, artifact policy in
`docs/HAZE4K_RUN_MANIFEST.md`, and method reasoning in the dated analysis docs.

## Current State

- Local workspace: `D:\Dehaze\Dehaze-Net`
- Local editing branch: `codex/haze4k-lf-residual-calibration`
- Conditional LF code branch/run lineage: `codex/haze4k-conditional-lf`
- ResidualCalib code branch/run lineage: `codex/haze4k-lf-residual-calibration`
- GitHub repo: `https://github.com/onenoober/Dehaze-Net` (private)
- Server SSH alias: `runyun-ts`
- Main server checkout: `/root/workspace/Dehaze-Net`
- Conditional LF checkout: `/root/workspace/Dehaze-Net-conditional-lf`
- Clean synced source checkout: `/root/workspace/Dehaze-Net-audit-sync`
  - Current ResidualCalib code applied as remote commit `a474953`, equivalent
    to local/GitHub commit `3b52a2b`.
  - Purpose: Git-backed source truth for the audit/metadata/doc sync work.
  - It symlinks `dataset/HAZE4K` and `experiment` to the main server checkout
    for dry-run validation without touching the older dirty training checkouts.
- Server env: `/opt/anaconda/envs/py310/bin/python`
- Local machine is for coding, docs, Git, and static checks only. Dry-run,
  smoke, training, benchmark, and evaluation run on the cloud CUDA server.
- Local commands are Windows PowerShell; server commands are Ubuntu/Linux. Use
  the PowerShell here-string SSH pattern below for multi-line server commands.

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

## Completed LF ResidualCalib Run

- Run ID: `DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654`
- Status: completed naturally on 2026-05-25; no tmux/process remained after
  completion and GPU returned to idle.
- Branch/commit: local and GitHub branch
  `codex/haze4k-lf-residual-calibration`, commit `3b52a2b`; remote applied
  commit `a474953` has the same tree.
- Remote checkout: `/root/workspace/Dehaze-Net-audit-sync`
- Log:
  `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/_run_logs/DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654.log`
- Artifact dir:
  `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654/`
- Fair config: `epochs=20`, `iters_per_epoch=5000`, total `100000`,
  `bs=16`, `patch_size=256`, `w_loss_CR=0.1`.
- Best checkpoint: `saved_model/best.pk` at 90k, PSNR `32.3936`,
  SSIM `0.9845`.
- Final checkpoint: `saved_model/latest.pk` at 100k, PSNR `32.3858`,
  SSIM `0.9846`; best values still came from 90k.
- Alpha diagnostics show the branch is active, not dead:
  `best.pk` alpha last mean/std/min/max
  `0.516481/0.010491/0.502225/0.581695`; `latest.pk`
  `0.516911/0.010565/0.502370/0.580890`.

Full-test and residual diagnostics:

- vs CR baseline full per-image:
  `experiment/HAZE4K/per_image_eval/CR-vs-ResidualCalib-full-20260525/`
  - mean delta PSNR `+0.1682 dB`, median `+0.2010 dB`
  - better/worse by PSNR `547/453`
  - weak-baseline mean delta `+0.6354 dB`
  - strong-baseline mean delta `-0.1171 dB`
- vs LF-v1 full per-image:
  `experiment/HAZE4K/per_image_eval/LF-v1-vs-ResidualCalib-full-20260525/`
  - mean delta PSNR `-0.0347 dB`, median `+0.0271 dB`
  - better/worse by PSNR `509/491`
  - weak-LF-v1-baseline mean delta `+0.3970 dB`
  - strong-LF-v1-baseline mean delta `-0.2374 dB`
- Residual diagnostics:
  - `experiment/HAZE4K/residual_diagnostic/CR-vs-ResidualCalib-20260525/`
    has wrong-direction count `163`, LF MSE improved/regressed `554/446`.
  - `experiment/HAZE4K/residual_diagnostic/LF-v1-vs-ResidualCalib-20260525/`
    has wrong-direction count `211`, LF MSE improved/regressed `512/488`.

Decision:

ResidualCalib is a useful positive ablation because it beats the CR baseline,
but it is not a replacement for LF-v1. The next LF optimization should directly
reduce residual wrong-direction cases and strong-baseline regressions; do not
restart pure mask stacking as the next step.

## LF ResidualSelector Ready State

- Local branch: `codex/haze4k-selector-oracle`
- Local/GitHub commit: `7c93000`
- Remote clean checkout: `/root/workspace/Dehaze-Net-audit-sync`
  - Branch: `codex/haze4k-selector-oracle`
  - Commit: `7c93000`
- New route docs:
  - `docs/HAZE4K_NEXT_ROUTE_REVIEW_20260525.md`
  - `docs/HAZE4K_LF_RESIDUAL_SELECTOR_PLAN_20260525.md`
- New read-only script:
  `code/analyze_selector_oracle.py`
- New launch script:
  `scripts/runyun-haze4k-lf-residual-selector-scout.sh`
- Selector/oracle diagnosis:
  `experiment/HAZE4K/selector_oracle/Baseline-LFv1-ResidualCalib-full-20260525/`
  - LF-v1 mean `32.4283 / 0.984454`
  - ResidualCalib mean `32.3936 / 0.984500`
  - LF-v1/ResidualCalib two-way oracle `33.0034 / 0.985299`,
    `+0.5751 dB` over LF-v1
  - baseline/LF-v1/ResidualCalib three-way oracle `33.2538 / 0.985637`,
    `+0.8255 dB` over LF-v1
- Old adverse 20-sample selector/oracle:
  `experiment/HAZE4K/selector_oracle/Baseline-LFv1-ResidualCalib-fixed20260522-20260525/`
  - LF-v1/ResidualCalib two-way oracle `31.8019 / 0.983492`,
    `+0.5439 dB` over LF-v1
  - three-way oracle `32.1661 / 0.983806`
- Interpretation: selector headroom is strong, but the best simple rule is
  GT-aware residual error ratio. Do not hard-code the oracle. Use it as evidence
  for a learned bounded selector.
- Implementation: `--lf_residual_selector` requires
  `--lf_residual_calibration`, starts near LF-v1 with
  `--lf_selector_init_bias 2.0`, and logs `LF_selector_*`.
- Dry-run passed on remote:
  `dryrun-H4K-LF-ResidualSelector-20260525`.
- 2-step smoke passed on remote:
  `smoke-H4K-LF-ResidualSelector-20260525-223437`.
  It wrote `saved_model/latest.pk` at step 2 and logged
  `LF_selector_mean/std/min/max =
  0.8807968/0.0/0.8807970/0.8807970`, plus `LF_alpha_mean=0.5`.
- Next action: after source sync, launch one fair 100k-target scout with
  standard internal 10k/20k/30k/50k gates. The run should not include
  Conditional LF, Haze-Aware Mask, TeacherGuard, LowFreqLoss, or CRPlus.

## Active LF ResidualSelector Run

- Run ID: `DEA-Net-LF-ResidualSelector-H4K-scout100k-20260525-223844`
- Status: stopped on 2026-05-26 00:27 CST after the 20k gate failed; do not
  resume this exact selector setting.
- tmux session: `h4k_lf_resselector_100k_20260525_223844`
- Launch script:
  `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/_run_logs/DEA-Net-LF-ResidualSelector-H4K-scout100k-20260525-223844.sh`
- Log:
  `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/_run_logs/DEA-Net-LF-ResidualSelector-H4K-scout100k-20260525-223844.log`
- Artifact dir:
  `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/DEA-Net-LF-ResidualSelector-H4K-scout100k-20260525-223844/`
- Verified launch config:
  `epochs=20`, `iters_per_epoch=5000`, total `100000`, `bs=16`,
  `patch_size=256`, `w_loss_CR=0.1`, `start_lr=0.0001`,
  `end_lr=0.000001`, checkpoint/eval every `10000`,
  `save_epoch_checkpoints=false`, `lf_residual_calibration=True`,
  `lf_residual_selector=True`, `lf_selector_init_bias=2.0`.
- Startup health: process and tmux were present; GPU showed about `13919 MiB`
  used and `87%` utilization shortly after launch.
- 10k gate, checked 2026-05-25 23:16 CST: `latest.pk` and `best.pk` both
  at step `10000`, PSNR `26.7739`, SSIM `0.9615`. This is below baseline
  by `-0.3362 dB`, above LF-v1 by `+0.5088 dB`, and above ResidualCalib by
  `+0.2073 dB`; SSIM is essentially tied with baseline but below LF-v1 and
  ResidualCalib. Continue to the 20k gate because it is not broken.
- 10k selector diagnostics: `LF_selector_mean/std/min/max =
  0.878648/0.000117/0.877970/0.878947`; `LF_alpha_mean/std/min/max =
  0.499702/0.000040/0.499460/0.499879`. The selector is still near
  initialization, so the 20k gate should check whether it becomes more
  selective.
- 20k gate, checked 2026-05-26 00:26 CST: `latest.pk` and `best.pk` both
  at step `20000`, PSNR `27.6830`, SSIM `0.9713`. This is clearly below
  baseline by `-1.2200 dB`, LF-v1 by `-1.1733 dB`, and ResidualCalib by
  `-0.8175 dB`; selector stats remained near-constant
  (`LF_selector_mean/std/min/max =
  0.879248/0.000051/0.878806/0.879565`; `LF_alpha_mean/std/min/max =
  0.499793/0.000094/0.499178/0.499984`). Stopped by PGID `50061` before
  30k; verification showed no matching process/tmux and GPU `0 MiB / 0%`.

Gate references for this run:

| Step | Baseline | LF-v1 | ResidualCalib | Decision Rule |
| ---: | --- | --- | --- | --- |
| 10000 | `27.1101 / 0.9615` | `26.2651 / 0.9631` | `26.5666 / 0.9621` | stop only if clearly broken or selector stats are unstable |
| 20000 | `28.9030 / 0.9713` | `28.8563 / 0.9751` | `28.5005 / 0.9720` | stop if clearly below both LF-v1 and ResidualCalib and selector remains near-constant |
| 30000 | `30.1143 / 0.9776` | `30.6253 / 0.9783` | `30.3852 / 0.9782` | hard gate; continue only if close to LF-v1 or diagnostically promising |
| 50000 | `31.2384 / 0.9817` | `31.3419 / 0.9817` | `31.1396 / 0.9808` | if below baseline, stop |

## Stopped LF-v2 Haze-Aware Mask Run

- Run ID: `DEA-Net-LF-HazeAwareMask-H4K-scout100k-20260524-152758`
- Status: stopped on 2026-05-24 at the 30k hard gate; do not continue this
  exact setting to 50k or 100k.
- Remote checkout: `/root/workspace/Dehaze-Net-lf-v2-verify`
  - This is an isolated copied checkout, not a Git repository.
  - It symlinks `dataset` and `experiment` to the main server checkout.
- tmux session: `h4k_lf_v2_hazeaware_100k_20260524-152758`
- Log:
  `/root/workspace/Dehaze-Net-lf-v2-verify/experiment/HAZE4K/_run_logs/DEA-Net-LF-HazeAwareMask-H4K-scout100k-20260524-152758.log`
- Artifact dir:
  `/root/workspace/Dehaze-Net-lf-v2-verify/experiment/HAZE4K/DEA-Net-LF-HazeAwareMask-H4K-scout100k-20260524-152758/`
- Launch config verified:
  `epochs=20`, `iters_per_epoch=5000`, total `100000`, `bs=16`,
  `patch_size=256`, `w_loss_CR=0.1`, `lf_prior_injection=pre_mix`,
  `lf_conditional_mask=True`, `lf_haze_aware_mask=True`,
  `lf_haze_mask_strength=1.0`.
- Startup health check: log, run dir, TensorBoard file, process, and GPU usage
  were present; GPU showed about `12909 MiB` and `97%` utilization shortly
  after launch.

Gate rules for this run:

| Step | Baseline Ref | LF-v1 Ref | Decision Rule |
| ---: | --- | --- | --- |
| 10000 | `27.1101 / 0.9615` | `26.2651 / 0.9631` | observed `27.5613 / 0.9571`; continue to 20k, but watch SSIM |
| 20000 | `28.9030 / 0.9713` | `28.8563 / 0.9751` | observed `28.6961 / 0.9724`; continue only to 30k hard gate |
| 30000 | `30.1143 / 0.9776` | `30.6253 / 0.9783` | observed `30.1157 / 0.9770`; stop, tied with baseline but far below LF-v1 |
| 50000 | `31.2384 / 0.9817` | `31.3419 / 0.9817` | not reached |

10k checkpoint diagnostics:

- `saved_model/latest.pk` and `best.pk` are from step `10000`.
- Metric: PSNR `27.5613`, SSIM `0.9571`.
- Compared with baseline 10k: `+0.4512 dB`, `-0.0044 SSIM`.
- Compared with LF-v1 10k: `+1.2962 dB`, `-0.0060 SSIM`.
- Mask stats from `latest.pk`: mean last `0.874442`, std last `0.000480`,
  min/max last `0.871182/0.875844`; tail-200 mean std `0.000190`.
- Interpretation: PSNR and mask movement justify continuing, but SSIM is
  worse than both references; 20k must show SSIM recovery or clear diagnostic
  value.

20k checkpoint diagnostics:

- `saved_model/latest.pk` and `best.pk` are from step `20000`.
- Metric: PSNR `28.6961`, SSIM `0.9724`.
- Compared with baseline 20k: `-0.2069 dB`, `+0.0011 SSIM`.
- Compared with LF-v1 20k: `-0.1602 dB`, `-0.0027 SSIM`.
- Mask stats from `latest.pk`: mean last `0.873570`, std last `0.000883`,
  min/max last `0.867528/0.875553`; tail-50 mean std `0.000724`.
- Interpretation: mixed soft gate. It is not strong enough to continue
  unattended, but mask selectivity is clearly more active than failed
  Conditional LF. Continue to 30k hard gate, then stop unless the curve becomes
  competitive with baseline and close enough to LF-v1.

30k checkpoint diagnostics:

- `saved_model/latest.pk` and `best.pk` are from step `30000`.
- Metric: PSNR `30.1157`, SSIM `0.9770`.
- Compared with baseline 30k: `+0.0014 dB`, `-0.0006 SSIM`.
- Compared with LF-v1 30k: `-0.5096 dB`, `-0.0013 SSIM`.
- Mask stats from `latest.pk`: mean last `0.873885`, std last `0.001312`,
  min/max last `0.866170/0.876420`; tail-50 mean std `0.001059`.
- Interpretation: haze-aware mask did activate, unlike failed Conditional LF,
  but the active mask did not translate into LF-v1-level image restoration.
  Record as useful negative evidence: simple dark-channel/luma mask selection
  is not enough to fix LF-v1's low-frequency direction errors.

Stop record:

- Stopped after reading 30k checkpoint and mask stats.
- Main train PID `17335`, PGID `17326`; stopped with `kill -TERM -- -17326`.
- Follow-up checks showed no matching process/tmux and GPU memory `0 MiB`.

## Stopped Conditional LF Run

- Run ID: `DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315`
- Status: stopped on 2026-05-24 at the 30k hard gate; do not continue this
  exact setting to 50k or 100k.
- Code used for the run: commit `09880be`
- Docs were synced to both server checkouts on 2026-05-24 after command
  validation. Do not change training code on the server outside a committed
  source-sync step.
- Remote checkout: `/root/workspace/Dehaze-Net-conditional-lf`
- Original log:
  `/root/workspace/Dehaze-Net-conditional-lf/experiment/HAZE4K/_run_logs/DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315.log`
- Resume-to-30k log:
  `/root/workspace/Dehaze-Net-conditional-lf/experiment/HAZE4K/_run_logs/DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315-resume-20260524-114820.log`
- Artifact dir:
  `/root/workspace/Dehaze-Net-conditional-lf/experiment/HAZE4K/DEA-Net-LF-ConditionalMask-H4K-scout100k-20260523-224315/`
- Saved checkpoint: `saved_model/latest.pk` and `saved_model/best.pk` at step
  `30000`
- Current metric points:

| Step | PSNR | SSIM | Note |
| ---: | ---: | ---: | --- |
| 10000 | 27.1085 | 0.9638 | fair 100k run |
| 20000 | 28.8571 | 0.9724 | fair 100k run |
| 30000 | 30.1830 | 0.9783 | 30k hard gate; latest/best saved |

Gate review on 2026-05-24:

| Step | Conditional LF | vs baseline | vs LF-v1 | Decision |
| ---: | --- | --- | --- | --- |
| 10000 | `27.1085 / 0.9638` | `-0.0016 / +0.0023` | `+0.8434 / +0.0007` | pass; not broken |
| 20000 | `28.8571 / 0.9724` | `-0.0459 / +0.0011` | `+0.0008 / -0.0027` | soft pass; only continue to the 30k hard gate |
| 30000 | `30.1830 / 0.9783` | `+0.0687 / +0.0007` | `-0.4423 / ~0.0000` | stop current setting; below LF-v1 and mask remains non-selective |

The 30k checkpoint does not justify continuing to 50k or 100k. It is slightly
above the baseline 30k point, but clearly below LF-v1 30k in PSNR. Mask
diagnostics from `latest.pk` still show an almost constant mask:
`LF_mask_mean` near `0.878747`, `std` near `0.000085`, min/max about
`0.877925/0.878916`, and scalar `lf_prior.gate` near `0.014778`. Record this
as "conditional mask not active/selective enough"; LF-v1 remains the current
positive LF candidate.

Historical pause: the run was first stopped near log step 21500 by process
group after confirming `MAIN_PID=39377` and `PGID=39375`. Resume-to-30k:
tmux `h4k_lf_condmask_100k_resume_20260524_114820` stopped automatically after
detecting `step :30000`; follow-up checks showed no matching process/tmux and
GPU memory `0 MiB`.

## Baseline References

- Baseline 100k scout:
  `DEA-Net-CR-H4K-Baseline-scout-20260520-101334`
  - 10k: PSNR `27.1101`, SSIM `0.9615`
  - best step 90000: PSNR `32.2255`, SSIM `0.9844`
  - 20k: PSNR `28.9030`, SSIM `0.9713`
  - 30k: PSNR `30.1143`, SSIM `0.9776`
  - 50k: PSNR `31.2384`, SSIM `0.9817`
  - 70k: PSNR `31.7662`, SSIM `0.9826`
  - 90k: PSNR `32.2255`, SSIM `0.9844`
  - 100k: PSNR `32.0952`, SSIM `0.9844`
- LF-v1 positive candidate:
  `DEA-Net-LF-H4K-scout-20260521-003100`
  - 10k: PSNR `26.2651`, SSIM `0.9631`
  - best step 90000: PSNR `32.4281`, SSIM `0.9845`
  - 20k: PSNR `28.8563`, SSIM `0.9751`
  - 30k: PSNR `30.6253`, SSIM `0.9783`
  - 50k: PSNR `31.3419`, SSIM `0.9817`
  - 70k: PSNR `31.9069`, SSIM `0.9836`
  - 90k: PSNR `32.4281`, SSIM `0.9845`
  - 100k: PSNR `32.3857`, SSIM `0.9845`
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

```powershell
@'
set -euo pipefail
cd /root/workspace/Dehaze-Net-conditional-lf
git remote set-url origin git@github.com:onenoober/Dehaze-Net.git
ssh -T git@github.com
git fetch --dry-run origin
git fetch origin
git pull --ff-only
'@ | ssh runyun-ts "tr -d '\r' | bash -s"
```

Run `git pull --ff-only` only after confirming the target checkout and dirty
state. For a non-mutating connectivity check, stop at `git fetch --dry-run
origin`.

For exact check/pause/resume templates, use `docs/WORKFLOW.md`.

## Documentation Map

For load-on-demand rules and future writing boundaries, read
`docs/README.md`.

Most common next files:

1. `docs/WORKFLOW.md`: exact local/server/GitHub/tmux/check/pause/resume
   command templates, including fair scout command skeletons.
2. `docs/EXPERIMENT_LOG.md`: chronological metrics and decisions.
3. `docs/HAZE4K_RUN_MANIFEST.md`: artifact keep/delete policy.
4. `docs/DEA_NET_LFCR_HAZE4K_PLAN.md`: main thesis route, stage plan,
   promotion rules, and reporting evidence chain.

Keep this file short; do not paste full experiment histories here.
