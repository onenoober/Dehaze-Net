# Current Project Context

This is the short handoff page for future DEA-Net work. Keep it compact. Use
`docs/README.md` to decide which deeper document to load next. Put detailed
metrics in `docs/EXPERIMENT_LOG.md`, artifact policy in
`docs/HAZE4K_RUN_MANIFEST.md`, and method reasoning in the dated analysis docs.

## Quick Handoff

- Current active training branch/source truth:
  `codex/haze4k-crplus-v2`. The cloud CRPlus-v2 run was launched from commit
  `24085db` (`Record local CRPlus-v2 scout launch`); later doc-only commits may
  exist on the same branch, so use `git log -1 --oneline` to verify.
- Active HAZE4K training run is on `runyun-ts` in
  `/root/workspace/Dehaze-Net-audit-sync`: run
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540`, tmux
  `h4k_crplusv2_100k_20260526-225540`, log
  `experiment/HAZE4K/_run_logs/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540.log`.
- The local WSL CRPlus-v2 scout
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-192721` was cancelled on
  2026-05-26 before the first 10k checkpoint. Do not resume it.
- Current positive model evidence remains LF-v1; ResidualCalib is a positive
  ablation but not a replacement.
- Latest selector evidence says the "proxy audit before selector-v2"
  recommendation was reliable, but strict CSV, rich CSV, and activation-forward
  deployable proxies all failed the pass line. Do not launch another
  LFResidualSelector 100k scout from oracle evidence alone.
- Before any new model/loss/selector/mask scout, use
  `docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md` to write the route card and
  mechanism-specific gate metrics.
- Local WSL is the durable workspace and experiment artifact home. GitHub is
  the lightweight code/docs/small-evidence relay. `runyun-ts` is a temporary
  compute node.

## Current State

- Local workspace: `D:\Dehaze\Dehaze-Net`
- Current local WSL workspace: `/home/ubuntu/workspace/Dehaze-Net`
- Local selector-audit branch: `codex/haze4k-rich-selector-proxy-audit`
- General research-sync branch: `codex/haze4k-research-sync`
- Conditional LF code branch/run lineage: `codex/haze4k-conditional-lf`
- ResidualCalib code branch/run lineage: `codex/haze4k-lf-residual-calibration`
- ResidualDirLoss code branch/run lineage: `codex/haze4k-residual-direction-loss`
- GitHub repo: `https://github.com/onenoober/Dehaze-Net` (private)
- Server SSH alias: `runyun-ts`
- Main server checkout: `/root/workspace/Dehaze-Net`
- Conditional LF checkout: `/root/workspace/Dehaze-Net-conditional-lf`
- Clean synced source checkout: `/root/workspace/Dehaze-Net-audit-sync`
  - Current branch `codex/haze4k-research-sync`; verify the exact
    commit with `git log -1 --oneline`.
  - Purpose: Git-backed source truth for the audit/metadata/doc sync work.
  - It symlinks `dataset/HAZE4K` and `experiment` to the main server checkout
    for dry-run validation without touching the older dirty training checkouts.
- Server env: `/opt/anaconda/envs/py310/bin/python`
- Local WSL env: `/home/ubuntu/miniconda3/envs/py310`.
  Validated on 2026-05-26 with PyTorch `2.11.0+cu128`, CUDA `12.8`,
  RTX 4080 SUPER, OpenCV `4.6.0`, NumPy `1.26.4`, and Matplotlib installed.
- Local WSL stores durable experiment artifacts under ignored paths such as
  `experiment/`, `trained_models/`, and `dataset/`.
- Cloud compute outputs should be synced back to local WSL by size and value:
  small logs, metric summaries, CSV/JSON summaries, scripts, and conclusions
  can be synced when they guide route decisions; checkpoints, full inference
  folders, datasets, large image grids, arrays, and archives are synced only
  when explicitly requested.
- Local project commands should run in WSL/bash. Windows PowerShell is still
  useful for `ssh runyun-ts` because that alias lives in Windows SSH config.

## 2026-05-26 Research Sync State

- Local WSL, GitHub `origin/codex/haze4k-research-sync`, and server clean
  checkout `/root/workspace/Dehaze-Net-audit-sync` were verified at commit
  `acafee751520f2cc6db7d615f954e80e12682b20`.
- Git tracks no files under `experiment/`; datasets, checkpoints, synced
  experiment byproducts, and temporary logs remain ignored.
- Small cloud evidence from the older dirty server checkout was copied into
  local ignored storage at `experiment/sync/runyun-ts-20260526/` and extracted
  into local `experiment/HAZE4K/`.
- Large cloud artifact candidates were inventoried but not synced. Use
  `experiment/sync/runyun-ts-20260526/large-artifacts.txt` as the request list
  if a later step needs selected checkpoints, full inference folders, large
  image grids, arrays, datasets, or archives.
- The older server checkout `/root/workspace/Dehaze-Net` remains dirty on
  `codex/haze4k-lf-prior` and should not be treated as source truth. Use it
  only for salvage or artifact inspection when explicitly intended.

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
- Historical note: this ready state led to the stopped selector run below.
  It is no longer the next action. Do not launch this exact selector setting
  again.

## Stopped LF ResidualSelector Run

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

## Current Residual-Direction Loss Scale Step

- Local branch: `codex/haze4k-residual-direction-loss`
- Rationale: the ResidualSelector 20k failure makes another selector or mask
  low-value. LF-v1 and ResidualCalib evidence still points to low-frequency
  residual direction as the next useful target.
- New read-only script:
  `code/analyze_residual_direction_loss_scale.py`
- New experiment card:
  `docs/HAZE4K_RESIDUAL_DIRECTION_LOSS_SCALE_PLAN_20260526.md`
- Remote loss-scale diagnostics completed on 2026-05-26 using LF-v1 best.pk:
  - train center-crop 256 output:
    `experiment/HAZE4K/loss_scale/residual-direction-lfv1-train256-20260526/`
  - test full-image 256 output:
    `experiment/HAZE4K/loss_scale/residual-direction-lfv1-test256-20260526/`
  - `w=0.005` weighted direction loss is about `0.13%` of L1 on train crops
    and about `0.67%` of L1 on test full images.
- Training options and launcher are prepared locally with default
  `w_loss_residual_dir=0.005`.
- Remote source-sync completed on `/root/workspace/Dehaze-Net-audit-sync`;
  branch `codex/haze4k-residual-direction-loss`, commit `f316354`.
- Dry-run passed:
  `dryrun-H4K-LF-ResidualDirLoss-args-20260526`.
- 2-step smoke passed:
  `smoke-H4K-LF-ResidualDirLoss-20260526`.
  It wrote `saved_model/latest.pk` at step `2`; checkpoint `loss_log` includes
  `ResidualDir`; post-check showed GPU `0 MiB / 0%`.
- The first fair scout below failed the 30k route-specific hard gate. Do not
  resume this exact `w_loss_residual_dir=0.005` setting.

## Stopped LF ResidualDirLoss Run

- Run ID: `DEA-Net-LF-ResidualDirLoss-w005-H4K-scout100k-20260526-103853`
- Status: stopped on 2026-05-26 about 12:56 CST after the 30k hard gate and
  route-specific diagnostic review failed; do not resume this exact setting.
- Local/GitHub/remote branch: `codex/haze4k-residual-direction-loss`
- Local/GitHub/remote commit: `521392c`
- Remote checkout: `/root/workspace/Dehaze-Net-audit-sync`
- tmux session: `h4k_lf_resdir_100k_20260526-103853`
- Launch script:
  `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/_run_logs/DEA-Net-LF-ResidualDirLoss-w005-H4K-scout100k-20260526-103853.sh`
- Log:
  `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/_run_logs/DEA-Net-LF-ResidualDirLoss-w005-H4K-scout100k-20260526-103853.log`
- Artifact dir:
  `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/DEA-Net-LF-ResidualDirLoss-w005-H4K-scout100k-20260526-103853/`
- Verified launch config:
  `epochs=20`, `iters_per_epoch=5000`, total `100000`, `bs=16`,
  `patch_size=256`, `w_loss_L1=1.0`, `w_loss_CR=0.1`,
  `w_loss_residual_dir=0.005`, `residual_dir_pool=8`,
  `residual_dir_warmup_steps=0`, `start_lr=0.0001`,
  `end_lr=0.000001`, checkpoint/eval every `10000`,
  `save_epoch_checkpoints=false`, `use_lf_prior=True`,
  `lf_prior_injection=pre_mix`, no ResidualCalib, no selector, no mask,
  no TeacherGuard, no LowFreqLoss, no CRPlus.
- Startup health: tmux/process present; GPU about `13193 MiB / 83%`;
  log reached step `129/100000` shortly after launch.
- Gate metrics:
  - 10k: `25.5050 / 0.9614`; below baseline, LF-v1, and ResidualCalib on
    PSNR, but SSIM did not collapse.
  - 20k: `28.9783 / 0.9733`; PSNR recovered above the 20k references, so the
    run was allowed to continue to the 30k hard gate.
  - 30k: `30.1058 / 0.9779`; essentially baseline-level
    (`30.1143 / 0.9776`) but clearly behind LF-v1
    (`30.6253 / 0.9783`) and ResidualCalib (`30.3852 / 0.9782`).
- Route-specific diagnostic review:
  - Output dirs:
    `experiment/HAZE4K/loss_scale/residual-dir-hardgate-review-20260526/`
    and
    `experiment/HAZE4K/residual_diagnostic/residual-dir-hardgate-review-20260526/`.
  - On the 64-image test subset, direct residual-direction loss was worse:
    ResidualDirLoss-30k `0.04645` vs LF-v1-best `0.03565` and
    ResidualCalib-best `0.03330`.
  - Direct residual cosine was worse:
    ResidualDirLoss-30k `0.95355` vs LF-v1-best `0.96435` and
    ResidualCalib-best `0.96670`.
  - Relative to CR on the same 64-image subset, wrong-direction count was
    `30/64` and LF MSE improved/regressed `17/47`, worse than LF-v1
    (`18/64`, `32/32`) and ResidualCalib (`22/64`, `26/38`).
  - Even though train checkpoint `loss_log` showed `ResidualDir` falling from
    `0.29394` to `0.02585`, the test-side mechanism metrics did not improve.
- Stop verification: before stopping the log had reached about
  `35776/100000`; `latest.pk` and `best.pk` remained at 30k. Stopped by
  `kill -TERM -- -7896`, then killed the tmux session. Verification showed no
  matching process/tmux and GPU `0 MiB / 0%`.
- Gate-rule update: future gates should use PSNR/SSIM as global guardrails plus
  mechanism-specific metrics chosen for the architecture. Residual-direction
  metrics are required for this route, but should not be blindly reused for
  unrelated routes.

## Selector Proxy Audit Follow-Up

- Local branch: `codex/haze4k-rich-selector-proxy-audit`
- Read-only scripts:
  - `code/analyze_selector_proxy_learning.py`
  - `code/analyze_selector_rich_proxy_learning.py`
  - `code/analyze_selector_activation_proxy_learning.py`
- Route review docs:
  - `docs/HAZE4K_SELECTOR_PROXY_AUDIT_20260526.md`
  - `docs/HAZE4K_RICH_SELECTOR_PROXY_AUDIT_20260526.md`
  - `docs/HAZE4K_ACTIVATION_SELECTOR_PROXY_AUDIT_20260526.md`
  - `docs/HAZE4K_SELECTOR_EVIDENCE_CLOSURE_20260526.md`
- Current strict local diagnostic artifact:
  `experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-strict-20260526/`
- Audit verdict: the prior recommendation is reliable only as a diagnostic-first
  step. The LF-v1/ResidualCalib oracle headroom is real, but it is not enough to
  justify another selector run because the strongest rules are GT-aware.
- Strictness update: safe output proxy features now come only from an explicit
  whitelist and script-generated candidate-output pairwise deltas. The script
  writes `feature_lists.json` and fails if a safe feature set contains a
  non-whitelisted feature. A manual regex check over strict stump/logistic
  outputs found `0` unsafe safe-feature rows.
- Strict local held-out proxy result on the existing full-test three-way CSV:
  - best safe proxy was `metadata_proxy` ridge logistic:
    `+0.0508 dB` vs LF-v1, oracle recovery `0.0876`, residual precision
    `0.5241`;
  - `output_plus_metadata_proxy` ridge logistic: `+0.0329 dB`,
    oracle recovery `0.0556`, residual precision `0.5236`;
  - `output_proxy` ridge logistic: `+0.0268 dB`, oracle recovery `0.0459`,
    residual precision `0.5231`;
  - GT-aware leakage check recovers the oracle (`+0.5796 dB`, recovery
    `0.9986`), confirming the target is real but not inference-safe.
- Strict-audit decision at this stage: do not launch selector-v2 from strict
  proxy evidence. The later rich and activation-forward audits below closed the
  remaining deployable-proxy path.
- Rich follow-up artifact:
  `experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-rich-20260526/`
  - uses 280 metadata-free rich output/agreement features plus degradation-held
    out split families: random, airlight leave-one, beta leave-one, and
    airlight/beta combo grouped 5-fold;
  - best random metadata-free rich row: `rich_output_proxy` ridge logistic,
    `+0.0956 dB`, oracle recovery `0.1643`, precision `0.6046`;
  - best held-out family rows: airlight `+0.0695 dB`, beta `+0.0755 dB`,
    combo grouped `+0.0656 dB`; all fail the pass line
    (`+0.12 dB`, recovery `0.20`, precision `0.65`);
  - GT-aware leakage still recovers the target, so the selector target exists
    but current inference-safe CSV-derived proxy evidence remains insufficient.
- Activation-forward artifact:
  `experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-activation-20260526/`
  - forwards the frozen CR baseline, LF-v1, and ResidualCalib best checkpoints
    over the full 1000-image HAZE4K test set and extracts inference-safe
    hazy-input, common activation, LF prior, ResidualCalib alpha/direction, and
    activation-disagreement features;
  - conclusive sample-size check passed: `1000` images, selector target
    positives/negatives `509/491`, minimum split class count `55`;
  - feature counts: activation-only `1418`, activation+strict output `1481`,
    activation+rich output `1698`;
  - best metadata-free activation rows still failed: random `+0.0748 dB`,
    recovery `0.1247`, precision `0.5978`; airlight `+0.0585/0.0922/0.5763`;
    beta `+0.0392/0.0630/0.6086`; combo grouped
    `+0.0544/0.0883/0.5855`;
  - manual safe-feature regex check found `0` leakage-like safe features, while
    GT-aware leakage still recovered near oracle (`+0.5796 dB`, recovery
    `0.9986`).
- Current selector-v2 decision: do not train. Strict, rich, and activation
  proxy audits all fail the predeclared pass line. Stop selector/structure
  search unless a future route changes the problem with an explicit
  supervised/distilled selector target and passes a fresh full-sample proxy
  audit first.
- Sample-size rule for future selector/proxy claims: use the full available
  evaluation set when feasible. Small fixed subsets are smoke/debug evidence
  only and must not be used to justify a training route without a predeclared,
  scientifically adequate sample-size rationale.

## Current CRPlus-v2 Frequency Curriculum Step

- Local branch: `codex/haze4k-crplus-v2`.
- Rationale: LF-v1 remains the positive candidate, but selector, mask,
  ResidualDirLoss, simple LowFreqLoss, and CRPlus-P1 low-pass negative evidence
  now make another LF structure run low-value. The next useful route is a
  no-inference-cost training loss that is frequency-aware and curriculum-safe.
- New read-only script:
  `code/analyze_crplus_v2_loss_scale.py`.
- New route card:
  `docs/HAZE4K_CRPLUS_V2_FREQ_CURRICULUM_PLAN_20260526.md`.
- New launchers:
  - `scripts/runyun-haze4k-crplus-v2-scale-diagnostic.sh`
  - `scripts/runyun-haze4k-crplus-v2-scout.sh`
- Local 64-image train center-crop scale diagnostic:
  `experiment/HAZE4K/loss_scale/crplus-v2-baseline-train64-20260526/`.
  Using the CR baseline best checkpoint, selected combined ratio loss was
  `0.189611`; weighted ratios to L1 were `0.0197/0.0592/0.0987/0.1975` for
  weights `0.001/0.003/0.005/0.01`. Absolute combined margin at `0.02` was
  near-zero (`0.000141`), so CRPlus-v2 is ratio-first and margin-diagnostic.
- Implemented default-off training support with `--w_loss_crplus_v2` and
  related CRPlus-v2 options. The first scout default is
  `w_loss_crplus_v2=0.003`.
- Local validation passed:
  - `python -m py_compile` over modified training and diagnostic files.
  - dry-run `dryrun-H4K-CRPlusV2-args-20260526`.
  - 2-step smoke `smoke-H4K-CRPlusV2-20260526`, checkpoint step `2`;
    `loss_log` includes `CRPlusV2` with tail `[1.4913553, 1.4035805]`.
- Branch `codex/haze4k-crplus-v2` was pushed to GitHub. The cloud run below was
  launched from code/docs commit `24085db`; later doc-only commits may update
  this context without changing the launched training code.
- Stopped local fair scout:
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-192721`.
  - tmux: `h4k_crplusv2_100k_20260526-192721`
  - log:
    `experiment/HAZE4K/_run_logs/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-192721.log`
  - launch env:
    `experiment/HAZE4K/_run_logs/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-192721.local_launch.env`
  - fair config: `epochs=20`, `iters_per_epoch=5000`, total `100000`,
    `bs=16`, `patch_size=256`, `w_loss_CR=0.1`,
    `w_loss_crplus_v2=0.003`, eval/checkpoint every `10000`,
    `save_epoch_checkpoints=false`, no LF prior or other auxiliary routes.
  - status: cancelled on 2026-05-26 after the cloud server became available;
    tmux/process were absent after stop and local GPU returned to idle.
  - final local evidence: log reached about step `1802/100000`; no 10k eval or
    checkpoint exists, only launch/loss/TensorBoard byproducts.
- Active cloud fair scout:
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540`.
  - remote checkout: `/root/workspace/Dehaze-Net-audit-sync`, branch
    `codex/haze4k-crplus-v2`, launch commit `24085db`.
  - tmux: `h4k_crplusv2_100k_20260526-225540`
  - log:
    `experiment/HAZE4K/_run_logs/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540.log`
  - fair config: `epochs=20`, `iters_per_epoch=5000`, total `100000`,
    `bs=16`, `patch_size=256`, `w_loss_CR=0.1`,
    `w_loss_crplus_v2=0.003`, eval/checkpoint every `10000`,
    `save_epoch_checkpoints=false`, no LF prior or other auxiliary routes.
  - startup health: `runyun-ts` reachable over Tailscale; tmux/process present;
    RTX 5090 GPU about `16483 MiB / 98%`; log reached step `247/100000` at
    about `3.3` steps/s during verification.
  - First gate is the 10k sanity gate. Use matched baseline 10k
    `27.1101 / 0.9615` as the reference and stop on a CRPlus-P1-like collapse.

## Model-Change Protocol

Future HAZE4K model changes should be written up in
`docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md` before a long scout starts. The route
card must name:

- the failure mode being targeted;
- the mechanism hypothesis;
- the exact code/loss change;
- the route-specific mechanism metrics;
- the gate rules for 10k/20k/30k/50k;
- the full-test artifact required before claiming success.

For this repo, PSNR/SSIM are necessary guardrails, but they are not enough on
their own. The mechanism metrics must match the route under test.

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
