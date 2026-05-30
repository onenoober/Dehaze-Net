# Current Project Context

This is the compact handoff page for future DEA-Net work. Read this first, then
use `docs/README.md` to choose the one or two deeper documents needed for the
task. Do not paste full experiment histories here; put run facts in
`docs/EXPERIMENT_LOG.md`, artifact policy in `docs/HAZE4K_RUN_MANIFEST.md`, and
route reasoning in the dated analysis docs.

## Quick Handoff

- Current HAZE4K official-checkpoint experiment state: no new official-PSNR
  training is authorized. The `HAZE4K official PSNR marginal improvement`
  route is paused/closed unless a fresh audit first finds a deployable
  no-change discriminative variable that passes strong-official/no-change,
  CR-strength, A/beta, content-cluster, and residual-low-energy held-out gates.
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
- Current user instruction for missing cloud dependencies: install required
  dependencies directly without asking for confirmation, then record durable
  server facts in `docs/CORE_SERVER_RUNBOOK.md` when useful for future runs.
- Current user instruction for in-flight runs: after a training, evaluation,
  or audit run has started, do not privately change its configuration, reduce
  its scope, swap probe/model heads, change feature sets/splits, stop it, or
  launch a reduced replacement unless the user explicitly approves the change
  first.
- Completed LFCR-v2 decay scout on `runyun-ts`:
  `DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455` in
  `/root/workspace/Dehaze-Net-audit-sync`, branch
  `codex/haze4k-lfcr-v2-decay` commit `9009515`. The fair 100k run completed
  with best/final checkpoint at step `100000`, PSNR/SSIM `32.1516 / 0.9844`;
  independent full-test verification gave `32.1518 / 0.9844`, LF gate
  `0.019099`. Compact final diagnostics are synced locally under
  `experiment/HAZE4K/lfcr_v2_decay_diagnostics/DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455-final-20260528-verify`
  (`5.2M`, no checkpoints). Conclusion: negative/neutral fair ablation. The
  mechanism was partly valid because the schedule turned off and some LF-v1
  regression rescue remained, but quality stayed below CR best, LF-v1,
  ResidualCalib, CRPlus-v2, and LFCR-v1 final. Do not promote or resume this
  exact schedule.
- Completed WaveletPreserve preflight audits on `runyun-ts`:
  `HAZE4K-wavelet-preserve-proxy-runyun-20260528-hazy` and
  `HAZE4K-wavelet-preserve-activation-proxy-runyun-20260528` in
  `/root/workspace/Dehaze-Net-audit-sync`. Both are diagnostic only and both
  returned `do_not_train_wavelet_preserve_yet`. Hazy-only wavelet features had
  some signal on the LF-v1 preserve/intervene target, but missed the pass line
  because random-split intervene precision was only `0.5541` against the
  required `0.60`, with airlight/beta held-out precision lower. Adding frozen
  CR/LF-v1 activation features worsened the primary target (`hazy_wavelet_plus_activation`
  random balanced accuracy `0.5360`, intervene precision `0.4757`). Do not
  launch a WaveletPreserve architecture scout from this evidence; only reopen
  with a changed supervised/distilled preservation target and a fresh
  full-sample proxy audit. Details:
  `docs/HAZE4K_WAVELET_PRESERVE_PROXY_AUDIT_20260528.md`.
- Completed supervised/distilled preserve-target preflight on `runyun-ts`:
  `HAZE4K-supervised-preserve-proxy-runyun-20260528-train-p4-sklearn-liblinear`
  in `/root/workspace/Dehaze-Net-audit-sync`, using HAZE4K train teacher
  labels from CR and LF-v1 best checkpoints. `scikit-learn==1.7.2` was
  installed in runyun `/opt/anaconda/envs/py310` for the final solver audit.
  The decisive patch target had `10055` samples (`4947` preserve, `5108`
  intervene). Best reliable `hazy_wavelet_plus_teacher_outputs` sklearn
  logistic random-image split failed the pass line: balanced accuracy
  `0.5889`, preserve recall `0.5951`, intervene precision `0.5934`,
  strong-CR regression intervene recall `0.5347`, despite positive simulated
  gain `+0.1955 dB`. A small MLP random-only check got closer but still missed
  preserve recall and strong-CR recall. Do not launch a supervised preserve
  head or WaveletPreserve scout from this target. Details:
  `docs/HAZE4K_SUPERVISED_PRESERVE_PROXY_AUDIT_20260528.md`.
- Completed residual-field confidence preflight on `runyun-ts`:
  `HAZE4K-residual-field-confidence-preflight-runyun-20260528-full` in
  `/root/workspace/Dehaze-Net-audit-sync`, branch
  `codex/haze4k-residual-field-confidence` commit `cbdb1c4`. The audit used
  HAZE4K train split, CR and LF-v1 best checkpoints at step `90000`, `3000`
  images, and `12000` patches. `scikit-learn==1.7.2` was already installed on
  runyun. The main continuous-confidence row
  `hazy_wavelet_plus_teacher_outputs` + `sklearn_hgb` had high simulated gain
  (`+0.8356 dB` vs LF-v1, recovery `0.7304`) and some confidence correlation
  (`0.3751`), but failed the pass line because LF-v1 gain preserve recall was
  only `0.6275 < 0.68` and intervention precision was only `0.5320 < 0.60`.
  Held-out airlight/beta rows also failed. Recommendation:
  `do_not_train_residual_field_confidence_yet`. Do not launch the prepared
  CR-reference residual-field 100k scout from this evidence. Details:
  `docs/HAZE4K_LF_RESIDUAL_FIELD_CONFIDENCE_PLAN_20260528.md`.
- Completed Depth-Guided LF-v1 preflight on `autodl-dehaze`:
  `HAZE4K-depth-guided-lf-preflight-autodl-20260529-204105` in
  `/root/autodl-tmp/workspace/Dehaze-Net`, branch
  `codex/haze4k-depth-guided-lf-preflight` commit `88cc498`. The audit used
  HAZE4K train split, CR and LF-v1 best checkpoints at step `90000`, `3000`
  images, `12000` patches, and `depth-anything/Depth-Anything-V2-Small-hf`
  cached relative depth maps. AutoDL needed `HF_ENDPOINT=https://hf-mirror.com`
  because direct `huggingface.co` access timed out. Main row
  `hazy_depth_plus_teacher_outputs` + `sklearn_hgb` had high simulated gain
  (`+0.8347 dB`) but failed the hard gates: preserve recall `0.6342 < 0.68`,
  intervention precision `0.5255 < 0.60`, and confidence correlation
  `0.4035 < 0.45`; airlight/beta held-out rows also failed. Shuffled-depth
  control was very close (`+0.8479 dB`, preserve `0.6273`, precision
  `0.5352`, corr `0.3968`), so depth did not add a reliable new signal.
  Recommendation: `do_not_train_depth_guided_lf_yet`. Do not launch
  Depth-Guided LF-v1 training from this evidence. Details:
  `docs/HAZE4K_DEPTH_GUIDED_LF_PREFLIGHT_PLAN_20260529.md`.
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
- LF-v2 multiscale bottleneck refiner cold-start route ran on
  `autodl-dehaze` because `runyun-ts` became unreachable before its result
  could be checked. Branch `codex/haze4k-lf-v2-mbr` is at commit `966d041`.
  AutoDL preflight passed on 2026-05-29 CST in
  `/root/autodl-tmp/workspace/Dehaze-Net`: static preflight reported parameter
  overhead `0.04395%`, latency overhead `1.3762%`, neutral-init max diff
  `0.0`, finite branch/backward stats; HAZE4K smoke wrote step `2` and
  recommended `preflight_passed_launch_allowed`. Replacement fair 100k scout
  was
  `DEA-Net-LF-v2-MBR-c8-H4K-scout100k-autodl-20260529-144500`, log
  `experiment/HAZE4K/_run_logs/DEA-Net-LF-v2-MBR-c8-H4K-scout100k-autodl-20260529-144500.log`,
  run dir
  `experiment/HAZE4K/DEA-Net-LF-v2-MBR-c8-H4K-scout100k-autodl-20260529-144500`.
  It is a from-scratch run using the formal 100k HAZE4K protocol with
  `--use_lf_prior --lf_multiscale_refiner --lf_mbr_channels 8
  --lf_mbr_pool_sizes 4,8,16`; it was stopped at the 30k hard gate on
  2026-05-29 after 10k `26.4345/0.9616`, 20k `28.2414/0.9730`, and 30k
  `29.8938/0.9776`. The 30k result is below CR baseline 30k `30.1143/0.9776`
  and LF-v1 30k `30.6253/0.9783`; checkpoint stats showed the MBR branch alive
  (`LF_mbr_std` tail about `0.203`) but the LF scalar gate was negative
  (`-0.02085`), so there was no clear mechanism reason to spend the rest of
  100k. Earlier runyun launch
  `DEA-Net-LF-v2-MBR-c8-H4K-scout100k-20260528-222223` failed before
  checkpoint, and runyun run `...20260528-223014` is currently
  unreachable/unknown; do not use either as evidence.
- New isolated warm-start fine-tuning route:
  `docs/HAZE4K_OFFICIAL_WARMSTART_FINETUNE_PLAN_20260528.md` on branch
  `codex/haze4k-official-warmstart-finetune`. This route starts from the
  official HAZE4K `.pth` via a converted step-0 training checkpoint and must
  not be mixed into cold-start fair-candidate tables. Its default staged
  fine-tune schedule is LF-only through the 10k gate, then LF plus
  bottleneck/fusion through the 30k gate, then full-model tiny-LR unfreeze.
- Completed local warm-start LF-v1 scout:
  `DEA-Net-OfficialWarmStart-LFv1-H4K-local-scout100k-20260528-152808` in
  local WSL `/home/ubuntu/workspace/Dehaze-Net`; no tmux/train process remains
  as of 2026-05-29 CST. Best checkpoint is step `90000`, PSNR/SSIM
  `34.2662 / 0.9886`; final step `100000` is `34.2631 / 0.9886`. Independent
  full-test per-image comparison against matched step-0 official-equivalent
  checkpoint gave step0 `34.2548 / 0.9885`, best90k `34.2661 / 0.9886`, mean
  delta `+0.0114 dB`, better/worse by PSNR `484/516`, and LF gate
  `-0.01404`. This is a tiny positive isolated warm-start result, not evidence
  for the cold-start fair-candidate table. Diagnostics live under
  `experiment/HAZE4K/official_warmstart_diagnostics/DEA-Net-OfficialWarmStart-LFv1-H4K-local-scout100k-20260528-152808-best-vs-step0-20260529`.
- Selector evidence is closed for now: strict CSV, rich CSV, and
  activation-forward deployable proxies all failed the pass line. Do not launch
  another LFResidualSelector 100k scout from oracle evidence alone.
- Before any new model/loss/selector/mask/guard scout, write or update a route
  card using `docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md`. The route must satisfy the
  most-valuable-attempt standard: highest route-decision value per training
  cost, with cheap preflight evidence, an earliest decisive gate, speed metrics,
  and mechanism metrics.
- Completed CBRFRC-v1 scout on `autodl-dehaze`:
  `DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811` in
  `/root/autodl-tmp/workspace/Dehaze-Net`, branch
  `codex/haze4k-cbrfrc-v1` commit `4634b13`, tmux `cbrfrc_v1_100k`, log
  `experiment/HAZE4K/_run_logs/DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811.log`.
  The route card is `docs/HAZE4K_CBRFRC_V1_PLAN_20260530.md`. Preflight passed
  only with `brf_preserve_warmup_steps=3000` and
  `brf_preserve_gate_weight=0.0`: full 1000-image zero-gate identity had
  PSNR/SSIM exactly matching CR J0 (`32.22545 / 0.984417`, delta `0.0`),
  bounded LF oracle headroom was `+10.6562 dB` with `0/250` strong-regression
  cases, and the fixed-patch 2k micro-overfit loss fell `0.032217 -> 0.023870`
  with residual cosine `0.0 -> 0.410177` and final `gate_lf_mean=0.007157`.
  The fair 100k scout completed on 2026-05-30: best checkpoint is step `10000`
  with `32.2237 / 0.9844`, final step `100000` is `32.1977 / 0.9844`, both
  below CR best `32.2255` and LF-v1 best `32.4281`. Full 1000-image BRF
  diagnostics for the best checkpoint are synced locally under
  `experiment/HAZE4K/brf_diagnostics/DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811-best-20260530-0830`:
  `delta_brf_vs_cr=-0.0007`, `delta_brf_vs_lfv1=-0.2035`,
  wrong-direction `557/1000`, mean residual cosine `-0.0506`, LF MSE
  improved/regressed `443/557`, LF-v1 gain preservation `50.1%`, and
  `gate_lf_mean=0.0327`. Conclusion: mechanism failed; do not run Stage C,
  joint fine-tune, or another CBRFRC-v1 variant without a changed target or
  loss design.
- Completed BRFRC-v2 representation Stage 0 audit on `autodl-dehaze`:
  `HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028` in
  `/root/autodl-tmp/workspace/Dehaze-Net`, branch
  `codex/haze4k-brfrc-v2-representation-audit` commit `a441148`. The audit
  used HAZE4K train split, CR and LF-v1 best checkpoints at step `90000`,
  `3000` images, and frozen CR/LF-v1 bottleneck/decoder features. Internal
  ridge probes beat output-only and shuffled controls on random residual
  direction and LF-v1 preservation, but every main feature row failed the
  written Stage 0 pass line because strong-CR preservation stayed far below
  `0.70` (`0.4081` for CR features, `0.4136` for LF-v1 features, `0.4473`
  for feature contrast) and CR-strength held-out rows collapsed. Recommendation:
  `do_not_train_brf_v2_representation_yet`. Do not implement BRFRC-v2-Rep or
  launch a 100k scout from this evidence. Compact local artifacts are under
  `experiment/HAZE4K/brfrc_v2_representation_audit/HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028`;
  large feature matrices remain on AutoDL only.
- Completed Strong-CR Abstention / Residual-SNR audit on `autodl-dehaze`:
  `HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full` in
  `/root/autodl-tmp/workspace/Dehaze-Net`, branch
  `codex/haze4k-strong-cr-abstention-snr-audit` commit `c436abe`. The audit
  used HAZE4K test split, existing CR/LF-v1/ResidualCalib/CBRFRC-v1
  checkpoints, and committed CRPlus-v2 route-evidence CSV metrics because the
  CRPlus-v2 checkpoint was not present on AutoDL. Recommendation:
  `do_not_train_abstention_brf_v3_yet`. Best deployable random row
  (`E_abstention_risk` + logistic) had intervention precision `0.6631` and
  shuffled gap `0.2578`, but missed strong-q4 preserve recall
  `0.7385 < 0.75`, strong false-intervention `0.2615 > 0.20`, LF-v1 gain
  preservation `0.5952 < 0.70`, and confidence correlation `0.3498 < 0.45`.
  CR-strength held-out strong-q4 collapsed (`0.2781` preserve recall,
  `0.7219` false intervention). SNR readouts support the "small residual"
  part but not high sign instability: strong-q4 target residual norm is only
  `0.2583` of weak-q1, while strong-q4 sign-flip rate is `0.0570`. Do not
  implement Abstention-First BRFRC-v3 or launch a 100k scout from this audit.
  Compact local artifacts are under
  `experiment/HAZE4K/strong_cr_abstention_snr_audit/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full`;
  GitHub-readable text package:
  `docs/ai_text_packages/20260530-strong-cr-abstention-snr-audit`.
- Completed Official-Centric Marginal Gain Audit on `autodl-dehaze`:
  `HAZE4K-official-marginal-gain-audit-autodl-20260530-hgb-full` in
  `/root/autodl-tmp/workspace/Dehaze-Net`, branch
  `codex/haze4k-official-marginal-gain-audit` commit `70267ca`. The audit
  used the 1000-image HAZE4K test join with official step0, official
  warm-start best, cold-start CR, LF-v1, ResidualCalib, committed CRPlus-v2
  route-evidence CSV metrics, and CBRFRC-v1 best, then ran the full HGB
  feature-set sweep over margins `0.15/0.20`. Recommendation:
  `do_not_train_official_adapter_yet`. Official oracle headroom is real
  (`34.2556 -> 34.9015`, mean `+0.6459 dB`; candidate gains `>=0.15 / 0.20 dB`
  on `523 / 493` images), but no deployable row passed (`0/448` HGB summary
  rows). The best eligible random deployable row (`A_official_output` /
  `C_official_internal`, margin `0.20`) only reached precision `0.6597`,
  strong-official no-change false intervention `0.4499`, all no-change false
  intervention `0.6050`, and shuffled gap `0.0462`. Leakage rows can separate
  the label but are not deployable. Do not write an official-default tiny
  adapter card or launch 100k official-PSNR training from this audit. Remote
  artifact:
  `experiment/HAZE4K/official_marginal_gain_audit/HAZE4K-official-marginal-gain-audit-autodl-20260530-hgb-full`.

## Storage And Server Roles

- Local WSL workspace: `/home/ubuntu/workspace/Dehaze-Net`.
- Local Windows workspace reference: `D:\Dehaze\Dehaze-Net`.
- GitHub repo: `https://github.com/onenoober/Dehaze-Net` (public as of
  2026-05-30). GitHub
  should carry code, docs, scripts, compact metrics, conclusions, and curated
  text-only AI project packages under `docs/ai_text_packages/`. When
  experiment-derived small text evidence is packaged for AI analysis, commit
  and push it to GitHub, audit source/local/remote parity, and record the audit
  in `docs/HAZE4K_RUN_MANIFEST.md` before calling the sync complete.
- Local WSL is the durable artifact home. Keep datasets, checkpoints, logs,
  plots, previews, and synced evidence under ignored paths such as
  `dataset/`, `trained_models/`, and `experiment/`. The only Git exception is
  a curated text-only copy under `docs/ai_text_packages/`; do not add raw
  experiment directories.
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
| Official-centric marginal gain route | Paused/closed. Completed full HGB audit on the 1000-image official join: oracle-with-no-change headroom is `+0.6459 dB`, but `0/448` deployable rows passed random plus official/CR-strength, airlight, beta, content-cluster, and residual-low-energy held-outs. Do not train an official adapter or continue ordinary official PSNR marginal tweaks unless a new deployable no-change variable is found first. | `docs/HAZE4K_OFFICIAL_MARGINAL_GAIN_AUDIT_20260530.md` |
| ResidualDirLoss | First `w_loss_residual_dir=0.005` fair scout failed the 30k hard gate and should not be resumed. | `docs/HAZE4K_RESIDUAL_DIRECTION_LOSS_SCALE_PLAN_20260526.md` |
| CRPlus-v2 | Completed first fair scout at 100k: `32.3633 / 0.9847`. It is positive versus CR baseline (`+0.1396 dB` full-test mean delta), but below LF-v1 (`-0.0633 dB`) and ResidualCalib (`-0.0286 dB`) in PSNR while slightly higher in SSIM. Treat as a positive CR-only component candidate, not an LF-v1 replacement. | `docs/HAZE4K_CRPLUS_V2_FREQ_CURRICULUM_PLAN_20260526.md` |
| LFCR-v1 w0.005 | Completed first high-upside combination scout at 100k: `32.2098 / 0.9844`, LF gate `0.0201`. Full diagnostics show LFCR vs LF-v1 mean `-0.2178 dB`, better/worse `461/539`; it improves `182/351` LF-v1 regression cases by at least `0.30 dB`, but loses at least `0.30 dB` on `264/453` LF-v1 gain cases. Treat as proof that CRPlus helps early/rescue behavior but is harmful as a full-run constant high weight. | `docs/HAZE4K_LFCR_V1_COMBINATION_PLAN_20260527.md` |
| LFCR-v2 decay | Completed fair 100k scout: final/best `32.1516 / 0.9844`, independent verify `32.1518 / 0.9844`, LF gate `0.019099`. Negative/neutral ablation: schedule-off mechanism partly worked and rescue cases remained, but final quality was below CR best, LF-v1, ResidualCalib, CRPlus-v2, and LFCR-v1 final. | `docs/HAZE4K_LFCR_V2_DECAY_PLAN_20260528.md` |
| Preserve/proxy family | WaveletPreserve, wavelet+activation, supervised preserve, and continuous confidence preflights all found some signal but failed preservation or intervention precision. Current feature/proxy family is not safe for a 100k scout. | `docs/HAZE4K_WAVELET_PRESERVE_PROXY_AUDIT_20260528.md`, `docs/HAZE4K_SUPERVISED_PRESERVE_PROXY_AUDIT_20260528.md`, `docs/HAZE4K_LF_RESIDUAL_FIELD_CONFIDENCE_PLAN_20260528.md` |
| ResidualFieldConfidence preflight | Continuous confidence has signal but is not safe enough: main random split `+0.8356 dB` simulated gain and `0.3751` confidence correlation, but preserve recall `0.6275` and intervention precision `0.5320` miss the pass line. Do not train LF-RFC v1 from this target. | `docs/HAZE4K_LF_RESIDUAL_FIELD_CONFIDENCE_PLAN_20260528.md` |
| Depth-Guided LF preflight | Frozen relative depth did not clear the stricter pass line: main random split `+0.8347 dB` simulated gain, but preserve recall `0.6342`, intervention precision `0.5255`, and confidence correlation `0.4035` failed; airlight/beta held-out rows failed and shuffled-depth control was too close. Do not train Depth-Guided LF-v1 from this target. | `docs/HAZE4K_DEPTH_GUIDED_LF_PREFLIGHT_PLAN_20260529.md` |
| Route evidence review / next route | Local aggregation across CR, LF-v1, ResidualCalib, CRPlus-v2, and LFCR-v1 confirms LF-v1 remains the best standalone mean-PSNR route, but all-five GT oracle reaches `33.5098` mean PSNR (`+1.0815 dB` over LF-v1). Later scoped LF-v2 MBR and Depth-Guided LF preflights/runs did not clear their gates, so no new cold-start LF route is currently authorized without a fresh route card and preflight. | `docs/HAZE4K_ROUTE_EVIDENCE_REVIEW_20260528.md`, `docs/HAZE4K_LF_V2_MULTISCALE_BOTTLENECK_REFINER_PLAN_20260528.md`, `docs/HAZE4K_DEPTH_GUIDED_LF_PREFLIGHT_PLAN_20260529.md` |

## Do Not Do

- Do not resume cancelled local CRPlus-v2 or failed ResidualDirLoss,
  ResidualSelector, Conditional LF, or Haze-Aware Mask runs.
- Do not launch selector-v2 from oracle evidence alone.
- Do not launch LF ResidualFieldConfidence / CR-reference residual-field 100k
  scout from the 2026-05-28 preflight; it failed preservation and intervention
  precision gates despite high simulated gain.
- Do not launch WaveletPreserve, supervised preserve-head, or another
  preserve/intervene proxy scout from the 2026-05-28 preflights.
- Do not launch Depth-Guided LF-v1 training from the 2026-05-29 depth preflight;
  frozen relative depth failed preservation, precision, correlation, and
  held-out stability gates, and shuffled-depth control was too close.
- Do not spend another 100k on LFCR weight/decay scheduling unless a new
  selectivity or representation mechanism is written and preflighted.
- Do not implement or launch BRFRC-v2-Rep training from the 2026-05-30 Stage 0
  representation audit; it found real internal feature signal, but failed the
  strong-CR preservation and CR-strength held-out gates.
- Do not implement or launch Abstention-First BRFRC-v3 from the completed
  2026-05-30 Strong-CR Abstention / Residual-SNR audit; deployable rows failed
  the no-change, LF-v1 preservation, confidence, and CR-strength held-out
  gates.
- Do not implement an official-default tiny adapter or launch new 100k
  official-PSNR training from the completed 2026-05-30 Official-Centric
  Marginal Gain Audit; headroom exists, but no deployable row passed the
  no-change and held-out gates.
- Do not reopen the HAZE4K official PSNR marginal improvement route with
  ordinary residual/selector/adapter/structure tweaks. Reopen only after a
  fresh deployable no-change variable passes the official-failure safety gates.
- Do not treat short-horizon smoke, dry-run, or changed-LR-horizon resumes as
  fair candidate evidence.
- Do not edit source directly on cloud servers for experiment variants. Make
  source changes locally, then commit/push/sync intentionally.
- Do not add files under `experiment/`, `dataset/`, or `trained_models/` to Git,
  except by copying curated text-only AI package evidence into
  `docs/ai_text_packages/` and committing that tracked copy.

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
7. `docs/ai_text_packages/`: GitHub-readable text-only project packages for
   external AI review.
8. `docs/DEA_NET_LFCR_HAZE4K_PLAN.md`: thesis route, stage plan, promotion
   rules, and reporting evidence chain.
9. `docs/HAZE4K_ROUTE_EVIDENCE_REVIEW_20260528.md`: current route evidence
   matrix, oracle headroom, failure modes, Pareto view, and next decision tree.
10. `docs/HAZE4K_BRFRC_V2_REPRESENTATION_AUDIT_PLAN_20260530.md`:
   completed Stage 0 representation audit; failed preservation/held-out gates,
   so no BRFRC-v2-Rep implementation or 100k scout is authorized from it.
11. `docs/HAZE4K_STRONG_CR_ABSTENTION_RESIDUAL_SNR_AUDIT_20260530.md`:
   completed diagnostic-only abstention/SNR route; failed deployable and
   CR-strength held-out gates, so no Abstention-First BRFRC-v3 implementation
   or 100k scout is authorized from it.
12. `docs/HAZE4K_OFFICIAL_WARMSTART_FINETUNE_PLAN_20260528.md`: isolated
   official-weight warm-start fine-tuning route, conversion script, gates, and
   DEA-Net reproduction pitfalls.
