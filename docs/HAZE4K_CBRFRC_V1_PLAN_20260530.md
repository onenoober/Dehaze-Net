# DEA-Net-CBRFRC-v1 Route Card

Date: 2026-05-30

Status: implementation route card. Do not launch the fair 100k scout until the
static checks, zero-gate identity check, bounded-oracle residual check, and
1k/2k micro-overfit check pass.

## Most Valuable Attempt

- Why this is the most valuable current attempt: LF-v1 remains the best
  standalone cold-start evidence, but its residual-direction diagnostics show
  wrong-direction and strong-baseline regression failures. CBRFRC tests a
  sharper causal mechanism instead of adding another selector or high-capacity
  refiner.
- Cheap preflight evidence: prior LF-v1 residual diagnostics reported
  wrong-direction count `160`, LF MSE improved/regressed `545/455`, and strong
  correlation between PSNR delta and residual cosine. The CBRFRC preflight must
  add zero-gate identity, bounded-oracle residual headroom, and micro-overfit
  evidence before training.
- Earliest decisive gate: 10k mechanism sanity. If the gate is dead, residual
  cosine does not rise, or LF MSE regressions look like LF-v1 again, stop.
- Expected training-time or attempt-count saving: a failed CBRFRC scout should
  decide whether baseline-relative applied residual learning is worth pursuing,
  instead of spending more runs on baseline-unaware LF branches.
- What success decides: baseline-relative applied residual correction reduces
  wrong-direction cases and strong-CR regressions while preserving LF-v1 gain
  cases.
- What failure decides: if applied residual supervision still regresses strong
  CR cases, the LF-v1 root cause is not solved by changing the residual anchor
  alone; future work should not simply add capacity to this corrector.
- Why a cheaper diagnostic is not enough: oracle and micro-overfit checks can
  show headroom and trainability, but only a fair 100k run can test whether
  preservation, color stability, and gain preservation hold across the full
  training horizon.

## Hypothesis

- Prior evidence: CR baseline is a strong stable reference; LF-v1 improves mean
  PSNR but hurts a meaningful strong-baseline subset. Later preserve/proxy and
  LFCR variants did not safely retain LF-v1 gains.
- Target failure mode: baseline-unaware LF correction predicts a hazy-to-clear
  low-frequency residual without knowing whether the CR output already solved
  the image.
- Mechanism hypothesis: if the corrector receives frozen CR output `J0` and
  learns the bounded applied residual `LP(GT)-LP(J0)`, residual cosine and LF
  MSE deltas should improve because the correction target is counterfactual and
  baseline-relative.

## Change

- Code branch: `codex/haze4k-cbrfrc-v1`.
- Primary variable: replace baseline-unaware LF correction with output-level
  Counterfactual Baseline-Relative Frequency Residual Corrector.
- Architecture/loss definition:
  - Frozen DEA-Net-CR baseline produces `J0`.
  - CBRFRC input includes hazy image, `J0`, `I-J0`, Laplacian low-pass bands,
    and optional haze cues.
  - Applied residuals are bounded:
    `C = gate * max_residual * tanh(raw)`.
  - Output is `J = clamp(J0 + C_lf + C_color + C_hf, 0, 1)`.
  - Loss supervises `LP(J)-LP(J0)` against `LP(GT)-LP(J0)`, not raw residuals.
- Enabled flags for the first scout: `--use_brf_frequency_corrector`,
  `--brf_freeze_baseline true`, `--brf_use_baseline_detach true`,
  `--brf_pyramid_type laplacian`, `--brf_hidden_channels 16`,
  `--brf_max_residual 0.08`, `--brf_max_color_residual 0.04`,
  `--brf_max_hf_residual 0.03`, `--brf_hf_scale 0.1`,
  `--brf_preserve_highfreq true`.
- Explicitly disabled related mechanisms for Run 1: no LF-v1 branch, no
  TeacherGuard, no selector-v2, no CRPlus-v2, no Mamba/DWT, no joint baseline
  fine-tuning, no haze prior unless Run 1 clears the 30k route-validity gate.

## References

- Baseline run/checkpoint: DEA-Net-CR best checkpoint, preferably best 90k
  training checkpoint used by existing CR reference diagnostics.
- Direct predecessor run/checkpoint: LF-v1 best checkpoint and existing
  ResidualCalib/CRPlus-v2/LFCR-v1 full-test CSVs for final comparisons.
- Matched gate reference table: use current `docs/CURRENT_CONTEXT.md` values
  and the existing per-image route evidence matrix until refreshed at gate time.

## Mechanism Metrics

| Metric | Why it matches this route | Gate subset | Full-test artifact |
| --- | --- | --- | --- |
| residual cosine | Tests whether applied residual points toward `GT-J0`. | 10k+ | `diagnose_brf_residual_direction.py` CSV |
| wrong-direction count | Directly targets LF-v1 residual direction failure. | 10k+ | summary JSON |
| LF MSE improved/regressed | Measures whether low-frequency correction improves `J0`. | 10k+ | per-image CSV |
| strong-CR delta/regression | Tests preservation of already-strong baseline outputs. | 30k+ | group summary |
| LF-v1 gain preservation | Tests whether CBRFRC keeps LF-v1 gains instead of only rescuing regressions. | 50k+ | per-image CR/LF-v1/BRF matrix |
| LF-v1 regression rescue | Tests whether CBRFRC improves LF-v1 failures. | 50k+ | per-image CR/LF-v1/BRF matrix |
| gate LF/HF stats | Distinguishes real correction from gate collapse or HF damage. | every eval | loss log and TensorBoard |
| residual norm ratio/error ratio | Separates under-correction, overshoot, and raw capacity failures. | 10k+ | diagnostics CSV |

## Fair Training Contract

- Dataset: HAZE4K.
- Total target: `epochs=20`, `iters_per_epoch=5000`, total `100000` steps.
- Batch/patch: `bs=16`, `patch_size=256`.
- Loss weights: `w_loss_L1=1.0`, `w_loss_CR=0.1`,
  `w_loss_brf_res_lf=0.10`, `w_loss_brf_dir=0.02`,
  `w_loss_brf_preserve=0.05`, `w_loss_brf_bound=0.01`,
  `w_loss_brf_color=0.02`, `brf_preserve_warmup_steps=1000`,
  `brf_preserve_gate_weight=0.0` if the fixed-patch micro-overfit confirms the
  direct preserve gate penalty collapses the gate.
- Eval/checkpoint cadence: every `10000` steps, `save_epoch_checkpoints=false`.

## Gates

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| 10000 | Must not collapse far below CR baseline. | `gate_lf_mean` in `0.01..0.15`, `gate_hf_mean < 0.03`, residual cosine rising, LF MSE trend better than LF-v1. | Stop if gates die, residual cosine stays non-positive, or LF MSE regressions mirror LF-v1. |
| 30000 | PSNR should approach CR baseline. | wrong-direction and strong-CR regression must be improving. | Stop if weak cases improve only by hurting strong cases. |
| 50000 | `mean PSNR >= CR + 0.10 dB` preferred. | LF-v1 gain preservation `>=70%`, LF-v1 regression rescue `>=25%`, wrong-direction count at least `15%` below LF-v1 trend. | Promote only if mechanism and quality agree. |
| 70000 | Curve stable or rising. | Gates neither collapse to 0 nor saturate; color/HF metrics stable. | Stop if quality gain comes with systematic color or strong-baseline damage. |
| 100000 | Full-test evidence required. | Compare against CR, LF-v1, ResidualCalib, CRPlus-v2, and LFCR-v1. | Decide Stage C only if mechanism gates pass. |

## Analysis Plan

- If stopped: label the failure as gate collapse, wrong residual direction,
  strong-baseline regression, color shift, or refiner-only gain.
- If promoted: run full per-image diagnostics against CR, LF-v1,
  ResidualCalib, CRPlus-v2, and LFCR-v1 before Stage C.
- Required docs to update: `docs/EXPERIMENT_LOG.md` for run facts,
  `docs/HAZE4K_RUN_MANIFEST.md` for artifact paths, and this route card for
  gate decisions.

## First Scout Command Shape

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net/code

RUN=DEA-Net-CBRFRC-v1-H4K-scout100k-$(date +%Y%m%d-%H%M%S)
mkdir -p ../experiment/HAZE4K/_run_logs

/root/miniconda3/envs/py310/bin/python train.py \
  --use_brf_frequency_corrector \
  --brf_baseline_checkpoint <CR_BASELINE_BEST_PK> \
  --brf_freeze_baseline true \
  --brf_use_baseline_detach true \
  --brf_hidden_channels 16 \
  --brf_pyramid_type laplacian \
  --brf_wavelet_levels 2 \
  --brf_gate_init -4.0 \
  --brf_hf_gate_init -5.0 \
  --brf_max_residual 0.08 \
  --brf_max_color_residual 0.04 \
  --brf_max_hf_residual 0.03 \
  --brf_hf_scale 0.1 \
  --brf_preserve_highfreq true \
  --w_loss_brf_res_lf 0.10 \
  --w_loss_brf_dir 0.02 \
  --w_loss_brf_preserve 0.05 \
  --w_loss_brf_bound 0.01 \
  --w_loss_brf_color 0.02 \
  --brf_lf_pool 8 \
  --brf_dir_norm_floor 0.01 \
  --brf_preserve_target_thr 0.015 \
  --brf_preserve_warmup_steps 1000 \
  --brf_preserve_gate_weight 0.0 \
  --model_name "$RUN" \
  --dataset HAZE4K \
  --epochs 20 \
  --iters_per_epoch 5000 \
  --bs 16 \
  --patch_size 256 \
  --num_workers 12 \
  --test_num_workers 4 \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor 2 \
  --w_loss_L1 1.0 \
  --w_loss_CR 0.1 \
  --start_lr 0.0001 \
  --end_lr 0.000001 \
  --exp_dir ../experiment/ \
  --checkpoint_interval_steps 10000 \
  --eval_interval_steps 10000 \
  --save_epoch_checkpoints false \
  --no_pdf_plots \
  --no_tqdm \
  2>&1 | tee "../experiment/HAZE4K/_run_logs/${RUN}.log"
```
