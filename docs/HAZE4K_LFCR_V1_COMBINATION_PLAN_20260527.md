# HAZE4K LFCR-v1 Combination Plan

Date: 2026-05-27

Status: route card for the first LF-v1 plus CRPlus-v2 combination scout.
Use this as the launch contract and gate definition. Run facts belong in
`docs/EXPERIMENT_LOG.md`; artifact retention belongs in
`docs/HAZE4K_RUN_MANIFEST.md`.

## Hypothesis

- Prior evidence:
  - LF-v1 remains the best standalone fair scout: best 90k
    `32.4281 / 0.9845`, full-test mean `32.4283 / 0.984454`.
  - CRPlus-v2 is a positive CR-only component: final 100k
    `32.3633 / 0.9847`, full-test mean delta `+0.1396 dB` over CR baseline,
    and higher SSIM than LF-v1.
  - CRPlus-v2 is complementary to LF-v1: on the `351` LF-v1 meaningful
    regression cases, CRPlus-v2 is at least `+0.3 dB` better than LF-v1 on
    `200` images and restores `108` images to baseline-or-better PSNR.
  - LF-v1/CRPlus-v2 oracle reaches `32.9373`, `+0.5090 dB` over LF-v1; the
    three-component LF-v1/ResidualCalib/CRPlus-v2 oracle reaches `33.2308`.
- Target failure mode:
  LF-v1 gives the best mean PSNR but has high per-image variance and severe
  wrong-direction or over-intervention cases. CRPlus-v2 gives a different
  frequency/contrastive training pressure that may reduce LF-v1 weak cases
  without adding inference-time parameters.
- Mechanism hypothesis:
  If LF-v1 is trained with a strong but bounded CRPlus-v2 frequency curriculum,
  LF-v1 should retain its low-frequency prior gains while CRPlus-v2 improves
  haze-relevant frequency separation and rescues part of LF-v1's regression
  set.

## Change

- Code branch: `codex/haze4k-lfcr-v1`.
- Primary variable: combine the existing LF-v1 architecture with CRPlus-v2
  training loss.
- Architecture/loss definition:

```text
DEA-Net-LF-v1:
  --use_lf_prior
  --lf_prior_channels 8
  --lf_prior_pool 8
  --lf_prior_gate_init 0.0
  --lf_prior_injection pre_mix

CRPlus-v2:
  distance = VGG_L1 + frequency_weight * FFT_amplitude_L1
                   + lowfreq_weight * low_frequency_L1
  loss = mean(clamp(d_pos / (d_neg + eps), max=ratio_cap))
```

- Enabled flags:
  - `--use_lf_prior`
  - `--w_loss_crplus_v2`
  - `--crplus_v2_negative_modes hazy,output_lowpass,under_dehazed_mix`
  - `--crplus_v2_start_negative_modes hazy,under_dehazed_mix`
  - `--crplus_v2_curriculum_steps 20000`
  - `--crplus_v2_frequency_weight 0.1`
  - `--crplus_v2_lowfreq_weight 0.1`
  - `--crplus_v2_under_dehazed_mix 0.5`
  - `--crplus_v2_ratio_cap 2.0`
- Weight selection:
  This route is not a conservative lite test. Use the largest diagnostic-safe
  weight among `0.003` and `0.005`. Prefer `0.005` if the LF-v1 scale
  diagnostic shows the weighted CRPlus-v2 objective is `<= 0.10` of L1 and no
  selected negative has a severe denominator trap. Fall back to `0.003` only
  if `0.005` violates that scale bound.
- Explicitly disabled related mechanisms:
  ResidualCalib, ResidualSelector, ConditionalMask, HazeAwareMask,
  TeacherGuard, LowFreqLoss, and ResidualDirLoss.

## References

- CR baseline best:
  `experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk`
  at step `90000`, `32.2255 / 0.9844`.
- LF-v1 best:
  `experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk`
  at step `90000`, `32.4281 / 0.9845`.
- CRPlus-v2 final:
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540`, step `100000`,
  `32.3633 / 0.9847`.

Matched gate references:

| Step | CR baseline | LF-v1 | CRPlus-v2 |
| ---: | --- | --- | --- |
| 10000 | `27.1101 / 0.9615` | `26.2651 / 0.9631` | `26.7627 / 0.9629` |
| 20000 | `28.9030 / 0.9713` | `28.8563 / 0.9751` | `29.2182 / 0.9714` |
| 30000 | `30.1143 / 0.9776` | `30.6253 / 0.9783` | `30.1416 / 0.9769` |
| 50000 | `31.2384 / 0.9817` | `31.3419 / 0.9817` | `31.3717 / 0.9824` |
| 90000 | `32.2255 / 0.9844` | `32.4281 / 0.9845` | `32.3067 / 0.9844` |
| 100000 | `32.0952 / 0.9844` | `32.3857 / 0.9845` | `32.3633 / 0.9847` |

## Mechanism Metrics

| Metric | Why it matches this route | Gate subset | Full-test artifact |
| --- | --- | --- | --- |
| CRPlus-v2 weighted objective / L1 ratio | keeps the added training pressure strong but bounded | LF-v1 scale diagnostic on train center crops and optional full test | `loss_scale/lfcr-v1-*` |
| LF scalar gate and loss log activity | checks whether LF-v1 remains active while CRPlus-v2 is active | checkpoint loss logs at each gate | run checkpoint/log |
| LF-v1 regression rescue count | direct target: rescue images where LF-v1 hurts baseline | full test after candidate checkpoint | `per_image_eval/LF-v1-vs-LFCR-v1-*` plus joined analysis |
| Strong-baseline regression count | guards against over-intervention on already-good images | full test after candidate checkpoint | `per_image_eval/CR-vs-LFCR-v1-*` |
| Residual cosine / wrong-direction count | verifies whether LF residual direction is improved or degraded | gate diagnostic if curve is mixed; full test after promotion | `residual_diagnostic/CR-vs-LFCR-v1-*` and `LF-v1-vs-LFCR-v1-*` |
| Frequency and low-frequency distance gaps | checks whether CRPlus-v2 improves its claimed mechanism | scale diagnostic at candidate checkpoint | `loss_scale/lfcr-v1-final-*` |
| PSNR/SSIM at matched steps | global quality guardrail and training-time efficiency signal | every 10k validation | run log and saved data |

## Fair Training Contract

- Dataset: HAZE4K.
- Total target: `100000` steps from launch.
- Batch/patch: `bs=16`, `patch_size=256`.
- Base losses: `w_loss_L1=1.0`, `w_loss_CR=0.1`.
- Added loss: diagnostic-selected `w_loss_crplus_v2`, preferably `0.005`.
- LR: `start_lr=0.0001`, `end_lr=0.000001`.
- Eval/checkpoint cadence: every `10000` steps.
- Epoch checkpoints: disabled.

## Gates

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| 10000 | Stop only on collapse: more than `0.8 dB` below both LF-v1 and CRPlus-v2 or unstable loss | CRPlus-v2 log must be active; LF gate must be finite | Continue if not collapsed |
| 20000 | Should be near or above LF-v1 20k and CRPlus-v2 20k on at least one metric | CRPlus-v2 active after warm curriculum; no exploding weighted ratio | Continue unless clearly below both references with no mechanism signal |
| 30000 | First hard gate: should be close to LF-v1 30k or show better SSIM and no collapse | If below LF-v1, run a small mechanism review before spending to 50k | Stop if quality is below both LF-v1 and CRPlus-v2 and mechanism metrics are worse |
| 50000 | Must be at least baseline-level and preferably beat LF-v1 50k or CRPlus-v2 50k | Check LF-v1 regression rescue and strong-baseline risk on a diagnostic subset if available | Continue only if it can plausibly beat LF-v1 best or reach same quality earlier |
| 70000 | Should be near LF-v1 best trajectory, with SSIM not worse than LF-v1 | Loss and LF gate should not show CRPlus suppressing LF | Stop if curve has flattened below LF-v1 with no mechanism win |
| 90000/100000 | Promote only if it beats LF-v1 mean PSNR, or reaches similar PSNR with materially better LF-v1 regression rescue / SSIM / training-time curve | Full per-image, residual, and CRPlus-v2 diagnostics required | Record as positive candidate, positive ablation, or negative fair ablation |

## Analysis Plan

- Before launch:
  run LF-v1 CRPlus-v2 scale diagnostic and select `w_loss_crplus_v2`.
- If stopped early:
  record whether the failure is from over-strong CRPlus pressure, LF gate
  suppression, weak frequency mechanism, or unchanged LF-v1 regression set.
- If promoted:
  run full-test comparisons against CR baseline, LF-v1, and CRPlus-v2; run
  residual-direction diagnostics and final CRPlus-v2 scale review.
- Required docs to update:
  `docs/EXPERIMENT_LOG.md`, `docs/CURRENT_CONTEXT.md`,
  `docs/HAZE4K_RUN_MANIFEST.md`, and this route card.
