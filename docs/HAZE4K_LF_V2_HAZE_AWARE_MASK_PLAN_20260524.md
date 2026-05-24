# HAZE4K LF-v2 Haze-Aware Mask Experiment Card

Date: 2026-05-24

## Purpose

LF-v1 is still the only positive LF candidate, but the root-cause diagnosis
shows that its regressions are dominated by wrong low-frequency direction:
strong-baseline and weak-failure hard cases have low-frequency MSE regressions
on `30/30` samples, while the two gain groups improve low-frequency MSE on
`30/30` samples. The next optimization should therefore target selective
low-frequency use, not simply smaller LF strength.

## Research-Grounded Hypothesis

Recent dehazing work points in the same direction:

- DEA-Net's value is lightweight detail-enhanced convolution plus
  content-guided attention; keep the backbone stable.
- Frequency-domain dehazing work supports using low-frequency/global context,
  but successful designs fuse frequency cues conditionally rather than adding a
  single global residual everywhere.
- Global-context dehazing work emphasizes that haze removal needs both global
  context and local detail; this matches the LF-v1 diagnosis.
- Frequency-based unpaired dehazing work warns that haze-related frequency
  changes can be entangled with content and color, so a low-pass image alone is
  not a reliable hard negative or universal correction.

Therefore LF-v2 should answer one narrow question:

> Can a haze-aware spatial mask keep LF-v1's weak-baseline gains while reducing
> low-frequency color/luma/dark-channel regressions on strong-baseline cases?

## Change

Add a controlled extension to `LowFrequencyPrior`:

```text
hazy low-pass RGB
  + low-frequency dark-channel cue
  + low-frequency luma cue
  + detached bottleneck hint
  -> tiny spatial mask
  -> x8 + scalar_gate * mask * LF_prior
  -> mix1
```

The first fair candidate uses:

```text
use_lf_prior=true
lf_prior_injection=pre_mix
lf_conditional_mask=true
lf_haze_aware_mask=true
lf_haze_mask_strength=1.0
lf_mask_hidden_channels=8
lf_mask_init_bias=2.0
lf_prior_channels=8
lf_prior_pool=8
lf_prior_gate_init=0.0
no new low-frequency loss
no teacher guard
no CRPlus negative change
```

This keeps the run close to LF-v1 and only changes the mask input. It avoids
the failed patterns already tested: over-constraining LF, moving to `post_mix`,
adding low-frequency L1, adding low-pass CR negatives, or using a strong early
teacher guard.

## Fair Training Contract

Formal scout must use the standard HAZE4K 100k target:

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

The helper script is:

```text
scripts/runyun-haze4k-lf-haze-aware-mask-scout.sh
```

## Revised Target Metrics

Do not promote a candidate on mean PSNR alone. A useful LF-v2 must satisfy all
or almost all of the following:

1. Mean PSNR: at 50k, at least baseline 50k `31.2384`, and preferably close to
   LF-v1 50k `31.3419`. At 100k/full-test, mean delta should be at least near
   LF-v1's `+0.2030 dB`.
2. Regression count: full-test `worse_030db_count` should be lower than LF-v1's
   `351`; strong-baseline `worse_030db_count` should fall materially below the
   LF-v1 reference.
3. Gain preservation: weak-baseline `better_030db_count` and mean delta should
   not collapse relative to LF-v1's weak-quartile gain (`+0.4921 dB` on the
   weakest quartile).
4. Root-cause metric: on the hard-case diagnostic set, strong-baseline and
   weak-failure low-frequency MSE/bias regressions should shrink, especially
   luma and RGB low-frequency bias.
5. Mask behavior: `LF_mask_std` must move above near-constant levels seen in
   the failed ConditionalMask run (`~0.000085`) without collapsing to all-zero
   or all-one behavior.

## Stop Gates

- 10k: stop only if clearly broken, e.g. far below both baseline and LF-v1 or
  mask statistics are degenerate and loss is unstable.
- 20k: continue only if roughly competitive with baseline/LF-v1 and mask
  statistics show real spatial variation.
- 30k: hard gate. If PSNR is clearly below LF-v1 and mask is still nearly
  constant, stop.
- 50k: must be at least baseline-level and preferably close to LF-v1; otherwise
  stop unless hard-case diagnostics show a strong regression-count reduction.

## Required Evaluation If Promoted

Run the upgraded per-image evaluator. It now reports baseline-strength bins:

```text
baseline_weakest_25
baseline_middle_50
baseline_strongest_25
```

Required artifacts:

- full `per_image_metrics.csv`
- `group_summary.csv`
- `summary.json`
- `hard_cases.json`
- fixed-sample visual comparison
- hard-case frequency decomposition against LF-v1 and baseline

## Decision

This is the highest-value next candidate because it directly targets the
verified failure mode: low-frequency direction errors caused by unconditional
LF injection. It is also the smallest useful change that remains aligned with
DEA-Net's lightweight thesis story.

## Run Result

Run: `DEA-Net-LF-HazeAwareMask-H4K-scout100k-20260524-152758`

Remote copy: `/root/workspace/Dehaze-Net-lf-v2-verify`

Log:
`/root/workspace/Dehaze-Net-lf-v2-verify/experiment/HAZE4K/_run_logs/DEA-Net-LF-HazeAwareMask-H4K-scout100k-20260524-152758.log`

Artifact:
`/root/workspace/Dehaze-Net-lf-v2-verify/experiment/HAZE4K/DEA-Net-LF-HazeAwareMask-H4K-scout100k-20260524-152758/`

The run used the fair 100k-target contract and was stopped at the 30k hard
gate.

| Step | PSNR | SSIM | Decision |
| ---: | ---: | ---: | --- |
| 10000 | 27.5613 | 0.9571 | continue; PSNR up, SSIM down |
| 20000 | 28.6961 | 0.9724 | mixed; continue only to 30k hard gate |
| 30000 | 30.1157 | 0.9770 | stop; tied baseline, far below LF-v1 |

30k comparison:

- Baseline 30k: `30.1143 / 0.9776`; LF-v2 delta `+0.0014 dB / -0.0006 SSIM`.
- LF-v1 30k: `30.6253 / 0.9783`; LF-v2 delta `-0.5096 dB / -0.0013 SSIM`.

Mask diagnostics at 30k from `latest.pk`:

- `LF_mask_mean` last `0.873885`.
- `LF_mask_std` last `0.001312`; tail-50 mean `0.001059`.
- `LF_mask_min/max` last `0.866170 / 0.876420`.

Conclusion:

The haze-aware mask did activate, so this is not the same failure as the
near-constant ConditionalMask run. However, the active mask only recovered the
baseline curve and lost the LF-v1 improvement. This means simple spatial
selection from low-pass RGB, dark-channel, and luma cues is insufficient; the
next root-cause hypothesis should focus on estimating the direction and
magnitude of the low-frequency correction, not only where to apply it.
