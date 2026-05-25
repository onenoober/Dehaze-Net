# HAZE4K Three-Way Output Analysis

Date: 2026-05-25

Scope: compare final test outputs from DEA-Net-CR baseline, LF-v1, and
ResidualCalib. The goal is to identify what visual/result patterns should guide
the next optimization step.

## Artifacts

Full 1000-image analysis:

```text
experiment/HAZE4K/three_way_eval/Baseline-LFv1-ResidualCalib-full-20260525/
```

Original 2026-05-22 fixed visual sample list, rerun with ResidualCalib added:

```text
experiment/HAZE4K/visual_compare/Baseline-LFv1-ResidualCalib-fixed20260522-20260525/
```

Preview contact sheet:

```text
experiment/HAZE4K/three_way_eval/Baseline-LFv1-ResidualCalib-preview-20260525/contact_sheet.jpg
```

Script:

```text
code/analyze_three_way_outputs.py
```

The script writes:

- `summary.json`
- `per_image_three_way_metrics.csv`
- `group_summary.csv`
- `hard_cases.json`
- `analysis_report.md`
- selected panels and heatmaps

## Full Test Summary

All three models use best 90k train checkpoints from the same fair 100k
HAZE4K scout protocol.

| Model | PSNR | SSIM |
| --- | ---: | ---: |
| DEA-Net-CR baseline | `32.2253` | `0.984417` |
| LF-v1 | `32.4283` | `0.984454` |
| ResidualCalib | `32.3936` | `0.984500` |

ResidualCalib is lower than LF-v1 in mean PSNR, but the per-image distribution
is not simply worse:

| Metric | Value |
| --- | ---: |
| LF-v1 mean delta vs baseline | `+0.2030 dB` |
| ResidualCalib mean delta vs baseline | `+0.1682 dB` |
| ResidualCalib mean delta vs LF-v1 | `-0.0347 dB` |
| ResidualCalib median delta vs LF-v1 | `+0.0271 dB` |
| ResidualCalib better/worse than LF-v1 by `0.30 dB` | `414 / 386` |

Winner counts by PSNR:

| Winner | Count |
| --- | ---: |
| baseline | `295` |
| LF-v1 | `322` |
| ResidualCalib | `383` |

This means ResidualCalib is not a uniformly weaker LF-v1. It wins more
individual images, but its failures are large enough to pull down the mean.

## Pattern Counts

| Pattern | Count | Meaning |
| --- | ---: | --- |
| `residual_beats_both` | `300` | ResidualCalib is meaningfully better than both baseline and LF-v1 |
| `lost_lfv1_gain` | `218` | LF-v1 gives a meaningful gain, but ResidualCalib loses it |
| `mitigates_lfv1_regression` | `110` | LF-v1 hurts baseline; ResidualCalib recovers part or most of that loss |
| `residual_worst` | `165` | ResidualCalib is meaningfully worse than both |
| `both_improve_baseline` | `82` | both LF variants help baseline |
| `both_regress_baseline` | `66` | both LF variants hurt baseline |
| `mixed_or_small` | `59` | no large directional difference |

The strongest signal is bifurcation: ResidualCalib creates a large
`residual_beats_both` group and a large `lost_lfv1_gain` / `residual_worst`
group at the same time.

## Baseline-Strength Split

| Group | LF-v1 Delta | ResidualCalib Delta | ResidualCalib vs LF-v1 |
| --- | ---: | ---: | ---: |
| weakest baseline 25% | `+0.4921` | `+0.6354` | `+0.1433` |
| middle baseline 50% | `+0.1859` | `+0.0773` | `-0.1086` |
| strongest baseline 25% | `-0.0520` | `-0.1171` | `-0.0651` |

ResidualCalib improves the weakest baseline cases more than LF-v1, but it is
less reliable on middle and already-strong cases. This supports a selective
route: use stronger calibration only where the baseline appears weak or the
predicted residual direction is reliable.

## Residual-Direction Signal

Wrong-direction counts against baseline:

| Model | Wrong-Direction Count |
| --- | ---: |
| LF-v1 | `160` |
| ResidualCalib | `163` |

ResidualCalib does not reduce wrong-direction count. It slightly improves mean
residual cosine overall, but the hard failures are still dominated by negative
or weak residual cosine and low-frequency MSE regression.

Correlation:

| Relation | Correlation |
| --- | ---: |
| ResidualCalib delta vs baseline, residual cosine | `0.8490` |
| ResidualCalib delta vs LF-v1, residual cosine | `0.5776` |

Residual direction remains the strongest actionable diagnostic. The next route
should not be another mask-only branch; it should predict or constrain whether
the LF residual direction is correct.

## Old Fixed Sample Set

The original `DEA-Net-CR-vs-LF-20260522` 20-sample list remains an adverse
visual subset.

| Model | Mean PSNR | Mean SSIM |
| --- | ---: | ---: |
| baseline | `31.5373` | `0.983465` |
| LF-v1 | `31.2580` | `0.982879` |
| ResidualCalib | `31.0085` | `0.982078` |

Winner counts on this subset:

| Winner | Count |
| --- | ---: |
| baseline | `7` |
| LF-v1 | `6` |
| ResidualCalib | `7` |

ResidualCalib is not a rescue for this old visual subset. It adds some wins,
but the mean drops further because several LF-v1-positive or baseline-strong
images are degraded more heavily.

Useful examples from the old subset:

- `384_0.97_0.82.png`: ResidualCalib mitigates a LF-v1 regression.
- `195_0.61_1.47.png`: LF-v1 gain is partially lost by ResidualCalib.
- `431_0.86_1.86.png`: ResidualCalib is a clear wrong-direction failure.
- `1000_0.73_1.8.png`: ResidualCalib beats both baseline and LF-v1.

## Visual Reading

From the generated panels:

- `residual_beats_both` cases often show ResidualCalib restoring global tone and
  contrast where LF-v1 either under-corrects or moves in the wrong direction.
- `mitigates_lfv1_regression` cases show ResidualCalib acting like a softer or
  better-directed correction than LF-v1.
- `lost_lfv1_gain` cases show the opposite: LF-v1 is visually and metrically
  closer to GT, while ResidualCalib shifts tone/color or leaves low-frequency
  haze correction in the wrong direction.
- `residual_worst` and strong-baseline regressions often look like
  over-intervention: baseline was already close, and ResidualCalib introduces
  visible tone/color or low-frequency contrast error.

## Optimization Implications

Do not promote ResidualCalib as the final LF replacement yet.

The next candidate should be selective. A better design target is:

1. Preserve LF-v1 when LF-v1 residual direction is clearly aligned.
2. Use ResidualCalib-style correction when LF-v1 is wrong or weak and
   calibration cosine is predicted reliable.
3. Suppress LF intervention on strong-baseline images unless confidence is high.
4. Add an explicit residual-direction or low-frequency consistency signal;
   mask location alone is not the bottleneck.

Concrete next route candidates:

- `ResidualConfidenceGate`: predict a confidence scalar from low-frequency input
  plus bottleneck hint; apply calibration only when confidence is high.
- `LF-v1/ResidualCalib selector`: learn a bounded mixture between the LF-v1
  adapter residual and calibrated residual, with identity-safe initialization.
- `StrongBaselineGuard`: detect already-clean/strong baseline-like cases and
  clamp LF residual magnitude.
- `ResidualDirectionLoss` as a small training-only term after verifying scale,
  encouraging low-frequency residual cosine without forcing large amplitude.

The minimum success criterion for a next run should not only be mean PSNR. It
should require fewer `lost_lfv1_gain` and `residual_worst` cases while retaining
most `residual_beats_both` cases.
