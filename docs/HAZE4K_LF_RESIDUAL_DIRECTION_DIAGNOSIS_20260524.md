# HAZE4K LF-v1 Residual Direction Diagnosis

Date: 2026-05-24

Purpose: verify whether LF-v1's next optimization target should move from
spatial mask selection to low-frequency residual direction and magnitude
calibration.

This is a read-only evaluation. No training was started.

## Setup

- Branch / commit: `codex/haze4k-residual-diagnostics` / `a989ee5`.
- Remote checkout: `/root/workspace/Dehaze-Net-audit-sync`.
- Script: `code/diagnose_lf_residual_direction.py`.
- Dataset: HAZE4K test, `1000` images.
- Baseline checkpoint:
  `experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk`
  at step `90000`.
- LF-v1 checkpoint:
  `experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk`
  at step `90000`.
- Low-frequency operator: average pool / upsample with `lowfreq_pool=8`.
- Output:
  `experiment/HAZE4K/residual_diagnostic/CR-vs-LF-v1-20260524/`.

The diagnostic compares:

```text
target_residual = LP(GT) - LP(baseline_output)
lf_residual     = LP(LF_output) - LP(baseline_output)
```

It then records cosine direction, residual norm ratio, residual error ratio,
low-frequency MSE change, luma-only variants, baseline-strength groups, and
hard cases.

## Main Result

LF-v1 still has a real full-test gain:

| Metric | Value |
| --- | ---: |
| Baseline mean PSNR / SSIM | `32.2253 / 0.9844` |
| LF-v1 mean PSNR / SSIM | `32.4283 / 0.9845` |
| Mean PSNR delta | `+0.2030 dB` |
| Median PSNR delta | `+0.1567 dB` |
| `>= +0.30 dB` / `<= -0.30 dB` | `453 / 351` |

The new evidence is that LF-v1's PSNR delta is strongly tied to residual
direction correctness:

| Diagnostic | Value |
| --- | ---: |
| Mean residual cosine | `0.3028` |
| Median residual cosine | `0.3421` |
| P10 residual cosine | `-0.1310` |
| Mean luma residual cosine | `0.3029` |
| Mean residual norm ratio | `0.5638` |
| P90 residual norm ratio | `0.8470` |
| Mean residual error ratio | `0.9914` |
| Wrong-direction count, cosine `< 0` | `160` |
| Low-frequency MSE improved / regressed | `545 / 455` |
| Luma low-frequency MSE improved / regressed | `547 / 453` |
| Corr(delta PSNR, residual cosine) | `0.8775` |
| Corr(delta PSNR, residual norm ratio) | `-0.2274` |
| Corr(delta PSNR, LF MSE delta) | `-0.5319` |

Interpretation: the dominant discriminator is direction, not just magnitude.
Magnitude is still relevant, especially on strong-baseline regressions, but the
largest positive/negative split is whether LF-v1 points the low-frequency
correction in the same direction as the GT residual.

## Baseline-Strength Groups

| Group | Count | Mean delta PSNR | Mean cosine | Mean norm ratio | LF MSE improved / regressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| weakest 25% | `250` | `+0.4921` | `0.2865` | `0.3501` | `153 / 97` |
| middle 50% | `500` | `+0.1859` | `0.2879` | `0.5806` | `268 / 232` |
| strongest 25% | `250` | `-0.0520` | `0.3490` | `0.7440` | `124 / 126` |

The weak-baseline group benefits most even though its average residual
magnitude ratio is smaller. This suggests LF-v1 often provides a useful partial
correction rather than a complete low-frequency replacement.

The strong-baseline group is different: the mean magnitude ratio is higher
(`0.7440`) and LF MSE is essentially split (`124 / 126`). This supports the
earlier observation that LF-v1 can over-intervene when baseline is already
strong.

## Hard-Case Evidence

Worst PSNR regressions are direction failures:

| File | Delta PSNR | Cosine | Norm ratio | LF MSE delta |
| --- | ---: | ---: | ---: | ---: |
| `946_0.65_0.58.png` | `-6.2238` | `-0.5088` | `1.5371` | `+0.00020041` |
| `390_0.55_1.07.png` | `-4.9448` | `-0.6894` | `0.9444` | `+0.00434916` |
| `49_0.7_1.47.png` | `-4.4989` | `-0.4811` | `1.0155` | `+0.00052973` |
| `167_0.53_1.34.png` | `-4.3617` | `-0.2592` | `1.1124` | `+0.00096928` |
| `947_0.79_1.07.png` | `-4.3226` | `-0.5642` | `0.9182` | `+0.00046619` |

Best PSNR gains are direction-aligned partial corrections:

| File | Delta PSNR | Cosine | Norm ratio | LF MSE delta |
| --- | ---: | ---: | ---: | ---: |
| `67_0.68_1.12.png` | `+6.4500` | `0.9110` | `0.7205` | `-0.00185286` |
| `412_0.52_0.64.png` | `+6.1472` | `0.8955` | `0.7280` | `-0.00069814` |
| `800_0.55_1.84.png` | `+5.8964` | `0.9335` | `0.6099` | `-0.00094372` |
| `402_0.5_1.99.png` | `+5.5931` | `0.9562` | `0.5254` | `-0.01220746` |

The worst and best sets are not separated by mask activation. They are
separated by whether the induced low-frequency residual points toward or away
from the target correction.

## Root-Cause Update

The previous conclusion was:

```text
LF-v1 needs conditional, region-aware use; simple global LF is too coarse.
```

The new conclusion is sharper:

```text
LF-v1 needs residual calibration. Spatial selection alone is insufficient
unless it also estimates whether the low-frequency residual direction and
magnitude are correct.
```

This explains the last two failed mask routes:

- Conditional LF did not learn meaningful spatial selection.
- Haze-Aware Mask did learn some spatial variation, but still fell far below
  LF-v1 because a dark-channel/luma mask can decide where to apply LF but cannot
  correct a wrong residual vector.

## Next Optimization Direction

Do not continue pure mask stacking. The next candidate should be one of:

1. **LF Residual Calibration module**
   - Keep `pre_mix`.
   - Replace `x8 + gate * prior` with a bounded calibrated residual:
     `x8 + gate * alpha(x8, low) * tanh(delta(low, x8))`.
   - `alpha` should be a small scalar or `B,1,H,W` amplitude map.
   - `delta` should be normalized or bounded so it cannot freely shift
     low-frequency color/luma in the wrong direction.
   - First version should avoid extra teacher, CRPlus, or lowfreq L1 losses.

2. **Residual direction/magnitude training loss**
   - No new inference parameters.
   - Compare `LP(out)-LP(hazy)` or `LP(out)-LP(baseline_ref)` against the target
     residual direction, not just `LP(out)` against `LP(GT)`.
   - Use cosine or normalized residual alignment plus a bounded magnitude
     penalty.
   - This must be treated as different from the failed LowFreqLoss; the failed
     version was direct low-frequency reconstruction, not residual calibration.

Preferred order:

1. Draft the LF Residual Calibration experiment card.
2. Implement the smallest structural version.
3. Smoke on remote CUDA.
4. Launch a fair 100k-target scout only after the card defines 10k/20k/30k/50k
   gates against baseline and LF-v1.

## Artifact Contract

Keep the diagnostic directory:

```text
experiment/HAZE4K/residual_diagnostic/CR-vs-LF-v1-20260524/
  analysis_report.md
  group_summary.csv
  hard_cases.json
  per_image_residual_metrics.csv
  summary.json
```

This directory is now part of the LF-v1 root-cause evidence chain and should be
preserved like the full per-image evaluation artifact.
