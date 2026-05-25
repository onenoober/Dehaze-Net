# HAZE4K LF Residual Calibration Experiment Card

Date: 2026-05-25

Purpose: define the next LF candidate after the LF-v1 residual direction
diagnosis. This card must be updated before training results are interpreted.

## Evidence

The read-only LF-v1 residual diagnostic showed:

- LF-v1 remains a real positive candidate: mean delta `+0.2030 dB` over the
  DEA-Net-CR baseline on 1000 HAZE4K test images.
- `corr(delta PSNR, residual cosine)=0.8775`, which is much stronger than the
  relation to residual norm ratio.
- Worst LF-v1 regressions usually have negative low-frequency residual cosine
  and positive LF MSE delta.
- Best LF-v1 gains usually have residual cosine near `0.9` and improved LF MSE.
- Conditional LF and Haze-Aware Mask already showed that spatial selection
  alone is not enough.

Therefore the next candidate should answer:

> Can a bounded residual calibration branch keep LF-v1's useful partial
> low-frequency correction while reducing wrong-direction residuals and
> over-intervention on strong-baseline images?

## Candidate

Name:

```text
DEA-Net-LF-ResidualCalib
```

Single changed mechanism:

```text
LF-v1:
  prior = adapter(lowpass(hazy))
  out = x8 + gate * prior

ResidualCalib:
  low_feat = low_encoder(lowpass(hazy))
  target_hint = target_hint_proj(x8.detach())
  direction = tanh(direction_head(low_feat, target_hint))
  alpha = sigmoid(alpha_head(low_feat, target_hint)) * alpha_max
  prior = direction * alpha
  out = x8 + gate * prior
```

The first version keeps:

- `lf_prior_injection=pre_mix`
- `lf_prior_channels=8`
- `lf_prior_pool=8`
- `lf_prior_gate_init=0.0`
- `w_loss_CR=0.1`
- no teacher guard
- no CRPlus negative change
- no low-frequency reconstruction loss
- no extra mask-only branch

## Why This Is Not Just Another Mask

The failed masks asked "where should LF be used?" but still multiplied the same
adapter residual. ResidualCalib asks two additional questions:

1. What bounded direction should the feature residual point to?
2. What local amplitude should it have?

This matches the diagnostic signal: the strongest positive/negative split is
residual direction correctness, not mask activation.

## Architecture Constraints

Keep the implementation local to `LowFrequencyPrior`.

Recommended first implementation:

- Add `--lf_residual_calibration`.
- Add `--lf_calib_hidden_channels` with default `8`.
- Add `--lf_calib_alpha_max` with default `1.0`.
- Reuse the low-frequency RGB input and detached bottleneck `target` hint.
- Use `tanh` for the direction branch.
- Use `sigmoid * alpha_max` for the amplitude branch.
- Keep the direction head normally initialized and set alpha-head weight to
  zero with bias `0.0`, so `gate_init=0.0` still gives identity at launch while
  the calibration branch can receive gradients once the scalar gate moves.
- Record alpha mean/std/min/max, similar to the existing LF mask stats.

Do not combine with `lf_conditional_mask` in the first scout. If a later hybrid
is needed, it should get a new card.

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

Use dry-run and 2-step smoke only to verify code paths. Any non-100k horizon
must be labeled invalid for fair comparison.

## Stop Gates

Reference curves:

| Step | Baseline | LF-v1 |
| ---: | --- | --- |
| 10000 | `27.1101 / 0.9615` | `26.2651 / 0.9631` |
| 20000 | `28.9030 / 0.9713` | `28.8563 / 0.9751` |
| 30000 | `30.1143 / 0.9776` | `30.6253 / 0.9783` |
| 50000 | `31.2384 / 0.9817` | `31.3419 / 0.9817` |
| 90000 | `32.2255 / 0.9844` | `32.4281 / 0.9845` |

Gate rules:

- 2-step smoke: must write checkpoint and alpha stats.
- 10k: stop only if clearly broken, e.g. more than `0.8 dB` below both
  references or alpha/gate is numerically unstable.
- 20k: stop if clearly below both references and alpha remains effectively
  uninformative.
- 30k: hard gate. Continue only if close to LF-v1 or if residual diagnostic on
  hard cases shows clear reduction of wrong-direction failures.
- 50k: must be at least baseline-level and preferably close to LF-v1. If below
  baseline, stop. If below LF-v1 but regression count is materially reduced,
  run full residual/per-image diagnosis before deciding.
- 100k: continue only from the same 100k-target run after 50k passes.

## Required Diagnostics

If the scout reaches 30k, run at least:

- same-protocol checkpoint eval
- `diagnose_lf_residual_direction.py` versus baseline and versus LF-v1
- fixed 20-sample visual compare if residual metrics are promising or mixed

Promotion requires one of:

- mean PSNR close to or above LF-v1 with no worse regression profile, or
- slightly lower mean PSNR but materially fewer LF-v1 worst regressions and
  better strong-baseline behavior.

## Failure Interpretation

| Observation | Interpretation | Next action |
| --- | --- | --- |
| alpha stays near constant and PSNR follows LF-v1 poorly | calibration branch did not learn useful amplitude | inspect alpha init / target hint, do not stack mask immediately |
| direction remains wrong on LF-v1 worst cases | low-frequency input lacks enough information | consider residual direction loss or richer frequency cues |
| strong-baseline regressions shrink but weak gains collapse | branch is too conservative | loosen alpha bound or warm-start from LF-v1 adapter |
| mean PSNR improves but residual errors remain high | metric gain may be non-LF side effect | require visual and residual hard-case review |
| result is below baseline by 30k | stop route unless smoke/config issue is found | record as failed structural ablation |

## Decision

Proceed to implementation only as a single-mechanism structural candidate:
`LF Residual Calibration`. Do not launch a formal scout until dry-run and
2-step smoke pass on the remote CUDA environment and this card is referenced in
`EXPERIMENT_LOG.md`.
