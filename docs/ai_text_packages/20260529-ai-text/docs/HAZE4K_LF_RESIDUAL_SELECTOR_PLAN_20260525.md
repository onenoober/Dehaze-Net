# HAZE4K LF Residual Selector Experiment Card

Date: 2026-05-25

Status: historical selector experiment card. It preserves the oracle evidence
and launch rationale for the first LFResidualSelector run. The fair run later
failed at the 20k gate, and the selector route is now summarized by
`docs/HAZE4K_SELECTOR_EVIDENCE_CLOSURE_20260526.md`.

Purpose: define the next LF candidate after selector/oracle diagnosis confirmed
that LF-v1 and ResidualCalib are strongly complementary.

## Evidence

Read-only selector/oracle outputs:

```text
experiment/HAZE4K/selector_oracle/Baseline-LFv1-ResidualCalib-full-20260525/
experiment/HAZE4K/selector_oracle/Baseline-LFv1-ResidualCalib-fixed20260522-20260525/
```

Full 1000-image result:

| Route | Mean PSNR | Mean SSIM | Delta vs LF-v1 |
| --- | ---: | ---: | ---: |
| LF-v1 | `32.4283` | `0.984454` | `0.0000` |
| ResidualCalib | `32.3936` | `0.984500` | `-0.0347` |
| LF-v1/ResidualCalib oracle | `33.0034` | `0.985299` | `+0.5751` |
| Baseline/LF-v1/ResidualCalib oracle | `33.2538` | `0.985637` | `+0.8255` |

Old adverse 20-sample subset:

| Route | Mean PSNR | Mean SSIM | Delta vs LF-v1 |
| --- | ---: | ---: | ---: |
| Baseline | `31.5373` | `0.983465` | `+0.2793` |
| LF-v1 | `31.2580` | `0.982879` | `0.0000` |
| ResidualCalib | `31.0085` | `0.982078` | `-0.2495` |
| LF-v1/ResidualCalib oracle | `31.8019` | `0.983492` | `+0.5439` |
| Baseline/LF-v1/ResidualCalib oracle | `32.1661` | `0.983806` | `+0.9080` |

The best one-rule selector used GT-aware residual error ratio, so it cannot be
used as an inference rule. It is still valuable: it proves the selector target
is real and should be learned or supervised, not hand-coded from test labels.

## Candidate

Name:

```text
DEA-Net-LF-ResidualSelector
```

Single changed mechanism:

```text
lf_prior = LF-v1 adapter(lowpass(hazy))
calib_prior = ResidualCalib direction/alpha prior
selector = sigmoid(selector_head(lowpass(hazy), x8.detach()))
prior = selector * lf_prior + (1 - selector) * calib_prior
out = x8 + gate * prior
```

Interpretation:

- `selector -> 1`: behave like LF-v1.
- `selector -> 0`: use ResidualCalib-style bounded correction.

First version should start close to LF-v1:

- Initialize selector final conv weight to zero.
- Initialize selector bias around `2.0`, so `sigmoid(2.0) ~= 0.88`.
- Keep `lf_prior_injection=pre_mix`.
- Keep `lf_prior_channels=8`, `lf_prior_pool=8`, `lf_prior_gate_init=0.0`.
- Keep `w_loss_CR=0.1`.
- Do not combine with Conditional LF mask, Haze-Aware Mask, TeacherGuard,
  LowFreqLoss, or CRPlus.

## Optional Training Signal

Because the oracle's best simple rule is GT-aware, the first selector scout may
include a small training-only selector target:

```text
target = 1 if LF-v1-style residual should be preferred
target = 0 if ResidualCalib-style residual should be preferred
```

However, do not load separate teacher models in the first run unless a cheap
offline target can be computed inside the forward/loss path. If implementation
risk is high, launch structure-only `LFResidualSelector` first and use selector
stats plus gate checkpoints as the gate decision.

## Logging

Record selector stats every training step alongside existing LF stats:

- `LF_selector_mean`
- `LF_selector_std`
- `LF_selector_min`
- `LF_selector_max`
- existing `LF_alpha_*` if calibration branch is active

Selector must move away from a near-constant value by the 20k/30k gate, unless
metrics are already clearly better than LF-v1.

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

## Stop Gates

Reference curves:

| Step | Baseline | LF-v1 | ResidualCalib |
| ---: | --- | --- | --- |
| 10000 | `27.1101 / 0.9615` | `26.2651 / 0.9631` | `26.5666 / 0.9621` |
| 20000 | `28.9030 / 0.9713` | `28.8563 / 0.9751` | `28.5005 / 0.9720` |
| 30000 | `30.1143 / 0.9776` | `30.6253 / 0.9783` | `30.3852 / 0.9782` |
| 50000 | `31.2384 / 0.9817` | `31.3419 / 0.9817` | `31.1396 / 0.9808` |
| 90000 | `32.2255 / 0.9844` | `32.4281 / 0.9845` | `32.3936 / 0.9845` |

Gate rules:

- 2-step smoke: must write checkpoint and selector stats.
- 10k: stop only if clearly broken or selector/alpha stats are numerically
  unstable.
- 20k: stop if clearly below both LF-v1 and ResidualCalib and selector remains
  near-constant.
- 30k: hard gate. Continue only if close to LF-v1 or if selector diagnostics
  show meaningful separation on ResidualCalib-positive groups.
- 50k: must be at least baseline-level and preferably close to LF-v1. If below
  baseline, stop.
- 90k/100k: promote only if it beats LF-v1 in mean PSNR, or matches LF-v1 while
  reducing `lost_lfv1_gain`, `residual_worst`, wrong-direction count, and
  strong-baseline regressions.

## Required Follow-Up Diagnostics

If the scout reaches 30k:

- same-protocol checkpoint eval
- selector stats from checkpoint/log
- full per-image comparison if 50k passes
- residual-direction diagnosis versus baseline and LF-v1
- fixed 20-sample three-way compare if metrics are mixed

## Decision

Proceed to implementation only as a bounded, LF-v1-initialized structural
candidate. The purpose is not to hand-code the oracle; it is to test whether a
small learned selector can recover part of the strong LF-v1/ResidualCalib
complementarity while preserving the current best LF-v1 behavior.
