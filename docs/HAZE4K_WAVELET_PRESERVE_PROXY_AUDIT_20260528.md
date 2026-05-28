# HAZE4K Wavelet Preserve Proxy Audit

Date: 2026-05-28

Status: diagnostic-only preflight. The current wavelet/degradation-aware
preservation proxy did not clear the written pass line, so do not launch a
WaveletPreserve architecture scout from this evidence.

## Purpose

LFCR-v2 showed that time-localized CRPlus-v2 pressure still destroys too many
LF-v1 gain cases. The next proposed architecture idea was a
wavelet/degradation-aware residual preservation block: use low/high frequency
and haze cues to decide when LF-v1-like low-frequency correction should be
preserved, suppressed, or calibrated.

This audit tests the cheapest necessary condition before any implementation:

> Can deployable, GT-free input or activation features distinguish LF-v1
> gain-preservation cases from LF-v1 regression/intervention cases?

No model training was started.

## Setup

Branch:

```text
codex/haze4k-wavelet-preserve-preflight
```

Run server:

```text
runyun-ts
```

Remote checkout:

```text
/root/workspace/Dehaze-Net-audit-sync
```

Script:

```text
code/analyze_wavelet_preserve_proxy.py
scripts/runyun-haze4k-wavelet-preserve-proxy.sh
```

Inputs:

- `model_per_image_matrix.csv` from
  `experiment/HAZE4K/route_evidence_review/HAZE4K-route-evidence-review-20260528`.
- HAZE4K hazy test images from
  `/root/workspace/Dehaze-Net/dataset/HAZE4K/test/haze`.
- Optional frozen activation checkpoints:
  - CR best:
    `/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk`
  - LF-v1 best:
    `/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk`

Primary target:

```text
extreme_preserve_vs_intervene
positive/preserve: LF-v1 delta vs CR >= +0.30 dB
negative/intervene: LF-v1 delta vs CR <= -0.30 dB
```

This gives `804` decisive samples: `453` preserve and `351` intervene.

Secondary target:

```text
oracle_lfv1_vs_best_alt
positive/preserve: LF-v1 >= best non-LF-v1 output
negative/intervene: best non-LF-v1 output > LF-v1
```

This gives all `1000` samples: `204` preserve and `796` intervene.

Pass line for the metadata-free primary target on random splits:

- mean gain vs LF-v1 at least `+0.10 dB`;
- oracle recovery at least `0.15`;
- intervene precision at least `0.60`;
- LF-v1 gain preserve recall at least `0.60`;
- balanced accuracy at least `0.60`.

The script also reports airlight-bin and beta-bin held-out splits to check
degradation generalization.

## Artifacts

Hazy-only wavelet proxy:

```text
experiment/HAZE4K/wavelet_preserve_proxy/HAZE4K-wavelet-preserve-proxy-runyun-20260528-hazy
```

Wavelet plus frozen activation proxy:

```text
experiment/HAZE4K/wavelet_preserve_proxy/HAZE4K-wavelet-preserve-activation-proxy-runyun-20260528
```

Each artifact includes:

- `summary.json`
- `summary.csv`
- `split_results.csv`
- `analysis_report.md`
- `wavelet_preserve_features.csv`

## Result

The hazy-only proxy extracted `309` metadata-free wavelet/degradation features.
It has real signal, but it does not pass:

| Target | Feature set | Split | Balanced acc | Gain vs LF-v1 | Recovery | Preserve recall | Intervene precision | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| extreme | hazy_wavelet | random | `0.6090` | `+0.5693` | `0.5185` | `0.6247` | `0.5541` | fail |
| extreme | hazy_wavelet | airlight-held-out | `0.5834` | `+0.5697` | `0.5154` | `0.5625` | `0.5233` | fail |
| extreme | hazy_wavelet | beta-held-out | `0.5804` | `+0.5464` | `0.4909` | `0.5814` | `0.5200` | fail |

The failure is not lack of gross signal: simulated gain is positive because
the action target has a large oracle. The failure is precision and
generalization. The proxy is still too willing to intervene on LF-v1-preserve
cases.

The activation-forward audit added `108` frozen CR/LF-v1 activation features.
It did not help. On the primary target:

| Target | Feature set | Split | Balanced acc | Gain vs LF-v1 | Recovery | Preserve recall | Intervene precision | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| extreme | activation_only | random | `0.5546` | `+0.4612` | `0.4271` | `0.5563` | `0.4703` | fail |
| extreme | hazy_wavelet_plus_activation | random | `0.5360` | `+0.5022` | `0.4600` | `0.5187` | `0.4757` | fail |
| extreme | hazy_wavelet_activation_plus_metadata | random | `0.5214` | `+0.4605` | `0.4123` | `0.5295` | `0.4526` | fail |

This means the simple "use wavelet cues plus bottleneck stats to gate LF
residuals" version is not a safe next 100k scout.

The secondary all-sample oracle target also does not justify training. Hazy
wavelet features recover some oracle gain, but LF-v1 preserve recall stays too
low (`0.4277` on random splits), so it would still destroy many LF-v1 winners.

## Decision

Do not train the current WaveletPreserve architecture candidate.

This result directly updates the next-route queue:

- Do not launch a wavelet/degradation-aware gate from hazy-only cues.
- Do not launch the same gate just because bottleneck activation features were
  added; activation features made the primary target worse in this audit.
- Keep wavelet/frequency decomposition as a research direction, but only with a
  changed target definition or a supervised/distilled preservation signal that
  passes a new full-sample proxy audit.

## Next Useful Route

The evidence now points away from another unsupervised/inference-safe gate. The
next valuable route, if we keep searching architecture, should change the
target rather than only the feature family:

1. Generate train-split pseudo-labels from existing teacher checkpoints, not
   from the HAZE4K test split.
2. Train or pretrain a small preservation/confidence head against that
   supervised target.
3. Before any 100k model scout, audit that head on held-out samples for:
   LF-v1 gain preservation, LF-v1 regression intervention precision,
   strong-CR regression control, and degradation-held-out stability.

Until that exists, LF-v1 remains the best standalone cold-start architecture
evidence, and the active official warm-start route should stay isolated from
these cold-start candidate tables.
