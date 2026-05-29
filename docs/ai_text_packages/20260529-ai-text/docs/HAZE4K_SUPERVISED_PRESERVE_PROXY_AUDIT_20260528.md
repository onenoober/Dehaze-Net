# HAZE4K Supervised Preserve Proxy Audit

Date: 2026-05-28

Status: diagnostic-only preflight. The supervised/distilled preservation
target did not clear the written pass line, so do not launch a preserve-head or
WaveletPreserve architecture scout from this evidence.

## Purpose

After LFCR-v2 and the unsupervised WaveletPreserve proxy both failed, the next
question was whether a stronger supervised target could make the LF-v1
preserve/intervene decision learnable.

This audit generated train-split pseudo-labels from existing CR and LF-v1
teacher checkpoints:

```text
positive / preserve: LF-v1 patch PSNR - CR patch PSNR >= +0.30 dB
negative / intervene: LF-v1 patch PSNR - CR patch PSNR <= -0.30 dB
```

No long model training was launched.

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

Scripts:

```text
code/analyze_supervised_preserve_proxy.py
scripts/runyun-haze4k-supervised-preserve-proxy.sh
```

Teacher checkpoints:

```text
/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk
/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk
```

Feature generation:

- HAZE4K train split.
- `3000` train images.
- `4` deterministic/random patches per image.
- `12000` total patches.
- `10055` decisive patch labels after the +/- `0.30 dB` margin:
  `4947` preserve, `5108` intervene.
- Features combine hazy wavelet/degradation cues and deployable teacher-output
  proxy cues from CR/LF-v1 outputs and residuals.

Dependency note:

- Installed `scikit-learn==1.7.2` into runyun `/opt/anaconda/envs/py310` after
  confirming it was missing.
- `scipy==1.15.3`, `joblib==1.5.3`, and `threadpoolctl==3.6.0` were installed
  as dependencies.

## Artifacts

Primary final artifact:

```text
experiment/HAZE4K/supervised_preserve_proxy/HAZE4K-supervised-preserve-proxy-runyun-20260528-train-p4-sklearn-liblinear
```

Large feature-generation artifact on runyun:

```text
/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/supervised_preserve_proxy/HAZE4K-supervised-preserve-proxy-runyun-20260528-train-p4-logistic/supervised_preserve_features.csv
```

The local compact sync keeps only summary files for the final sklearn audit.
The `170M` feature CSV remains on runyun and should not be committed.

## Pass Line

Primary random-image split requirements:

- gain vs LF-v1 at least `+0.05 dB`;
- oracle recovery at least `0.20`;
- balanced accuracy at least `0.62`;
- intervene precision at least `0.62`;
- preserve recall at least `0.65`;
- strong-CR regression intervene recall at least `0.60`.

Airlight and beta held-out splits also need stability:

- held-out intervene precision at least `0.56`;
- held-out preserve recall at least `0.56`;
- held-out balanced accuracy at least `0.56`.

## Results

The best reliable linear solver was sklearn `LogisticRegression` with
`solver=liblinear`, `class_weight=balanced`, and `C=1.0`.

| Feature set | Head | Split | Gain vs LF-v1 | Recovery | Balanced acc | Preserve recall | Intervene precision | Strong CR recall | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hazy_wavelet_plus_teacher_outputs | sklearn_logistic | random-image | `+0.1955` | `0.2288` | `0.5889` | `0.5951` | `0.5934` | `0.5347` | fail |
| hazy_wavelet_plus_teacher_outputs | sklearn_logistic | airlight-held-out | `+0.1806` | `0.2129` | `0.5815` | `0.5847` | `0.5905` | `0.5339` | fail |
| hazy_wavelet_plus_teacher_outputs | sklearn_logistic | beta-held-out | `+0.1881` | `0.2131` | `0.5845` | `0.5878` | `0.5927` | `0.5813` | fail |

For comparison, a small GPU MLP on random-image splits improved the random
metrics but still missed the pass line:

```text
balanced_accuracy=0.6355
gain_vs_lfv1=+0.2979
oracle_recovery=0.3489
preserve_recall=0.6429
intervene_precision=0.6412
strong_cr_regression_intervene_recall=0.5805
```

It missed both preserve recall (`0.6429 < 0.65`) and strong-CR regression
recall (`0.5805 < 0.60`) before any degradation-held-out stability check.

## Decision

Do not train the current supervised preserve-head route.

The target is more informative than the previous hazy-only wavelet proxy, but
it is still not selective enough. The reliable sklearn audit fails on balanced
accuracy, preserve recall, intervene precision, and strong-CR regression
control. The MLP random split gets closer, but still misses key guardrails and
has not demonstrated held-out degradation stability.

## Next Direction

The next architecture search should move away from simple preserve/intervene
classification from patch-level teacher deltas. More promising options:

- predict a continuous correction magnitude or residual confidence instead of
  a binary preserve/intervene label;
- pretrain a compact distillation head on teacher residual fields, then audit
  whether its output correlates with LF residual direction and magnitude;
- revisit architectural changes only when the preflight target passes random
  and degradation-held-out stability, especially strong-CR regression control.

Until then, LF-v1 remains the best standalone cold-start architecture evidence,
with ResidualCalib and CRPlus-v2 retained as positive components rather than a
deployable replacement.
