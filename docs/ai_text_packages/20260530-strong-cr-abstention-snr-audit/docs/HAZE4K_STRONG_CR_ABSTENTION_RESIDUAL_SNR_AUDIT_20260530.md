# HAZE4K Strong-CR Abstention / Residual-SNR Audit

Date: 2026-05-30

Status: completed diagnostic audit. The AutoDL audit ran on 2026-05-30 and
failed the deployable pass line, so this card does not authorize a new model
implementation or any 100k training run.

Execution boundary: run audit extraction, probe fitting, and evaluation on
`autodl-dehaze` in `/root/autodl-tmp/workspace/Dehaze-Net`. Local WSL work is
limited to documentation, code edits, Git, packaging, and lightweight static
checks unless explicitly requested otherwise.

## Stage 0 Result

Run:
`HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full`.

Branch and commit at run time:
`codex/haze4k-strong-cr-abstention-snr-audit` / `c436abe`.

Artifacts:

- AutoDL:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/strong_cr_abstention_snr_audit/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full`
- Local compact sync:
  `experiment/HAZE4K/strong_cr_abstention_snr_audit/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full`
- GitHub text package:
  `docs/ai_text_packages/20260530-strong-cr-abstention-snr-audit`

Recommendation: `do_not_train_abstention_brf_v3_yet`.

Key result:

| Row | strong_q4 preserve | strong false intervention | precision | LF-v1 gain keep | corr | sim vs LF-v1 | shuffle gap | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `E_abstention_risk` + logistic random | `0.7385` | `0.2615` | `0.6631` | `0.5952` | `0.3498` | `+0.1174` | `0.2578` | fail |
| `D_feature_contrast` + logistic random | `0.7121` | `0.2879` | `0.6333` | `0.5949` | `0.3099` | `+0.0919` | `0.2279` | fail |
| `A_output` + ridge random | `0.7850` | `0.2150` | `0.5373` | `0.3241` | `0.1427` | `-0.0698` | `0.1085` | fail |
| `F_diagnostic_leakage` + hgb random | `1.0000` | `0.0000` | `1.0000` | `0.6804` | `0.5271` | `+0.2620` | `0.5800` | diagnostic-only |

The best deployable random row has real signal: intervention precision clears
`0.60`, simulated PSNR is above LF-v1, and the shuffled-control gap is large.
It still fails the written route gate because strong-q4 preserve recall is
below `0.75`, strong false-intervention is above `0.20`, LF-v1 gain
preservation is below `0.70`, and confidence correlation is below `0.45`.
CR-strength held-out strong-q4 is the decisive failure for the same row:
preserve recall `0.2781`, false intervention `0.7219`, precision `0.3744`,
and shuffled gap only `0.0059`.

Residual-SNR readouts:

- strong_q4 target residual norm / weak_q1 target residual norm: `0.2583`.
- strong_q4 sign flip rate: `0.0570`.
- Label counts: preserve `610`, intervene `386`, ignore `4`.

Interpretation:

The strong-CR target residual is much smaller than weak-CR residuals, which
supports the abstention-first concern. However, sign flip is not high; the
main blocker is that deployable features still do not identify strong-CR
no-change cases robustly under CR-strength held-out evaluation. Diagnostic
leakage rows show the target is separable when GT-derived information is
allowed, but that signal cannot enter a deployable predictor.

Decision:

Do not implement Abstention-First BRFRC-v3 from this audit. A future attempt
would need a new deployable no-change signal or a different target, not a
larger residual head.

## Most Valuable Attempt

- Why this is the most valuable current attempt:
  BRFRC-v2 representation audit proved frozen CR/LF-v1 internal features carry
  residual-direction signal, but failed because strong-CR preservation stayed
  far below the written gate. The next cheapest useful question is therefore
  whether the system can reliably decide "do not touch CR" before asking it to
  predict a residual.
- Cheap preflight evidence:
  Use existing CR, LF-v1, ResidualCalib, CRPlus-v2, CBRFRC-v1 checkpoints and
  the BRFRC-v2 representation-audit feature pattern. No new trainable image
  model is introduced.
- Earliest decisive gate:
  This audit itself. If no-change risk is not readable with small probes and
  held-out splits, do not spend a 100k run on another residual head.
- Expected training-time or attempt-count saving:
  A failed audit blocks BRFRC-v3 before model coding and prevents another
  capacity/loss variant that repeats the strong-CR no-regression failure.
- What success decides:
  Strong-CR abstention/no-regression risk is readable from deployable features
  with enough precision and held-out stability to justify an abstention-first
  model card.
- What failure decides:
  The missing signal is not merely residual direction; the strong-CR region may
  be too low-SNR or unstable for residual correction with current features.
- Why a cheaper diagnostic is not enough:
  Output-level CBRFRC-v1 and BRFRC-v2 representation probes already answered
  identity, trainability, and residual-direction readability. The remaining
  cheap decision requires explicit abstention labels, residual-SNR analysis,
  and strong-CR held-out reporting.

## Hypothesis

- Prior evidence:
  CBRFRC-v1 full-test diagnostics showed wrong-direction `557/1000`,
  mean residual cosine `-0.0506`, LF MSE improved/regressed `443/557`, and
  LF-v1 gain preservation `50.1%`. BRFRC-v2 Stage 0 internal ridge probes
  improved residual direction and LF-v1 preservation, but random strong-CR
  preservation was only `0.4081` for CR features, `0.4136` for LF-v1 features,
  and `0.4473` for feature contrast, below the required `0.70`.
- Target failure mode:
  Residual predictors may be forced to act on strong-CR images or tiny/unstable
  low-frequency residuals where the safest output is CR identity.
- Mechanism hypothesis:
  If strong-CR residuals are low-SNR, then small classifiers should first learn
  abstention/no-regression risk from CR/LF-v1 features and residual-risk
  statistics; if they cannot, a residual head should not be trained.

## Stage 0 Change

- Code branch:
  `codex/haze4k-strong-cr-abstention-snr-audit`.
- Primary variable:
  Target changes from residual prediction to abstention/no-regression risk
  prediction.
- Explicitly disabled related mechanisms:
  No 100k training, no BRFRC-v3 implementation, no selector-v2, no new deep
  MLP, no joint fine-tuning, no CRPlus schedule, no TeacherGuard, no Mamba/DWT
  branch.
- Small probe family:
  logistic regression, ridge classifier, histogram gradient boosting, and a
  calibrated classifier. These are diagnostic probes, not hidden refiners.

## Inputs And Features

Use image-level HAZE4K diagnostics first, with optional patch-level work only
if the image-level route passes. The first AutoDL run may use the HAZE4K test
split because the committed route-evidence CSV contains CRPlus-v2 full-test
metrics while AutoDL does not currently have the CRPlus-v2 100k checkpoint.
This remains diagnostic-only and is not a publishable model result.

| Group | Inputs | Deployability |
| --- | --- | --- |
| A. output-only | `I`, `J0`, `I-J0`, `LP(I)`, `LP(J0)`, HF stats | Deployable |
| B. CR features | CR `bottleneck/dec3/dec2/dec1` stats | Deployable if CR forward is available |
| C. LF-v1 features | LF-v1 `bottleneck/dec3/dec2/dec1` stats | Deployable only if LF-v1/reference forward is allowed |
| D. feature contrast | CR/LF-v1 feature deltas, cosine, norm, channel stats | Deployable with both forwards |
| E. residual-risk stats | LF-v1 residual norm, local LF variance, output residual norms, color/luma LF bias, no-reference CR-strength proxies | Deployable |
| Diagnostic-only | `||LP(GT)-LP(J0)||`, residual cosine/error ratio vs GT, true CR PSNR bin, true candidate margins | Label/SNR analysis only |

GT-related features are forbidden in deployable predictor rows. They may be
used only for label construction, SNR diagnosis, and leakage upper bounds.

## Labels

The audit is abstention-first:

| Label | Definition |
| --- | --- |
| strong/no-change preserve | CR is strong or all candidates are unsafe/low gain, especially when intervention has `>= 0.10` or `>= 0.30 dB` regression risk. |
| safe intervene | At least one existing candidate gains `>= +0.30 dB` over CR without obvious LF MSE, color/luma, or SSIM degradation. |
| ambiguous / ignore | Candidate gaps are below `0.10 dB`, residual norm is too small outside strong-preserve cases, or residual sign is unstable across low-pass kernels. |

The ignore zone is required. It prevents tiny and sign-unstable residuals from
becoming false supervision targets.

## Residual-SNR Diagnostics

Report by CR-strength quartile:

| Metric | Purpose |
| --- | --- |
| `||LP(GT)-LP(J0)||` | True target residual energy. |
| `||LP(LF-v1)-LP(J0)||` | Deployable residual-action proxy. |
| residual cosine stability | Whether residual direction is stable enough to supervise. |
| sign flip rate across low-pass kernels | Detect low-SNR target instability. |
| LF MSE gain distribution | Separate LF improvement from visual PSNR noise. |
| PSNR gain distribution | Identify weak/strong CR intervention behavior. |
| color/luma LF bias | Catch tone/color regressions. |

Key root-cause readouts:

- `strong_q4 target residual norm / weak_q1 target residual norm`
- `strong_q4 sign_flip_rate`

If strong_q4 residual norm is small, sign flips are high, and PSNR gains are
near zero-mean, the route should conclude that strong-CR residual correction is
ill-posed rather than blaming only head capacity.

## Pass Line

All criteria must pass on an eligible deployable feature row before writing an
Abstention-First BRFRC-v3 model card:

| Metric | Pass line |
| --- | --- |
| strong_q4 preserve recall | `>= 0.75` |
| strong_q4 false-intervention rate | `<= 0.20` |
| intervention precision | `>= 0.60`, preferably `>= 0.65` |
| LF-v1 gain preservation | `>= 0.70` |
| LF-v1 regression rescue | not lower than ResidualCalib / CRPlus-v2 rescue level from existing evidence |
| airlight / beta held-out | no material collapse |
| CR-strength held-out | strong_q4 reported separately, not averaged away |
| confidence correlation | `>= 0.45` |
| shuffled-control gap | clearly above shuffled controls |
| simulated PSNR | close to LF-v1, preferably LF-v1 `+0.05 dB` or better |

Simulated PSNR is diagnostic only. It decides whether a training route is worth
writing; it is not a publishable model result.

## If Stage 0 Passes

Only then write a separate model card for:

```text
Abstention-First BRFRC-v3

CR identity highway:
    J = J0 by default

Shared trunk:
    CR/LF-v1 internal features + output features

Abstention head:
    predicts p_preserve / p_intervene / uncertainty

Residual head:
    active only when p_intervene is high

Output:
    J = J0 + (1 - p_preserve) * q * bounded_residual
```

First loss priority:

```text
L_no_regression > L_preserve_LFv1_gain > L_residual_direction > L_reconstruction
```

This differs from CBRFRC-v1 and BRFRC-v2 by making "avoid bad intervention"
the primary task.

## Required Updates After Stage 0

- Append audit run facts and decision to `docs/EXPERIMENT_LOG.md`.
- Add compact artifact paths and keep/delete decisions to
  `docs/HAZE4K_RUN_MANIFEST.md`.
- Update `docs/CURRENT_CONTEXT.md` with the executable conclusion.
- If the audit passes, write the separate Abstention-First BRFRC-v3 route card
  before implementation or formal scout.
