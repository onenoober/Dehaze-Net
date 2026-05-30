# HAZE4K BRFRC-v2 Representation Audit Plan

Date: 2026-05-30

Status: completed Stage 0 diagnostic audit. The audit failed the written
strong-CR preservation and CR-strength held-out gates, so this card does not
authorize BRFRC-v2-Rep implementation or a 100k HAZE4K scout.

Execution boundary: run audit extraction, probe training, smoke tests,
evaluation, and any later formal scout on `autodl-dehaze` in
`/root/autodl-tmp/workspace/Dehaze-Net`. Local WSL work is limited to
documentation, code edits, Git, and lightweight static checks unless explicitly
requested otherwise.

## Stage 0 Result

- Run:
  `HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028` on
  `autodl-dehaze`, branch/commit
  `codex/haze4k-brfrc-v2-representation-audit` / `a441148`.
- Artifact:
  `experiment/HAZE4K/brfrc_v2_representation_audit/HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028`.
- Recommendation:
  `do_not_train_brf_v2_representation_yet`.
- Main read:
  frozen CR/LF-v1 internal features contain real residual-direction signal
  beyond output-only and shuffled controls, but they do not preserve strong CR
  cases reliably enough to justify a trainable BRFRC-v2-Rep head.
- Best random ridge rows:
  CR features cosine/wrong/LF-improve/LF-v1-preserve/strong-CR/corr
  `0.6176/0.0302/0.6933/0.7739/0.4081/0.5371`; LF-v1 features
  `0.6411/0.0262/0.7267/0.8310/0.4136/0.5859`; feature contrast
  `0.6488/0.0222/0.7284/0.8321/0.4473/0.5604`.
- Failure gate:
  strong-CR preservation required `>= 0.70`, but all main feature rows stayed
  below `0.45`; CR-strength held-out rows also collapsed. No Stage 1 model
  card, implementation, or formal scout is authorized from this evidence.

## Most Valuable Attempt

- Why this is the most valuable current attempt:
  CBRFRC-v1 proved identity safety, bounded oracle headroom, and fixed-patch
  trainability, but its fair full-test mechanism failed. The best checkpoint
  stayed below CR and LF-v1 while the residual direction diagnostics were
  negative. The next cheapest high-value question is whether DEA-Net internal
  representations contain the missing residual direction and reliability
  signal before spending another 100k training run.
- Cheap preflight evidence:
  Use existing CR and LF-v1 checkpoints plus `DEANet.forward_with_features`
  outputs (`dec1`, `dec2`, `dec3`, `bottleneck`) to train small probes on
  frozen features.
- Earliest decisive gate:
  Stage 0 itself. If internal features do not beat output-only and shuffled
  controls on direction, preservation, reliability, and held-out stability, do
  not implement or train BRFRC-v2-Rep.
- Expected training-time or attempt-count saving:
  A failed audit stops the representation route before any formal scout and
  prevents another CBRFRC-v1 capacity or gate/loss variant.
- What success decides:
  Bottleneck/decoder features carry deployable signal for low-frequency
  residual direction, reliability, LF-v1 gain preservation, and strong-CR
  preservation beyond `I/J0/I-J0/LP/HF`.
- What failure decides:
  The current DEA-Net CR/LF-v1 representation is still not sufficient for a
  safe residual corrector; future work should change target, pretraining, or
  backbone scale instead of adding a new head over these features.
- Why a cheaper diagnostic is not enough:
  Output-level CBRFRC-v1 already supplied the cheaper oracle and micro-overfit
  diagnostics. The remaining cheap decision is feature-readability with
  controls and held-out splits.

## Hypothesis

- Prior evidence:
  CBRFRC-v1 best checkpoint was step `10000`, `32.2237 / 0.9844`; final step
  `100000` was `32.1977 / 0.9844`. Full-test diagnostics for the best
  checkpoint reported `delta_brf_vs_cr=-0.0007`,
  `delta_brf_vs_lfv1=-0.2035`, wrong-direction `557/1000`, mean residual
  cosine `-0.0506`, LF MSE improved/regressed `443/557`, and LF-v1 gain
  preservation `50.1%`.
- Target failure mode:
  Output-level observations can be identity-safe and trainable but still fail
  to infer the sign, magnitude, and reliability of the applied residual
  `LP(GT)-LP(J0)`.
- Mechanism hypothesis:
  If frozen DEA-Net bottleneck/decoder features encode the missing
  restoration state, small probes over those features should improve residual
  direction, LF MSE improvement, LF-v1 gain preservation, and strong-CR
  preservation beyond output-only and shuffled-feature controls.

## Stage 0 Change

- Code branch:
  `codex/haze4k-brfrc-v2-representation-audit`.
- Primary variable:
  Frozen feature source for a diagnostic probe, not a training architecture.
- Source checkpoints:
  CR best checkpoint and LF-v1 best checkpoint matched to the existing route
  evidence where possible.
- Feature source:
  `DEANet.forward_with_features`, returning `dec1`, `dec2`, `dec3`, and
  `bottleneck`.
- Explicitly disabled related mechanisms:
  No 100k scout, no BRFRC-v2-Rep model implementation, no selector, no haze
  prior, no CRPlus schedule, no TeacherGuard, no Mamba/DWT branch, and no
  joint fine-tuning during Stage 0.

## Feature Groups

| Group | Inputs | Purpose |
| --- | --- | --- |
| A. output-only baseline | `I`, `J0`, `I-J0`, `LP(I)`, `LP(J0)`, `HF` | Reproduce visible CBRFRC-v1 information as the lower bound. |
| B. CR frozen features | CR `bottleneck/dec3/dec2/dec1` plus group A | Test whether CR internal features add direction/reliability signal. |
| C. LF-v1 features | LF-v1 `bottleneck/dec3/dec2/dec1` plus group A | Test whether LF-v1 gains are readable from its own representation. |
| D. CR plus LF-v1 feature contrast | feature deltas, cosine, norms, channel statistics plus group A | Test whether gain/regression signal appears in representation differences. |
| Control | shuffled versions of B/C/D feature blocks | Reject feature sets that only match a shuffled-feature control. |

## Probe Family

Use small probes only:

- linear, ridge, or logistic probes for linearly readable signal;
- tiny MLP or 1x1-conv probes for lightweight nonlinear signal;
- tiny patch-level spatial probe, bounded so it cannot become a hidden
  refiner.

Do not use a large model probe. The goal is to audit whether the frozen
representation contains signal, not to train a deployable corrector early.

## Splits

| Split | Purpose |
| --- | --- |
| random image split | Baseline generalization. |
| airlight held-out | Prevent learning only the airlight distribution. |
| beta held-out | Prevent learning only haze-density intervals. |
| CR-strength held-out | Test whether strong-CR preservation generalizes. |

## Audit Targets

| Target | Definition | Why required |
| --- | --- | --- |
| residual direction | Cosine between predicted/applied low-frequency residual and `LP(GT)-LP(J0)` | Direction is the central mechanism variable from LF-v1 and CBRFRC-v1 diagnostics. |
| LF MSE improvement | Whether corrected `LP(J)` is closer to `LP(GT)` than `LP(J0)` | Avoid probes that point roughly right but do not reduce LF error. |
| LF-v1 gain preservation | Recall on samples where LF-v1 has large gain over CR | Prevent repeating rescue-without-preservation failures. |
| strong-CR preservation | Recall on already-strong CR samples | Protect the stable baseline region. |
| confidence/reliability | Correlation between predicted confidence and actual LF/PSNR improvement | Ensure reliability is meaningful instead of a decorative mask. |

## Stage 0 Pass Line

All criteria must pass before writing a BRFRC-v2-Rep model card:

| Metric | Pass line |
| --- | --- |
| residual cosine on random split | `>= 0.20` and clearly above output-only probe |
| wrong-direction rate | `<= 35%` and clearly lower than output-only |
| LF MSE improved/regressed | at least `55/45` |
| LF-v1 gain preservation recall | `>= 0.70` |
| strong-CR preservation recall | `>= 0.70` |
| intervention precision | `>= 0.60` |
| confidence correlation | `>= 0.45` |
| held-out stability | airlight, beta, and CR-strength held-out rows must not materially collapse |
| shuffled-control gap | main feature groups must clearly outperform shuffled-feature controls |

If the audit only shows simulated gain but misses preservation, intervention
precision, confidence correlation, or shuffled-control separation, the route
is diagnostic-only and must not train.

## If Stage 0 Passes

Only then create:

```text
docs/HAZE4K_BRFRC_V2_REPRESENTATION_PLAN_20260530.md
```

The first model card should keep one primary variable:

```text
Frozen/reference CR forward:
    I -> J0
    I -> {F_bottleneck, F_dec3, F_dec2, F_dec1}

Shared representation trunk:
    concat/project(F_bottleneck, F_dec3, pooled I, pooled J0, I-J0,
                   optional LF-v1 feature contrast if Stage 0 proves useful)

Residual head:
    predict bounded feature residual dF or bounded low-frequency residual dJ_lf

Reliability head:
    predict q in [0, 1] or log_sigma from the shared trunk

Identity highway:
    output = J0 + q * bounded_residual
```

First implementation constraints:

- insertion preference: bottleneck / x8 before `mix1`, then `dec3` /
  `x_level3_mix`;
- do not place the first version only at final RGB output level;
- initialize correction exactly or near-exactly zero;
- keep residual and reliability heads on a shared trunk;
- do not add selector, haze prior, CRPlus schedule, TeacherGuard, Mamba/DWT,
  or joint baseline fine-tuning in the first scout.

## Stage 1 Preflight If Allowed

| Preflight | Pass line |
| --- | --- |
| neutral-init equivalence | max absolute `out-J0` is zero or tiny numerical noise |
| parameter overhead | preferably `<= +3%` |
| latency overhead | preferably `<= +8%` |
| branch activity | residual trunk, q head, and residual output finite, nonzero, and not saturated |
| fixed-patch 2k overfit | loss decreases, residual cosine rises, q does not collapse |
| identity stress test | full 1000-image zero-correction output equals CR |

Passing these checks only authorizes the formal scout; it is not success
evidence by itself.

## Stage 2 Scout Contract If Allowed

The formal candidate must run on AutoDL with the project non-negotiable HAZE4K
protocol:

```text
dataset = HAZE4K
epochs = 20
iters_per_epoch = 5000
total steps = 100000
bs = 16
patch_size = 256
w_loss_L1 = 1.0
w_loss_CR = 0.1
start_lr = 0.0001
end_lr = 0.000001
eval/checkpoint every 10000
save_epoch_checkpoints = false
```

The `10k`, `20k`, `30k`, `50k`, `70k`, and `90k` checks are internal gates
inside the same 100k run, not separate short-horizon launches.

## Stage 2 Mechanism Metrics If Allowed

| Metric | Purpose |
| --- | --- |
| residual cosine | Test low-frequency direction against `LP(GT)-LP(J0)`. |
| wrong-direction count | Directly track the CBRFRC-v1 failure mode. |
| LF MSE improved/regressed | Verify low-frequency error actually decreases. |
| residual norm/error ratios | Separate under-correction, overshoot, and direction failure. |
| LF-v1 gain preservation | Prevent losing LF-v1 wins. |
| LF-v1 regression rescue | Preserve useful rescue behavior. |
| strong-CR regression count | Protect the strongest CR quartile. |
| q mean/std/p10/p90 | Detect reliability collapse or all-open behavior. |
| q-improvement correlation | Test whether reliability predicts actual improvement. |
| calibration ECE/Brier | Check confidence calibration. |
| feature ablation delta | Prove gains come from representation features. |
| cost metrics | Track params, latency, VRAM, and iteration speed. |

## Stage 2 Gates If Allowed

| Step | Continue rule |
| ---: | --- |
| 10000 | No quality collapse; residual and q heads active; mean residual cosine positive; wrong-direction trending below CBRFRC-v1 `557/1000`; q not collapsed; LF MSE split near `50/50`. |
| 20000 | PSNR trajectory not like failed LF-v2 MBR; residual cosine rising; LF MSE improved greater than regressed; strong-CR regression not expanding; q/improvement correlation positive. |
| 30000 | First hard gate: near CR 30k quality, mean residual cosine `>= 0.15`, wrong-direction rate `<= 35%`, LF MSE improved/regressed `>= 55/45`, LF-v1 gain preservation `>= 65%`, strong-CR preservation `>= 70%`, q correlation `>= 0.35`. |
| 50000 | Promotion gate: PSNR at least CR `+0.10 dB` or very close to LF-v1 trajectory; LF-v1 gain preservation `>= 70%`; rescue not achieved by sacrificing gains; wrong-direction continues down; strong-CR mean delta nonnegative or regression count clearly below LF-v1. |
| 70000/90000 | Curve not degrading; q not late-collapsing; LF-v1 gain preservation and strong-CR control remain stable; best checkpoint is worth full-test diagnostics. |
| 100000 | Final full-test comparisons against CR, LF-v1, ResidualCalib, and CRPlus-v2, plus per-image matrix, residual diagnostic, q calibration report, cost report, and hard-case grid. |

## Required Updates After Stage 0

- Append audit run facts and decision to `docs/EXPERIMENT_LOG.md`.
- Add compact artifact paths and keep/delete decisions to
  `docs/HAZE4K_RUN_MANIFEST.md`.
- Update `docs/CURRENT_CONTEXT.md` with the executable conclusion.
- If Stage 0 passes, write the separate model route card before any
  implementation or formal scout.
