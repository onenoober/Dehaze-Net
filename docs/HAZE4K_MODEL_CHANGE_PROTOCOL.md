# HAZE4K Model Change Protocol

Date: 2026-05-26

Purpose: define the required evidence chain for any future HAZE4K model,
architecture, loss, selector, mask, or guard change. The goal is to make each
candidate scientifically reviewable and compute-efficient rather than a
PSNR/SSIM-only trial.

## Core Rule

Do not launch a long fair scout until the experiment card states:

1. the failure mode being targeted;
2. the proposed mechanism;
3. the exact code/config change;
4. the mechanism-specific metrics expected to improve;
5. the matched checkpoints, references, and gate rules that will decide stop or
   continue.

PSNR and SSIM are global quality guardrails. They are necessary but not
sufficient. A candidate must also be judged against the mechanism it claims to
improve.

## Most Valuable Attempt Standard

The earlier rule "choose the highest-upside attempt, and make failure
informative" is directionally right but incomplete. It can still waste training
if the route only has a large possible payoff at 100k and no early way to
decide whether the mechanism is working.

For this project, the most valuable attempt is the candidate with the highest
route-decision value per unit of training cost. It should either get a better
checkpoint earlier, or reduce the number of future training attempts by making
the next decision clear at an early gate.

A candidate may be called the most valuable current attempt only if all of
these constraints are written before launch:

1. **Known target**: it targets a documented failure mode, complementarity gap,
   or training-speed bottleneck of the current best evidence, not a generic
   hope for higher PSNR.
2. **Cheap preflight**: it has a diagnostic from existing checkpoints, a scale
   test, proxy audit, subset analysis, or prior full-test split that makes the
   first fair scout worth the compute.
3. **Decision value**: the route card says what we will learn if it succeeds
   and what we will learn if it fails. A failed run should narrow the next
   choice, for example by deciding between lower weight, schedule, selector,
   or abandoning the mechanism.
4. **Earliest decisive gate**: the route card names the earliest gate expected
   to be informative and what evidence is required there. A route that can
   only be judged at 100k needs stronger preflight evidence than a route with
   a credible 20k or 30k mechanism check.
5. **Primary variable**: the first fair scout changes one primary variable
   whenever possible. A combination is allowed only when prior diagnostics show
   a specific complementary failure pattern and the interaction itself is the
   primary variable.
6. **Speed metric**: the test records training-efficiency evidence, such as
   matched-step curve, time-to-threshold, steps-to-baseline, iteration speed,
   or whether it reaches the current reference quality earlier.
7. **Mechanism metric**: the test includes route-specific mechanism metrics
   that can explain a win or a failure, not only PSNR/SSIM.
8. **Stop value**: if the candidate fails a written gate, the stop reason must
   state what future attempt is now deprioritized or what constrained variant
   is justified next.

The standard is not "safe and conservative." A high-upside route can be the
right choice when its failure would be decisive. The standard is also not
"try the largest possible change." If a route lacks preflight evidence,
mechanism metrics, or an early decision gate, it is not the most valuable
attempt even if the imagined final gain is large.

## Required Evidence Chain

Every new route should follow this order:

0. **Candidate value**
   - State why this is the most valuable current attempt under the standard
     above.
   - Name the cheaper diagnostic already done, the earliest decisive gate, and
     the decision that will be made if the route fails.
   - State how the route could reduce future attempts or training time.

1. **Observed failure**
   - Name the prior run or diagnostic that motivates the route.
   - Include concrete run IDs, artifact paths, and metrics.
   - State why the failure is not already explained by a simpler issue such as
     unfair schedule, dead branch, bad resume, or logging artifact.

2. **Mechanism hypothesis**
   - State one sentence in the form:
     `If we change X, metric family Y should improve because failure mode Z is
     being targeted.`
   - Keep the first fair scout to one primary variable whenever possible.

3. **Architecture or loss change**
   - Describe the exact insertion point or loss definition.
   - List all enabled feature flags.
   - List all intentionally disabled related mechanisms, so interaction effects
     are clear.

4. **Mechanism metrics**
   - Define metrics that match the route's target.
   - Define whether each metric is gate-only, full-test required, or merely
     diagnostic.
   - Define the subset or split used for fast gates and the full-test artifact
     required before making a final claim.

5. **Fair training protocol**
   - Use the standard HAZE4K 100k target unless explicitly labeled smoke or
     diagnostic.
   - Keep `epochs * iters_per_epoch = 100000` across resume.
   - Keep baseline/LF-v1/predecessor references matched by gate step when
     possible.

6. **Gate decision**
   - Compare PSNR/SSIM against baseline and direct predecessor.
   - Compare mechanism metrics against the route's stated target.
   - Continue past a weak quality gate only if the mechanism metrics provide a
     clear reason to spend the next block of compute.

7. **Post-gate analysis**
   - If stopped, record why the mechanism failed, not only the image metrics.
   - If promoted, record which mechanism metrics improved and whether the gain
     holds on full-test and hard cases.

## Mechanism Metric Examples

These examples are not a fixed checklist. Use the metrics that match the route.

| Route type | Expected mechanism | Useful metrics |
| --- | --- | --- |
| residual direction / residual calibration | Low-frequency correction points in the right direction and has a usable magnitude. | residual-direction loss, residual cosine, wrong-direction count, LF MSE delta, residual norm ratio, residual error ratio, strong-baseline regression count |
| selector / mask | The model makes non-trivial input-dependent choices and improves the intended case groups. | selector/mask mean/std/min/max, entropy or variance, branch-selection distribution, target-group delta PSNR, hardcase selection accuracy when an oracle label is available |
| teacher / no-regression guard | The guard is active at the intended step and reduces regressions without suppressing gains. | guard loss, active guard weight, guarded-regression count, strong-baseline regression count, gain/loss split by teacher strength |
| low-frequency reconstruction | The low-frequency image target improves without causing color/tone artifacts. | LF L1/MSE, luma LF MSE, color/luma bias, delta-E or objective visual metrics, PSNR/SSIM split by baseline strength |
| insertion-point or backbone change | The changed feature path carries useful signal and does not simply slow or destabilize training. | matched-step curve, branch/gate activation stats, feature-path ablation, parameter/runtime change, per-group gains/regressions |

## Standard Test Metrics

Every fair candidate gate should report the smallest useful set from these
families. Route-specific metrics may add to the list, but should not replace
the image-quality and speed evidence.

| Metric family | Required use | Examples |
| --- | --- | --- |
| image quality | gate and final | PSNR/SSIM at matched steps versus CR baseline, LF-v1 or current best, and direct predecessor |
| training efficiency | gate and final | time to gate, iteration speed, steps-to-baseline, earliest step matching a reference, curve area when available |
| route activity | gate and final | enabled flags, current scheduled weights, scalar gates, loss-component tails, non-finite checks |
| per-image split | final, and gate when cheap | mean/median delta, better/worse counts, gain/regression counts at `0.10 dB` and `0.30 dB`, weak/strong reference quartiles |
| regression control | final, and gate when cheap | strong-baseline regression count, current-best regression rescue count, current-best gain preservation |
| mechanism diagnostics | route-specific | residual cosine/wrong-direction count, LF MSE delta, CRPlus-v2 loss-scale and active negatives, selector entropy, guard active count |
| cost and deployability | final, and gate if affected | parameter change, VRAM, inference speed, training speed, added inference-time modules |

## Gate Policy

The fair run still uses validation every `10000` steps. Gates should be written
before launch.

| Step | Role | Required decision evidence |
| ---: | --- | --- |
| 10000 | sanity gate | Training health, PSNR/SSIM collapse check, proof that the new branch/loss is active, and current training speed. |
| 20000 | early trajectory gate | Matched PSNR/SSIM against baseline and predecessor, first mechanism metric check, and whether the route is improving time-to-quality. |
| 30000 | first hard gate | Continue only if close to predecessor, clearly improving the stated mechanism, or producing decisive route information worth the next block. |
| 50000 | promotion gate | Must be at least close to baseline and preferably close to predecessor, with mechanism metrics and regression control not worse. |
| 70000 | late confirmation | Continue only if both quality and mechanism metrics remain plausible. |
| 90000 | best-checkpoint check | Compare against known best-step behavior and decide whether full-test analysis is worth running. |
| 100000 | final scout point | Record final/best checkpoints, full-test metrics, mechanism metrics, cost metrics, and what next attempt is now justified or ruled out. |

For emergency compute saving, a route may be stopped using a small fixed subset
or gate-only diagnostic. Such a stop is valid for resource decisions. Formal
claims should still be backed by full-test or a clearly labeled diagnostic
artifact.

## Experiment Card Template

Use this outline for every future dated route plan.

```text
# <Route Name>

## Most Valuable Attempt

- Why this is the most valuable current attempt:
- Cheap preflight evidence:
- Earliest decisive gate:
- Expected training-time or attempt-count saving:
- What success decides:
- What failure decides:
- Why a cheaper diagnostic is not enough:

## Hypothesis

- Prior evidence:
- Target failure mode:
- Mechanism hypothesis:

## Change

- Code branch:
- Primary variable:
- Architecture/loss definition:
- Enabled flags:
- Explicitly disabled related mechanisms:

## References

- Baseline run/checkpoint:
- Direct predecessor run/checkpoint:
- Matched gate reference table:

## Mechanism Metrics

| Metric | Why it matches this route | Gate subset | Full-test artifact |
| --- | --- | --- | --- |
| ... | ... | ... | ... |

## Fair Training Contract

- Dataset:
- Total target:
- Batch/patch:
- Loss weights:
- Eval/checkpoint cadence:

## Gates

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| 10000 | ... | ... | ... |
| 20000 | ... | ... | ... |
| 30000 | ... | ... | ... |
| 50000 | ... | ... | ... |

## Analysis Plan

- If stopped:
- If promoted:
- Required docs to update:
```

## Decision Language

Use precise labels:

- `positive candidate`: beats the main reference on fair full-test and has no
  unresolved mechanism contradiction.
- `positive ablation`: improves an important reference or mechanism but is not
  the main replacement.
- `negative fair ablation`: fair run failed a written gate.
- `diagnostic only`: invalid or short schedule, smoke, dry-run, subset-only, or
  changed LR horizon.
- `inconclusive`: evidence is insufficient; state exactly what is missing.

Avoid saying a route is "promising" based only on PSNR/SSIM or only on the
training loss. The decision must name both image-quality evidence and
mechanism-specific evidence.
