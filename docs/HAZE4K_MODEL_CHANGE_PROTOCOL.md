# HAZE4K Model Change Protocol

Date: 2026-05-26

Purpose: define the required evidence chain for any future HAZE4K model,
architecture, loss, selector, mask, or guard change. The goal is to make each
candidate scientifically reviewable rather than a PSNR/SSIM-only trial.

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

## Required Evidence Chain

Every new route should follow this order:

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

## Gate Policy

The fair run still uses validation every `10000` steps. Gates should be written
before launch.

| Step | Role | Required decision evidence |
| ---: | --- | --- |
| 10000 | sanity gate | Training health, PSNR/SSIM collapse check, and proof that the new branch/loss is active. |
| 20000 | early trajectory gate | Matched PSNR/SSIM against baseline and predecessor, plus first mechanism metric check. |
| 30000 | first hard gate | Continue only if close to predecessor or clearly improving the stated mechanism. |
| 50000 | promotion gate | Must be at least close to baseline and preferably close to predecessor, with mechanism metrics not worse. |
| 70000 | late confirmation | Continue only if both quality and mechanism metrics remain plausible. |
| 90000 | best-checkpoint check | Compare against known best-step behavior and decide whether full-test analysis is worth running. |
| 100000 | final scout point | Record final/best checkpoints, full-test metrics, mechanism metrics, and decision. |

For emergency compute saving, a route may be stopped using a small fixed subset
or gate-only diagnostic. Such a stop is valid for resource decisions. Formal
claims should still be backed by full-test or a clearly labeled diagnostic
artifact.

## Experiment Card Template

Use this outline for every future dated route plan.

```text
# <Route Name>

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
