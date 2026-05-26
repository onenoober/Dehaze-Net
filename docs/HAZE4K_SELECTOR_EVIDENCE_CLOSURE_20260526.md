# HAZE4K Selector Evidence Closure

Date: 2026-05-26

## Final Read

The selector target is real, but the deployable proxy is not good enough.

Evidence chain:

- LF-v1 is the current positive model candidate.
- ResidualCalib is a positive ablation and shows real oracle headroom with
  LF-v1.
- The GT-aware oracle is strong, so the routing target exists.
- Strict CSV proxy, rich CSV proxy, and frozen activation proxy all failed the
  predeclared pass line using full-test evidence: strict/rich used the
  1000-image three-way CSV with held-out splits, and activation-forward used
  frozen checkpoint features from all 1000 test images.
- ResidualSelector failed at the 20k gate.
- ResidualDirLoss failed at the 30k hard gate on full validation PSNR/SSIM;
  the 64-image residual review was supporting mechanism evidence, not the sole
  basis for the stop.

## Decision

Do not launch selector-v2 from the current evidence.

The only legitimate reopen condition is a changed problem definition, such as
an explicit supervised or distilled selector target, followed by a fresh
full-sample proxy audit that satisfies the same pass line.

## Sample-Size Rule

Future selector/proxy claims should follow this policy:

- use the full available evaluation set when feasible;
- otherwise predeclare a scientifically adequate sample size;
- treat smaller subsets as smoke/debug evidence only;
- never use a tiny subset to justify a training route.

## Keep

- LF-v1 positive result
- ResidualCalib positive ablation
- Oracle headroom
- Strict/rich/activation proxy failures
- ResidualSelector failure
- ResidualDirLoss failure

## Stop

- No selector-v2 100k scout from oracle evidence alone
- No further selector/structure search unless the target definition changes
