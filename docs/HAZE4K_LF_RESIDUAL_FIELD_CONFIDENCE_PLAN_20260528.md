# HAZE4K LF Residual Field Confidence Plan

Date: 2026-05-28

Status: route card plus preflight contract. Do not launch the 100k scout unless
the residual-field confidence preflight returns
`proceed_to_cr_ref_residual_field_scout`.

## Most Valuable Attempt

- Why this is the most valuable current attempt:
  LFCR-v1/v2 proved that CRPlus-style pressure can rescue some LF-v1
  regressions but destroys too many LF-v1 wins. Wavelet and supervised
  preserve/intervene proxy audits also failed because the decision was too
  imprecise. A continuous residual-field confidence target tests whether the
  missing signal is learnable before spending another long run.
- Cheap preflight evidence:
  Use CR and LF-v1 best checkpoints on HAZE4K train patches to compute the
  oracle continuous blend coefficient between CR and LF-v1 outputs, then audit
  whether hazy wavelet plus teacher-output proxy features can predict that
  coefficient on random-image and degradation-held-out splits.
- Earliest decisive gate:
  Preflight is the first gate. If it fails, do not train. If it passes, the
  100k scout uses 10k as sanity, 30k as first hard gate, and 50k as promotion
  gate.
- Expected training-time or attempt-count saving:
  A failed preflight stops the route before any 100k compute. A passed preflight
  justifies exactly one scoped scout and deprioritizes another binary selector,
  mask, or CRPlus schedule sweep.
- What success decides:
  Continuous residual confidence is learnable enough to try a CR-reference
  residual-field training signal on top of ResidualCalib.
- What failure decides:
  The next search should not train another LF preserve/selector/guard head from
  the current feature family; move to a different target or a larger backbone
  only after documenting that cost.
- Why a cheaper diagnostic is not enough:
  Existing binary preserve/intervene audits already failed. This preflight is
  the cheaper diagnostic; the 100k scout is conditional on it.

## Hypothesis

- Prior evidence:
  LF-v1 remains the best cold-start standalone candidate at `32.4281 / 0.9845`.
  ResidualCalib is a positive ablation at `32.3936 / 0.9845` but still loses
  too many LF-v1 wins. LFCR-v2 decay is closed as negative/neutral at
  `32.1516 / 0.9844`, while preserving some LF-v1 regression rescue.
- Target failure mode:
  Current routes can rescue LF-v1 failures but cannot preserve enough LF-v1
  gains or strong-CR cases.
- Mechanism hypothesis:
  If the model is trained with a CR-baseline referenced residual-field
  alignment signal, then the LF residual branch should point closer to the
  useful low-frequency correction direction without applying a binary
  preserve/intervene switch.

## Change

- Code branch:
  `codex/haze4k-residual-field-confidence`.
- Primary variable:
  CR-reference residual-field auxiliary loss applied to the ResidualCalib
  architecture, conditional on the continuous-confidence preflight passing.
- Preflight definition:
  Compute per-patch CR, LF-v1, and oracle continuous blend metrics:

```text
out(c) = (1 - c) * CR + c * LF-v1
c_oracle = argmin_c MSE(out(c), GT), clipped to [0, 1]
```

  Predict `c_oracle` from deployable-ish hazy wavelet features plus teacher
  output proxy features, then simulate the predicted continuous blend.
- 100k scout definition if preflight passes:

```text
DEA-Net-LF-ResidualCalib:
  --use_lf_prior
  --lf_residual_calibration
  --lf_prior_injection pre_mix

CR-reference residual-field loss:
  pred = LP(out) - LP(CR_teacher(hazy))
  target = LP(GT) - LP(CR_teacher(hazy))
  loss = 1 - cosine(pred, target)
         + magnitude_weight * smooth_l1(norm(pred) / norm(target), 1)
```

- Enabled flags for the first scout:
  `--use_lf_prior`, `--lf_residual_calibration`,
  `--w_loss_cr_ref_residual`, and `--cr_ref_checkpoint`.
- Explicitly disabled related mechanisms:
  CRPlus-v2, residual selector, conditional mask, haze-aware mask,
  TeacherGuard, LowFreqLoss, and the old ResidualDirLoss.

## References

- CR baseline:
  `DEA-Net-CR-H4K-Baseline-scout-20260520-101334`, best step `90000`,
  `32.2255 / 0.9844`.
- LF-v1:
  `DEA-Net-LF-H4K-scout-20260521-003100`, best step `90000`,
  `32.4281 / 0.9845`.
- ResidualCalib:
  `DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654`, best step `90000`,
  `32.3936 / 0.9845`.
- LFCR-v2 decay:
  `DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455`, final/best
  `32.1516 / 0.9844`.

Matched gate references:

| Step | CR | LF-v1 | ResidualCalib |
| ---: | --- | --- | --- |
| 10000 | `27.1101 / 0.9615` | `26.2651 / 0.9631` | `26.5666 / 0.9621` |
| 20000 | `28.9030 / 0.9713` | `28.8563 / 0.9751` | `28.5005 / 0.9720` |
| 30000 | `30.1143 / 0.9776` | `30.6253 / 0.9783` | TBD |
| 50000 | `31.2384 / 0.9817` | `31.3419 / 0.9817` | `31.1396 / 0.9808` |
| 90000 | `32.2255 / 0.9844` | `32.4281 / 0.9845` | `32.3936 / 0.9845` |

## Preflight Pass Line

The main row is `hazy_wavelet_plus_teacher_outputs` with `sklearn_hgb` on
random-image splits. It must pass all:

- gain vs LF-v1 at least `+0.05 dB`;
- oracle recovery at least `0.20`;
- LF-v1 gain preserve recall at least `0.68`;
- LF-v1 regression improve recall at least `0.55`;
- intervention precision at least `0.60`;
- strong-CR regression improve recall at least `0.55`;
- predicted-vs-oracle confidence correlation at least `0.35`.

Airlight and beta held-out stability must also satisfy:

- gain vs LF-v1 at least `0.0 dB`;
- preserve recall at least `0.62`;
- regression improve recall at least `0.45`;
- predicted-vs-oracle confidence correlation at least `0.25`.

If any line fails, record the diagnostic and do not launch the 100k scout.

## Mechanism Metrics

| Metric | Why it matches this route | Gate subset | Full-test artifact |
| --- | --- | --- | --- |
| preflight gain/recovery | Checks whether continuous confidence is learnable before training. | train patch audit | residual-field preflight artifact |
| preserve recall on LF-v1 gains | Guards against repeating LFCR gain destruction. | preflight and full test | pairwise/per-image CSV |
| regression improvement on LF-v1 losses | Checks whether the route still rescues known LF-v1 failures. | preflight and full test | pairwise/per-image CSV |
| strong-CR regression control | Checks already-good samples. | preflight and full test | grouped pairwise CSV |
| residual cosine / LF MSE delta | Direct target of CR-reference residual-field loss. | 30k if mixed, final required | residual diagnostic artifact |
| CRRefResidual loss scale | Ensures the auxiliary loss is active but not dominating. | checkpoint loss log | checkpoint `loss_log` |
| LF alpha/gate stats | Checks ResidualCalib branch activity. | every checkpoint | checkpoint `loss_log` |

## Fair Training Contract

- Dataset: HAZE4K.
- Total target: `20 * 5000 = 100000` steps.
- Batch/patch: `bs=16`, `patch_size=256`.
- Base losses: `w_loss_L1=1.0`, `w_loss_CR=0.1`.
- Added loss if preflight passes:
  `w_loss_cr_ref_residual=0.001`, `cr_ref_residual_pool=8`,
  `cr_ref_residual_magnitude_weight=0.25`.
- LR: `start_lr=0.0001`, `end_lr=0.000001`.
- Eval/checkpoint cadence: every `10000` steps.
- Epoch checkpoints: disabled.

## Gates

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| preflight | Must pass the written continuous-confidence line. | Must be stable on random and held-out splits. | Stop if failed. |
| 10000 | Stop only on collapse worse than both LF-v1 and ResidualCalib by more than `0.8 dB`. | CRRefResidual and LF alpha stats must be finite and active. | Continue if healthy. |
| 20000 | Should be near ResidualCalib 20k and not far below LF-v1. | Loss should not dominate L1/CR; alpha should not be degenerate. | Continue only if trajectory or mechanism is plausible. |
| 30000 | First hard gate: should approach LF-v1/ResidualCalib or show clear residual-direction improvement. | If PSNR is weak, run compact residual diagnostic. | Stop if below references and mechanism is not improved. |
| 50000 | Must be at least baseline-level and preferably close to LF-v1. | Gain preservation and strong-CR risk must not look worse than ResidualCalib. | Continue only if it can plausibly challenge LF-v1 or teach a decisive route lesson. |
| 90000/100000 | Promotion requires beating or approaching LF-v1 with better preservation/regression control. | Full per-image and residual diagnostics required. | Record as positive candidate, positive ablation, or negative fair ablation. |

## Analysis Plan

- If preflight fails:
  update this card, `EXPERIMENT_LOG.md`, and `CURRENT_CONTEXT.md`; do not train.
- If preflight passes:
  run smoke/dry-run on runyun, then launch the fair 100k scout in tmux using
  `scripts/runyun-haze4k-lf-residual-field-confidence-scout.sh`.
- If scout reaches 30k or final:
  run pairwise per-image comparison against CR, LF-v1, and ResidualCalib, plus
  residual direction diagnostics.
