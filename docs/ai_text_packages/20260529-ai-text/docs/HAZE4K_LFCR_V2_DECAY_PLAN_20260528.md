# HAZE4K LFCR-v2 Decayed CRPlus Schedule

Date: 2026-05-28

Status: completed negative/neutral fair ablation. The schedule-off mechanism
was partly validated, but final quality was insufficient, so this exact decay
setting is closed and should not be promoted.

## Launch Record

- Smoke:
  `smoke-H4K-LFCR-v2-decay-runyun-20260528-091330` passed on `runyun-ts`.
  The diagnostic schedule used decay steps `1/2`; checkpoint `loss_log`
  recorded `CRPlusV2_weight [0.005]` before the weight reached zero, confirming
  the schedule disables CRPlus-v2 after the tiny decay window.
- Fair run:
  `DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455`.
- Server and checkout:
  `runyun-ts`, `/root/workspace/Dehaze-Net-audit-sync`.
- Branch/commit:
  `codex/haze4k-lfcr-v2-decay` / `9009515`.
- tmux/log:
  `h4k_lfcr_v2_decay_20260528_091455`,
  `experiment/HAZE4K/_run_logs/DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455.log`.
- Startup health:
  tmux/process active, GPU about `17263 MiB / 88%`, log reached about step
  `89/100000` with the declared fair config.

## Final Result And Decision

- Final checkpoint:
  `best.pk` and `latest.pk` both reached step `100000`.
- Final run metric:
  `32.1516 / 0.9844`.
- Independent verification:
  full-test mean `32.1518 / 0.9844`, LF scalar gate `0.019099`.
- Compact artifact:
  `experiment/HAZE4K/lfcr_v2_decay_diagnostics/DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455-final-20260528-verify`
  (`5.2M`, `48` files, no checkpoints).
- Decision:
  negative/neutral fair ablation. The route is useful evidence but not a
  promotable model candidate.

The mechanism read is mixed. The schedule did remove late CRPlus-v2 pressure
and retained some LF-v1 regression rescue: among `351` LF-v1 regression cases,
`186` improved over LF-v1 by at least `0.30 dB`, and `96` were fully rescued
against CR. However, this did not preserve LF-v1's broad gains: among `453`
LF-v1 gain cases, LFCR-v2 lost at least `0.30 dB` on `316`.

Full-test pairwise mean delta PSNR is negative against all relevant
references: CR baseline best `-0.0735`, LF-v1 best `-0.2765`, ResidualCalib
best `-0.2417`, CRPlus-v2 final `-0.2131`, and LFCR-v1 final `-0.0587`.
Residual diagnostics also remained weak: LF-v1 to LFCR-v2 cosine `0.1960`,
norm ratio `0.5780`, wrong direction `264/1000`, and LF MSE
improved/regressed `413/587`. Final loss-scale review still selected mostly
`under_dehazed_mix` at margin `0.02` (`35.9%`; `hazy` `2.3%`;
`output_lowpass` `0.0%`).

Conclusion: time-localized CRPlus-v2 pressure is not enough to combine the
early/rescue behavior of LFCR-v1 with LF-v1's final quality. Do not spend
another blind 100k run on this exact decay family; only reopen with a changed
guard/selectivity hypothesis or a cheaper preflight that directly addresses
the LF-v1 gain-case loss.

## Most Valuable Attempt

- Why this is the most valuable current attempt:
  LFCR-v1 proved that CRPlus-v2 can accelerate early training and rescue part
  of LF-v1's regression set, but constant high pressure suppresses LF-v1 late.
  A decayed schedule tests that exact failure with one primary variable. It is
  higher-value than another constant-weight search because it can validate or
  reject the "early help, late harm" explanation directly.
- Cheap preflight evidence:
  LFCR-v1 full diagnostics show best 10k trajectory, LF gate suppression,
  `182/351` large LF-v1 regression cases improved by at least `0.30 dB`, and
  late CRPlus-v2 pressure dominated by `under_dehazed_mix`.
- Earliest decisive gate:
  20k verifies that the schedule turns off cleanly; 30k is the first hard gate
  for LF gate recovery and trajectory recovery versus LFCR-v1/LF-v1.
- Expected training-time or attempt-count saving:
  If the 20k/30k gates fail, deprioritize constant CRPlus-v2 weight search and
  avoid spending more 100k runs on weight-only variants. If they pass, the next
  search narrows to decay window/min-weight tuning instead of broad mechanism
  exploration.
- What success decides:
  CRPlus-v2 is useful as an early training curriculum for LF-v1, and the next
  optimization should tune schedule shape or minimum late weight.
- What failure decides:
  The LFCR complementarity is not solved by time-localized CRPlus-v2 pressure;
  the next useful route should move to selective/guarded application or stop
  prioritizing CRPlus-v2-on-LF-v1.
- Why a cheaper diagnostic is not enough:
  The key claim is training dynamics under a changing loss weight. Existing
  checkpoints can reveal the failure pattern, but only a scheduled fair scout
  can test whether early acceleration remains after late pressure is removed.

## Hypothesis

- Prior evidence:
  LFCR-v1 constant `w_loss_crplus_v2=0.005` reached only
  `32.2098 / 0.9844` at 100k, below LF-v1, ResidualCalib, CRPlus-v2, and the
  CR best checkpoint. However, it beat all matched 10k references and the
  full-test diagnostics show real rescue behavior on LF-v1 regression cases.
- Target failure mode:
  constant high CRPlus-v2 pressure suppresses LF-v1's useful low-frequency
  branch late in training.
- Mechanism hypothesis:
  If CRPlus-v2 is kept strong only early and then decayed away, early
  optimization speed should be retained while LF-v1's late low-frequency
  correction and scalar gate recover.

## Change

- Code branch:
  `codex/haze4k-lfcr-v2-decay`.
- Primary variable:
  CRPlus-v2 weight schedule.
- Architecture/loss definition:
  same LF-v1 architecture and CRPlus-v2 loss as LFCR-v1. Only the effective
  CRPlus-v2 loss weight changes over step:
  `0.005` through step `10000`, linearly decays to `0.0` by step `20000`, then
  stays disabled.
- Enabled flags:
  `use_lf_prior=true`, `lf_prior_channels=8`, `lf_prior_pool=8`,
  `lf_prior_injection=pre_mix`, `w_loss_crplus_v2=0.005`,
  `crplus_v2_weight_schedule=linear_decay`,
  `crplus_v2_weight_decay_start_step=10000`,
  `crplus_v2_weight_decay_end_step=20000`, `crplus_v2_min_weight=0.0`.
- Explicitly disabled related mechanisms:
  ResidualCalib, selector, teacher guard, residual direction loss, low-frequency
  reconstruction loss, haze-aware mask, and conditional mask.

## Review Decision

The route passes the "is tuning still worth trying" review for one reason:
LFCR-v1 is not a pure negative. It improved `182/351` LF-v1 regression cases by
at least `0.30 dB`, fully rescued `84`, and gave the best 10k trajectory.

It does not justify another constant-weight search as the first follow-up.
Constant `0.005` lowered the LF gate (`0.0201` vs LF-v1 `0.0339`), worsened
strong CR regressions (`-0.1789 dB` vs LF-v1 `-0.0520 dB`), and left final
CRPlus selected pressure mostly on `under_dehazed_mix` (`343/1000` active).
Therefore the most valuable next trial is a time-localized schedule, not a
lower constant weight. A constant `0.003` remains a secondary fallback only if
the schedule fails.

## References

- CR baseline:
  `DEA-Net-CR-H4K-Baseline-scout-20260520-101334`, best 90k
  `32.2255 / 0.9844`.
- LF-v1:
  `DEA-Net-LF-H4K-scout-20260521-003100`, best 90k
  `32.4281 / 0.9845`.
- ResidualCalib:
  `DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654`, best 90k
  `32.3936 / 0.9845`.
- CRPlus-v2:
  `DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540`, final/best 100k
  `32.3633 / 0.9847`.
- Direct predecessor:
  `DEA-Net-LFCR-v1-w005-H4K-scout100k-20260527-231728`, final/best 100k
  `32.2098 / 0.9844`.
- LFCR-v1 final diagnostics:
  `experiment/HAZE4K/lfcr_v1_diagnostics/DEA-Net-LFCR-v1-w005-100k-20260528-084016`.
- LFCR-v2 final diagnostics:
  `experiment/HAZE4K/lfcr_v2_decay_diagnostics/DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455-final-20260528-verify`.

## Matched Gate References

| Step | CR | LF-v1 | CRPlus-v2 | LFCR-v1 |
| ---: | --- | --- | --- | --- |
| 10000 | `27.1101 / 0.9615` | `26.2651 / 0.9631` | `26.7627 / 0.9629` | `27.3004 / 0.9660` |
| 20000 | TBD | TBD | `29.2182 / 0.9714` | `28.0104 / 0.9648` |
| 30000 | TBD | `30.6253 / 0.9776` | `30.1416 / 0.9769` | `30.2119 / 0.9752` |
| 50000 | `31.2384 / 0.9817` | `31.3419 / 0.9817` | `31.3717 / 0.9824` | `30.8358 / 0.9797` |
| 90000 | `32.2255 / 0.9844` | `32.4281 / 0.9845` | `32.3067 / 0.9844` | `32.0620 / 0.9842` |
| 100000 | `32.0952 / 0.9844` | final checkpoint not promotion source | `32.3633 / 0.9847` | `32.2098 / 0.9844` |

## Mechanism Metrics

| Metric | Why it matches this route | Gate subset | Full-test artifact |
| --- | --- | --- | --- |
| matched-step PSNR/SSIM and time-to-quality | Primary goal is faster useful quality, not only final quality. | every 10k eval | `saved_data/log.txt` plus run log |
| `CRPlusV2_weight` log | Confirms scheduled loss is active early and off after 20k. | checkpoint/loss log | `loss_log` in checkpoint |
| LF scalar gate | Tests whether late CRPlus pressure suppression is removed. | every checkpoint | pairwise/per-image summary |
| LFCR vs LF-v1 mean delta | Direct predecessor quality comparison. | every 10k eval | full-test pairwise CSV |
| LF-v1 regression rescue count | Checks retained complementarity. | 30k/50k diagnostic if needed | full-test pairwise CSV |
| strong CR regression count | Checks whether the schedule avoids the v1 strong-sample harm. | 30k/50k diagnostic if needed | full-test pairwise CSV |
| residual cosine / LF MSE delta | Verifies LFCR changes do not push LF-v1 in the wrong direction. | later gate diagnostic | residual-direction artifact |
| CRPlus-v2 final loss-scale activity | Checks whether under-dehazed-mix pressure is no longer active late. | final only | loss-scale artifact |

## Fair Training Contract

- Dataset: HAZE4K.
- Total target: `20 * 5000 = 100000` steps.
- Batch/patch: `bs=16`, `patch_size=256`.
- Loss weights:
  `w_loss_L1=1.0`, `w_loss_CR=0.1`, CRPlus-v2 scheduled from `0.005` to `0`.
- Optimizer schedule:
  `start_lr=0.0001`, `end_lr=0.000001`.
- Eval/checkpoint cadence:
  every `10000` steps, `save_epoch_checkpoints=false`.

## Gates

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| 10000 | Should be near LFCR-v1 10k and above collapse line. | `CRPlusV2_weight` still near `0.005`; LF gate nonzero. | Continue if PSNR/SSIM healthy and scheduled loss active. |
| 20000 | Must not trail LFCR-v1 badly; compare to CRPlus-v2 20k. | `CRPlusV2_weight` reaches `0.0`; later logs should stop appending CRPlusV2. | Continue if schedule works and quality is plausible. |
| 30000 | Prefer close to LF-v1/CRPlus-v2 30k, or clear recovery from LFCR-v1 30k. | LF gate should be trending closer to LF-v1 than LFCR-v1. | Stop if worse than LFCR-v1 and LF gate remains suppressed. |
| 50000 | Must recover above LFCR-v1 50k and preferably be close to CR/LF-v1/CRPlus-v2. | Strong-sample risk should not exceed LFCR-v1 pattern if a diagnostic is run. | Continue only if image trajectory and mechanism are both plausible. |
| 90000/100000 | Promotion requires beating or approaching LF-v1, or materially better speed/rescue evidence with no strong regression. | Full diagnostics required. | Record as positive candidate, positive ablation, or negative fair ablation. |

## Analysis Plan

- Completed:
  full-test pairwise diagnostics against CR, LF-v1, ResidualCalib, CRPlus-v2,
  and LFCR-v1; residual direction diagnostics; and CRPlus-v2 loss-scale
  diagnostics.
- Archived in:
  `docs/EXPERIMENT_LOG.md`, `docs/CURRENT_CONTEXT.md`,
  `docs/HAZE4K_RUN_MANIFEST.md`, and this route card.
