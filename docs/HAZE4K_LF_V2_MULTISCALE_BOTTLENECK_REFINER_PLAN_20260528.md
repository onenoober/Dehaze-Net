# HAZE4K LF-v2 Multiscale Bottleneck Refiner Plan

Date: 2026-05-28

Status: stopped at the 30k hard gate on `autodl-dehaze` after
`runyun-ts` became unreachable. The replacement scout did not approach LF-v1
30k quality and did not show enough mechanism evidence to justify the remaining
100k compute. Current run state belongs in `docs/CURRENT_CONTEXT.md`; run facts
belong in `docs/EXPERIMENT_LOG.md`.

## Most Valuable Attempt

- Why this is the most valuable current attempt:
  LF-v1 remains the best cold-start standalone checkpoint, but LFCR-v1,
  LFCR-v2, WaveletPreserve, supervised preserve, and ResidualFieldConfidence
  all show the same failure: they can rescue some LF-v1 regressions but cannot
  preserve enough LF-v1 wins. The next attempt should change the representation
  that produces the low-frequency residual, not add another decision head over
  current features.
- Cheap preflight evidence:
  route evidence review, LF-v1 residual-direction diagnosis, ResidualCalib,
  LFCR-v2 final diagnostics, WaveletPreserve proxy audit, supervised preserve
  proxy audit, and ResidualFieldConfidence preflight.
- Earliest decisive gate:
  implementation preflight before training; then the 30k hard gate for a fair
  100k scout if preflight passes.
- Expected training-time or attempt-count saving:
  a failed scoped refiner would deprioritize more LF bottleneck architecture
  tweaks and move the project toward isolated warm-start or larger-backbone
  routes. A passed route would make the next attempt an LF-v2 refinement rather
  than another selector/proxy/CRPlus schedule search.
- What success decides:
  whether a small multiscale/frequency feature path can improve LF residual
  direction and LF-v1 gain preservation while keeping DEA-Net entrypoints and
  cost nearly stable.
- What failure decides:
  if neutral-init, branch activity, cost, and 30k preservation gates fail, stop
  scoped LF bottleneck architecture tweaks for now.
- Why a cheaper diagnostic is not enough:
  train-free proxies over current outputs already failed. The open question is
  whether a changed residual feature generator can learn a safer correction.

## Hypothesis

- Prior evidence:
  LF-v1 best 90k `32.4281 / 0.9845` remains the current best standalone
  checkpoint. ResidualCalib is positive but lower than LF-v1. LFCR-v2 final
  `32.1516 / 0.9844` shows scheduling CRPlus pressure is not enough. RFC has
  high simulated gain but failed preservation and precision gates.
- Target failure mode:
  current LF-side routes over-correct samples where LF-v1 or CR is already
  reliable; proxies can detect some regressions but cannot intervene precisely.
- Mechanism hypothesis:
  If a small multiscale/frequency refiner is inserted at the LF-v1 bottleneck,
  residual direction and LF MSE should improve because the branch can model
  haze-scale structure and detail-preserving skip paths before the residual is
  produced, instead of choosing among already flawed outputs afterward.

## Change

- Code branch:
  `codex/haze4k-lf-v2-mbr`.
- Primary variable:
  add one scoped LF-v2 multiscale bottleneck refiner path.
- Architecture definition:
  a compact branch near the existing LF-v1 bottleneck or `mix1` region that
  combines a low-resolution or wavelet-like context path with a detail-preserving
  skip and a learnable residual gate initialized near neutral.
- Enabled flags:
  `--use_lf_prior --lf_multiscale_refiner --lf_mbr_channels 8
  --lf_mbr_pool_sizes 4,8,16 --lf_prior_gate_init 0.0
  --lf_prior_injection pre_mix`. The official `code/train.py` and
  `code/eval.py` entrypoints stay stable; the new path is default-off.
- Explicitly disabled related mechanisms:
  CRPlus-v2 schedule, WaveletPreserve head, supervised preserve head,
  ResidualFieldConfidence loss/head, selector-v2, new teacher guard, and large
  Transformer/Mamba/diffusion backbone replacement.

## Implementation Notes

- `code/model/backbone_train.py` adds a default-off `MultiscaleBottleneckRefiner`
  inside `LowFrequencyPrior`.
- The branch builds low-pass features from pool sizes `4,8,16`, adds a
  low-frequency detail difference, fuses them with a detached bottleneck target
  hint, and produces a gated residual at the same insertion point as LF-v1.
- Neutral initialization is supplied by the existing scalar LF gate
  (`--lf_prior_gate_init 0.0`), so the candidate should match LF-v1 output
  before training while still reporting branch statistics.
- `code/preflight_lf_v2_mbr.py` checks parameter overhead, inference latency,
  neutral-init equivalence, branch non-degeneracy, and random backward health.
- `scripts/runyun-haze4k-lf-v2-mbr-preflight.sh` runs the static preflight and
  a cloud HAZE4K smoke. `scripts/runyun-haze4k-lf-v2-mbr-scout.sh` is the fair
  100k launcher and must only be used after preflight passes.

## References

- Baseline run/checkpoint:
  DEA-Net-CR best 90k `32.2255 / 0.9844`.
- Direct predecessor run/checkpoint:
  LF-v1 best 90k `32.4281 / 0.9845`.
- Mechanism references:
  ResidualCalib best 90k `32.3936 / 0.9845`; CRPlus-v2 final `32.3633 /
  0.9847`; LFCR-v2 final `32.1516 / 0.9844`; RFC preflight preserve recall
  `0.6275` and intervention precision `0.5320`.
- Matched gate reference table:
  fill exact LF-v1, CR, and direct predecessor gate rows from
  `docs/EXPERIMENT_LOG.md` before launch.

## Preflight

Preflight must pass before any fair 100k training launch:

| Check | Pass line | Stop meaning |
| --- | --- | --- |
| Parameter budget | Prefer `<= +3%` versus LF-v1; explain any exception. | If much larger, this is no longer a lightweight LF route. |
| Inference latency | Prefer `<= +8%` versus LF-v1 on the same GPU and input size. | If slower, defer until a stronger reason exists. |
| Neutral-init output | In neutral mode, fixed minibatch output should match LF-v1 within tiny numerical tolerance. | If not neutral, the route risks confounding architecture with initialization shock. |
| Branch activity | Gate/activation stats must be finite and non-degenerate after smoke or early diagnostic steps. | If dead or saturated, do not spend 100k. |
| Training smoke | No non-finite loss; checkpoint and eval path still work. | If smoke fails, fix implementation before route evaluation. |

Preflight result on `runyun-ts`:

- Run id:
  `HAZE4K-lf-v2-mbr-preflight-runyun-20260528-222102`.
- Checkout:
  `/root/workspace/Dehaze-Net-audit-sync`, branch
  `codex/haze4k-lf-v2-mbr`, commit `cc6b3c4`.
- Static cost and neutrality:
  LF-v1 params `7790690`; LF-v2 MBR params `7794114`; overhead `0.04395%`.
  LF-v1 latency `18.2281 ms`; MBR latency `18.5519 ms`; overhead `1.7766%`.
  Neutral-init max abs diff `0.0`.
- Branch/backward health:
  branch mean/std/min/max `0.002305 / 0.202575 / -0.380796 / 0.388539`;
  random backward finite; gate grad abs `0.000421999`.
- HAZE4K smoke:
  `smoke-H4K-LF-v2-MBR-runyun-20260528-222111` wrote step `2`, with
  `lf_mbr_std_tail 0.1993472`.
- Recommendation:
  `preflight_passed_launch_allowed`.

Replacement preflight result on `autodl-dehaze`:

- Context:
  `runyun-ts` became unreachable before the active runyun scout result could be
  checked, so the route was restarted from scratch on AutoDL.
- Checkout:
  `/root/autodl-tmp/workspace/Dehaze-Net`, branch `codex/haze4k-lf-v2-mbr`,
  commit `966d041`.
- Static cost and neutrality:
  LF-v1 params `7790690`; LF-v2 MBR params `7794114`; overhead `0.04395%`.
  LF-v1 latency `18.6813 ms`; MBR latency `18.9383 ms`; overhead `1.3762%`.
  Neutral-init max abs diff `0.0`.
- Branch/backward health:
  branch mean/std/min/max `0.002309 / 0.202572 / -0.383039 / 0.378891`;
  random backward finite; gate grad abs `0.00039898`.
- HAZE4K smoke:
  `smoke-H4K-LF-v2-MBR-autodl-` wrote step `2`, with
  `lf_mbr_std_tail 0.1993409`.
- Recommendation:
  `preflight_passed_launch_allowed`.

## Mechanism Metrics

| Metric | Why it matches this route | Gate subset | Full-test artifact |
| --- | --- | --- | --- |
| residual cosine vs GT correction | Directly tests low-frequency residual direction. | 30k compact diagnostic if quality is plausible. | residual diagnostic CSV/MD |
| wrong-direction count | Explains strong regressions better than global PSNR alone. | 30k compact diagnostic. | full-test residual diagnostic |
| LF MSE improved/regressed | Tests whether the refiner improves low-frequency reconstruction. | 30k compact diagnostic. | full-test LF MSE table |
| LF-v1 gain preservation | Main failure in LFCR-v1/v2 and RFC. | 30k if cheap, final required. | per-image pairwise CSV |
| LF-v1 regression rescue | Keep the useful part of LFCR/RFC evidence. | 30k if cheap, final required. | per-image pairwise CSV |
| strong-CR regression count | Guards against harming already reliable CR cases. | 30k if cheap, final required. | group breakdown |
| branch gate/activation stats | Proves the new path is alive but not saturated. | every gate. | checkpoint/log summary |
| params, latency, VRAM | Keeps the route lightweight and deployable. | preflight and final. | cost report |

## Fair Training Contract

- Dataset:
  HAZE4K.
- Total target:
  `epochs=20`, `iters_per_epoch=5000`, total `100000` steps.
- Batch/patch:
  `bs=16`, `patch_size=256`.
- Loss weights:
  keep baseline `w_loss_L1=1.0`, `w_loss_CR=0.1`; do not enable CRPlus-v2 or
  preservation losses in the first scout.
- Eval/checkpoint cadence:
  every `10000` steps; `save_epoch_checkpoints=false`.

Current scout:

- Stopped fair run:
  `DEA-Net-LF-v2-MBR-c8-H4K-scout100k-autodl-20260529-144500`.
- Cloud path:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-v2-MBR-c8-H4K-scout100k-autodl-20260529-144500`.
- Log:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/_run_logs/DEA-Net-LF-v2-MBR-c8-H4K-scout100k-autodl-20260529-144500.log`.
- Launch note:
  earlier run `DEA-Net-LF-v2-MBR-c8-H4K-scout100k-20260528-222223`
  stopped around step `487` because the launch was not detached; it produced no
  checkpoint and is diagnostic only. Later runyun run
  `DEA-Net-LF-v2-MBR-c8-H4K-scout100k-20260528-223014` became unreachable
  before a result could be checked, so it is unknown and not evidence.
- Stop note:
  AutoDL scout was stopped on 2026-05-29 after the 30k checkpoint. Validation
  curve was 10k `26.4345 / 0.9616`, 20k `28.2414 / 0.9730`, and 30k
  `29.8938 / 0.9776`. The 30k PSNR was below CR baseline 30k `30.1143` and
  LF-v1 30k `30.6253`. Checkpoint stats showed the branch was alive
  (`LF_mbr_std` tail about `0.203`) but the LF scalar gate was negative
  (`-0.02085`), so the route failed the hard gate without a clear mechanism
  reason to continue to 100k.

## Gates

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| preflight | No quality claim. | Cost, neutral-init, branch activity, and smoke pass. | Stop if any preflight pass line fails. |
| 10000 | No collapse; should be in the same broad trajectory family as LF-v1/CR. | Branch alive; no saturated gate; no non-finite stats. | Stop only for collapse or dead branch. |
| 20000 | Should be clearly above CR collapse line and not obviously worse than failed LF routes. | First branch stats should show useful variance. | Continue only if quality and activity are plausible. |
| 30000 | First hard gate: should approach LF-v1 30k or show a clear mechanism improvement worth more compute. | Residual cosine/wrong-direction/LF MSE and LF-v1 gain preservation must improve versus the most relevant failed route. | Stop if below references and mechanism is not improved. |
| 50000 | Must be close to LF-v1 or show strong mechanism evidence. | Preservation and strong-CR regression cannot be worse than LFCR-v2/RFC patterns. | Stop if it repeats rescue-without-preservation. |
| 90000 | Compare against LF-v1 best-step behavior. | Full or compact diagnostics should justify final evaluation. | Continue to final only if promotable or diagnostically decisive. |
| 100000 | Final/best comparison against CR, LF-v1, ResidualCalib, CRPlus-v2, and LFCR-v2. | Full per-image, residual, and cost diagnostics required. | Promote only if mean quality or preservation/residual mechanism is clearly better. |

Gate result on 2026-05-29: the AutoDL replacement scout failed the 30k hard
gate and was stopped. This deprioritizes another small neutral LF bottleneck
refiner with the same insertion point and no stronger preservation objective.

## Analysis Plan

- If stopped:
  record whether the failure was cost, dead branch, training curve, residual
  direction, LF MSE, LF-v1 gain preservation, or strong-CR regression. State
  which future route is deprioritized.
- If promoted:
  sync compact diagnostics, update `EXPERIMENT_LOG.md`,
  `HAZE4K_RUN_MANIFEST.md`, this route card, and the main LFCR plan. Then run
  full per-image comparison and visual review before any larger training claim.
- Required docs to update:
  `docs/CURRENT_CONTEXT.md`, `docs/EXPERIMENT_LOG.md`,
  `docs/HAZE4K_RUN_MANIFEST.md`, this route card, and
  `docs/DEA_NET_LFCR_HAZE4K_PLAN.md`.
