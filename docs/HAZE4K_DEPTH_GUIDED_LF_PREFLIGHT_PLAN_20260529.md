# HAZE4K Depth-Guided LF-v1 Preflight

Date: 2026-05-29

Status: route card for a diagnostic preflight only. Do not launch a 100k
Depth-Guided LF-v1 scout unless this preflight writes
`preflight_passed_depth_guided_lf_scout_allowed`.

## Most Valuable Attempt

- Why this is the most valuable current attempt:
  LF-v1 remains the best cold-start standalone evidence, but its gains are
  highly conditional. The root diagnosis showed that LF-v1 delta PSNR is tied
  to low-frequency residual direction, while later preserve/selector/confidence
  preflights failed because they over-intervened on LF-v1 gain cases. A frozen
  relative depth prior is a new physical cue for haze thickness/transmission
  trend, not another selector over the same feature family.
- Cheap preflight evidence:
  Generate or reuse cached Depth Anything V2 Small relative depth maps for
  HAZE4K train hazy images, add depth statistics to the existing continuous
  CR/LF-v1 blend-confidence audit, and test random-image plus airlight/beta
  held-out splits before any model training.
- Earliest decisive gate:
  The preflight is the first gate. If it fails any main or held-out stability
  line, do not train Depth-Guided LF-v1.
- Expected training-time or attempt-count saving:
  A failed preflight stops the route before a 100k scout. A passed preflight
  justifies exactly one neutral-init LF-v1 branch modification and blocks
  another CRPlus schedule or binary selector attempt.
- What success decides:
  Frozen relative depth adds learnable signal for LF residual confidence,
  correction magnitude, or residual-direction risk beyond the current
  hazy/teacher-output proxy family.
- What failure decides:
  Depth maps from hazy images are not selective enough for HAZE4K LF-v1
  residual control; do not spend 100k on a depth-conditioned LF branch without
  a changed depth target or distillation strategy.
- Why a cheaper diagnostic is not enough:
  Existing wavelet, supervised preserve, and ResidualFieldConfidence preflights
  were the cheaper diagnostics over current features. The remaining cheap test
  is whether a frozen external depth prior changes the proxy audit.

## Hypothesis

- Prior evidence:
  LF-v1 best 90k remains `32.4281 / 0.9845`, with full-test mean delta about
  `+0.2030 dB` over CR. Its residual diagnosis found
  `corr(delta PSNR, residual cosine)=0.8775`. ResidualFieldConfidence had high
  simulated gain but failed preserve recall and intervention precision.
- Target failure mode:
  The current LF-side routes rescue some LF-v1 regressions but destroy too many
  LF-v1 gain cases and strong-CR cases.
- Mechanism hypothesis:
  If relative depth supplies a deployable haze-thickness/transmission trend,
  then continuous LF residual confidence should preserve more LF-v1 wins while
  still intervening on wrong-direction or over-amplitude residuals.

## Preflight Change

- Code branch:
  `codex/haze4k-depth-guided-lf-preflight`.
- Primary variable:
  Add cached frozen relative depth features to the continuous LF-v1 confidence
  proxy audit.
- Implementation:
  `code/analyze_depth_guided_lf_preflight.py` uses
  `depth-anything/Depth-Anything-V2-Small-hf` through `transformers`, caches
  one depth map per HAZE4K train hazy image, and evaluates continuous
  CR/LF-v1 blend confidence on train patches.
- AutoDL entrypoint:
  `scripts/autodl-haze4k-depth-guided-lf-preflight.sh`.
- Explicitly disabled related mechanisms:
  No DEA-Net architecture change, no 100k training, no CRPlus schedule, no
  binary selector, no Mamba/Transformer backbone replacement.

## Preflight Pass Line

The main row is:

```text
feature_set=hazy_depth_plus_teacher_outputs
head=sklearn_hgb
split_family=random_image
```

It must pass all:

- simulated gain vs LF-v1 at least `+0.25 dB`;
- LF-v1 gain preserve recall at least `0.68`;
- intervention precision at least `0.60`;
- LF-v1 regression improve recall at least `0.60`;
- strong-CR regression improve recall at least `0.60`;
- predicted-vs-oracle confidence correlation at least `0.45`.

Airlight and beta held-out stability must also satisfy:

- gain vs LF-v1 at least `0.0 dB`;
- preserve recall at least `0.62`;
- regression improve recall at least `0.45`;
- predicted-vs-oracle confidence correlation at least `0.30`.

Depth ablations are diagnostic but required in the artifact:

- `hazy_wavelet`;
- `hazy_wavelet_plus_teacher_outputs`;
- `depth_only`;
- `hazy_depth`;
- `hazy_depth_plus_teacher_outputs`;
- `hazy_shuffled_depth_plus_teacher_outputs`.

The shuffled-depth row is a sanity check. If it matches or beats the true-depth
row, the route should be treated as diagnostic failure even if the main row is
numerically close.

## If Preflight Passes

Only then write the model route card and implement the smallest LF-v1 branch
change:

```text
low = avgpool(hazy)
d = avgpool(norm_depth)
g = DepthGate([low, d, grad(d), target_hint])
delta = DepthResidual([low, d, target_hint])
prior = base_prior + depth_gate * g * delta
out = bottleneck + lf_gate * prior
```

Required implementation constraints:

- neutral init: the depth branch final layer is zero-init and step-0 output must
  match LF-v1 within tiny numerical tolerance;
- frozen/cached depth is counted as extra inference cost;
- no CRPlus-v2 or LFCR schedule is enabled in the first Depth-Guided LF scout;
- record depth gate stats, LF gate stats, residual cosine, LF MSE delta,
  strong-CR regression count, LF-v1 gain preservation, and inference latency.

## Fair Scout Gates If Allowed

The first fair scout must use the standard HAZE4K 100k target. The internal
gates are:

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| 10000 | No collapse; training health only. | LF gate/depth gate finite and non-degenerate. | Continue only if the branch is alive and stable. |
| 30000 | Must not be below LF-v1 30k. | Depth gate must not be dead; early residual metrics should not worsen. | Stop if below LF-v1 with no mechanism improvement. |
| 50000 | Must start improving failure metrics, not only approach PSNR. | Gain preservation and strong-CR control must improve versus failed routes. | Continue only if it can plausibly challenge LF-v1. |
| 100000 | Mean PSNR should beat LF-v1 by at least `+0.05 dB`. | Strong-CR delta near zero or positive, fewer LF-v1 gain losses, regression rescue at least CRPlus-v2-level, wrong-direction count down. | Promote only with both quality and mechanism evidence. |

## Run Command

Run on AutoDL inside tmux:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net
tmux new-session -d -s dg_lf_preflight \
  'bash scripts/autodl-haze4k-depth-guided-lf-preflight.sh'
```

## Required Updates After Completion

- Append run facts and decision to `docs/EXPERIMENT_LOG.md`.
- Add the compact artifact path and keep/delete decision to
  `docs/HAZE4K_RUN_MANIFEST.md`.
- Update `docs/CURRENT_CONTEXT.md` only with the current executable conclusion.
