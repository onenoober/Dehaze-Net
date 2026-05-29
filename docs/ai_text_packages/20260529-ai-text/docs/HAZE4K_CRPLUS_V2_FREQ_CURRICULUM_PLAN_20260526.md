# HAZE4K CRPlus-v2 Frequency Curriculum Plan

Date: 2026-05-26

Status: first CRPlus-v2 fair scout completed. It is positive versus CR
baseline but is not a replacement for LF-v1; use this card as the route
definition, final evidence, and next-step boundary. Live run status is not
stored here; verify `docs/CURRENT_CONTEXT.md`, `docs/EXPERIMENT_LOG.md`, and
live server logs before claiming a run is active.

Purpose: define the next non-LF training route after LF-v1, ResidualCalib,
ResidualSelector, and ResidualDirLoss evidence. This card is diagnostic-first:
do not launch a long scout until the loss-scale report passes.

## Hypothesis

- Prior evidence:
  - DEA-Net-CR baseline best 90k: `32.2255 / 0.9844`.
  - LF-v1 best 90k: `32.4281 / 0.9845`, mean full-test gain about
    `+0.2030 dB`, but with high per-image variance.
  - ResidualCalib is a positive ablation, best 90k `32.3936 / 0.9845`, but it
    remains below LF-v1 and still has wrong-direction plus strong-baseline
    regression cases.
  - ResidualSelector, safe selector proxies, direct ResidualDirLoss, simple
    low-frequency L1, and the first low-pass CR negative all failed their
    gates.
- Target failure mode: current routes either over-intervene on strong samples
  or apply low-frequency pressure that does not transfer to better test-side
  mechanism metrics.
- Mechanism hypothesis: if CR uses a bounded margin/curriculum objective over
  VGG, Fourier amplitude, and low-frequency feature distances, the model should
  learn more haze-relevant separation without adding inference parameters or
  forcing direct low-frequency reconstruction.

## Change

- Code branch: `codex/haze4k-crplus-v2`.
- Primary variable: training loss only; no architecture change for the first
  candidate.
- Current step: read-only diagnostic script
  `code/analyze_crplus_v2_loss_scale.py`.
- Candidate loss shape, if the diagnostic passes:

```text
d_pos = distance(out, clear)
d_neg = distance(out, negative)
loss_ratio = mean(clamp(d_pos / (d_neg + eps), max=ratio_cap))
loss_margin = mean(relu(margin + d_pos - d_neg))  # diagnostic or late-stage

distance = VGG_L1 + frequency_weight * FFT_amplitude_L1
                 + lowfreq_weight * low_frequency_L1
```

Use ratio-first because a smoke-scale check showed that absolute combined
margin can be too easy and produce zero selected loss when VGG distance
dominates. Margin remains useful as a diagnostic and possible late-stage
hard-negative gate.

- Candidate negatives to audit before training:
  - `hazy`
  - `hazy_lowpass`
  - `output_lowpass`
  - `under_dehazed_mix`
- Explicitly disabled related mechanisms for the first training candidate:
  LF prior, ResidualCalib, ResidualSelector, ConditionalMask, HazeAwareMask,
  TeacherGuard, LowFreqLoss, and ResidualDirLoss.

## References

- Baseline run/checkpoint:
  `experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk`
  at step `90000`.
- LF-v1 reference:
  `experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk`
  at step `90000`.
- Failed predecessor to avoid repeating:
  `DEA-Net-CRPlus-P1-w005-H4K-scout-20260523-011100`, stopped at 10k with
  `24.9623 / 0.9504` after using the simple low-pass negative design.

Matched gate references:

| Step | Baseline | LF-v1 |
| ---: | --- | --- |
| 10000 | `27.1101 / 0.9615` | `26.2651 / 0.9631` |
| 20000 | `28.9030 / 0.9713` | `28.8563 / 0.9751` |
| 30000 | `30.1143 / 0.9776` | `30.6253 / 0.9783` |
| 40000 | `30.3812 / 0.9804` | `29.7611 / 0.9776` |
| 50000 | `31.2384 / 0.9817` | `31.3419 / 0.9817` |
| 90000 | `32.2255 / 0.9844` | `32.4281 / 0.9845` |

## Mechanism Metrics

| Metric | Why it matches this route | Gate subset | Full-test artifact |
| --- | --- | --- | --- |
| weighted CRPlus-v2 / L1 ratio | prevents the new loss from dominating the baseline training objective | train 256 center crops and optional test 256 full images | loss-scale `summary.json` |
| positive vs negative distance gap | checks whether negatives are usable and not denominator-collapse traps | same as scale diagnostic | per-image scale CSV |
| margin active fraction | verifies nonzero training signal without making every sample a hard violation | same as scale diagnostic | per-image scale CSV |
| frequency amplitude distance | matches the proposed frequency-aware mechanism | gate diagnostic and full-test eval if trained | loss-scale and post-gate analysis |
| per-image gain/regression split | checks whether CRPlus-v2 reduces regressions rather than shifting averages only | after candidate checkpoint | full per-image eval |

## Diagnostic Pass Line

The first training implementation is allowed only if:

- selected negatives give a nonzero bounded-ratio signal;
- selected weighted loss ratio to L1 is preferably `<= 0.05`, and must be
  `<= 0.10` for a practical first smoke weight;
- no selected negative has a strongly negative p10 distance gap that indicates
  most samples are closer to the negative than the clear target;
- the result suggests a staged curriculum, not an immediate hard-negative mix
  that repeats the CRPlus-P1 failure.

## Fair Training Contract

If promoted, the first fair scout must use:

```text
epochs=20
iters_per_epoch=5000
total steps=100000
bs=16
patch_size=256
w_loss_L1=1.0
w_loss_CR=0.1
start_lr=0.0001
end_lr=0.000001
checkpoint_interval_steps=10000
eval_interval_steps=10000
save_epoch_checkpoints=false
```

First candidate should be `DEA-Net-CR + CRPlus-v2`, with no LF prior. Only
combine with LF-v1 after CRPlus-v2 independently passes the fair scout or gives
a clearly positive mechanism result.

## Gates

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| 10000 | do not trail baseline 10k by more than about `0.3 dB` unless the mechanism is unusually strong | loss logs show CRPlus-v2 active but modest | stop on CRPlus-P1-like collapse |
| 20000 | should be close to baseline 20k | frequency/margin diagnostics must not indicate runaway pressure | continue only if curve is not clearly broken |
| 30000 | should be close to baseline 30k | per-gate diagnostic subset should show better frequency separation or reduced regressions | first hard gate |
| 50000 | must be at least close to baseline 50k | mechanism metrics should not be worse than baseline | stop if below baseline with no mechanism gain |
| 90000/100000 | promote only if full-test beats baseline or gives a strong regression/visual-quality improvement | full per-image plus frequency analysis required | consider LFCR combination only after this |

## Diagnostic Commands

Local WSL smoke:

```bash
cd /home/ubuntu/workspace/Dehaze-Net/code
/home/ubuntu/miniconda3/envs/py310/bin/python analyze_crplus_v2_loss_scale.py \
  --dataset HAZE4K \
  --split train \
  --checkpoint ../experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk \
  --model_label DEA-Net-CR-baseline \
  --output_dir ../experiment/HAZE4K/loss_scale/crplus-v2-baseline-train-smoke-20260526 \
  --max_images 8 \
  --patch_size 256
```

Remote diagnostic:

```bash
cd /root/workspace/Dehaze-Net-audit-sync
bash scripts/runyun-haze4k-crplus-v2-scale-diagnostic.sh
```

## Local Diagnostic Result

Artifact:

```text
experiment/HAZE4K/loss_scale/crplus-v2-baseline-train64-20260526/
```

Setup:

- local WSL `/home/ubuntu/workspace/Dehaze-Net`
- checkpoint:
  `experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk`
- split: HAZE4K train
- images: `64`
- patch: center-crop `256`
- selected negatives: `hazy,output_lowpass,under_dehazed_mix`
- candidate objective: bounded ratio

Main result:

- mean L1: `0.009603`
- selected combined ratio loss: `0.189611`
- selected combined margin loss at margin `0.02`: `0.000141`
- candidate weighted ratios to L1:
  - `w=0.001`: `0.0197`
  - `w=0.003`: `0.0592`
  - `w=0.005`: `0.0987`
  - `w=0.010`: `0.1975`

Interpretation:

- Absolute combined margin is too weak for the first version because VGG
  distance makes most negatives easy; keep margin as a diagnostic, not the
  first training objective.
- Ratio objective has usable signal.
- `w_loss_crplus_v2=0.003` is the first scout default because it is close to
  the preferred `<= 0.05` ratio while staying below the hard `<= 0.10` limit.
  `0.005` is still feasible by scale but should be reserved for a later
  stronger variant.

## Implementation Validation

Implemented default-off training support:

- `--w_loss_crplus_v2`
- `--crplus_v2_negative_modes`
- `--crplus_v2_start_negative_modes`
- `--crplus_v2_curriculum_steps`
- `--crplus_v2_lowpass_pool`
- `--crplus_v2_frequency_weight`
- `--crplus_v2_lowfreq_weight`
- `--crplus_v2_under_dehazed_mix`
- `--crplus_v2_ratio_cap`

Validation:

- `python -m py_compile` passed for the modified training and diagnostic files.
- Local dry-run passed with CRPlus-v2 enabled.
- Local 2-step smoke passed:
  `experiment/HAZE4K/smoke-H4K-CRPlusV2-20260526/`.
  It wrote `saved_model/latest.pk` at step `2`; checkpoint `loss_log`
  contains `CRPlusV2`, with tail values `[1.4913553, 1.4035805]`.

## Runyun Scout Evidence

Run:

```text
DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540
```

- Server: `runyun-ts`, checkout `/root/workspace/Dehaze-Net-audit-sync`.
- Launch commit: `24085db`.
- Synced compact local evidence:
  `experiment/HAZE4K/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540/`
  and
  `experiment/HAZE4K/_run_logs/DEA-Net-CRPlusV2-w003-H4K-scout100k-20260526-225540.log`.
- Large `best.pk` and `latest.pk` were not copied locally because they are
  checkpoint-sized artifacts; they remain on runyun unless explicitly needed.
- 2026-05-27 50k resume: resumed from 40k to 50k in
  `h4k_crplusv2_resume50k_20260527-143152`; watcher found the 50k eval log,
  verified checkpoint step `50000`, and stopped training.
- 2026-05-27 100k resume: after the 50k pass, resumed again from checkpoint
  step `50000` in `h4k_crplusv2_resume100k_20260527-161004` with the same
  `epochs=20`, `iters_per_epoch=5000` 100k horizon.
- 2026-05-27 final check: no matching tmux/train process, GPU idle, and
  `best.pk` / `latest.pk` both at step `100000`.

Validation curve:

| Step | PSNR | SSIM |
| ---: | ---: | ---: |
| 10000 | `26.7627` | `0.9629` |
| 20000 | `29.2182` | `0.9714` |
| 30000 | `30.1416` | `0.9769` |
| 40000 | `30.6141` | `0.9795` |
| 50000 | `31.3717` | `0.9824` |
| 60000 | `31.5598` | `0.9824` |
| 70000 | `31.9543` | `0.9838` |
| 80000 | `32.1779` | `0.9842` |
| 90000 | `32.3067` | `0.9844` |
| 100000 | `32.3633` | `0.9847` |

Interpretation:

- The route did not show the CRPlus-P1 collapse pattern.
- Final PSNR/SSIM is positive versus CR baseline best `32.2255 / 0.9844`,
  but below LF-v1 best `32.4281 / 0.9845` and ResidualCalib best
  `32.3936 / 0.9845` in PSNR.
- The final SSIM is slightly higher than LF-v1 and ResidualCalib, but the
  decision metric cannot be SSIM alone because PSNR and per-image distribution
  still favor LF-v1/ResidualCalib.

## Final Diagnostics

Artifacts synced locally under ignored `experiment/HAZE4K/`:

- `per_image_eval/CR-vs-CRPlusV2-full-100k-20260527/`
- `per_image_eval/LF-v1-vs-CRPlusV2-full-100k-20260527/`
- `per_image_eval/ResidualCalib-vs-CRPlusV2-full-100k-20260527/`
- `loss_scale/crplus-v2-final-test-100k-20260527/`

Per-image full-test comparison:

| Baseline | Mean Delta PSNR | Median Delta PSNR | Mean Delta SSIM | Better / Worse | Weak-Baseline Delta | Strong-Baseline Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CR baseline | `+0.1396` | `+0.1559` | `+0.000271` | `552 / 448` | `+0.4462` | `+0.0921` |
| LF-v1 | `-0.0633` | `-0.0296` | `+0.000235` | `488 / 512` | `+0.2187` | `-0.1116` |
| ResidualCalib | `-0.0286` | `-0.1235` | `+0.000189` | `461 / 539` | `+0.4367` | `-0.1092` |

Loss-scale full-test review:

- Mean L1: `0.021967`.
- Selected ratio objective: `0.357142`; at trained weight `0.003`, this is
  about `0.0488` of L1.
- Selected combined margin loss at margin `0.02`: `0.001901`.
- `under_dehazed_mix` remains the main active selected negative at margin
  `0.02` with active fraction `0.345`; `hazy` is mostly easy at `0.024`, and
  `output_lowpass` is easy in the combined distance at this margin.

Final interpretation:

- CRPlus-v2 is a real, useful CR-only route: it beats CR baseline by
  `+0.1396 dB` full-test mean delta and has no inference-time architecture
  cost.
- It is not the best current standalone model: PSNR is still below LF-v1 and
  ResidualCalib, and the per-image pattern shows weak-sample compensation with
  strong-sample regression against the LF-family checkpoints.
- The most valuable next use is as a component or ablation input, not as a
  replacement for LF-v1. A lighter CRPlus-v2-lite or LF+CRPlus combination
  should be justified by reducing strong-case regressions and avoiding a second
  redundant VGG-heavy constraint.

## Analysis Plan

- If diagnostic fails: record the negative-set or scale failure, do not add a
  training loss.
- If diagnostic passes: implement default-off training options and one fair
  scout launcher, then dry-run and 2-step smoke before any long run.
- Required docs to update after any run: `docs/EXPERIMENT_LOG.md`,
  `docs/CURRENT_CONTEXT.md`, and `docs/HAZE4K_RUN_MANIFEST.md`.
