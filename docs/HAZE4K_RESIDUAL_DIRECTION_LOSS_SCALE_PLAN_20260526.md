# HAZE4K Residual Direction Loss Scale Plan

Date: 2026-05-26

Purpose: re-check the current route decision after the failed
`LFResidualSelector` run, then define the first safe step before any new
training.

## Verdict

The route review still holds.

Do not start another selector, spatial mask, stronger LF clamp, CRPlus negative,
or large backbone replacement as the next action. The highest-value next step is
a read-only scale diagnosis for a residual-direction training loss.

The candidate loss is different from the failed low-frequency L1 loss:

```text
r_pred = lowpass(out) - lowpass(hazy)
r_gt   = lowpass(clear) - lowpass(hazy)
loss_dir = mean(1 - cosine(r_pred, r_gt))
```

This tests direction agreement of the dehazing residual, not direct
low-frequency reconstruction amplitude.

## Why This Is Reasonable

Local evidence:

- LF-v1 is still the best positive LF candidate: best 90k
  `32.4281 / 0.9845`.
- ResidualCalib is positive versus the CR baseline, but not better than LF-v1:
  best 90k `32.3936 / 0.9845`.
- ResidualSelector failed at the 20k gate: `27.6830 / 0.9713`, clearly below
  baseline, LF-v1, and ResidualCalib, while selector stats stayed nearly
  constant.
- The strongest root-cause signal remains residual direction: LF-v1 full-test
  `corr(delta PSNR, residual cosine)=0.8775`, with `160` wrong-direction cases.
- Simple low-frequency L1 already failed, so the next loss must avoid forcing
  amplitude reconstruction.

Research sanity check:

- Recent dehazing work still supports frequency or wavelet reasoning, but most
  strong routes combine frequency cues with spatial/global context instead of
  applying a blind low-frequency branch.
- The practical implication for this fork is narrow: preserve DEA-Net's
  lightweight baseline and target the diagnosed LF residual failure mode.

## Added Tool

Read-only script:

```text
code/analyze_residual_direction_loss_scale.py
```

It computes:

- model PSNR and L1 on sampled images
- low-frequency L1
- residual-direction cosine loss
- residual cosine and residual norms
- candidate weighted direction-loss ratios versus L1

Outputs:

```text
per_image_loss_scale.csv
summary.json
analysis_report.md
```

## Scale Diagnostic Results

Remote checkout:

```text
/root/workspace/Dehaze-Net-audit-sync
```

Checkpoint:

```text
/root/workspace/Dehaze-Net/experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk
```

Train center-crop sample:

```text
experiment/HAZE4K/loss_scale/residual-direction-lfv1-train256-20260526/
```

- Images: `256`
- Patch size: `256`
- Mean L1: `0.009006`
- Mean residual-direction loss: `0.002301`
- Mean residual cosine: `0.997699`
- `w=0.001` ratio to L1: `0.000256`
- `w=0.003` ratio to L1: `0.000767`
- `w=0.005` ratio to L1: `0.001278`
- `w=0.010` ratio to L1: `0.002555`

Test full-image sample:

```text
experiment/HAZE4K/loss_scale/residual-direction-lfv1-test256-20260526/
```

- Images: `256`
- Mean L1: `0.029399`
- Mean residual-direction loss: `0.039146`
- Mean residual cosine: `0.960854`
- `w=0.001` ratio to L1: `0.001332`
- `w=0.003` ratio to L1: `0.003995`
- `w=0.005` ratio to L1: `0.006658`
- `w=0.010` ratio to L1: `0.013315`

Interpretation:

- The scale check supports a small first training weight.
- `w=0.005` is a conservative default: it is below `1%` of L1 on the sampled
  test full images and far below `1%` on train center crops.
- `w=0.01` is still small by this diagnostic, but the first run should avoid
  unnecessary pressure after the failed LowFreqLoss route.

## Training Candidate Prepared

Default-off training options:

```text
--w_loss_residual_dir
--residual_dir_pool
--residual_dir_warmup_steps
--residual_dir_target_norm_floor
```

Launcher:

```text
scripts/runyun-haze4k-lf-residual-dir-loss-scout.sh
```

First candidate default:

```text
LF-v1 + residual-direction loss
w_loss_residual_dir=0.005
residual_dir_pool=8
residual_dir_warmup_steps=0
```

## Remote Scale-Diagnostic Command

Run from the clean server checkout after syncing this branch:

```bash
cd /root/workspace/Dehaze-Net-audit-sync/code
/opt/anaconda/envs/py310/bin/python analyze_residual_direction_loss_scale.py \
  --dataset HAZE4K \
  --split train \
  --checkpoint ../experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk \
  --model_label DEA-Net-LF-v1 \
  --use_lf_prior \
  --lf_prior_channels 8 \
  --lf_prior_pool 8 \
  --lf_prior_gate_init 0.0 \
  --lf_prior_injection pre_mix \
  --patch_size 256 \
  --max_images 256 \
  --lowfreq_pool 8 \
  --candidate_weights 0.001,0.003,0.005,0.01 \
  --output_dir ../experiment/HAZE4K/loss_scale/residual-direction-lfv1-train256-20260526
```

Optional test-side check:

```bash
cd /root/workspace/Dehaze-Net-audit-sync/code
/opt/anaconda/envs/py310/bin/python analyze_residual_direction_loss_scale.py \
  --dataset HAZE4K \
  --split test \
  --checkpoint ../experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk \
  --model_label DEA-Net-LF-v1 \
  --use_lf_prior \
  --lf_prior_channels 8 \
  --lf_prior_pool 8 \
  --lf_prior_gate_init 0.0 \
  --lf_prior_injection pre_mix \
  --max_images 256 \
  --lowfreq_pool 8 \
  --candidate_weights 0.001,0.003,0.005,0.01 \
  --output_dir ../experiment/HAZE4K/loss_scale/residual-direction-lfv1-test256-20260526
```

## Promotion Rule

Only implement the training loss if the scale report shows that a small weight
keeps the weighted direction loss modest relative to L1.

Initial interpretation:

- `ratio_to_l1 <= 0.05`: safe enough for a first smoke/dry-run candidate.
- `0.05 < ratio_to_l1 <= 0.15`: possible, but start at the smallest weight and
  add a warmup.
- `ratio_to_l1 > 0.15`: do not train directly; change normalization or
  re-check the loss definition.

If promoted, the first training candidate should be exactly:

```text
LF-v1 + small residual-direction loss
```

No selector, no ResidualCalib, no CRPlus, no mask, no teacher guard.

## Stop Rules For Future Training

If a training candidate is implemented after the scale check, it must use the
standard fair HAZE4K 100k target and the usual gates:

- 10k/20k: stop if it repeats the LowFreqLoss collapse pattern.
- 30k: should be close to LF-v1 30k `30.6253 / 0.9783`.
- 50k: must be at least baseline 50k `31.2384 / 0.9817`, preferably near
  LF-v1 50k `31.3419 / 0.9817`.
- 90k/100k: promote only if it beats LF-v1, or matches LF-v1 while reducing
  wrong-direction and strong-baseline regressions.
