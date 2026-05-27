# HAZE4K Next Route Review

Date: 2026-05-25

Status: historical route review. It records why selector/oracle and
ResidualDirLoss were worth testing at that time. Current route status is in
`docs/CURRENT_CONTEXT.md`; selector-specific closure is in
`docs/HAZE4K_SELECTOR_EVIDENCE_CLOSURE_20260526.md`.

Purpose: combine the current HAZE4K experiment evidence with related dehazing
research, then decide the highest-value next attempt.

## Verdict

Do not start another pure mask, stronger LF clamp, post-mix, teacher-guard, or
low-pass negative run.

The most valuable next step is:

1. Run a read-only selector/oracle diagnosis on the existing baseline, LF-v1,
   and ResidualCalib outputs.
2. If the oracle/proxy signal is strong, implement one bounded
   `LFResidualSelector` candidate that starts close to LF-v1 and learns when to
   use ResidualCalib-style correction or suppress LF intervention.
3. Keep a small residual-direction loss as the next loss-side candidate, but
   only after an offline loss-scale check.

This route follows the data: the current bottleneck is not "whether frequency
information helps"; it is whether the LF residual is directionally correct and
whether the model can avoid intervening on already strong baseline cases.

## Research Reading

The related research supports a frequency/physics/contrastive direction, but it
does not support blindly adding a bigger branch.

- DEA-Net is already a strong lightweight baseline because DEConv and CGA
  improve feature learning without simply increasing model depth/width, and
  DEConv can be re-parameterized without extra inference cost:
  <https://arxiv.org/abs/2301.04805>.
- Dark channel prior and atmospheric-scattering methods remain useful because
  they provide physical interpretation of haze thickness and transmission, but
  their assumptions can break in sky or complex scenes:
  <https://people.csail.mit.edu/kaiming/cvpr09/index.html>.
- AECR-Net and C2PNet show that contrastive regularization is a valid dehazing
  training idea, especially when negatives are task-aware or curriculum-aware:
  <https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Contrastive_Learning_for_Compact_Single_Image_Dehazing_CVPR_2021_paper.html>
  and <https://cvpr.thecvf.com/virtual/2023/poster/22682>.
- Focal Frequency Loss shows a general restoration principle: spatial losses can
  miss frequency-domain errors, and adaptive frequency weighting can complement
  spatial objectives: <https://arxiv.org/abs/2012.12821>.
- Recent dehazing papers continue to move toward frequency-aware, wavelet, or
  spectrum-guided designs, including HAA-Net, MWA-Net, SAD-Net, and MCRFS-Net:
  <https://arxiv.org/abs/2407.11505>,
  <https://link.springer.com/article/10.1007/s40747-025-02076-4>,
  <https://www.nature.com/articles/s41598-025-92061-1>,
  <https://www.nature.com/articles/s41598-025-08690-z>.

The important local implication is narrow: frequency-domain reasoning is
defensible for the thesis, but the next contribution should stay lightweight and
diagnostic. A large frequency Transformer or diffusion route would be less
valuable here because it would abandon DEA-Net's existing strength and consume
too much training budget.

## Local Evidence

Current fair 100k HAZE4K results:

| Route | Best PSNR / SSIM | Decision |
| --- | ---: | --- |
| DEA-Net-CR baseline | `32.2255 / 0.9844` | strong reference |
| LF-v1 | `32.4281 / 0.9845` | current best positive LF candidate |
| ResidualCalib | `32.3936 / 0.9845` | positive ablation, below LF-v1 |
| Conservative LF | `32.1083 / 0.9843` | failed |
| Conditional LF | 30k `30.1830 / 0.9783` | stopped; mask near constant |
| Haze-Aware Mask | 30k `30.1157 / 0.9770` | stopped; mask active but ineffective |
| CRPlus low-pass negative | 10k `24.9623 / 0.9504` | failed |
| LowFreqLoss / LF+LowFreqLoss | below references | failed |
| TeacherGuard / PostMix | below references | failed |

The strongest diagnostic facts:

- LF-v1 full-test gain over baseline: mean `+0.2030 dB`, better/worse `549/451`.
- LF-v1 residual diagnosis: `corr(delta PSNR, residual cosine)=0.8775`,
  wrong-direction count `160`.
- ResidualCalib full-test gain over baseline: mean `+0.1682 dB`, better/worse
  `547/453`.
- ResidualCalib vs LF-v1: mean `-0.0347 dB`, median `+0.0271 dB`, better/worse
  `509/491`.
- Three-way winner counts: baseline `295`, LF-v1 `322`, ResidualCalib `383`.
- Three-way pattern counts: `residual_beats_both=300`,
  `lost_lfv1_gain=218`, `mitigates_lfv1_regression=110`,
  `residual_worst=165`.

This is a selector-shaped problem. ResidualCalib wins many individual images but
loses mean PSNR because the failures are large. LF-v1 has the best current mean,
but still hurts a large minority of images. A better route should choose or
blend interventions, not make every image accept the same LF correction.

## Route Ranking

### 1. Highest value: read-only selector/oracle diagnosis

Before a new training run, compute an upper bound and a proxy-feasibility check
from existing outputs.

Questions:

- What PSNR/SSIM would an oracle get if it could choose baseline, LF-v1, or
  ResidualCalib per image?
- What PSNR/SSIM would a safer two-way oracle get between LF-v1 and
  ResidualCalib only?
- Are winner groups separable using cheap cues already available at inference:
  low-frequency residual cosine proxy, dark/luma statistics, bottleneck feature
  norm, LF gate/alpha stats, or baseline-output confidence proxies?
- Does the fixed adverse 20-sample list remain adverse after oracle selection?

Decision:

- If oracle gain over LF-v1 is large and winner groups are separable, implement
  `LFResidualSelector`.
- If oracle gain is small, do not train a selector; move to residual-direction
  loss diagnosis instead.

This is the cheapest high-information step because it uses existing artifacts
and can prevent another expensive weak run.

### 2. Best training candidate: `LFResidualSelector`

Design target:

```text
lf_prior = adapter(lowpass(hazy))                 # LF-v1 residual
calib_prior = residual_calib(lowpass(hazy), x8)   # ResidualCalib residual
mix = confidence(lowpass(hazy), x8.detach())
prior = mix * lf_prior + (1 - mix) * calib_prior
out = x8 + gate * prior
```

Safer initialization:

- Start close to LF-v1, not identity and not ResidualCalib.
- Initialize `mix` with a high bias, e.g. sigmoid around `0.85-0.90`, so the
  run begins near the current best positive candidate.
- Keep the scalar LF gate behavior comparable to LF-v1.
- Bound the calibration residual amplitude; do not reuse the failed strong
  clamp/dropout/gate-L2 conservative bundle.
- Log `mix` mean/std/min/max and alpha stats.

Why this is better than another mask:

- Conditional LF failed because mask was nearly constant.
- Haze-Aware Mask failed even after the mask became active.
- Three-way analysis shows the useful choice is between residual types and
  intervention strength, not only where to apply a single residual.

Minimum success criteria:

- 30k: should be close to LF-v1 30k `30.6253 / 0.9783`; if it is only
  baseline-level, stop unless read-only diagnostics show a major regression
  reduction.
- 50k: must be at least baseline 50k `31.2384 / 0.9817`, and preferably near
  LF-v1 50k `31.3419 / 0.9817`.
- 90k/100k: promote only if it beats LF-v1 in mean PSNR, or matches LF-v1 while
  materially reducing `lost_lfv1_gain`, `residual_worst`, wrong-direction count,
  and strong-baseline regressions.

## 3. Loss-side candidate: residual-direction loss

This should not be the first long run unless selector analysis is weak or
negative.

Rationale:

- Previous low-frequency L1 failed because it forced amplitude and pixel-level
  lowpass agreement.
- The diagnostic signal points more specifically to direction: residual cosine
  tracks delta PSNR much more strongly.

Candidate training-only loss:

```text
r_pred = lowpass(out) - lowpass(hazy)
r_gt = lowpass(clear) - lowpass(hazy)
loss_dir = 1 - cosine(r_pred, r_gt)
```

Guardrails:

- First run an offline scale check on existing checkpoints.
- Use a small weight and optional warmup.
- Do not combine with CRPlus or selector in the first test.
- Stop at 10k/20k if it repeats the LowFreqLoss collapse pattern.

## Routes To Pause

- Pure spatial mask stacking: already tested as Conditional LF and Haze-Aware
  Mask; it does not fix the residual-direction problem.
- Stronger LF regularization: Conservative LF and LF+LowFreqLoss already show
  over-constraint risk.
- CRPlus low-pass negative: the current negative design collapsed at 10k.
  Revisit only after feature-distance/loss-scale diagnosis.
- TeacherGuard: the existing setting failed early and should not be reused.
- PostMix: the insertion move alone fell below both baseline and LF-v1 by 50k.
- Large backbone replacement: not aligned with the current graduation route,
  DEA-Net's lightweight advantage, or the available evidence.

## Recommended Execution Order

1. Add/read-only `analyze_selector_oracle.py`.
2. Run it on:
   `Baseline-LFv1-ResidualCalib-full-20260525`,
   `CR-vs-LF-v1-full-20260523`,
   `CR-vs-ResidualCalib-full-20260525`, and the old fixed 20-sample subset.
3. If oracle gain is meaningful, create an experiment card for
   `LFResidualSelector`.
4. Implement the selector locally on a new branch; dry-run and 2-step smoke on
   `/root/workspace/Dehaze-Net-audit-sync`.
5. Launch one fair 100k-target scout with standard internal 10k/20k/30k/50k
   gates.
6. If selector fails, stop LF architecture work and run offline scale diagnosis
   for residual-direction loss before any new training.

## Thesis Positioning

The current thesis story is already viable:

```text
Strong DEA-Net-CR baseline
-> lightweight low-frequency prior gives measurable but uneven gain
-> residual-direction diagnosis explains wins and failures
-> selector/calibration attempt targets the identified failure mode
```

Even if the next selector does not beat LF-v1, it can still strengthen the
paper if it produces a clean failure analysis: frequency priors help, but
dehazing gains depend on residual direction reliability and strong-case
protection.
