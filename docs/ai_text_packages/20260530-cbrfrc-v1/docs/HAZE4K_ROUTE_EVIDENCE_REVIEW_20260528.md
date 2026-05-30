# HAZE4K Route Evidence Review

Date: 2026-05-28

Status: system evidence review for choosing the next HAZE4K model route,
updated after LFCR-v2 decay, WaveletPreserve, supervised preserve, and
ResidualFieldConfidence evidence. It is not a route card and does not authorize
a new long scout by itself. Current active run state still belongs in
`docs/CURRENT_CONTEXT.md`; single-run facts belong in
`docs/EXPERIMENT_LOG.md`.

## Purpose

The goal is to stop treating the existing HAZE4K work as a list of unrelated
model swaps. This review reorganizes the evidence by mechanism:

1. what each route actually proved;
2. which failure modes recur across routes;
3. where there is real complementarity versus only GT-oracle headroom;
4. which next attempt has the highest route-decision value per training cost.

Scope is HAZE4K only. Literature trends should be used as hypothesis sources,
not as permission to replace the current DEA-Net/LF evidence chain with a large
new backbone.

## Supplemental Local Aggregation

A local read-only aggregation was generated from existing 1000-image full-test
per-image CSVs. No model forward pass, training, or cloud checkpoint sync was
run for this aggregation.

Artifact:

```text
experiment/HAZE4K/route_evidence_review/HAZE4K-route-evidence-review-20260528/
```

Inputs:

- `per_image_eval/CR-vs-LF-v1-full-20260523/per_image_metrics.csv`
- `per_image_eval/CR-vs-ResidualCalib-full-20260525/per_image_metrics.csv`
- `per_image_eval/CR-vs-CRPlusV2-full-100k-20260527/per_image_metrics.csv`
- `lfcr_v1_diagnostics/DEA-Net-LFCR-v1-w005-100k-20260528-084016/pair_cr_vs_lfcr/per_image_metrics.csv`

Main outputs:

- `model_per_image_matrix.csv`: joined CR, LF-v1, ResidualCalib, CRPlus-v2,
  and LFCR-v1 per-image PSNR/SSIM table.
- `model_summary.csv`: mean metrics, winner count, weak/strong CR split.
- `pairwise_delta_matrix.csv`: all pairwise mean deltas and gain/loss counts.
- `lfv1_complementarity_summary.csv`: rescue/preservation metrics relative to
  LF-v1.
- `oracle_summary.json`: GT-oracle ceilings for candidate sets.
- `group_breakdown.csv`: airlight, beta, and CR-strength group deltas.
- `top_route_conflict_cases.csv`: high-conflict samples for future visual
  review.

## Executive Read

The current best standalone evidence remains **LF-v1**:

| Model | Mean PSNR | Mean SSIM | Delta vs CR | Winner Count | Weak CR Delta | Strong CR Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CR baseline | `32.2253` | `0.9844` | `0.0000` | `187` | `0.0000` | `0.0000` |
| LF-v1 | `32.4283` | `0.9845` | `+0.2030` | `204` | `+0.4921` | `-0.0520` |
| ResidualCalib | `32.3936` | `0.9845` | `+0.1682` | `250` | `+0.6354` | `-0.1171` |
| CRPlus-v2 | `32.3649` | `0.9847` | `+0.1396` | `176` | `+0.4462` | `+0.0921` |
| LFCR-v1 w0.005 | `32.2105` | `0.9844` | `-0.0148` | `183` | `+0.2136` | `-0.1789` |

This table says two things at once:

- LF-v1 is the best single checkpoint by mean PSNR and should stay the main
  standalone reference.
- The winner counts are spread across all five models, including CR and
  LFCR-v1. There is real per-image complementarity, but it is not currently
  deployable by the failed selector/proxy evidence.

The GT-oracle ceiling is large:

| Oracle set | Mean PSNR | Gain vs CR | Gain vs LF-v1 | Winner counts |
| --- | ---: | ---: | ---: | --- |
| LF-v1 or ResidualCalib | `33.0034` | `+0.7781` | `+0.5751` | LF-v1 `491`, ResidualCalib `509` |
| LF-v1 or CRPlus-v2 | `32.9373` | `+0.7120` | `+0.5090` | LF-v1 `512`, CRPlus-v2 `488` |
| LF-v1, ResidualCalib, or CRPlus-v2 | `33.2308` | `+1.0055` | `+0.8026` | LF-v1 `347`, ResidualCalib `361`, CRPlus-v2 `292` |
| CR, LF-v1, ResidualCalib, or CRPlus-v2 | `33.3975` | `+1.1722` | `+0.9692` | CR `224`, LF-v1 `255`, ResidualCalib `297`, CRPlus-v2 `224` |
| all five current outputs | `33.5098` | `+1.2845` | `+1.0815` | CR `187`, LF-v1 `204`, ResidualCalib `250`, CRPlus-v2 `176`, LFCR-v1 `183` |

Read this as an upper-bound warning, not a selector go-ahead. The selector
route already failed strict/rich/activation deployable proxy audits. The oracle
headroom says the target is real; it does not say the current model can infer
the target safely.

## Route Coordinates

| Route family | Mechanism tested | Evidence class | Current decision |
| --- | --- | --- | --- |
| CR baseline | Original DEA-Net-CR training reference | Strong baseline | Keep as the fixed comparison anchor. |
| LF-v1 | Lightweight bottleneck low-frequency prior | Positive candidate | Keep as the current best standalone route. |
| Conservative LF / PostMix / LowFreqLoss | Gate limiting, insertion move, direct low-frequency loss | Negative ablations | Do not repeat simple LF weakening, insertion move, or low-frequency L1. |
| Conditional LF / HazeAwareMask | Spatial selection over LF correction | Negative fair ablations | Do not restart simple mask routes; selection alone did not fix direction/magnitude. |
| ResidualCalib | Direction/magnitude calibration of LF residual | Positive ablation | Keep as mechanism evidence; not a replacement for LF-v1. |
| ResidualDirLoss | Direct residual direction loss | Negative fair ablation | Do not resume current loss/scale; direct direction pressure failed the 30k gate. |
| Selector/proxy | Per-image route choice among LF-family outputs | Closed diagnostic | Reopen only with a changed supervised/distilled target and fresh full-sample proxy audit. |
| CRPlus-v2 | Frequency/degradation-aware contrastive curriculum loss | Positive CR-only component | Keep as no-inference-cost loss ablation and possible curriculum signal. |
| LFCR-v1 constant w0.005 | LF-v1 plus constant high CRPlus-v2 pressure | Negative fair ablation with useful mechanism evidence | Do not repeat constant high weight; it proves early help and late suppression. |
| LFCR-v2 decay | Time-localized CRPlus-v2 on LF-v1 | Negative/neutral fair ablation with useful mechanism evidence | Do not continue this schedule family without a changed selectivity mechanism. |
| WaveletPreserve proxy | Hazy wavelet/degradation features and frozen activation features for preserve/intervene choice | Failed preflight | Do not train current WaveletPreserve gate or wavelet+activation preserve scout. |
| Supervised preserve proxy | CR/LF-v1 teacher-labeled patch preserve/intervene target | Failed preflight | Do not train this preserve-head target; the reliable head missed preservation and strong-CR recall. |
| ResidualFieldConfidence | Continuous CR-reference residual-field confidence target | Failed preflight with signal | Do not launch LF-RFC v1; high simulated gain was not selective enough. |

## Failure Modes

### 1. Weak-sample help versus strong-sample damage

The current family repeatedly helps weak CR samples but risks strong CR samples.
This split is more informative than only reading global PSNR:

- LF-v1: weak CR `+0.4921`, strong CR `-0.0520`.
- ResidualCalib: weak CR `+0.6354`, strong CR `-0.1171`.
- CRPlus-v2: weak CR `+0.4462`, strong CR `+0.0921`.
- LFCR-v1: weak CR `+0.2136`, strong CR `-0.1789`.

CRPlus-v2 is the only current route with a positive strong-CR mean delta, which
is why it remains useful even though its global PSNR is below LF-v1. LFCR-v1 is
the opposite: it keeps some weak-sample help but makes the strong-sample risk
larger.

### 2. LF-v1 gains are hard to preserve

LF-v1 has `453` images with at least `+0.30 dB` over CR and `351` images with
at least `-0.30 dB` regression. Candidate behavior on those two sets:

| Candidate | Rescue >=0.30 on LF-v1 regressions | Full rescue vs CR | Preserve within 0.10 on LF-v1 gains | Lose >=0.30 on LF-v1 gains |
| --- | ---: | ---: | ---: | ---: |
| ResidualCalib | `188` | `108` | `200` | `218` |
| CRPlus-v2 | `200` | `108` | `157` | `267` |
| LFCR-v1 w0.005 | `182` | `84` | `163` | `264` |

The common pattern is rescue without preservation. Routes can fix a meaningful
part of LF-v1's regression set, but they destroy too much of LF-v1's gain set.
The next successful route must explicitly preserve LF-v1 wins, not only improve
LF-v1 losses.

### 3. Direction and low-frequency MSE remain causal signals

The LF-v1 root diagnosis showed `corr(delta PSNR, residual cosine)=0.8775`.
LFCR-v1 repeats the same story: compared with LF-v1, LFCR-v1 has lower LF gate,
wrong-direction count `245/1000`, LF MSE regressed/improved `537/463`, and
mean `LFCR - LF-v1 = -0.2178 dB`.

Therefore, the next LF-side route should not be another mask or scalar-gate
tweak. It should only be considered if it directly reduces one of:

- wrong residual direction;
- low-frequency MSE regression;
- strong-baseline regressions;
- LF-v1 gain destruction.

### 4. Selector headroom is real but currently unusable

The local aggregation strengthens the selector story: all-five oracle reaches
`33.5098`, which is `+1.0815 dB` over LF-v1. But the deployable proxy audits
already failed. This creates a hard boundary:

- It is valid to write about oracle complementarity and failure to deploy it.
- It is not valid to launch selector-v2 from oracle evidence alone.
- A future selector route must change the target definition, for example a
  supervised/distilled selector target, then pass a fresh full-sample proxy
  audit before any 100k training.

### 5. CRPlus-on-LF scheduling is now closed as a first-order fix

LFCR-v1 constant `w=0.005` is not a pure negative. It gave the best early 10k
trajectory and rescued `182/351` large LF-v1 regressions by at least `0.30 dB`.
But it ended below LF-v1, CRPlus-v2, ResidualCalib, and even the CR best
checkpoint. Final diagnostics show lowered LF gate and late selected CRPlus
pressure dominated by `under_dehazed_mix`.

LFCR-v2 decay was the right decisive follow-up, and it failed as a final model:
best/final `32.1516 / 0.9844`, independent verify `32.1518 / 0.9844`, below
CR best, LF-v1, ResidualCalib, CRPlus-v2, and LFCR-v1 final. The schedule-off
mechanism partly worked and LF-v1 regression rescue remained, but it still lost
at least `0.30 dB` on `316/453` LF-v1 gain cases.

This closes blind CRPlus-on-LF weight and decay scheduling as the next most
valuable route. CRPlus-v2 remains useful as CR-only evidence and as a
no-inference-cost ablation, but the next LF-side attempt must change the
representation that produces the residual or its preservation behavior, not
only the CRPlus timing.

## Literature Cross-Check, 2023-2026

Last checked: 2026-05-28. The literature below is used as hypothesis evidence,
not as a replacement for this repository's per-image HAZE4K evidence chain.
Where possible, prefer official proceedings/project pages; arXiv is used for
recent work without a stable proceedings page.

### Solution Maturity Against Current Bottlenecks

| Core bottleneck from this review | Recent literature signal | Local cross-check | Current answer |
| --- | --- | --- | --- |
| A stronger backbone might improve global PSNR. | DehazeFormer and later Transformer/Mamba image-restoration work show that better global modeling can lift dehazing/restoration benchmarks. | Local failures are not capacity-only: LF-v1 is already positive, CRPlus-v2 helps strong-CR cases, and oracle winners are spread across existing outputs. | A large backbone replacement is still too expensive for the next step. A scoped bottleneck/frequency representation change is now reasonable if it keeps DEA-Net entrypoints and has strict cost and preservation gates. |
| Low-frequency and frequency signals help but can damage strong cases. | Wavelet/Fourier/dual-domain/Retinex papers repeatedly separate low-frequency haze, detail restoration, and color/illumination correction. | This matches LF-v1, ResidualCalib, and CRPlus-v2 positives, but failed LowFreqLoss, HazeAwareMask, ResidualDirLoss, LFCR-v1, and LFCR-v2 show that coarse LF pressure is unsafe. | The next usable form should be a representation-level LF/multiscale refiner with detail-preserving skip, not another scalar LF loss, mask, or schedule. |
| The model needs to know when not to apply a correction. | PromptIR, DA-CLIP, PTTD, HazeCLIP, LMHaze, and other degradation-aware/prompt/MoE work support input-conditioned restoration. | Selector, WaveletPreserve, supervised preserve, and continuous-confidence proxies all failed preservation or precision gates. Current features cannot infer the preserve/intervene target safely. | Do not train another head over current features. Reopen only if the representation itself changes or a new proxy clears a full-sample audit. |
| Strong CR and LF-v1 wins must be preserved. | Recent signal-preservation and teacher/pseudo-label work supports guarding or selecting reliable regions/features instead of always rewriting them. | TeacherGuard failed, LFCR-v1/v2 lost too many LF-v1 gain cases, and RFC over-intervened despite high simulated gain. | Preservation is mandatory for the next architecture route, not a post-hoc diagnostic. |
| Real-world dehazing generalization may need priors, prompts, or pseudo-labels. | CORUN, Dehaze-RetinexGAN, PromptHaze, PTTD, and Diff-Dehazer address unpaired/real-world haze with physical, Retinex, prompt, or diffusion priors. | HAZE4K here is a supervised synthetic benchmark with strict PSNR/SSIM gates. These papers help later real-world discussion, but they do not explain current HAZE4K per-image regressions by themselves. | Defer as a separate phase. Do not mix real-world adaptation with the current LFCR-v2 HAZE4K decision. |
| Diffusion/generative models may improve perceptual quality. | Diffusion dehazing/restoration papers use strong generative priors and sometimes frequency/physics guidance. | Current promotion is still PSNR/SSIM plus mechanism diagnostics; diffusion can hallucinate or trade fidelity for perceptual quality and usually adds inference cost. | Not a near-term solution for this HAZE4K route. Keep only as later visual-refiner or real-world reference. |

### Cross-Validated Conclusions

1. **Frequency evidence is real.** Independent frequency, wavelet, Retinex, and
   physics-guided papers agree with the local observation that haze removal is
   not purely high-level semantic restoration. This supports keeping LF-v1,
   ResidualCalib, and CRPlus-v2 as meaningful evidence.
2. **Frequency evidence does not license coarse low-frequency pressure.** The
   local failures of LowFreqLoss, HazeAwareMask, ResidualDirLoss, and LFCR-v1
   are consistent with recent papers that treat low/high frequency differently
   instead of using a single global LF penalty.
3. **Degradation-aware routing is plausible but not yet deployable here.**
   Prompt/MoE/CLIP-style work supports the idea of input-conditioned behavior,
   but this repo already tested inference-safe selector proxies and they failed.
   Literature changes the next target design, not the current decision.
4. **Preservation is the missing metric.** Recent restoration work increasingly
   emphasizes signal preservation, pseudo-label reliability, and adaptive
   conditioning. Locally, the decisive missing quantity is not only better mean
   PSNR; it is LF-v1 gain preservation and strong-CR regression control.
5. **The next attempt should change representation, not another decision head.**
   LFCR-v2, WaveletPreserve, supervised preserve, and RFC all found signal but
   failed preservation/selectivity. The best current route-decision value is a
   small LF/multiscale bottleneck refiner that can alter the residual features
   themselves while preserving LF-v1 wins by construction and by gate.

### What Counts As A Solved Problem Here

| Problem | Solved enough for next action? | Required evidence before a 100k scout |
| --- | --- | --- |
| LFCR constant-weight suppression | Yes. LFCR-v1 diagnosed it and LFCR-v2 decay tested the obvious schedule follow-up. | Do not spend another 100k on weight/decay search without a new selectivity mechanism. |
| Low-frequency direction/magnitude errors | Partially. Literature and local residual diagnostics agree on the target. | A route card for a representation-level residual refiner plus cheap cost/smoke checks before any 100k scout. |
| Strong-case/no-regression guard | Partially. Literature supports the idea; local TeacherGuard setting failed and preserve proxies were not safe. | Preserve metrics must be embedded in the next architecture gates; do not train a standalone guard head from current features. |
| Deployable selector | No. Oracle headroom is real, but current strict/rich/activation, wavelet, supervised, and continuous-confidence proxies failed. | A changed representation or target plus full-sample proxy audit that clears the written pass line. |
| Scoped bottleneck representation change | Yes for planning, not for training. | A dated route card with parameter/runtime limits, neutral-init smoke, branch-activity checks, and LF-v1 gain-preservation gates. |
| Large backbone replacement | No for the current phase. | Only after the scoped representation route and isolated warm-start route fail or plateau, with complexity/runtime and per-image split gates. |
| Diffusion or real-world prompt adaptation | No for HAZE4K PSNR route. | Separate objective: real-world or perceptual benchmark, not mixed into current HAZE4K decision. |

### Integrated Literature Index

| ID | Source | What it contributes | Use in this repo |
| --- | --- | --- | --- |
| L1 | [DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention](https://arxiv.org/abs/2301.04805) | Defines the upstream DEA-Net/DEAB family that this fork extends. | Keep CR as the fixed reproducible anchor; compare small mechanism routes against it. |
| L2 | [Vision Transformers for Single Image Dehazing / DehazeFormer](https://arxiv.org/abs/2204.03883) and [TIP record](https://doi.org/10.1109/TIP.2023.3256763) | Shows that Transformer design can be adapted to dehazing, but also that generic Swin choices needed task-specific changes. | Evidence against blindly swapping in a generic large backbone without route-specific diagnostics. |
| L3 | [PromptIR: Prompting for All-In-One Image Restoration](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html) | Uses learned prompts to encode degradation-specific information for blind restoration. | Inspiration for future selector/guard target design, not a go-ahead for current failed proxies. |
| L4 | [DA-CLIP / Controlling Vision-Language Models for Multi-Task Image Restoration](https://openreview.net/forum?id=t3vnnLeajU) | Learns degradation-aware CLIP features for restoration conditioning. | If selector is reopened, consider a learned degradation embedding as a candidate feature family, then audit it before training. |
| L5 | [Prompt-Based Test-Time Real Image Dehazing](https://eccv.ecva.net/virtual/2024/poster/303) | Uses test-time feature-statistic adaptation guided by visual prompts for real hazy images. | Useful later for real-world adaptation; do not mix with current supervised HAZE4K route gates. |
| L6 | [WaveDH: Wavelet Sub-bands Guided ConvNet for Efficient Image Dehazing](https://arxiv.org/abs/2404.01604) | Uses wavelet sub-bands to build a compact frequency-aware dehazing model. | Supports frequency decomposition as a small-route idea, but local gates must still protect LF-v1 wins. |
| L7 | [DFP-Net: unsupervised dual-branch frequency-domain processing for single image dehazing](https://doi.org/10.1016/j.engappai.2024.109012) | Separates high- and low-frequency processing and fuses frequency/spatial information. | Supports dual-branch frequency reasoning; does not support repeating simple LF L1. |
| L8 | [WTCL-Dehaze: Wavelet Transform and Contrastive Learning](https://arxiv.org/abs/2410.04762) | Combines wavelet decomposition with contrastive learning for real-world robustness. | Supports CRPlus-style frequency contrast as a mechanism, especially as no-inference-cost training evidence. |
| L9 | [MambaIR: A Simple Baseline for Image Restoration with State-Space Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02740.pdf) | Shows efficient long-range restoration backbones with local enhancement and channel interaction. | Backbone reference only; not a near-term replacement unless current small routes are exhausted. |
| L10 | [U-shaped Vision Mamba for Single Image Dehazing](https://arxiv.org/abs/2402.04139) and [LMHaze](https://arxiv.org/abs/2410.16095) | Applies Mamba to dehazing; LMHaze adds haze-intensity-aware MoE behavior. | Supports the "input-conditioned correction" idea, but selector deployability must be proven locally. |
| L11 | [WDMamba: Wavelet Degradation Prior Meets Vision Mamba](https://arxiv.org/abs/2505.04369) | Reports a haze-specific wavelet prior where haze information is concentrated in low-frequency components, followed by detail enhancement. | Strong literature match for residual direction/magnitude and LF gain-preservation metrics. |
| L12 | [Real-world Image Dehazing with Coherence-based Pseudo Labeling and Cooperative Unfolding Network](https://nips.cc/virtual/2024/poster/95790) | Combines physical unfolding with mean-teacher pseudo labels and coherence weighting for real-world dehazing. | Teacher/pseudo-label inspiration for a redesigned guard; current TeacherGuard setting remains failed. |
| L13 | [Dehaze-RetinexGAN](https://ojs.aaai.org/index.php/AAAI/article/view/32862) | Uses Retinex decomposition, DTCWT attention, and weakly supervised adversarial learning for real-world dehazing. | Supports illumination/reflectance and frequency decomposition ideas; better suited to real-world phase. |
| L14 | [PromptHaze](https://ojs.aaai.org/index.php/AAAI/article/view/33024) | Uses depth prompts from a foundation depth model for real-world dehazing. | Possible future degradation cue; not usable as HAZE4K selector evidence without a proxy audit. |
| L15 | [Diff-Dehazer](https://ojs.aaai.org/index.php/AAAI/article/view/32469) and [RSHazeDiff](https://arxiv.org/abs/2405.09083) | Uses diffusion priors, unpaired training, and/or Fourier-aware guidance. | Defer for real-world/perceptual work; not the next PSNR-first HAZE4K route. |
| L16 | [Beware of Aliases - Signal Preservation is Crucial for Robust Image Restoration](https://arxiv.org/abs/2406.07435) | Argues that restoration robustness depends on preserving signal paths, using frequency-domain operations to reduce aliasing. | Supports making preservation an explicit gate: LF-v1 gain cases and strong-CR cases must not be destroyed. |

The main research story is not "DEA-Net needs a bigger architecture." It is:
low-frequency and frequency-aware mechanisms contain useful signals, but the
hard part is **when not to apply them and how to preserve already-correct
outputs**.

## Pareto View To Maintain

Future reports should keep these Pareto axes:

1. final PSNR/SSIM versus training curve speed;
2. weak-CR gain versus strong-CR regression;
3. LF-v1 regression rescue versus LF-v1 gain preservation;
4. oracle headroom versus deployable proxy quality;
5. inference cost versus mean PSNR;
6. mechanism health versus global score.

Under this view:

- LF-v1 is the best standalone mean-PSNR point.
- CRPlus-v2 is Pareto-relevant because it has no inference-time cost and the
  best strong-CR behavior.
- ResidualCalib is Pareto-relevant as a mechanism ablation and weak-sample
  booster, but not as an LF-v1 replacement.
- LFCR-v1 and LFCR-v2 are not Pareto-relevant as final models, but they are
  route-decision relevant because they show CRPlus-on-LF can rescue some
  regressions while destroying too many LF-v1 gains.

## Next Decision Tree

### Immediate next step

The next most valuable cold-start architecture direction is **LF-v2
multiscale bottleneck refiner**: a small frequency/multiscale feature path near
the LF-v1 bottleneck or `mix1` region, with a detail-preserving skip and
near-neutral initialization. It should change the representation that produces
the residual, not add another selector/proxy over the same outputs.

This direction has the best current route-decision value because:

- the oracle target is real, but every current deployable head over existing
  features over-intervenes;
- residual direction and LF MSE remain causal signals, so changing the residual
  feature generator is more informative than another label/head attempt;
- WaveDH/WDMamba/MambaIR-style literature supports small frequency or
  long-range restoration modules, but the local evidence argues for a scoped
  DEA-Net insertion instead of a full backbone swap;
- a failed scoped refiner would decisively deprioritize more bottleneck LF
  architecture tweaks and move the project toward warm-start/backbone-scale
  routes.

Use the dated route card
`docs/HAZE4K_LF_V2_MULTISCALE_BOTTLENECK_REFINER_PLAN_20260528.md`. Before any
100k scout, pass its cheap checks:

- parameter/runtime budget: preferably `<= +3%` parameters and `<= +8%`
  inference latency versus LF-v1, or explain the exception;
- neutral-init smoke: the added branch must not perturb LF-v1 output before
  training beyond tiny numerical tolerance if configured as residual-gated;
- branch activity: after smoke/early steps, its gate/activation statistics must
  be non-degenerate;
- first hard gate: by 30k it must be close to LF-v1 trajectory or show
  materially better residual direction/LF MSE and LF-v1 gain preservation.

### Lower-priority follow-up families

1. **Representation-aware preservation guard**
   A guard may still be useful, but only coupled to a changed feature path or a
   new target that passes preflight. Do not train a standalone preserve head
   from the current wavelet/activation/teacher-output feature families.

2. **Residual direction/magnitude correction v2**
   Still valid only if it changes the architecture or target enough to reduce
   wrong-direction count or LF MSE regression. Another scalar loss is low value.

3. **Supervised/distilled selector target**
   Oracle headroom remains large, but current selector/proxy evidence blocks
   training. Reopen only with a changed representation or a proxy audit that
   passes preservation and precision gates.

4. **CRPlus-v2-lite or late-minweight schedule**
   Low priority after LFCR-v2. Keep CRPlus-v2 as a CR-only ablation unless a
   new mechanism explicitly protects LF-v1 gain cases.

## What Not To Do Next

- Do not start a large Transformer/Mamba/diffusion backbone replacement as the
  next cold-start attempt; the next architecture step should stay scoped and
  gated.
- Do not launch another selector-v2 from oracle headroom alone.
- Do not run another constant high-weight LFCR scout.
- Do not run another LFCR-v2-style weight/decay scout without a new selectivity
  mechanism.
- Do not train WaveletPreserve, supervised preserve, or RFC heads from the
  failed 2026-05-28 preflights.
- Do not add a new LF mask without a residual-direction/magnitude target.
- Do not treat SSIM-only gains as promotion when PSNR and per-image splits show
  strong LF-v1 or CR regressions.

## Reporting Value

This review can support a thesis or paper narrative:

- DEA-Net-CR is a strong reproducible baseline.
- LF-v1 is a small, positive low-frequency route.
- ResidualCalib and CRPlus-v2 are positive but incomplete mechanism ablations.
- Selector/oracle evidence shows why naive averages hide real complementarity.
- Failed routes are informative because they identify preservation, direction,
  and deployability as the bottlenecks.
- The next experiment should not be a random module swap or another decision
  head; it should test whether a small multiscale/frequency bottleneck refiner
  can improve the LF residual representation while preserving LF-v1 wins.
