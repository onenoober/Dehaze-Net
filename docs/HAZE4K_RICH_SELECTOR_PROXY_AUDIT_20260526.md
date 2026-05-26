# HAZE4K Rich Selector Proxy Audit

Date: 2026-05-26

Purpose: re-check whether the next best step after the strict selector proxy
audit should be a richer proxy learnability audit, then run that audit locally
before any selector-v2 training.

## Review Verdict

The previous analysis is reasonable with one boundary:

- Reasonable: current evidence and recent restoration research both support a
  degradation/quality-aware proxy audit before any new selector-v2 training.
- Boundary: metadata is diagnostic only, because HAZE4K filename airlight/beta
  is not a deployable real-image signal.
- Boundary: full-test three-way artifacts currently provide per-image CSV
  statistics but not all 1000 candidate output images, so this run is a
  CSV-derived rich proxy audit. A heavier activation-forward audit would be a
  separate final attempt.

Relevant research context:

- DEA-Net shows that detail-enhanced attention is a strong lightweight dehazing
  baseline, so this fork should prefer small, evidence-gated additions over a
  full backbone rewrite.
- DehazeFormer and TCL-Net support the idea that dehazing benefits from
  restoration-specific design and frequency-aware signals.
- PromptIR and DFPIR support using degradation-aware or prompt-like features for
  restoration control, rather than expecting a plain selector to infer reliable
  routing from weak statistics.

## Local Evidence Before This Audit

- LF-v1 remains the current positive model candidate.
- ResidualCalib is a positive ablation, but below LF-v1 on mean PSNR.
- LF-v1/ResidualCalib oracle headroom is large: about `+0.58 dB`.
- Strict safe output proxy failed: best metadata-free strict output proxy was
  only `+0.0268 dB`, oracle recovery `0.0459`, precision `0.5231`.
- GT-aware leakage ceiling remains near oracle, proving the target exists but
  current inference-safe signals do not recover it.

## Added Tool

New read-only script:

```text
code/analyze_selector_rich_proxy_learning.py
```

It imports the strict audit helpers and evaluates richer GT-free feature sets:

- `strict_output_proxy`: the strict 63-feature output proxy baseline.
- `agreement_proxy`: candidate-agreement ratios, normalized deltas, and
  direction flags between baseline/LF-v1/ResidualCalib output statistics.
- `rich_output_proxy`: 280 metadata-free features, including strict output
  features, agreement features, cross-candidate ranges/std/ranks, and
  baseline-context interactions.
- `metadata_diagnostic`: HAZE4K airlight/beta/height/width only; diagnostic,
  not a proceed condition.
- `gt_diagnostic_leakage_check`: leakage ceiling only.

Split families:

- `random_stratified`: five label-stratified random splits.
- `airlight_leave_one`: leave one airlight bin out.
- `beta_leave_one`: leave one beta bin out.
- `degradation_combo_group5`: group 5-fold split over airlight/beta bin
  combinations.

Proceed rule:

```text
metadata-free proxy must pass every required split family:
gain >= +0.12 dB
oracle recovery >= 0.20
residual precision >= 0.65
```

## Local Command

```powershell
python code\analyze_selector_rich_proxy_learning.py `
  --three_way_csv experiment\HAZE4K\three_way_eval\Baseline-LFv1-ResidualCalib-full-20260525\per_image_three_way_metrics.csv `
  --output_dir experiment\HAZE4K\selector_proxy\Baseline-LFv1-ResidualCalib-full-rich-20260526 `
  --splits 5 `
  --threshold_steps 7 `
  --logistic_steps 200 `
  --valid_fraction 0.30 `
  --seed 20260526
```

Artifact:

```text
experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-rich-20260526/
```

Outputs:

```text
summary.json
allowed_feature_sets.json
feature_lists.json
aggregate_summary.csv
split_metrics.csv
stump_rules.csv
logistic_coefficients.csv
analysis_report.md
```

## Result

Best metadata-free row by split family:

| Split Family | Best Feature Set | Model | Gain vs LF-v1 | Oracle Recovery | Precision | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `random_stratified` | `rich_output_proxy` | ridge logistic | `+0.0956 dB` | `0.1643` | `0.6046` | fail |
| `airlight_leave_one` | `rich_output_proxy` | ridge logistic | `+0.0695 dB` | `0.1208` | `0.5858` | fail |
| `beta_leave_one` | `agreement_proxy` | ridge logistic | `+0.0755 dB` | `0.1235` | `0.5861` | fail |
| `degradation_combo_group5` | `rich_output_proxy` | ridge logistic | `+0.0656 dB` | `0.1077` | `0.5805` | fail |

Leakage ceiling still works:

- `gt_diagnostic_leakage_check` under random split remains near oracle:
  `+0.5796 dB` for stump and `+0.5770 dB` for ridge logistic.

Safety check:

- Manual regex check over rich audit `stump_rules.csv` and
  `logistic_coefficients.csv` found `0` unsafe rows inside metadata-free safe
  feature sets.

## Decision

Do not launch selector-v2.

The richer metadata-free proxy improved over strict output proxy on random
splits (`+0.0956 dB` vs `+0.0268 dB`), but it still missed all three pass
lines and weakened under degradation-held-out splits. The follow-up
activation-forward audit then checked the strongest remaining deployable-proxy
route identified in this repo and also failed the pass line. Current
selector-v2 training should not start from these evidence sets.

The practical next step is to stop selector-v2 search and consolidate the
paper story around LF-v1, ResidualCalib, oracle headroom, strict/rich/
activation proxy failure, and the failed ResidualSelector/ResidualDirLoss
branches.
