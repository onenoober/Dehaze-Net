# HAZE4K Selector Proxy Audit

Date: 2026-05-26

Purpose: review whether the previous recommendation is reliable, then define
the next evidence gate before any new selector training run.

## Reliability Verdict

The recommendation is reliable only as a diagnostic-first step.

It is reliable to stop launching new 100k scouts until selector proxy
learnability is checked. It is not reliable to treat the LF-v1/ResidualCalib
oracle alone as enough evidence for another selector run, because the strongest
selector rule found so far is GT-aware and cannot be used at inference.

## Evidence Review

Positive evidence:

- LF-v1 remains the current positive model candidate: best 90k
  `32.4281 / 0.9845`, above the CR baseline `32.2255 / 0.9844`.
- ResidualCalib is a positive ablation: best 90k `32.3936 / 0.9845`,
  above CR baseline but below LF-v1.
- LF-v1 and ResidualCalib are strongly complementary. The full-test two-way
  oracle reaches `33.0034 / 0.985299`, `+0.5751 dB` over LF-v1. The three-way
  baseline/LF-v1/ResidualCalib oracle reaches `33.2538 / 0.985637`.

Negative evidence:

- The first LFResidualSelector fair run failed at the 20k gate:
  `27.6830 / 0.9713`, clearly below baseline, LF-v1, and ResidualCalib.
  Selector stats stayed near constant around `0.879`, so the branch did not
  learn meaningful selection.
- The first residual-direction loss run failed the 30k hard gate. It tied the
  baseline but remained behind LF-v1 by `-0.5195 dB`, and its route-specific
  residual-direction metrics were worse than LF-v1 and ResidualCalib on the
  64-image review subset.
- Existing selector/oracle analysis says the best simple rule is
  `rescalib_from_lfv1_residual_error_ratio`, which is GT-aware.

Conclusion: the target is real, but the usable proxy has not been proven.

## Added Tool

New read-only script:

```text
code/analyze_selector_proxy_learning.py
```

It reads an existing three-way per-image CSV and evaluates whether proxy
features can recover the LF-v1/ResidualCalib oracle on held-out splits.

Feature sets:

- `output_proxy`: output-derived statistics and pairwise deltas that do not use
  GT directly. The script now builds this set from an explicit whitelist only:
  per-output luma/saturation/dark-channel/edge/Laplacian/high-frequency
  summaries plus candidate-output pairwise deltas generated inside the script.
  It does not auto-discover existing CSV `*_delta` columns.
- `metadata_proxy`: synthetic HAZE4K filename metadata such as airlight, beta,
  height, and width. Useful for diagnosis, but less portable to real images.
- `output_plus_metadata_proxy`: union of the above.
- `gt_diagnostic_leakage_check`: GT-aware residual diagnostics. This is a
  leakage ceiling, not an inference-time selector input.

Models:

- always choose LF-v1;
- always choose ResidualCalib;
- two-way oracle;
- decision stump trained on the train split and evaluated on held-out images;
- ridge logistic selector trained on the train split and evaluated on held-out
  images.

Outputs:

```text
summary.json
feature_lists.json
aggregate_summary.csv
split_metrics.csv
stump_rules.csv
logistic_coefficients.csv
analysis_report.md
```

## Local Audit Result

Strict local command, using the existing full-test three-way CSV:

```powershell
python code\analyze_selector_proxy_learning.py `
  --three_way_csv experiment\HAZE4K\three_way_eval\Baseline-LFv1-ResidualCalib-full-20260525\per_image_three_way_metrics.csv `
  --output_dir experiment\HAZE4K\selector_proxy\Baseline-LFv1-ResidualCalib-full-strict-20260526 `
  --splits 5 `
  --threshold_steps 9 `
  --logistic_steps 250 `
  --valid_fraction 0.30 `
  --seed 20260526
```

Artifact:

```text
experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-strict-20260526/
```

Strictness checks:

- `output_proxy` now has `63` features instead of the earlier `66`.
- `output_plus_metadata_proxy` now has `67` features instead of the earlier
  `70`.
- The script validates that safe feature sets contain only whitelisted
  inference-safe features before writing results.
- A manual regex check over `stump_rules.csv` and `logistic_coefficients.csv`
  found `0` unsafe safe-feature rows.

Result:

| Feature Set | Best Safe Model | Held-Out Gain vs LF-v1 | Oracle Recovery | Residual Precision | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `metadata_proxy` | ridge logistic | `+0.0508 dB` | `0.0876` | `0.5241` | fail |
| `output_plus_metadata_proxy` | ridge logistic | `+0.0329 dB` | `0.0556` | `0.5236` | fail |
| `output_proxy` | ridge logistic | `+0.0268 dB` | `0.0459` | `0.5231` | fail |
| `gt_diagnostic_leakage_check` | decision stump | `+0.5796 dB` | `0.9986` | `0.9960` | leakage ceiling only |

Pass line before selector-v2:

```text
safe held-out gain >= +0.12 dB
oracle recovery >= 0.20
residual precision >= 0.65
```

No safe feature set passed.

## Rich Proxy Follow-Up

Follow-up doc:

```text
docs/HAZE4K_RICH_SELECTOR_PROXY_AUDIT_20260526.md
```

Follow-up artifact:

```text
experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-rich-20260526/
```

The rich follow-up added 280 metadata-free CSV-derived output/agreement
features and evaluated random plus degradation-held-out split families. Best
metadata-free rows still failed:

| Split Family | Feature Set | Gain vs LF-v1 | Oracle Recovery | Precision |
| --- | --- | ---: | ---: | ---: |
| `random_stratified` | `rich_output_proxy` | `+0.0956 dB` | `0.1643` | `0.6046` |
| `airlight_leave_one` | `rich_output_proxy` | `+0.0695 dB` | `0.1208` | `0.5858` |
| `beta_leave_one` | `agreement_proxy` | `+0.0755 dB` | `0.1235` | `0.5861` |
| `degradation_combo_group5` | `rich_output_proxy` | `+0.0656 dB` | `0.1077` | `0.5805` |

This improves over strict output proxy but remains below the pass line.

## Activation Proxy Follow-Up

Follow-up doc:

```text
docs/HAZE4K_ACTIVATION_SELECTOR_PROXY_AUDIT_20260526.md
```

Follow-up artifact:

```text
experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-activation-20260526/
```

The activation follow-up forwarded frozen CR baseline, LF-v1, and ResidualCalib
checkpoints over the full 1000-image HAZE4K test set and extracted non-GT
hazy-input, common activation, LF prior, ResidualCalib alpha/direction, and
activation-disagreement features.

The sample-size gate passed:

- images: `1000`;
- selector target positives/negatives: `509 / 491`;
- minimum split class count: `55`.

Best metadata-free activation rows still failed:

| Split Family | Feature Set | Gain vs LF-v1 | Oracle Recovery | Precision |
| --- | --- | ---: | ---: | ---: |
| `random_stratified` | `activation_plus_rich_output_proxy` | `+0.0748 dB` | `0.1247` | `0.5978` |
| `airlight_leave_one` | `activation_plus_rich_output_proxy` | `+0.0585 dB` | `0.0922` | `0.5763` |
| `beta_leave_one` | `activation_plus_strict_output_proxy` | `+0.0392 dB` | `0.0630` | `0.6086` |
| `degradation_combo_group5` | `activation_plus_rich_output_proxy` | `+0.0544 dB` | `0.0883` | `0.5855` |

Manual regex checking found `0` leakage-like safe features. The GT-aware
leakage ceiling still works (`+0.5796 dB`, recovery `0.9986`), so the target
exists but deployable activation/output proxies still do not recover it.

## Decision

Do not launch another LFResidualSelector 100k scout yet.

Strict, rich, and activation-forward proxy audits all failed the predeclared
safe pass line. The reliable next step is to stop selector/structure search for
now and consolidate the thesis story:

- baseline CR is strong;
- LF-v1 gives a real but uneven `+0.2030 dB` full-test gain;
- ResidualCalib is a positive ablation but not a replacement;
- selector/oracle analysis proves there is headroom, while strict/rich/
  activation proxy audits explain why naive learned selection failed;
- ResidualDirLoss confirms that training-loss improvement alone does not
  guarantee test-side residual mechanism improvement.

Only reopen selector-v2 if the proposal changes the problem, such as adding an
explicit supervised or distilled selector target, and then passes a fresh
full-sample proxy audit before any 100k training run.

Future proxy conclusions must use a scientifically adequate sample size. Small
subsets are valid for smoke/debug only and cannot justify a new training route.

## Formal Follow-Up Gate

Before any future selector-v2 run, write a new experiment card using
`docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md`. It must include:

- the proxy features used by the selector;
- whether the selector has an explicit supervised/distilled target;
- held-out proxy audit results;
- the same fair 100k training contract;
- gate rules that include both PSNR/SSIM and selector-specific diagnostics.

No selector-v2 training should start from oracle evidence alone.
