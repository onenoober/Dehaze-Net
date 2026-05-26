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
  GT directly.
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
aggregate_summary.csv
split_metrics.csv
stump_rules.csv
logistic_coefficients.csv
analysis_report.md
```

## Local Audit Result

Local command, using the existing full-test three-way CSV:

```powershell
python code\analyze_selector_proxy_learning.py `
  --three_way_csv experiment\HAZE4K\three_way_eval\Baseline-LFv1-ResidualCalib-full-20260525\per_image_three_way_metrics.csv `
  --output_dir experiment\HAZE4K\selector_proxy\Baseline-LFv1-ResidualCalib-full-20260526 `
  --splits 5 `
  --threshold_steps 9 `
  --logistic_steps 250 `
  --valid_fraction 0.30 `
  --seed 20260526
```

Artifact:

```text
experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-20260526/
```

Result:

| Feature Set | Best Safe Model | Held-Out Gain vs LF-v1 | Oracle Recovery | Residual Precision | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `metadata_proxy` | ridge logistic | `+0.0508 dB` | `0.0876` | `0.5241` | fail |
| `output_plus_metadata_proxy` | ridge logistic | `+0.0323 dB` | `0.0551` | `0.5306` | fail |
| `output_proxy` | ridge logistic | `+0.0300 dB` | `0.0529` | `0.5227` | fail |
| `gt_diagnostic_leakage_check` | decision stump | `+0.5796 dB` | `0.9986` | `0.9960` | leakage ceiling only |

Pass line before selector-v2:

```text
safe held-out gain >= +0.12 dB
oracle recovery >= 0.20
residual precision >= 0.65
```

No safe feature set passed.

## Decision

Do not launch another LFResidualSelector 100k scout yet.

The reliable next step is one of these two:

1. Improve proxy evidence first:
   - extract stronger non-GT features, such as bottleneck confidence summaries,
     LF branch activation traces, local contrast, dark-channel/luma statistics,
     and uncertainty from both candidate outputs;
   - rerun `analyze_selector_proxy_learning.py`;
   - proceed only if the safe proxy pass line is met.

2. If selector proxy evidence remains weak, stop LF architecture search for now
   and consolidate the thesis story:
   - baseline CR is strong;
   - LF-v1 gives a real but uneven `+0.2030 dB` full-test gain;
   - ResidualCalib is a positive ablation but not a replacement;
   - selector/oracle analysis proves there is headroom, while proxy audit shows
     why naive learned selection failed;
   - ResidualDirLoss confirms that training-loss improvement alone does not
     guarantee test-side residual mechanism improvement.

## Formal Follow-Up Gate

Before any future selector-v2 run, write a new experiment card using
`docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md`. It must include:

- the proxy features used by the selector;
- whether the selector has an explicit supervised/distilled target;
- held-out proxy audit results;
- the same fair 100k training contract;
- gate rules that include both PSNR/SSIM and selector-specific diagnostics.

No selector-v2 training should start from oracle evidence alone.
