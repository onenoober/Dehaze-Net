# HAZE4K Activation Selector Proxy Audit

Date: 2026-05-26

Status: detailed activation-proxy evidence. The current selector route decision
is summarized in `docs/HAZE4K_SELECTOR_EVIDENCE_CLOSURE_20260526.md`; load this
file only when checking the frozen-forward activation audit, sample-size rule,
or non-GT deployable feature results.

Purpose: run the final non-training selector proxy check by forwarding frozen
CR baseline, LF-v1, and ResidualCalib checkpoints and testing whether internal
inference-time activation features can learn the LF-v1 vs ResidualCalib choice.

## Sample-Size Rule

This audit is meant to support a route decision, so it cannot use a tiny fixed
subset. The script treats a run as conclusive only if all of the following hold:

- images `>= 800`;
- selector target positives and negatives each `>= 120`;
- every usable split has at least `20` samples in its smallest train/valid
  target class.

Runs below this rule are smoke tests only, even if their metrics look good.
Future selector/proxy attempts should follow the same principle: use the full
available evaluation set when feasible, or predeclare a scientifically justified
sample size and mark smaller runs as diagnostic only.

## Added Tool

New read-only script:

```text
code/analyze_selector_activation_proxy_learning.py
```

It does not train. It loads the three frozen checkpoints, forwards HAZE4K test
images, records non-GT activation summaries, and reuses the same held-out audit
logic as the rich proxy check.

Safe feature sets:

- `activation_only_proxy`: hazy-input low-frequency summaries, common backbone
  activation summaries, LF prior injection summaries, ResidualCalib alpha and
  direction summaries, and activation disagreement features.
- `activation_plus_strict_output_proxy`: activation features plus strict
  metadata-free output features.
- `activation_plus_rich_output_proxy`: activation features plus rich
  metadata-free output/agreement features.
- `rich_output_reference`: the 280-feature CSV-derived rich proxy reference.

Diagnostic-only feature sets:

- `metadata_diagnostic`: HAZE4K airlight/beta/shape metadata.
- `gt_diagnostic_leakage_check`: GT-aware leakage ceiling.

The safe feature whitelist rejects names containing PSNR/SSIM, GT/clear,
MAE/RMSE, Delta-E, explicit error/bias fields, residual cosine, residual norm
ratio, or LF MSE terms.

## Local Command

```powershell
python code\analyze_selector_activation_proxy_learning.py `
  --three_way_csv experiment\HAZE4K\three_way_eval\Baseline-LFv1-ResidualCalib-full-20260525\per_image_three_way_metrics.csv `
  --dataset_root dataset\HAZE4K\test `
  --output_dir experiment\HAZE4K\selector_proxy\Baseline-LFv1-ResidualCalib-full-activation-20260526 `
  --baseline_checkpoint experiment\HAZE4K\DEA-Net-CR-H4K-Baseline-scout-20260520-101334\saved_model\best.pk `
  --lfv1_checkpoint experiment\HAZE4K\DEA-Net-LF-H4K-scout-20260521-003100\saved_model\best.pk `
  --residual_checkpoint experiment\HAZE4K\DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654\saved_model\best.pk `
  --splits 5 `
  --threshold_steps 7 `
  --logistic_steps 220 `
  --valid_fraction 0.30 `
  --seed 20260526
```

Artifact:

```text
experiment/HAZE4K/selector_proxy/Baseline-LFv1-ResidualCalib-full-activation-20260526/
```

Outputs:

```text
activation_features.csv
joined_features.csv
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

The run is conclusive by the sample-size rule:

- images: `1000`;
- selector target positives/negatives: `509 / 491`;
- minimum split class count: `55`.

Feature counts:

| Feature Set | Count |
| --- | ---: |
| `activation_only_proxy` | `1418` |
| `activation_plus_strict_output_proxy` | `1481` |
| `activation_plus_rich_output_proxy` | `1698` |
| `rich_output_reference` | `280` |
| `metadata_diagnostic` | `4` |
| `gt_diagnostic_leakage_check` | `19` |

Best metadata-free activation row by split family:

| Split Family | Best Feature Set | Model | Gain vs LF-v1 | Oracle Recovery | Precision | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `random_stratified` | `activation_plus_rich_output_proxy` | ridge logistic | `+0.0748 dB` | `0.1247` | `0.5978` | fail |
| `airlight_leave_one` | `activation_plus_rich_output_proxy` | ridge logistic | `+0.0585 dB` | `0.0922` | `0.5763` | fail |
| `beta_leave_one` | `activation_plus_strict_output_proxy` | ridge logistic | `+0.0392 dB` | `0.0630` | `0.6086` | fail |
| `degradation_combo_group5` | `activation_plus_rich_output_proxy` | ridge logistic | `+0.0544 dB` | `0.0883` | `0.5855` | fail |

Pass line:

```text
gain >= +0.12 dB
oracle recovery >= 0.20
residual precision >= 0.65
```

No metadata-free activation proxy passed.

Reference rows:

- Best random `rich_output_reference` row: `+0.0908 dB`, oracle recovery
  `0.1569`, precision `0.5914`; still below the pass line.
- GT-aware leakage check still recovers the target: random split stump
  `+0.5796 dB`, oracle recovery `0.9986`, precision `0.9960`.

Safety check:

- Manual regex check over safe feature lists found `0` leakage-like safe
  features.

## Interpretation

The activation-forward audit was the best remaining deployable-proxy attempt:
it used frozen internal signals that are unavailable to the CSV-only rich proxy
but still available at inference. It improved neither random nor
degradation-held-out results enough to justify a new selector-v2 100k scout.

The selector target remains real because the oracle and GT-aware leakage
ceiling are strong. The blocker is still proxy learnability from deployable
signals.

## Decision

Do not train selector-v2 from the current evidence.

The selector/structure-search branch should stop here unless a future proposal
changes the problem definition, for example by adding an explicit supervised or
distilled selector target and then passing a fresh full-sample proxy audit.
The highest-value near-term work is to consolidate the paper evidence chain:
LF-v1 positive result, ResidualCalib positive ablation, oracle headroom,
strict/rich/activation proxy failure, ResidualSelector failure, and
ResidualDirLoss failure.
