# Selector Proxy Learnability Audit

## Verdict

- Recommendation: `do_not_train_selector_v2_yet`
- Reason: Safe proxies did not recover enough oracle headroom. Improve proxy features or add an explicit supervised/distilled selector target before training.
- Images: `1000`
- Splits: `5`
- Validation fraction: `0.3`
- Minimum pass line: gain `>= 0.120` dB, oracle recovery `>= 0.20`, residual precision `>= 0.65`

## Feature Sets

- `output_proxy`: `63` features
- `metadata_proxy`: `4` features
- `output_plus_metadata_proxy`: `67` features
- `gt_diagnostic_leakage_check`: `19` features

## Cross-Validated Results

| feature_set | model_type | name | gain_vs_lfv1_psnr_mean | gain_vs_lfv1_psnr_std | oracle_recovery_mean | residual_precision_mean | residual_recall_mean | two_way_accuracy_mean | lost_lfv1_gain_selected_count_mean | residual_worst_selected_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| output_proxy | oracle | two_way_oracle | 0.580463 | 0.040757 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| metadata_proxy | oracle | two_way_oracle | 0.580463 | 0.040757 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| output_plus_metadata_proxy | oracle | two_way_oracle | 0.580463 | 0.040757 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| gt_diagnostic_leakage_check | oracle | two_way_oracle | 0.580463 | 0.040757 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| gt_diagnostic_leakage_check | stump | decision_stump | 0.579646 | 0.040439 | 0.998623 | 0.995991 | 0.971242 | 0.983333 | 0.000000 | 0.000000 |
| gt_diagnostic_leakage_check | logistic | ridge_logistic | 0.577691 | 0.040191 | 0.995278 | 0.972464 | 0.959477 | 0.965333 | 0.200000 | 0.000000 |
| metadata_proxy | logistic | ridge_logistic | 0.050798 | 0.031514 | 0.087647 | 0.524052 | 0.537255 | 0.518667 | 30.400000 | 23.800000 |
| metadata_proxy | stump | decision_stump | 0.048985 | 0.034752 | 0.082885 | 0.527020 | 0.516340 | 0.517333 | 28.400000 | 22.600000 |
| output_plus_metadata_proxy | logistic | ridge_logistic | 0.032925 | 0.051902 | 0.055645 | 0.523551 | 0.750327 | 0.522667 | 47.200000 | 34.000000 |
| output_proxy | logistic | ridge_logistic | 0.026770 | 0.049555 | 0.045864 | 0.523091 | 0.707190 | 0.520667 | 44.400000 | 32.600000 |
| output_proxy | stump | decision_stump | 0.020581 | 0.034463 | 0.031993 | 0.519414 | 0.640523 | 0.514000 | 41.000000 | 30.800000 |
| output_plus_metadata_proxy | stump | decision_stump | 0.020581 | 0.034463 | 0.031993 | 0.519414 | 0.640523 | 0.514000 | 41.000000 | 30.800000 |

## Top Held-Out Stumps

| feature_set | feature | op | threshold | train_gain_vs_lfv1_psnr | valid_gain_vs_lfv1_psnr | valid_oracle_recovery | valid_residual_precision | valid_residual_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995088 | 0.547870 | 0.635574 | 0.997931 | 0.993289 | 0.967320 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995422 | 0.560317 | 0.607046 | 0.997973 | 0.986667 | 0.967320 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.994807 | 0.568832 | 0.586665 | 0.999190 | 1.000000 | 0.973856 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.994350 | 0.587035 | 0.543617 | 0.998890 | 1.000000 | 0.967320 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995422 | 0.595339 | 0.525327 | 0.999129 | 1.000000 | 0.980392 |
| metadata_proxy | height | >= | 480.000000 | 0.045597 | 0.092061 | 0.144548 | 0.562914 | 0.555556 |
| metadata_proxy | height | >= | 440.000000 | 0.054794 | 0.070602 | 0.134279 | 0.553333 | 0.542484 |
| output_proxy | lfv1_dark_channel_mean | <= | 0.304383 | 0.077994 | 0.067322 | 0.110676 | 0.532338 | 0.699346 |
| output_plus_metadata_proxy | lfv1_dark_channel_mean | <= | 0.304383 | 0.077994 | 0.067322 | 0.110676 | 0.532338 | 0.699346 |
| output_proxy | residual_lfv1_luma_std_delta | >= | -0.000308 | 0.096112 | 0.051609 | 0.081032 | 0.504950 | 0.666667 |
| output_plus_metadata_proxy | residual_lfv1_luma_std_delta | >= | -0.000308 | 0.096112 | 0.051609 | 0.081032 | 0.504950 | 0.666667 |
| metadata_proxy | height | >= | 480.000000 | 0.063913 | 0.049324 | 0.081088 | 0.513514 | 0.496732 |

## Top Logistic Coefficients

| feature_set | feature | weight | abs_weight | prob_threshold |
| --- | --- | --- | --- | --- |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.574373 | 0.574373 | 0.526329 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.573437 | 0.573437 | 0.500000 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.568510 | 0.568510 | 0.542176 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.567993 | 0.567993 | 0.526329 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.567910 | 0.567910 | 0.502697 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.565008 | 0.565008 | 0.500000 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.560186 | 0.560186 | 0.502697 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.559871 | 0.559871 | 0.542176 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.554925 | 0.554925 | 0.530809 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.551210 | 0.551210 | 0.530809 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.532993 | 0.532993 | 0.542176 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.529410 | 0.529410 | 0.502697 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.529000 | 0.529000 | 0.500000 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.525759 | 0.525759 | 0.526329 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.512774 | 0.512774 | 0.530809 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_residual_error_ratio | -0.322242 | 0.322242 | 0.542176 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_residual_error_ratio | -0.311868 | 0.311868 | 0.530809 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_residual_error_ratio | -0.310523 | 0.310523 | 0.502697 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_residual_error_ratio | -0.304119 | 0.304119 | 0.500000 |
| gt_diagnostic_leakage_check | residual_color_regression_vs_lfv1 | -0.298343 | 0.298343 | 0.526329 |

## Interpretation

- `gt_diagnostic_leakage_check` is a leakage ceiling. It is useful for explaining headroom, but it is not an inference-time selector input.
- `output_proxy` and `output_plus_metadata_proxy` are the scientifically relevant sets for deciding whether another selector run is justified.
- A selector-v2 training run should start only if safe proxy results pass the written line; otherwise the next useful step is better proxy extraction or thesis documentation, not another 100k scout.
