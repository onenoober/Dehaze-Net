# Selector Proxy Learnability Audit

## Verdict

- Recommendation: `do_not_train_selector_v2_yet`
- Reason: Safe proxies did not recover enough oracle headroom. Improve proxy features or add an explicit supervised/distilled selector target before training.
- Images: `1000`
- Splits: `2`
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
| output_proxy | oracle | two_way_oracle | 0.622585 | 0.014307 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| metadata_proxy | oracle | two_way_oracle | 0.622585 | 0.014307 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| output_plus_metadata_proxy | oracle | two_way_oracle | 0.622585 | 0.014307 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| gt_diagnostic_leakage_check | oracle | two_way_oracle | 0.622585 | 0.014307 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| gt_diagnostic_leakage_check | stump | decision_stump | 0.621310 | 0.014264 | 0.997952 | 0.989978 | 0.967320 | 0.978333 | 0.000000 | 0.000000 |
| gt_diagnostic_leakage_check | logistic | ridge_logistic | 0.617361 | 0.016589 | 0.991520 | 0.939532 | 0.964052 | 0.950000 | 0.500000 | 0.500000 |
| metadata_proxy | stump | decision_stump | 0.070693 | 0.021369 | 0.112818 | 0.538214 | 0.526144 | 0.528333 | 28.500000 | 20.000000 |
| output_plus_metadata_proxy | logistic | ridge_logistic | 0.069628 | 0.031192 | 0.113048 | 0.539717 | 0.702614 | 0.543333 | 42.500000 | 27.500000 |
| output_proxy | stump | decision_stump | 0.063063 | 0.009148 | 0.101683 | 0.544530 | 0.503268 | 0.531667 | 31.500000 | 19.000000 |
| output_plus_metadata_proxy | stump | decision_stump | 0.063063 | 0.009148 | 0.101683 | 0.544530 | 0.503268 | 0.531667 | 31.500000 | 19.000000 |
| output_proxy | logistic | ridge_logistic | 0.054371 | 0.013593 | 0.087879 | 0.525061 | 0.833333 | 0.531667 | 53.000000 | 34.500000 |
| metadata_proxy | logistic | ridge_logistic | 0.052098 | 0.035992 | 0.082396 | 0.512547 | 0.444444 | 0.510000 | 26.000000 | 18.000000 |

## Top Held-Out Stumps

| feature_set | feature | op | threshold | train_gain_vs_lfv1_psnr | valid_gain_vs_lfv1_psnr | valid_oracle_recovery | valid_residual_precision | valid_residual_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995088 | 0.547870 | 0.635574 | 0.997931 | 0.993289 | 0.967320 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995422 | 0.560317 | 0.607046 | 0.997973 | 0.986667 | 0.967320 |
| metadata_proxy | height | >= | 480.000000 | 0.045597 | 0.092061 | 0.144548 | 0.562914 | 0.555556 |
| output_proxy | residual_baseline_luma_std_delta | >= | 0.000479 | 0.076692 | 0.072211 | 0.118714 | 0.578571 | 0.529412 |
| output_plus_metadata_proxy | residual_baseline_luma_std_delta | >= | 0.000479 | 0.076692 | 0.072211 | 0.118714 | 0.578571 | 0.529412 |
| output_proxy | residual_lfv1_luma_std_delta | >= | 0.000724 | 0.088889 | 0.053915 | 0.084653 | 0.510490 | 0.477124 |
| output_plus_metadata_proxy | residual_lfv1_luma_std_delta | >= | 0.000724 | 0.088889 | 0.053915 | 0.084653 | 0.510490 | 0.477124 |
| metadata_proxy | height | >= | 480.000000 | 0.063913 | 0.049324 | 0.081088 | 0.513514 | 0.496732 |

## Top Logistic Coefficients

| feature_set | feature | weight | abs_weight | prob_threshold |
| --- | --- | --- | --- | --- |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.320493 | 0.320493 | 0.532136 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.318243 | 0.318243 | 0.532136 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.316579 | 0.316579 | 0.516249 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.315630 | 0.315630 | 0.516249 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.285481 | 0.285481 | 0.532136 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.282829 | 0.282829 | 0.516249 |
| gt_diagnostic_leakage_check | residual_color_regression_vs_lfv1 | -0.209172 | 0.209172 | 0.532136 |
| gt_diagnostic_leakage_check | residual_color_regression_vs_lfv1 | -0.206718 | 0.206718 | 0.516249 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_residual_error_ratio | -0.171394 | 0.171394 | 0.532136 |
| gt_diagnostic_leakage_check | residual_luma_abs_bias_regression_vs_lfv1 | -0.167434 | 0.167434 | 0.532136 |
| gt_diagnostic_leakage_check | residual_luma_abs_bias_regression_vs_lfv1 | -0.165356 | 0.165356 | 0.516249 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_luma_residual_cosine | 0.162751 | 0.162751 | 0.532136 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_residual_cosine | 0.161066 | 0.161066 | 0.532136 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_residual_error_ratio | -0.159943 | 0.159943 | 0.516249 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_residual_cosine | 0.154997 | 0.154997 | 0.516249 |
| gt_diagnostic_leakage_check | rescalib_from_baseline_luma_residual_cosine | 0.152244 | 0.152244 | 0.516249 |
| gt_diagnostic_leakage_check | residual_dark_channel_abs_bias_regression_vs_lfv1 | -0.150767 | 0.150767 | 0.532136 |
| gt_diagnostic_leakage_check | residual_dark_channel_abs_bias_regression_vs_lfv1 | -0.145609 | 0.145609 | 0.516249 |
| gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | -0.143779 | 0.143779 | 0.532136 |
| gt_diagnostic_leakage_check | lfv1_from_baseline_residual_cosine | -0.140219 | 0.140219 | 0.516249 |

## Interpretation

- `gt_diagnostic_leakage_check` is a leakage ceiling. It is useful for explaining headroom, but it is not an inference-time selector input.
- `output_proxy` and `output_plus_metadata_proxy` are the scientifically relevant sets for deciding whether another selector run is justified.
- A selector-v2 training run should start only if safe proxy results pass the written line; otherwise the next useful step is better proxy extraction or thesis documentation, not another 100k scout.
