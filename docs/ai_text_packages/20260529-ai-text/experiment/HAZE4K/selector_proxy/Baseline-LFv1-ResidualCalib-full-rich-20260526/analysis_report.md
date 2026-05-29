# Rich Selector Proxy Learnability Audit

## Verdict

- Recommendation: `do_not_train_selector_v2_yet`
- Reason: No metadata-free rich proxy passed the required random and degradation-held-out audit lines. Keep improving proxy evidence or stop selector-v2 training plans.
- Images: `1000`
- Split families: `airlight_leave_one, beta_leave_one, degradation_combo_group5, random_stratified`
- Minimum pass line: gain `>= 0.120` dB, oracle recovery `>= 0.20`, residual precision `>= 0.65`
- Proceed requires a metadata-free proxy to pass all required families: `random_stratified, airlight_leave_one, beta_leave_one, degradation_combo_group5`

## Feature Sets

- `strict_output_proxy`: `63` features
- `agreement_proxy`: `63` features
- `rich_output_proxy`: `280` features
- `metadata_diagnostic`: `4` features
- `gt_diagnostic_leakage_check`: `19` features

## Best Metadata-Free Row By Split Family

| split_family | feature_set | model_type | name | gain_vs_lfv1_psnr_mean | oracle_recovery_mean | residual_precision_mean | residual_recall_mean | two_way_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airlight_leave_one | rich_output_proxy | logistic | ridge_logistic | 0.069537 | 0.120801 | 0.585802 | 0.477282 | 0.553434 |
| beta_leave_one | agreement_proxy | logistic | ridge_logistic | 0.075459 | 0.123503 | 0.586131 | 0.591424 | 0.578134 |
| degradation_combo_group5 | rich_output_proxy | logistic | ridge_logistic | 0.065619 | 0.107675 | 0.580452 | 0.551321 | 0.568004 |
| random_stratified | rich_output_proxy | logistic | ridge_logistic | 0.095588 | 0.164328 | 0.604612 | 0.569935 | 0.587333 |

## Top Aggregate Rows

| split_family | feature_set | model_type | name | gain_vs_lfv1_psnr_mean | oracle_recovery_mean | residual_precision_mean | residual_recall_mean | two_way_accuracy_mean | lost_lfv1_gain_selected_count_mean | residual_worst_selected_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_stratified | strict_output_proxy | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | agreement_proxy | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | rich_output_proxy | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | metadata_diagnostic | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | stump | decision_stump | 0.579646 | 0.998623 | 0.995991 | 0.971242 | 0.983333 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | logistic | ridge_logistic | 0.577033 | 0.994078 | 0.969702 | 0.952941 | 0.960667 | 0.200000 | 0.000000 |
| random_stratified | rich_output_proxy | logistic | ridge_logistic | 0.095588 | 0.164328 | 0.604612 | 0.569935 | 0.587333 | 27.600000 | 18.400000 |
| random_stratified | agreement_proxy | logistic | ridge_logistic | 0.077921 | 0.130656 | 0.582397 | 0.603922 | 0.574667 | 30.200000 | 23.600000 |
| random_stratified | metadata_diagnostic | logistic | ridge_logistic | 0.045286 | 0.079028 | 0.517243 | 0.503268 | 0.514667 | 28.400000 | 22.600000 |
| random_stratified | metadata_diagnostic | stump | decision_stump | 0.041807 | 0.071084 | 0.526114 | 0.528105 | 0.516667 | 29.000000 | 25.000000 |
| random_stratified | strict_output_proxy | stump | decision_stump | 0.035245 | 0.056834 | 0.534394 | 0.620915 | 0.528000 | 39.000000 | 27.200000 |
| random_stratified | agreement_proxy | stump | decision_stump | 0.035225 | 0.056860 | 0.532262 | 0.619608 | 0.526000 | 39.000000 | 27.200000 |
| random_stratified | strict_output_proxy | logistic | ridge_logistic | 0.027706 | 0.048068 | 0.525227 | 0.661438 | 0.522667 | 41.000000 | 30.200000 |
| random_stratified | strict_output_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | agreement_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | rich_output_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | metadata_diagnostic | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | strict_output_proxy | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | agreement_proxy | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | rich_output_proxy | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | metadata_diagnostic | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | gt_diagnostic_leakage_check | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |

## Top Held-Out Stumps

| split_family | split | feature_set | feature | op | threshold | valid_gain_vs_lfv1_psnr | valid_oracle_recovery | valid_residual_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airlight_leave_one | 0.65<=A<0.75 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | <= | -0.000000 | 0.686758 | 0.998703 | 0.981308 |
| degradation_combo_group5 | 2 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.993819 | 0.654853 | 0.999627 | 0.989691 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995088 | 0.635574 | 0.997931 | 0.993289 |
| airlight_leave_one | A>=0.95 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | <= | -0.000000 | 0.629639 | 0.999868 | 0.986111 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995141 | 0.617509 | 0.999236 | 1.000000 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995422 | 0.607046 | 0.997973 | 0.986667 |
| beta_leave_one | 1.1<=beta<1.4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995704 | 0.601617 | 0.998966 | 1.000000 |
| beta_leave_one | 0.8<=beta<1.1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.996598 | 0.597714 | 0.999456 | 0.989691 |
| random_stratified | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.994807 | 0.586665 | 0.999190 | 1.000000 |
| degradation_combo_group5 | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | <= | -0.000000 | 0.582411 | 0.998090 | 0.981651 |
| beta_leave_one | 1.4<=beta<1.7 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.998851 | 0.580137 | 0.995926 | 0.962264 |
| beta_leave_one | beta<0.8 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.988522 | 0.571087 | 0.995256 | 1.000000 |
| airlight_leave_one | 0.85<=A<0.95 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995704 | 0.551758 | 0.998052 | 0.988636 |
| random_stratified | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.994350 | 0.543617 | 0.998890 | 1.000000 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | <= | -0.000003 | 0.541316 | 0.989919 | 0.992537 |
| degradation_combo_group5 | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995704 | 0.541125 | 0.998290 | 0.990099 |
| random_stratified | 2 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.995422 | 0.525327 | 0.999129 | 1.000000 |
| beta_leave_one | beta>=1.7 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.993941 | 0.524256 | 0.998523 | 1.000000 |
| airlight_leave_one | 0.75<=A<0.85 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.986527 | 0.493398 | 0.993319 | 1.000000 |
| degradation_combo_group5 | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 0.991464 | 0.473169 | 0.995396 | 0.989130 |

## Top Logistic Coefficients

| split_family | split | feature_set | feature | weight | abs_weight | prob_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| airlight_leave_one | 0.75<=A<0.85 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.480520 | 0.480520 | 0.500000 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.480072 | 0.480072 | 0.513869 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.479476 | 0.479476 | 0.529758 |
| degradation_combo_group5 | 2 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.478173 | 0.478173 | 0.500000 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.477212 | 0.477212 | 0.500000 |
| beta_leave_one | 1.4<=beta<1.7 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.476986 | 0.476986 | 0.500000 |
| airlight_leave_one | 0.75<=A<0.85 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.476679 | 0.476679 | 0.500000 |
| beta_leave_one | beta<0.8 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.476499 | 0.476499 | 0.516913 |
| random_stratified | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.475372 | 0.475372 | 0.500000 |
| beta_leave_one | beta<0.8 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.475099 | 0.475099 | 0.516913 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.475018 | 0.475018 | 0.529758 |
| random_stratified | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.474764 | 0.474764 | 0.543054 |
| degradation_combo_group5 | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.474224 | 0.474224 | 0.500232 |
| degradation_combo_group5 | 2 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.474144 | 0.474144 | 0.500000 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.473822 | 0.473822 | 0.513869 |
| airlight_leave_one | 0.65<=A<0.75 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.473810 | 0.473810 | 0.523867 |
| degradation_combo_group5 | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.472858 | 0.472858 | 0.500232 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.472092 | 0.472092 | 0.500000 |
| beta_leave_one | 1.4<=beta<1.7 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.471609 | 0.471609 | 0.500000 |
| degradation_combo_group5 | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.471168 | 0.471168 | 0.524016 |
| random_stratified | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.469746 | 0.469746 | 0.500000 |
| airlight_leave_one | 0.85<=A<0.95 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.469612 | 0.469612 | 0.497580 |
| airlight_leave_one | 0.65<=A<0.75 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.469519 | 0.469519 | 0.523867 |
| airlight_leave_one | A>=0.95 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.469374 | 0.469374 | 0.500000 |

## Interpretation

- Metadata-derived feature sets are diagnostic only and cannot justify selector-v2.
- `gt_diagnostic_leakage_check` is a leakage ceiling, not an inference-time selector input.
- A selector-v2 route card should be written only after a metadata-free proxy passes both random and degradation-held-out split families.
