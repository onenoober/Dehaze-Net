# Activation Selector Proxy Learnability Audit

## Verdict

- Recommendation: `do_not_train_selector_v2_yet`
- Reason: No frozen-activation, GT-free proxy passed the required random and degradation-held-out audit lines. Stop selector-v2 training plans or move to explicitly supervised/distilled selector targets.
- Sample-size verdict: `conclusive_sample_size`
- Images: `1000`; positive/negative target counts: `509` / `491`
- Minimum conclusive sample rule: images `>= 800`, class counts `>= 120`, min split class count `>= 20`
- Split families: `airlight_leave_one, beta_leave_one, degradation_combo_group5, random_stratified`
- Minimum pass line: gain `>= 0.120` dB, oracle recovery `>= 0.20`, residual precision `>= 0.65`
- Proceed requires a metadata-free activation proxy to pass all required families: `random_stratified, airlight_leave_one, beta_leave_one, degradation_combo_group5`

## Feature Sets

- `activation_only_proxy`: `1418` features
- `activation_plus_strict_output_proxy`: `1481` features
- `activation_plus_rich_output_proxy`: `1698` features
- `rich_output_reference`: `280` features
- `metadata_diagnostic`: `4` features
- `gt_diagnostic_leakage_check`: `19` features

## Best Metadata-Free Activation Row By Split Family

| split_family | feature_set | model_type | name | gain_vs_lfv1_psnr_mean | oracle_recovery_mean | residual_precision_mean | residual_recall_mean | two_way_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airlight_leave_one | activation_plus_rich_output_proxy | logistic | ridge_logistic | 0.058521 | 0.092154 | 0.576282 | 0.299936 | 0.526037 |
| beta_leave_one | activation_plus_strict_output_proxy | logistic | ridge_logistic | 0.039153 | 0.063011 | 0.608620 | 0.231032 | 0.526134 |
| degradation_combo_group5 | activation_plus_rich_output_proxy | logistic | ridge_logistic | 0.054380 | 0.088350 | 0.585543 | 0.272498 | 0.532017 |
| random_stratified | activation_plus_rich_output_proxy | logistic | ridge_logistic | 0.074846 | 0.124713 | 0.597754 | 0.364706 | 0.551333 |

## Top Aggregate Rows

| split_family | feature_set | model_type | name | gain_vs_lfv1_psnr_mean | oracle_recovery_mean | residual_precision_mean | residual_recall_mean | two_way_accuracy_mean | lost_lfv1_gain_selected_count_mean | residual_worst_selected_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_stratified | activation_only_proxy | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | activation_plus_strict_output_proxy | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | activation_plus_rich_output_proxy | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | rich_output_reference | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | metadata_diagnostic | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | oracle | two_way_oracle | 0.580463 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | stump | decision_stump | 0.579646 | 0.998623 | 0.995991 | 0.971242 | 0.983333 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | logistic | ridge_logistic | 0.576961 | 0.993901 | 0.963621 | 0.956863 | 0.959333 | 0.200000 | 0.000000 |
| random_stratified | rich_output_reference | logistic | ridge_logistic | 0.090755 | 0.156866 | 0.591400 | 0.626144 | 0.586667 | 31.800000 | 21.200000 |
| random_stratified | activation_plus_rich_output_proxy | logistic | ridge_logistic | 0.074846 | 0.124713 | 0.597754 | 0.364706 | 0.551333 | 18.600000 | 13.000000 |
| random_stratified | activation_only_proxy | logistic | ridge_logistic | 0.068724 | 0.117056 | 0.589863 | 0.269281 | 0.528667 | 13.000000 | 10.800000 |
| random_stratified | activation_plus_strict_output_proxy | logistic | ridge_logistic | 0.066532 | 0.114079 | 0.595568 | 0.274510 | 0.534667 | 12.800000 | 10.800000 |
| random_stratified | rich_output_reference | stump | decision_stump | 0.045081 | 0.072991 | 0.544238 | 0.606536 | 0.532667 | 37.000000 | 26.600000 |
| random_stratified | metadata_diagnostic | logistic | ridge_logistic | 0.042293 | 0.073335 | 0.515585 | 0.498039 | 0.512000 | 28.400000 | 22.600000 |
| random_stratified | metadata_diagnostic | stump | decision_stump | 0.041807 | 0.071084 | 0.526114 | 0.528105 | 0.516667 | 29.000000 | 25.000000 |
| random_stratified | activation_only_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | activation_plus_strict_output_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | activation_plus_rich_output_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | rich_output_reference | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | metadata_diagnostic | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.490000 | 0.000000 | 0.000000 |
| random_stratified | activation_only_proxy | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | activation_plus_strict_output_proxy | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | activation_plus_rich_output_proxy | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | rich_output_reference | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | metadata_diagnostic | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| random_stratified | gt_diagnostic_leakage_check | baseline | always_residual | -0.039514 | -0.070639 | 0.510000 | 1.000000 | 0.510000 | 64.600000 | 51.000000 |
| degradation_combo_group5 | activation_only_proxy | oracle | two_way_oracle | 0.574803 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| degradation_combo_group5 | activation_plus_strict_output_proxy | oracle | two_way_oracle | 0.574803 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| degradation_combo_group5 | activation_plus_rich_output_proxy | oracle | two_way_oracle | 0.574803 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

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
| airlight_leave_one | 0.75<=A<0.85 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.421030 | 0.421030 | 0.528665 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.421002 | 0.421002 | 0.515938 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.419537 | 0.419537 | 0.528941 |
| degradation_combo_group5 | 2 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.418759 | 0.418759 | 0.510235 |
| beta_leave_one | 1.4<=beta<1.7 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.418149 | 0.418149 | 0.500000 |
| airlight_leave_one | 0.75<=A<0.85 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.418007 | 0.418007 | 0.528665 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.417899 | 0.417899 | 0.500000 |
| beta_leave_one | beta<0.8 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.417756 | 0.417756 | 0.518743 |
| random_stratified | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.416964 | 0.416964 | 0.500000 |
| beta_leave_one | beta<0.8 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.416333 | 0.416333 | 0.518743 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.416150 | 0.416150 | 0.528941 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.415987 | 0.415987 | 0.515938 |
| degradation_combo_group5 | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.415657 | 0.415657 | 0.501628 |
| degradation_combo_group5 | 2 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.415606 | 0.415606 | 0.510235 |
| random_stratified | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.415583 | 0.415583 | 0.542859 |
| degradation_combo_group5 | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.414697 | 0.414697 | 0.501628 |
| airlight_leave_one | 0.65<=A<0.75 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.414445 | 0.414445 | 0.522754 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.413945 | 0.413945 | 0.500000 |
| beta_leave_one | 1.4<=beta<1.7 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.413804 | 0.413804 | 0.500000 |
| degradation_combo_group5 | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.412631 | 0.412631 | 0.489154 |
| random_stratified | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.412515 | 0.412515 | 0.500000 |
| airlight_leave_one | 0.85<=A<0.95 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.411588 | 0.411588 | 0.500358 |
| airlight_leave_one | A>=0.95 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.411250 | 0.411250 | 0.500000 |
| airlight_leave_one | 0.65<=A<0.75 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.411194 | 0.411194 | 0.522754 |
| beta_leave_one | beta>=1.7 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.410913 | 0.410913 | 0.500000 |
| random_stratified | 4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.410567 | 0.410567 | 0.542859 |
| beta_leave_one | 0.8<=beta<1.1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.409604 | 0.409604 | 0.500000 |
| degradation_combo_group5 | 3 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.409187 | 0.409187 | 0.489154 |
| beta_leave_one | 1.1<=beta<1.4 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.408601 | 0.408601 | 0.500000 |
| random_stratified | 2 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.407666 | 0.407666 | 0.502117 |

## Interpretation Rules

- Safe activation feature sets are inference-time only: hazy input summaries, frozen internal activations, LF prior statistics, and output/agreement proxies.
- Metadata and GT-aware feature sets are diagnostic only and cannot justify selector-v2.
- Any run below the sample-size rule is smoke-only even if a metric looks favorable.
- A selector-v2 route card should be written only after a metadata-free activation proxy passes random and degradation-held-out split families.
