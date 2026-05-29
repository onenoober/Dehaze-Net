# Activation Selector Proxy Learnability Audit

## Verdict

- Recommendation: `do_not_train_selector_v2_underpowered_sample`
- Reason: The activation audit ran on too few or too imbalanced images for a conclusive selector decision. Treat it as smoke only.
- Sample-size verdict: `underpowered_smoke_only`
- Images: `80`; positive/negative target counts: `39` / `41`
- Minimum conclusive sample rule: images `>= 800`, class counts `>= 120`, min split class count `>= 20`
- Split families: `airlight_leave_one, degradation_combo_group5, random_stratified`
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
| airlight_leave_one | activation_only_proxy | logistic | ridge_logistic | 0.042367 | 0.075569 | 0.727273 | 0.800000 | 0.772727 |
| degradation_combo_group5 | activation_plus_strict_output_proxy | logistic | ridge_logistic | 0.178890 | 0.227718 | 0.627404 | 0.698684 | 0.662500 |
| random_stratified | activation_only_proxy | logistic | ridge_logistic | 0.449799 | 0.706989 | 0.904545 | 0.791667 | 0.854167 |

## Top Aggregate Rows

| split_family | feature_set | model_type | name | gain_vs_lfv1_psnr_mean | oracle_recovery_mean | residual_precision_mean | residual_recall_mean | two_way_accuracy_mean | lost_lfv1_gain_selected_count_mean | residual_worst_selected_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_stratified | activation_only_proxy | oracle | two_way_oracle | 0.636050 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | activation_plus_strict_output_proxy | oracle | two_way_oracle | 0.636050 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | activation_plus_rich_output_proxy | oracle | two_way_oracle | 0.636050 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | rich_output_reference | oracle | two_way_oracle | 0.636050 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | metadata_diagnostic | oracle | two_way_oracle | 0.636050 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | oracle | two_way_oracle | 0.636050 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | logistic | ridge_logistic | 0.634723 | 0.997913 | 0.923077 | 1.000000 | 0.958333 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | stump | decision_stump | 0.631442 | 0.992754 | 0.857143 | 1.000000 | 0.916667 | 0.000000 | 0.000000 |
| random_stratified | activation_only_proxy | logistic | ridge_logistic | 0.449799 | 0.706989 | 0.904545 | 0.791667 | 0.854167 | 1.000000 | 0.000000 |
| random_stratified | activation_plus_strict_output_proxy | logistic | ridge_logistic | 0.422574 | 0.664784 | 0.900000 | 0.750000 | 0.833333 | 1.000000 | 0.000000 |
| random_stratified | activation_plus_rich_output_proxy | logistic | ridge_logistic | 0.422574 | 0.664784 | 0.900000 | 0.750000 | 0.833333 | 1.000000 | 0.000000 |
| random_stratified | rich_output_reference | logistic | ridge_logistic | 0.317802 | 0.502656 | 0.682692 | 0.708333 | 0.687500 | 2.500000 | 0.500000 |
| random_stratified | rich_output_reference | stump | decision_stump | 0.288104 | 0.454674 | 0.583333 | 0.583333 | 0.583333 | 1.500000 | 2.000000 |
| random_stratified | activation_plus_strict_output_proxy | stump | decision_stump | 0.117672 | 0.184052 | 0.464286 | 0.333333 | 0.479167 | 2.000000 | 0.500000 |
| random_stratified | metadata_diagnostic | logistic | ridge_logistic | 0.094746 | 0.146050 | 0.578947 | 0.916667 | 0.625000 | 4.000000 | 2.500000 |
| random_stratified | metadata_diagnostic | stump | decision_stump | 0.080739 | 0.124336 | 0.575188 | 0.958333 | 0.625000 | 4.500000 | 2.500000 |
| random_stratified | activation_only_proxy | stump | decision_stump | 0.063668 | 0.100333 | 0.464286 | 0.416667 | 0.479167 | 2.000000 | 1.500000 |
| random_stratified | activation_only_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.500000 | 0.000000 | 0.000000 |
| random_stratified | activation_plus_strict_output_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.500000 | 0.000000 | 0.000000 |
| random_stratified | activation_plus_rich_output_proxy | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.500000 | 0.000000 | 0.000000 |
| random_stratified | rich_output_reference | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.500000 | 0.000000 | 0.000000 |
| random_stratified | metadata_diagnostic | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.500000 | 0.000000 | 0.000000 |
| random_stratified | gt_diagnostic_leakage_check | baseline | always_lfv1 | 0.000000 | 0.000000 |  | 0.000000 | 0.500000 | 0.000000 | 0.000000 |
| random_stratified | activation_only_proxy | baseline | always_residual | -0.116236 | -0.186492 | 0.500000 | 1.000000 | 0.500000 | 5.500000 | 4.000000 |
| random_stratified | activation_plus_strict_output_proxy | baseline | always_residual | -0.116236 | -0.186492 | 0.500000 | 1.000000 | 0.500000 | 5.500000 | 4.000000 |
| random_stratified | activation_plus_rich_output_proxy | baseline | always_residual | -0.116236 | -0.186492 | 0.500000 | 1.000000 | 0.500000 | 5.500000 | 4.000000 |
| random_stratified | rich_output_reference | baseline | always_residual | -0.116236 | -0.186492 | 0.500000 | 1.000000 | 0.500000 | 5.500000 | 4.000000 |
| random_stratified | metadata_diagnostic | baseline | always_residual | -0.116236 | -0.186492 | 0.500000 | 1.000000 | 0.500000 | 5.500000 | 4.000000 |
| random_stratified | gt_diagnostic_leakage_check | baseline | always_residual | -0.116236 | -0.186492 | 0.500000 | 1.000000 | 0.500000 | 5.500000 | 4.000000 |
| degradation_combo_group5 | activation_only_proxy | oracle | two_way_oracle | 0.676795 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

## Top Held-Out Stumps

| split_family | split | feature_set | feature | op | threshold | valid_gain_vs_lfv1_psnr | valid_oracle_recovery | valid_residual_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | <= | 0.000025 | 0.732709 | 0.984514 | 0.904762 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 1.011515 | 0.640455 | 0.992857 | 0.857143 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | <= | 1.011515 | 0.622429 | 0.992651 | 0.857143 |
| degradation_combo_group5 | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | <= | -0.000009 | 0.608798 | 0.999085 | 1.000000 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | <= | -0.000021 | 0.560634 | 1.000000 | 1.000000 |
| degradation_combo_group5 | 0 | rich_output_reference | residual_lfv1_highfreq_abs_mean_ratio | >= | 1.002629 | 0.476432 | 0.640165 | 0.636364 |
| random_stratified | 0 | rich_output_reference | residual_lfv1_lap_var_higher_flag | >= | 1.000000 | 0.361028 | 0.575768 | 0.666667 |
| degradation_combo_group5 | 1 | rich_output_reference | residual_lfv1_lap_var_higher_flag | >= | 1.000000 | 0.239268 | 0.392658 | 0.631579 |
| random_stratified | 1 | rich_output_reference | residual_lfv1_highfreq_abs_mean_higher_flag | >= | 1.000000 | 0.215180 | 0.333579 | 0.500000 |
| random_stratified | 1 | metadata_diagnostic | beta | <= | 1.712500 | 0.198664 | 0.307976 | 0.571429 |
| degradation_combo_group5 | 1 | activation_only_proxy | lfv1_baseline_down1_spatial_std_mean_abs_delta | >= | 0.000163 | 0.197869 | 0.324718 | 0.714286 |
| degradation_combo_group5 | 1 | activation_plus_strict_output_proxy | lfv1_baseline_down1_spatial_std_mean_abs_delta | >= | 0.000163 | 0.197869 | 0.324718 | 0.714286 |
| random_stratified | 1 | activation_plus_strict_output_proxy | residual_lfv1_lap_var_delta | >= | 0.000815 | 0.162090 | 0.251278 | 0.500000 |
| random_stratified | 0 | activation_only_proxy | lfv1_baseline_mix1_channel_mean_std_abs_delta | >= | 0.015510 | 0.073254 | 0.116825 | 0.428571 |
| random_stratified | 0 | activation_plus_strict_output_proxy | lfv1_baseline_mix1_channel_mean_std_abs_delta | >= | 0.015510 | 0.073254 | 0.116825 | 0.428571 |
| degradation_combo_group5 | 1 | metadata_diagnostic | beta | <= | 1.502500 | 0.067386 | 0.110585 | 0.625000 |
| random_stratified | 1 | activation_only_proxy | residual_lf_calib_direction_active_frac_001 | <= | 0.965068 | 0.054082 | 0.083840 | 0.500000 |
| airlight_leave_one | A<0.65 | rich_output_reference | residual_lfv1_luma_std_abs_delta_x_baseline_lap_var | <= | 0.000775 | -0.025490 | -0.045467 | 0.538462 |
| random_stratified | 0 | metadata_diagnostic | beta | <= | 1.655000 | -0.037185 | -0.059303 | 0.578947 |
| degradation_combo_group5 | 0 | metadata_diagnostic | airlight | >= | 0.630000 | -0.110829 | -0.148917 | 0.400000 |

## Top Logistic Coefficients

| split_family | split | feature_set | feature | weight | abs_weight | prob_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.275632 | 0.275632 | 0.472375 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.273866 | 0.273866 | 0.472375 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.271444 | 0.271444 | 0.343532 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.268459 | 0.268459 | 0.343532 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.256863 | 0.256863 | 0.368185 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.255325 | 0.255325 | 0.368185 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.253414 | 0.253414 | 0.584771 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.252345 | 0.252345 | 0.584771 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.249789 | 0.249789 | 0.472375 |
| degradation_combo_group5 | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_cosine | 0.243916 | 0.243916 | 0.606817 |
| degradation_combo_group5 | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_luma_residual_cosine | 0.238376 | 0.238376 | 0.606817 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.238053 | 0.238053 | 0.343532 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.232055 | 0.232055 | 0.584771 |
| degradation_combo_group5 | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | -0.226976 | 0.226976 | 0.606817 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.219902 | 0.219902 | 0.368185 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | -0.214652 | 0.214652 | 0.584771 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | -0.213877 | 0.213877 | 0.472375 |
| random_stratified | 0 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | -0.212480 | 0.212480 | 0.343532 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | rescalib_from_baseline_residual_cosine | 0.209135 | 0.209135 | 0.584771 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_baseline_residual_cosine | 0.206902 | 0.206902 | 0.368185 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_lf_mse_delta | -0.206702 | 0.206702 | 0.368185 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | rescalib_from_baseline_luma_residual_cosine | 0.202888 | 0.202888 | 0.584771 |
| degradation_combo_group5 | 1 | gt_diagnostic_leakage_check | rescalib_from_lfv1_residual_error_ratio | -0.201625 | 0.201625 | 0.606817 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | residual_color_regression_vs_lfv1 | -0.200620 | 0.200620 | 0.472375 |
| random_stratified | 1 | gt_diagnostic_leakage_check | rescalib_from_baseline_luma_residual_cosine | 0.197704 | 0.197704 | 0.368185 |
| airlight_leave_one | A<0.65 | gt_diagnostic_leakage_check | residual_color_regression_vs_lfv1 | -0.194254 | 0.194254 | 0.584771 |
| degradation_combo_group5 | 1 | metadata_diagnostic | beta | -0.191094 | 0.191094 | 0.449813 |
| random_stratified | 1 | gt_diagnostic_leakage_check | residual_color_regression_vs_lfv1 | -0.189159 | 0.189159 | 0.368185 |
| degradation_combo_group5 | 0 | gt_diagnostic_leakage_check | rescalib_from_baseline_residual_cosine | 0.188410 | 0.188410 | 0.472375 |
| random_stratified | 0 | gt_diagnostic_leakage_check | residual_color_regression_vs_lfv1 | -0.187097 | 0.187097 | 0.343532 |

## Interpretation Rules

- Safe activation feature sets are inference-time only: hazy input summaries, frozen internal activations, LF prior statistics, and output/agreement proxies.
- Metadata and GT-aware feature sets are diagnostic only and cannot justify selector-v2.
- Any run below the sample-size rule is smoke-only even if a metric looks favorable.
- A selector-v2 route card should be written only after a metadata-free activation proxy passes random and degradation-held-out split families.
