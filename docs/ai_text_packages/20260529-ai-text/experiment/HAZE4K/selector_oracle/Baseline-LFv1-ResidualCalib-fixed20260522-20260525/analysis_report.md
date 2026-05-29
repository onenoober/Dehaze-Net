# HAZE4K Selector Oracle Analysis

## Summary

- Input CSV: `../experiment/HAZE4K/three_way_eval/Baseline-LFv1-ResidualCalib-full-20260525/per_image_three_way_metrics.csv`
- Images: `20`
- LF-v1 mean PSNR: `31.2580`
- ResidualCalib mean PSNR: `31.0085`
- Two-way oracle mean PSNR: `31.8019`; gain vs LF-v1 `+0.5439` dB
- Three-way oracle mean PSNR: `32.1661`; gain vs LF-v1 `+0.9080` dB
- Best one-rule selector mean PSNR: `31.8019`; gain vs LF-v1 `+0.5439` dB
- Recommendation: **continue_to_lf_residual_selector**

## Model And Oracle Rows

| name | num_images | mean_psnr | mean_ssim | gain_vs_lfv1_psnr | gain_vs_baseline_psnr | residual_selection_count | lfv1_selection_count | baseline_selection_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 31.537306 | 0.983465 | 0.279285 | 0.000000 |  |  |  |
| lfv1 | 20 | 31.258021 | 0.982879 | 0.000000 | -0.279285 |  |  |  |
| residual | 20 | 31.008494 | 0.982078 | -0.249527 | -0.528812 |  |  |  |
| always_lfv1 | 20 | 31.258021 | 0.982879 | 0.000000 | -0.279285 | 0 | 20 | 0 |
| always_residual | 20 | 31.008494 | 0.982078 | -0.249527 | -0.528812 | 20 | 0 | 0 |
| two_way_oracle_lfv1_residual | 20 | 31.801927 | 0.983492 | 0.543906 | 0.264621 | 9 | 11 | 0 |
| three_way_oracle_baseline_lfv1_residual | 20 | 32.166057 | 0.983806 | 0.908036 | 0.628751 | 7 | 6 | 7 |

## Top Rule Selectors

| feature | op | threshold | feature_class | mean_psnr | gain_vs_lfv1_psnr | residual_selection_count | residual_precision_vs_lfv1 | residual_recall_vs_lfv1 | lost_lfv1_gain_selected_count | residual_worst_selected_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rescalib_from_lfv1_residual_error_ratio | <= | 0.982439 | gt_aware_diagnostic | 31.801927 | 0.543906 | 9 | 1.000000 | 1.000000 | 0 | 0 |
| rescalib_from_lfv1_lf_mse_delta | <= | -0.000005 | gt_aware_diagnostic | 31.801927 | 0.543906 | 9 | 1.000000 | 1.000000 | 0 | 0 |
| rescalib_from_lfv1_residual_error_ratio | <= | 1.008260 | gt_aware_diagnostic | 31.797706 | 0.539685 | 10 | 0.900000 | 1.000000 | 0 | 0 |
| rescalib_from_lfv1_lf_mse_delta | <= | 0.000001 | gt_aware_diagnostic | 31.797706 | 0.539685 | 10 | 0.900000 | 1.000000 | 0 | 0 |
| rescalib_from_lfv1_residual_cosine | >= | 0.260592 | gt_aware_diagnostic | 31.794216 | 0.536195 | 11 | 0.818182 | 1.000000 | 0 | 0 |
| rescalib_from_lfv1_residual_error_ratio | <= | 1.034600 | gt_aware_diagnostic | 31.794216 | 0.536195 | 11 | 0.818182 | 1.000000 | 0 | 0 |
| rescalib_from_lfv1_lf_mse_delta | <= | 0.000004 | gt_aware_diagnostic | 31.794216 | 0.536195 | 11 | 0.818182 | 1.000000 | 0 | 0 |
| rescalib_from_lfv1_lf_mse_delta | <= | -0.000039 | gt_aware_diagnostic | 31.789689 | 0.531668 | 8 | 1.000000 | 0.888889 | 0 | 0 |
| residual_color_regression_vs_lfv1 | <= | -0.038266 | gt_aware_diagnostic | 31.789689 | 0.531668 | 8 | 1.000000 | 0.888889 | 0 | 0 |
| rescalib_from_lfv1_residual_error_ratio | <= | 0.960023 | gt_aware_diagnostic | 31.785849 | 0.527828 | 8 | 1.000000 | 0.888889 | 0 | 0 |

## Top Correlations With ResidualCalib Minus LF-v1

| feature | feature_class | corr_with_residual_delta_lfv1_psnr | abs_corr |
| --- | --- | --- | --- |
| rescalib_from_lfv1_residual_error_ratio | gt_aware_diagnostic | -0.992802 | 0.992802 |
| residual_color_regression_vs_lfv1 | gt_aware_diagnostic | -0.907547 | 0.907547 |
| rescalib_from_lfv1_residual_cosine | gt_aware_diagnostic | 0.883741 | 0.883741 |
| rescalib_from_lfv1_luma_residual_cosine | gt_aware_diagnostic | 0.874542 | 0.874542 |
| rescalib_from_baseline_residual_error_ratio | gt_aware_diagnostic | -0.757592 | 0.757592 |
| rescalib_from_baseline_residual_cosine | gt_aware_diagnostic | 0.703042 | 0.703042 |
| rescalib_from_baseline_luma_residual_cosine | gt_aware_diagnostic | 0.696228 | 0.696228 |
| residual_luma_abs_bias_regression_vs_lfv1 | gt_aware_diagnostic | -0.695634 | 0.695634 |
| residual_dark_channel_abs_bias_regression_vs_lfv1 | gt_aware_diagnostic | -0.673830 | 0.673830 |
| rescalib_from_lfv1_lf_mse_delta | gt_aware_diagnostic | -0.622848 | 0.622848 |
| residual_edge_error_regression_vs_lfv1 | gt_aware_diagnostic | -0.609059 | 0.609059 |
| residual_lfv1_dark_channel_mean_delta | proxy_or_output_feature | 0.577455 | 0.577455 |

## Groups With Largest Two-Way Oracle Gain

| group_name | group_value | num_images | lfv1_mean_psnr | residual_mean_psnr | two_way_oracle_mean_psnr | two_way_oracle_gain_vs_lfv1 | residual_winner_count | lfv1_winner_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pattern | mitigates_lfv1_regression | 1 | 31.656524 | 34.281129 | 34.281129 | 2.624606 | 1 | 0 |
| airlight_bin | A>=0.95 | 2 | 31.907779 | 31.398005 | 33.220082 | 1.312303 | 1 | 1 |
| beta_bin | 0.8<=beta<1.1 | 2 | 36.187886 | 37.334498 | 37.500188 | 1.312303 | 1 | 1 |
| pattern | residual_beats_both | 7 | 28.965700 | 30.109809 | 30.109809 | 1.144109 | 7 | 0 |
| winner_by_psnr | residual | 7 | 28.965700 | 30.109809 | 30.109809 | 1.144109 | 7 | 0 |
| baseline_strength_bin | baseline_middle_50 | 10 | 31.850400 | 32.057118 | 32.802984 | 0.952584 | 6 | 4 |
| airlight_bin | 0.75<=A<0.85 | 5 | 32.404896 | 32.472062 | 33.033808 | 0.628912 | 3 | 2 |
| airlight_bin | 0.85<=A<0.95 | 6 | 27.235917 | 26.915493 | 27.846437 | 0.610520 | 3 | 3 |
| beta_bin | beta>=1.7 | 5 | 27.458090 | 27.407939 | 28.053948 | 0.595858 | 3 | 2 |
| beta_bin | 1.4<=beta<1.7 | 5 | 30.519390 | 30.520962 | 31.094821 | 0.575431 | 2 | 3 |
| beta_bin | 1.1<=beta<1.4 | 5 | 32.287370 | 31.822334 | 32.766784 | 0.479414 | 3 | 2 |
| winner_by_psnr | baseline | 7 | 31.978203 | 30.942829 | 32.388111 | 0.409908 | 2 | 5 |

## Interpretation

The two-way oracle has enough headroom and a simple rule recovers some of it, so a bounded LFResidualSelector is justified. Keep the first implementation close to LF-v1 and use gate checks to avoid large ResidualCalib false positives.
