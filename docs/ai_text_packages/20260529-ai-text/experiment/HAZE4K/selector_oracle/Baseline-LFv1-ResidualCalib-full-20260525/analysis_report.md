# HAZE4K Selector Oracle Analysis

## Summary

- Input CSV: `../experiment/HAZE4K/three_way_eval/Baseline-LFv1-ResidualCalib-full-20260525/per_image_three_way_metrics.csv`
- Images: `1000`
- LF-v1 mean PSNR: `32.4283`
- ResidualCalib mean PSNR: `32.3936`
- Two-way oracle mean PSNR: `33.0034`; gain vs LF-v1 `+0.5751` dB
- Three-way oracle mean PSNR: `33.2538`; gain vs LF-v1 `+0.8255` dB
- Best one-rule selector mean PSNR: `33.0026`; gain vs LF-v1 `+0.5743` dB
- Recommendation: **continue_to_lf_residual_selector**

## Model And Oracle Rows

| name | num_images | mean_psnr | mean_ssim | gain_vs_lfv1_psnr | gain_vs_baseline_psnr | residual_selection_count | lfv1_selection_count | baseline_selection_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1000 | 32.225315 | 0.984417 | -0.202959 | 0.000000 |  |  |  |
| lfv1 | 1000 | 32.428274 | 0.984454 | 0.000000 | 0.202959 |  |  |  |
| residual | 1000 | 32.393553 | 0.984500 | -0.034721 | 0.168238 |  |  |  |
| always_lfv1 | 1000 | 32.428274 | 0.984454 | 0.000000 | 0.202959 | 0 | 1000 | 0 |
| always_residual | 1000 | 32.393553 | 0.984500 | -0.034721 | 0.168238 | 1000 | 0 | 0 |
| two_way_oracle_lfv1_residual | 1000 | 33.003395 | 0.985299 | 0.575120 | 0.778079 | 509 | 491 | 0 |
| three_way_oracle_baseline_lfv1_residual | 1000 | 33.253764 | 0.985637 | 0.825490 | 1.028449 | 383 | 322 | 295 |

## Top Rule Selectors

| feature | op | threshold | feature_class | mean_psnr | gain_vs_lfv1_psnr | residual_selection_count | residual_precision_vs_lfv1 | residual_recall_vs_lfv1 | lost_lfv1_gain_selected_count | residual_worst_selected_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rescalib_from_lfv1_residual_error_ratio | <= | 0.995704 | gt_aware_diagnostic | 33.002620 | 0.574345 | 501 | 0.992016 | 0.976424 | 0 | 0 |
| rescalib_from_lfv1_lf_mse_delta | <= | -0.000002 | gt_aware_diagnostic | 33.001924 | 0.573649 | 501 | 0.984032 | 0.968566 | 0 | 0 |
| rescalib_from_lfv1_residual_error_ratio | <= | 1.017611 | gt_aware_diagnostic | 32.999409 | 0.571134 | 550 | 0.923636 | 0.998035 | 1 | 0 |
| rescalib_from_lfv1_lf_mse_delta | <= | 0.000007 | gt_aware_diagnostic | 32.997475 | 0.569200 | 550 | 0.923636 | 0.998035 | 3 | 1 |
| rescalib_from_lfv1_residual_error_ratio | <= | 0.972752 | gt_aware_diagnostic | 32.996255 | 0.567980 | 450 | 0.997778 | 0.882122 | 0 | 0 |
| rescalib_from_lfv1_residual_error_ratio | <= | 1.036053 | gt_aware_diagnostic | 32.989440 | 0.561165 | 600 | 0.848333 | 1.000000 | 3 | 1 |
| rescalib_from_lfv1_lf_mse_delta | <= | -0.000012 | gt_aware_diagnostic | 32.989431 | 0.561156 | 451 | 0.995565 | 0.882122 | 0 | 0 |
| rescalib_from_lfv1_residual_error_ratio | <= | 0.953340 | gt_aware_diagnostic | 32.982871 | 0.554597 | 400 | 1.000000 | 0.785855 | 0 | 0 |
| rescalib_from_lfv1_lf_mse_delta | <= | 0.000017 | gt_aware_diagnostic | 32.978209 | 0.549935 | 600 | 0.846667 | 0.998035 | 19 | 13 |
| rescalib_from_lfv1_residual_cosine | >= | 0.319541 | gt_aware_diagnostic | 32.976689 | 0.548414 | 501 | 0.906188 | 0.891945 | 8 | 6 |

## Top Correlations With ResidualCalib Minus LF-v1

| feature | feature_class | corr_with_residual_delta_lfv1_psnr | abs_corr |
| --- | --- | --- | --- |
| rescalib_from_lfv1_residual_error_ratio | gt_aware_diagnostic | -0.976031 | 0.976031 |
| rescalib_from_lfv1_residual_cosine | gt_aware_diagnostic | 0.858046 | 0.858046 |
| rescalib_from_lfv1_luma_residual_cosine | gt_aware_diagnostic | 0.853838 | 0.853838 |
| residual_color_regression_vs_lfv1 | gt_aware_diagnostic | -0.842249 | 0.842249 |
| residual_luma_abs_bias_regression_vs_lfv1 | gt_aware_diagnostic | -0.721554 | 0.721554 |
| residual_dark_channel_abs_bias_regression_vs_lfv1 | gt_aware_diagnostic | -0.664155 | 0.664155 |
| rescalib_from_baseline_residual_error_ratio | gt_aware_diagnostic | -0.631230 | 0.631230 |
| rescalib_from_lfv1_lf_mse_delta | gt_aware_diagnostic | -0.586015 | 0.586015 |
| rescalib_from_baseline_residual_cosine | gt_aware_diagnostic | 0.577606 | 0.577606 |
| rescalib_from_baseline_luma_residual_cosine | gt_aware_diagnostic | 0.575296 | 0.575296 |
| rescalib_from_lfv1_residual_norm_ratio | gt_aware_diagnostic | -0.407191 | 0.407191 |
| residual_edge_error_regression_vs_lfv1 | gt_aware_diagnostic | -0.372387 | 0.372387 |

## Groups With Largest Two-Way Oracle Gain

| group_name | group_value | num_images | lfv1_mean_psnr | residual_mean_psnr | two_way_oracle_mean_psnr | two_way_oracle_gain_vs_lfv1 | residual_winner_count | lfv1_winner_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pattern | residual_beats_both | 300 | 31.724011 | 33.132212 | 33.132212 | 1.408201 | 300 | 0 |
| pattern | mitigates_lfv1_regression | 110 | 31.267099 | 32.502461 | 32.502461 | 1.235362 | 110 | 0 |
| winner_by_psnr | residual | 383 | 31.768410 | 33.000397 | 33.000397 | 1.231986 | 383 | 0 |
| baseline_strength_bin | baseline_weakest_25 | 250 | 26.130813 | 26.274161 | 26.819685 | 0.688872 | 141 | 109 |
| airlight_bin | 0.65<=A<0.75 | 194 | 32.723981 | 32.858842 | 33.411631 | 0.687650 | 106 | 88 |
| airlight_bin | A>=0.95 | 126 | 30.336150 | 30.418116 | 30.965872 | 0.629722 | 71 | 55 |
| beta_bin | 1.1<=beta<1.4 | 199 | 32.267985 | 32.244656 | 32.870224 | 0.602240 | 103 | 96 |
| beta_bin | 0.8<=beta<1.1 | 188 | 32.791666 | 32.786041 | 33.389706 | 0.598040 | 96 | 92 |
| baseline_strength_bin | baseline_middle_50 | 500 | 32.512646 | 32.404056 | 33.104958 | 0.592312 | 247 | 253 |
| beta_bin | 1.4<=beta<1.7 | 199 | 31.368530 | 31.486321 | 31.951040 | 0.582510 | 105 | 94 |
| beta_bin | beta<0.8 | 196 | 35.387845 | 35.308221 | 35.961654 | 0.573809 | 93 | 103 |
| airlight_bin | 0.85<=A<0.95 | 175 | 31.489419 | 31.520560 | 32.042253 | 0.552835 | 90 | 85 |

## Interpretation

The two-way oracle has enough headroom and a simple rule recovers some of it, so a bounded LFResidualSelector is justified. Keep the first implementation close to LF-v1 and use gate checks to avoid large ResidualCalib false positives.
