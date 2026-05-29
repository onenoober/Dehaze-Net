# Objective Visual Comparison Analysis

- Compare dir: `../experiment/HAZE4K/root_cause/LF-v1-hardcases-20260524/compare`
- Baseline: `DEA-Net-CR`
- Current: `DEA-Net-LF-v1`
- Samples: `150`

## Summary

- `num_samples`: `150`
- `mean_input_psnr`: `17.284548`
- `mean_baseline_psnr`: `31.853458`
- `mean_current_psnr`: `32.059294`
- `mean_delta_psnr`: `0.205836`
- `mean_input_ssim`: `0.827467`
- `mean_baseline_ssim`: `0.981280`
- `mean_current_ssim`: `0.980992`
- `mean_delta_ssim`: `-0.000288`
- `mean_delta_mae_improvement`: `0.000794`
- `mean_delta_e_improvement`: `0.064513`
- `mean_luma_abs_bias_improvement`: `0.001294`
- `mean_saturation_abs_bias_improvement`: `-0.000341`
- `mean_dark_channel_abs_bias_improvement`: `0.001594`
- `mean_edge_error_improvement`: `0.000189`

Tag counts:

- `color_shift_risk`: `9`
- `current_better_objective`: `60`
- `current_worse_objective`: `60`
- `mixed_or_neutral`: `30`
- `oversmooth_or_texture_loss_risk`: `3`
- `residual_haze_or_low_contrast_risk`: `1`
- `saturation_shift_risk`: `3`
- `tone_shift_risk`: `3`

## Top 20 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 67_0.68_1.12.png | 6.4425 | 0.0061 | 0.0224 | 2.0983 | current_better_objective |
| 800_0.55_1.84.png | 5.8847 | 0.0034 | 0.0130 | 1.3172 | current_better_objective |
| 402_0.5_1.99.png | 5.5923 | 0.0141 | 0.0296 | 2.9956 | current_better_objective;saturation_shift_risk |
| 411_0.76_1.73.png | 5.3245 | 0.0056 | 0.0246 | 2.2527 | current_better_objective |
| 90_0.89_0.67.png | 4.7950 | 0.0133 | 0.0166 | 1.8526 | current_better_objective |
| 415_0.56_1.96.png | 4.6138 | 0.0068 | 0.0111 | 1.1099 | current_better_objective |
| 212_0.54_1.78.png | 4.5239 | 0.0031 | 0.0189 | 1.7246 | current_better_objective |
| 799_0.52_1.97.png | 4.4133 | 0.0040 | 0.0127 | 1.2824 | current_better_objective |
| 404_0.71_1.78.png | 4.2075 | -0.0177 | 0.0089 | 0.6420 | current_better_objective;saturation_shift_risk |
| 828_0.63_0.64.png | 3.9045 | 0.0006 | 0.0035 | 0.3053 | current_better_objective |
| 422_0.65_0.92.png | 3.6252 | 0.0008 | 0.0126 | 0.9762 | current_better_objective |
| 624_0.58_0.68.png | 3.5036 | 0.0008 | 0.0034 | 0.3282 | current_better_objective |
| 698_0.51_1.13.png | 3.3669 | 0.0004 | 0.0032 | 0.3762 | current_better_objective |
| 430_0.66_1.24.png | 3.2533 | 0.0072 | 0.0097 | 0.9055 | current_better_objective |
| 178_0.64_1.88.png | 3.1753 | 0.0041 | 0.0140 | 0.2844 | current_better_objective;saturation_shift_risk |
| 410_0.98_1.98.png | 3.1626 | 0.0025 | 0.0180 | 1.5873 | current_better_objective |
| 13_0.52_1.9.png | 3.1398 | 0.0034 | 0.0141 | 1.3543 | current_better_objective |
| 650_0.56_0.65.png | 2.9730 | 0.0005 | 0.0018 | 0.1564 | current_better_objective |
| 409_0.54_1.68.png | 2.7605 | 0.0051 | 0.0159 | 1.4989 | current_better_objective |
| 89_0.63_1.05.png | 2.7306 | 0.0157 | 0.0148 | 1.5943 | current_better_objective |

## Top 20 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 946_0.65_0.58.png | -6.1608 | -0.0036 | -0.0059 | -0.5642 | current_worse_objective |
| 390_0.55_1.07.png | -4.9427 | -0.0168 | -0.0347 | -3.2043 | current_worse_objective;color_shift_risk;tone_shift_risk;oversmooth_or_texture_loss_risk |
| 107_0.89_1.5.png | -4.0178 | -0.0093 | -0.0110 | -0.8375 | current_worse_objective |
| 127_0.57_0.92.png | -3.9977 | -0.0018 | -0.0094 | -0.9021 | current_worse_objective |
| 950_0.57_1.28.png | -3.8683 | -0.0006 | -0.0071 | -0.6536 | current_worse_objective |
| 560_0.74_1.77.png | -3.7083 | -0.0033 | -0.0037 | -0.3365 | current_worse_objective |
| 958_0.9_0.64.png | -3.5791 | -0.0014 | -0.0040 | -0.4508 | current_worse_objective |
| 951_0.57_0.85.png | -3.1623 | 0.0000 | -0.0047 | -0.3850 | current_worse_objective |
| 287_0.77_1.14.png | -2.9617 | -0.0001 | -0.0072 | -0.5741 | current_worse_objective |
| 579_0.52_0.61.png | -2.8090 | -0.0007 | -0.0038 | -0.3777 | current_worse_objective |
| 960_0.78_0.97.png | -2.7889 | -0.0013 | -0.0029 | -0.3644 | current_worse_objective |
| 509_0.95_1.52.png | -2.5714 | -0.0063 | -0.0098 | -0.9783 | current_worse_objective |
| 389_0.75_1.27.png | -2.4685 | -0.0094 | -0.0232 | -2.4619 | current_worse_objective;color_shift_risk;tone_shift_risk;oversmooth_or_texture_loss_risk |
| 512_0.94_1.48.png | -2.4376 | -0.0057 | -0.0092 | -0.9182 | current_worse_objective |
| 288_0.82_1.17.png | -2.4336 | 0.0001 | -0.0063 | -0.4786 | current_worse_objective |
| 254_0.7_0.56.png | -2.3993 | -0.0015 | -0.0031 | -0.2411 | current_worse_objective |
| 128_0.65_0.67.png | -2.3146 | -0.0004 | -0.0046 | -0.4185 | current_worse_objective |
| 690_0.62_0.99.png | -2.2719 | -0.0003 | -0.0019 | -0.1944 | current_worse_objective |
| 966_0.54_1.62.png | -2.2692 | 0.0008 | -0.0025 | -0.1918 | current_worse_objective |
| 391_0.87_1.8.png | -2.2448 | -0.0116 | -0.0212 | -2.3823 | current_worse_objective;color_shift_risk;tone_shift_risk;oversmooth_or_texture_loss_risk |

## Top 20 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 390_0.55_1.07.png | -4.9427 | -0.0168 | -0.0347 | -3.2043 | current_worse_objective;color_shift_risk;tone_shift_risk;oversmooth_or_texture_loss_risk |
| 389_0.75_1.27.png | -2.4685 | -0.0094 | -0.0232 | -2.4619 | current_worse_objective;color_shift_risk;tone_shift_risk;oversmooth_or_texture_loss_risk |
| 391_0.87_1.8.png | -2.2448 | -0.0116 | -0.0212 | -2.3823 | current_worse_objective;color_shift_risk;tone_shift_risk;oversmooth_or_texture_loss_risk |
| 346_0.77_1.92.png | -1.9267 | -0.0143 | -0.0130 | -1.4013 | current_worse_objective;color_shift_risk |
| 348_0.72_1.07.png | -1.5076 | -0.0098 | -0.0107 | -1.1848 | current_worse_objective;color_shift_risk |
| 476_0.91_1.92.png | -2.1528 | -0.0070 | -0.0122 | -1.1755 | current_worse_objective;color_shift_risk |
| 347_0.92_1.53.png | -1.1275 | -0.0113 | -0.0106 | -1.1625 | current_worse_objective;color_shift_risk |
| 118_0.68_1.96.png | -1.6753 | -0.0009 | -0.0104 | -1.0742 | current_worse_objective;color_shift_risk |
| 117_0.69_1.13.png | -2.0534 | -0.0003 | -0.0106 | -1.0312 | current_worse_objective;color_shift_risk |
| 509_0.95_1.52.png | -2.5714 | -0.0063 | -0.0098 | -0.9783 | current_worse_objective |
| 512_0.94_1.48.png | -2.4376 | -0.0057 | -0.0092 | -0.9182 | current_worse_objective |
| 127_0.57_0.92.png | -3.9977 | -0.0018 | -0.0094 | -0.9021 | current_worse_objective |
| 107_0.89_1.5.png | -4.0178 | -0.0093 | -0.0110 | -0.8375 | current_worse_objective |
| 109_0.74_1.74.png | -1.6249 | -0.0006 | -0.0097 | -0.8217 | current_worse_objective |
| 433_0.95_1.27.png | -1.6503 | -0.0068 | -0.0067 | -0.7844 | current_worse_objective |
| 170_0.96_1.28.png | -1.5599 | -0.0116 | -0.0058 | -0.6857 | current_worse_objective |
| 436_0.98_1.16.png | -1.5737 | -0.0028 | -0.0059 | -0.6778 | current_worse_objective |
| 950_0.57_1.28.png | -3.8683 | -0.0006 | -0.0071 | -0.6536 | current_worse_objective |
| 853_1.0_1.75.png | -1.6179 | -0.0071 | -0.0060 | -0.6428 | current_worse_objective;residual_haze_or_low_contrast_risk |
| 480_0.54_0.98.png | -1.6075 | -0.0043 | -0.0069 | -0.6364 | current_worse_objective |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
