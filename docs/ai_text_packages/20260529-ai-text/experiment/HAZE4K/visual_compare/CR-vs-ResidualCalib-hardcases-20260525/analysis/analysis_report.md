# Objective Visual Comparison Analysis

- Compare dir: `../experiment/HAZE4K/visual_compare/CR-vs-ResidualCalib-hardcases-20260525`
- Baseline: `CR-baseline`
- Current: `ResidualCalib`
- Samples: `26`

## Summary

- `num_samples`: `26`
- `mean_input_psnr`: `17.540998`
- `mean_baseline_psnr`: `31.243544`
- `mean_current_psnr`: `31.286253`
- `mean_delta_psnr`: `0.042709`
- `mean_input_ssim`: `0.834696`
- `mean_baseline_ssim`: `0.984884`
- `mean_current_ssim`: `0.983186`
- `mean_delta_ssim`: `-0.001698`
- `mean_delta_mae_improvement`: `-0.003845`
- `mean_delta_e_improvement`: `-0.366250`
- `mean_luma_abs_bias_improvement`: `-0.003804`
- `mean_saturation_abs_bias_improvement`: `-0.001360`
- `mean_dark_channel_abs_bias_improvement`: `-0.001364`
- `mean_edge_error_improvement`: `0.000729`

Tag counts:

- `color_shift_risk`: `11`
- `current_better_objective`: `13`
- `current_worse_objective`: `12`
- `mixed_or_neutral`: `1`
- `oversmooth_or_texture_loss_risk`: `1`
- `residual_haze_or_low_contrast_risk`: `6`
- `tone_shift_risk`: `5`

## Top 10 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 402_0.5_1.99.png | 9.7348 | 0.0368 | 0.0475 | 4.6862 | current_better_objective |
| 401_0.7_0.6.png | 9.5965 | 0.0053 | 0.0128 | 1.3291 | current_better_objective |
| 800_0.55_1.84.png | 7.5636 | 0.0035 | 0.0150 | 1.4890 | current_better_objective |
| 403_0.91_0.7.png | 7.5601 | 0.0027 | 0.0112 | 1.1653 | current_better_objective |
| 422_0.65_0.92.png | 6.5717 | 0.0022 | 0.0181 | 1.6604 | current_better_objective |
| 799_0.52_1.97.png | 6.5242 | 0.0047 | 0.0162 | 1.6383 | current_better_objective |
| 67_0.68_1.12.png | 5.2659 | 0.0029 | 0.0185 | 1.7701 | current_better_objective |
| 217_0.86_1.31.png | 4.9611 | 0.0023 | 0.0141 | 1.2599 | current_better_objective |
| 553_0.51_1.76.png | 3.6946 | 0.0002 | 0.0063 | 0.5373 | current_better_objective |
| 510_0.71_1.55.png | 3.0192 | 0.0001 | 0.0087 | 0.7347 | current_better_objective |

## Top 10 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 219_0.62_1.27.png | -8.8183 | -0.0027 | -0.0314 | -2.8841 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 390_0.55_1.07.png | -8.0041 | -0.0478 | -0.0645 | -6.5542 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk;oversmooth_or_texture_loss_risk |
| 220_0.68_1.35.png | -7.5572 | -0.0015 | -0.0228 | -2.0707 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 886_0.51_0.87.png | -6.8707 | -0.0013 | -0.0094 | -0.8861 | current_worse_objective |
| 88_0.63_0.94.png | -6.1857 | -0.0021 | -0.0205 | -1.9185 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 218_0.53_1.18.png | -6.0676 | -0.0063 | -0.0341 | -3.2433 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 127_0.57_0.92.png | -6.0408 | -0.0049 | -0.0154 | -1.4652 | current_worse_objective;color_shift_risk;residual_haze_or_low_contrast_risk |
| 191_0.86_1.88.png | -5.7111 | -0.0062 | -0.0165 | -1.7191 | current_worse_objective;color_shift_risk |
| 344_0.82_1.51.png | -4.5579 | -0.0109 | -0.0166 | -1.6380 | current_worse_objective;color_shift_risk |
| 343_0.83_1.07.png | -4.0266 | -0.0070 | -0.0141 | -1.0520 | current_worse_objective;color_shift_risk |

## Top 10 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 390_0.55_1.07.png | -8.0041 | -0.0478 | -0.0645 | -6.5542 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk;oversmooth_or_texture_loss_risk |
| 218_0.53_1.18.png | -6.0676 | -0.0063 | -0.0341 | -3.2433 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 219_0.62_1.27.png | -8.8183 | -0.0027 | -0.0314 | -2.8841 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 220_0.68_1.35.png | -7.5572 | -0.0015 | -0.0228 | -2.0707 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 88_0.63_0.94.png | -6.1857 | -0.0021 | -0.0205 | -1.9185 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 411_0.76_1.73.png | -2.1550 | -0.0098 | -0.0184 | -1.8295 | current_worse_objective;color_shift_risk |
| 191_0.86_1.88.png | -5.7111 | -0.0062 | -0.0165 | -1.7191 | current_worse_objective;color_shift_risk |
| 344_0.82_1.51.png | -4.5579 | -0.0109 | -0.0166 | -1.6380 | current_worse_objective;color_shift_risk |
| 127_0.57_0.92.png | -6.0408 | -0.0049 | -0.0154 | -1.4652 | current_worse_objective;color_shift_risk;residual_haze_or_low_contrast_risk |
| 412_0.52_0.64.png | -2.8293 | -0.0043 | -0.0119 | -1.1616 | current_worse_objective;color_shift_risk |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
