# Objective Visual Comparison Analysis

- Compare dir: `../experiment/HAZE4K/visual_compare/LF-v1-vs-ResidualCalib-hardcases-20260525`
- Baseline: `LF-v1`
- Current: `ResidualCalib`
- Samples: `26`

## Summary

- `num_samples`: `26`
- `mean_input_psnr`: `17.540998`
- `mean_baseline_psnr`: `31.898121`
- `mean_current_psnr`: `31.286253`
- `mean_delta_psnr`: `-0.611867`
- `mean_input_ssim`: `0.834696`
- `mean_baseline_ssim`: `0.985347`
- `mean_current_ssim`: `0.983186`
- `mean_delta_ssim`: `-0.002160`
- `mean_delta_mae_improvement`: `-0.006033`
- `mean_delta_e_improvement`: `-0.624304`
- `mean_luma_abs_bias_improvement`: `-0.006302`
- `mean_saturation_abs_bias_improvement`: `-0.000007`
- `mean_dark_channel_abs_bias_improvement`: `-0.005844`
- `mean_edge_error_improvement`: `0.000243`

Tag counts:

- `color_shift_risk`: `8`
- `current_better_objective`: `13`
- `current_worse_objective`: `13`
- `residual_haze_or_low_contrast_risk`: `4`
- `saturation_shift_risk`: `1`
- `tone_shift_risk`: `8`

## Top 10 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 510_0.71_1.55.png | 6.0278 | 0.0031 | 0.0190 | 1.7272 | current_better_objective |
| 401_0.7_0.6.png | 5.4270 | 0.0071 | 0.0080 | 0.7469 | current_better_objective |
| 553_0.51_1.76.png | 5.3413 | 0.0006 | 0.0096 | 0.8315 | current_better_objective |
| 960_0.78_0.97.png | 5.0342 | 0.0023 | 0.0060 | 0.6333 | current_better_objective |
| 958_0.9_0.64.png | 4.7408 | 0.0017 | 0.0057 | 0.6215 | current_better_objective |
| 49_0.7_1.47.png | 4.4314 | 0.0014 | 0.0111 | 0.9940 | current_better_objective |
| 556_0.61_0.91.png | 4.1876 | -0.0004 | 0.0047 | 0.3820 | current_better_objective |
| 402_0.5_1.99.png | 4.1424 | 0.0228 | 0.0180 | 1.6906 | current_better_objective |
| 403_0.91_0.7.png | 2.9671 | 0.0059 | 0.0053 | 0.5130 | current_better_objective |
| 422_0.65_0.92.png | 2.9465 | 0.0014 | 0.0055 | 0.6842 | current_better_objective |

## Top 10 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 412_0.52_0.64.png | -8.9599 | -0.0056 | -0.0242 | -2.2355 | current_worse_objective;color_shift_risk;tone_shift_risk |
| 411_0.76_1.73.png | -7.4795 | -0.0154 | -0.0430 | -4.0822 | current_worse_objective;color_shift_risk;tone_shift_risk |
| 343_0.83_1.07.png | -7.3164 | -0.0107 | -0.0204 | -2.5901 | current_worse_objective;color_shift_risk;tone_shift_risk;saturation_shift_risk |
| 219_0.62_1.27.png | -7.1134 | -0.0039 | -0.0266 | -2.5098 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 220_0.68_1.35.png | -7.0387 | -0.0022 | -0.0224 | -2.0360 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 886_0.51_0.87.png | -5.9393 | -0.0014 | -0.0085 | -0.8081 | current_worse_objective |
| 218_0.53_1.18.png | -5.8710 | -0.0089 | -0.0366 | -3.5251 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 344_0.82_1.51.png | -5.6048 | -0.0133 | -0.0205 | -2.1008 | current_worse_objective;color_shift_risk;tone_shift_risk |
| 390_0.55_1.07.png | -3.0614 | -0.0310 | -0.0297 | -3.3498 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 191_0.86_1.88.png | -2.3514 | -0.0056 | -0.0063 | -0.8531 | current_worse_objective |

## Top 10 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 411_0.76_1.73.png | -7.4795 | -0.0154 | -0.0430 | -4.0822 | current_worse_objective;color_shift_risk;tone_shift_risk |
| 218_0.53_1.18.png | -5.8710 | -0.0089 | -0.0366 | -3.5251 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 390_0.55_1.07.png | -3.0614 | -0.0310 | -0.0297 | -3.3498 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 343_0.83_1.07.png | -7.3164 | -0.0107 | -0.0204 | -2.5901 | current_worse_objective;color_shift_risk;tone_shift_risk;saturation_shift_risk |
| 219_0.62_1.27.png | -7.1134 | -0.0039 | -0.0266 | -2.5098 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 412_0.52_0.64.png | -8.9599 | -0.0056 | -0.0242 | -2.2355 | current_worse_objective;color_shift_risk;tone_shift_risk |
| 344_0.82_1.51.png | -5.6048 | -0.0133 | -0.0205 | -2.1008 | current_worse_objective;color_shift_risk;tone_shift_risk |
| 220_0.68_1.35.png | -7.0387 | -0.0022 | -0.0224 | -2.0360 | current_worse_objective;color_shift_risk;tone_shift_risk;residual_haze_or_low_contrast_risk |
| 191_0.86_1.88.png | -2.3514 | -0.0056 | -0.0063 | -0.8531 | current_worse_objective |
| 886_0.51_0.87.png | -5.9393 | -0.0014 | -0.0085 | -0.8081 | current_worse_objective |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
