# Objective Visual Comparison Analysis

- Compare dir: `/root/workspace/Dehaze-Net/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-Conservative-20260522`
- Baseline: `DEA-Net-CR`
- Current: `DEA-Net-LF-Conservative`
- Samples: `20`

## Summary

- `num_samples`: `20`
- `mean_input_psnr`: `16.957819`
- `mean_baseline_psnr`: `31.520396`
- `mean_current_psnr`: `30.737899`
- `mean_delta_psnr`: `-0.782497`
- `mean_input_ssim`: `0.848667`
- `mean_baseline_ssim`: `0.982670`
- `mean_current_ssim`: `0.982192`
- `mean_delta_ssim`: `-0.000478`
- `mean_delta_mae_improvement`: `-0.001668`
- `mean_delta_e_improvement`: `-0.147932`
- `mean_luma_abs_bias_improvement`: `-0.001998`
- `mean_saturation_abs_bias_improvement`: `-0.000999`
- `mean_dark_channel_abs_bias_improvement`: `-0.002632`
- `mean_edge_error_improvement`: `-0.000629`

Tag counts:

- `current_better_objective`: `4`
- `current_worse_objective`: `14`
- `mixed_or_neutral`: `2`
- `residual_haze_or_low_contrast_risk`: `1`

## Top 8 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 195_0.61_1.47.png | 1.2976 | 0.0013 | 0.0060 | 0.5499 | current_better_objective |
| 573_0.91_1.54.png | 1.1647 | 0.0019 | 0.0019 | 0.1854 | current_better_objective |
| 715_0.63_1.36.png | 1.0200 | 0.0000 | 0.0007 | 0.0645 | current_better_objective |
| 80_0.74_1.76.png | 0.8401 | 0.0049 | 0.0085 | 0.7894 | current_better_objective |
| 147_0.87_1.14.png | -0.0117 | -0.0004 | -0.0006 | 0.1331 | mixed_or_neutral |
| 1000_0.73_1.8.png | -0.1712 | -0.0009 | -0.0008 | -0.1695 | mixed_or_neutral |
| 904_0.82_1.3.png | -0.3203 | 0.0000 | -0.0007 | -0.0496 | current_worse_objective |
| 336_0.94_1.46.png | -0.3850 | -0.0002 | -0.0021 | -0.1576 | current_worse_objective |

## Top 8 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 952_0.97_1.33.png | -3.5632 | -0.0019 | -0.0078 | -0.8197 | current_worse_objective;residual_haze_or_low_contrast_risk |
| 763_0.82_1.68.png | -2.3363 | -0.0034 | -0.0018 | -0.2036 | current_worse_objective |
| 858_0.8_1.72.png | -2.2963 | -0.0022 | -0.0041 | -0.3660 | current_worse_objective |
| 479_0.79_0.62.png | -2.2739 | -0.0031 | -0.0102 | -0.9209 | current_worse_objective |
| 28_0.81_1.96.png | -2.0983 | -0.0022 | -0.0070 | -0.6893 | current_worse_objective |
| 9_0.85_1.67.png | -1.3830 | 0.0005 | -0.0039 | -0.1820 | current_worse_objective |
| 384_0.97_0.82.png | -1.2780 | -0.0006 | -0.0027 | -0.3382 | current_worse_objective |
| 668_0.69_1.25.png | -1.0085 | -0.0004 | -0.0014 | -0.1787 | current_worse_objective |

## Top 8 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 479_0.79_0.62.png | -2.2739 | -0.0031 | -0.0102 | -0.9209 | current_worse_objective |
| 952_0.97_1.33.png | -3.5632 | -0.0019 | -0.0078 | -0.8197 | current_worse_objective;residual_haze_or_low_contrast_risk |
| 28_0.81_1.96.png | -2.0983 | -0.0022 | -0.0070 | -0.6893 | current_worse_objective |
| 858_0.8_1.72.png | -2.2963 | -0.0022 | -0.0041 | -0.3660 | current_worse_objective |
| 384_0.97_0.82.png | -1.2780 | -0.0006 | -0.0027 | -0.3382 | current_worse_objective |
| 241_0.85_0.72.png | -0.6606 | 0.0013 | -0.0034 | -0.2962 | current_worse_objective |
| 431_0.86_1.86.png | -0.7232 | -0.0035 | -0.0027 | -0.2257 | current_worse_objective |
| 763_0.82_1.68.png | -2.3363 | -0.0034 | -0.0018 | -0.2036 | current_worse_objective |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
