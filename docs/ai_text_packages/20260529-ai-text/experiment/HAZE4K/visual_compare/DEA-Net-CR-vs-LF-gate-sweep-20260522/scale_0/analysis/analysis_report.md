# Objective Visual Comparison Analysis

- Compare dir: `/root/workspace/Dehaze-Net/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-gate-sweep-20260522/scale_0`
- Baseline: `DEA-Net-CR`
- Current: `DEA-Net-LF gate x0`
- Samples: `20`

## Summary

- `num_samples`: `20`
- `mean_input_psnr`: `16.957819`
- `mean_baseline_psnr`: `31.520396`
- `mean_current_psnr`: `31.252323`
- `mean_delta_psnr`: `-0.268072`
- `mean_input_ssim`: `0.848667`
- `mean_baseline_ssim`: `0.982670`
- `mean_current_ssim`: `0.981998`
- `mean_delta_ssim`: `-0.000671`
- `mean_delta_mae_improvement`: `-0.000244`
- `mean_delta_e_improvement`: `-0.042975`
- `mean_luma_abs_bias_improvement`: `0.000107`
- `mean_saturation_abs_bias_improvement`: `-0.000920`
- `mean_dark_channel_abs_bias_improvement`: `-0.001001`
- `mean_edge_error_improvement`: `0.000179`

Tag counts:

- `current_better_objective`: `5`
- `current_worse_objective`: `8`
- `mixed_or_neutral`: `7`

## Top 8 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 195_0.61_1.47.png | 2.2251 | 0.0024 | 0.0097 | 0.8706 | current_better_objective |
| 241_0.85_0.72.png | 1.3815 | 0.0055 | 0.0043 | 0.5047 | current_better_objective |
| 80_0.74_1.76.png | 0.7262 | -0.0130 | 0.0026 | 0.0637 | current_better_objective |
| 620_0.56_0.87.png | 0.7153 | -0.0009 | 0.0001 | 0.0496 | current_better_objective |
| 526_0.73_0.69.png | 0.3286 | -0.0003 | 0.0004 | 0.0502 | current_better_objective |
| 336_0.94_1.46.png | 0.2053 | 0.0028 | -0.0025 | -0.1740 | mixed_or_neutral |
| 858_0.8_1.72.png | 0.1928 | -0.0015 | 0.0008 | 0.1493 | mixed_or_neutral |
| 668_0.69_1.25.png | 0.0863 | -0.0001 | 0.0001 | -0.0240 | mixed_or_neutral |

## Top 8 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -2.9436 | -0.0014 | -0.0055 | -0.7771 | current_worse_objective |
| 9_0.85_1.67.png | -2.2388 | -0.0010 | -0.0049 | -0.7065 | current_worse_objective |
| 715_0.63_1.36.png | -2.0769 | -0.0008 | -0.0010 | -0.1071 | current_worse_objective |
| 952_0.97_1.33.png | -0.9000 | -0.0011 | -0.0004 | -0.1790 | current_worse_objective |
| 763_0.82_1.68.png | -0.7737 | -0.0004 | -0.0007 | -0.0743 | current_worse_objective |
| 1000_0.73_1.8.png | -0.6736 | -0.0007 | -0.0023 | -0.1986 | current_worse_objective |
| 904_0.82_1.3.png | -0.6566 | -0.0006 | -0.0012 | -0.1312 | current_worse_objective |
| 431_0.86_1.86.png | -0.4351 | -0.0011 | -0.0026 | -0.2265 | current_worse_objective |

## Top 8 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -2.9436 | -0.0014 | -0.0055 | -0.7771 | current_worse_objective |
| 9_0.85_1.67.png | -2.2388 | -0.0010 | -0.0049 | -0.7065 | current_worse_objective |
| 431_0.86_1.86.png | -0.4351 | -0.0011 | -0.0026 | -0.2265 | current_worse_objective |
| 1000_0.73_1.8.png | -0.6736 | -0.0007 | -0.0023 | -0.1986 | current_worse_objective |
| 952_0.97_1.33.png | -0.9000 | -0.0011 | -0.0004 | -0.1790 | current_worse_objective |
| 336_0.94_1.46.png | 0.2053 | 0.0028 | -0.0025 | -0.1740 | mixed_or_neutral |
| 479_0.79_0.62.png | -0.2526 | -0.0023 | -0.0022 | -0.1702 | mixed_or_neutral |
| 904_0.82_1.3.png | -0.6566 | -0.0006 | -0.0012 | -0.1312 | current_worse_objective |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
