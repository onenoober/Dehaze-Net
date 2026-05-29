# Objective Visual Comparison Analysis

- Compare dir: `/root/workspace/Dehaze-Net/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-gate-sweep-20260522/scale_0p25`
- Baseline: `DEA-Net-CR`
- Current: `DEA-Net-LF gate x0.25`
- Samples: `20`

## Summary

- `num_samples`: `20`
- `mean_input_psnr`: `16.957819`
- `mean_baseline_psnr`: `31.520396`
- `mean_current_psnr`: `31.253256`
- `mean_delta_psnr`: `-0.267140`
- `mean_input_ssim`: `0.848667`
- `mean_baseline_ssim`: `0.982670`
- `mean_current_ssim`: `0.982033`
- `mean_delta_ssim`: `-0.000636`
- `mean_delta_mae_improvement`: `-0.000227`
- `mean_delta_e_improvement`: `-0.041478`
- `mean_luma_abs_bias_improvement`: `0.000010`
- `mean_saturation_abs_bias_improvement`: `-0.000952`
- `mean_dark_channel_abs_bias_improvement`: `-0.001128`
- `mean_edge_error_improvement`: `0.000172`

Tag counts:

- `current_better_objective`: `5`
- `current_worse_objective`: `9`
- `mixed_or_neutral`: `6`

## Top 8 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 195_0.61_1.47.png | 2.2104 | 0.0024 | 0.0096 | 0.8655 | current_better_objective |
| 241_0.85_0.72.png | 1.3137 | 0.0054 | 0.0041 | 0.4821 | current_better_objective |
| 80_0.74_1.76.png | 0.7797 | -0.0128 | 0.0031 | 0.1135 | current_better_objective |
| 620_0.56_0.87.png | 0.7299 | -0.0009 | 0.0002 | 0.0510 | current_better_objective |
| 526_0.73_0.69.png | 0.3207 | -0.0003 | 0.0004 | 0.0498 | current_better_objective |
| 858_0.8_1.72.png | 0.2998 | -0.0012 | 0.0009 | 0.1650 | mixed_or_neutral |
| 336_0.94_1.46.png | 0.1597 | 0.0027 | -0.0030 | -0.2224 | mixed_or_neutral |
| 668_0.69_1.25.png | 0.0710 | -0.0001 | 0.0001 | -0.0260 | mixed_or_neutral |

## Top 8 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -3.0148 | -0.0014 | -0.0056 | -0.7978 | current_worse_objective |
| 9_0.85_1.67.png | -2.1384 | -0.0009 | -0.0047 | -0.6785 | current_worse_objective |
| 715_0.63_1.36.png | -2.0414 | -0.0008 | -0.0010 | -0.1050 | current_worse_objective |
| 952_0.97_1.33.png | -0.9342 | -0.0011 | -0.0004 | -0.1864 | current_worse_objective |
| 763_0.82_1.68.png | -0.8309 | -0.0003 | -0.0007 | -0.0807 | current_worse_objective |
| 904_0.82_1.3.png | -0.6694 | -0.0007 | -0.0012 | -0.1362 | current_worse_objective |
| 1000_0.73_1.8.png | -0.6491 | -0.0006 | -0.0023 | -0.1914 | current_worse_objective |
| 431_0.86_1.86.png | -0.4316 | -0.0011 | -0.0024 | -0.2147 | current_worse_objective |

## Top 8 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -3.0148 | -0.0014 | -0.0056 | -0.7978 | current_worse_objective |
| 9_0.85_1.67.png | -2.1384 | -0.0009 | -0.0047 | -0.6785 | current_worse_objective |
| 336_0.94_1.46.png | 0.1597 | 0.0027 | -0.0030 | -0.2224 | mixed_or_neutral |
| 431_0.86_1.86.png | -0.4316 | -0.0011 | -0.0024 | -0.2147 | current_worse_objective |
| 479_0.79_0.62.png | -0.3167 | -0.0024 | -0.0025 | -0.1924 | current_worse_objective |
| 1000_0.73_1.8.png | -0.6491 | -0.0006 | -0.0023 | -0.1914 | current_worse_objective |
| 952_0.97_1.33.png | -0.9342 | -0.0011 | -0.0004 | -0.1864 | current_worse_objective |
| 904_0.82_1.3.png | -0.6694 | -0.0007 | -0.0012 | -0.1362 | current_worse_objective |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
