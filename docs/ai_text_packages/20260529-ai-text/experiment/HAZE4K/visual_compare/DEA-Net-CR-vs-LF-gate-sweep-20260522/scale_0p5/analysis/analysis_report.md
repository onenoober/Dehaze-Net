# Objective Visual Comparison Analysis

- Compare dir: `/root/workspace/Dehaze-Net/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-gate-sweep-20260522/scale_0p5`
- Baseline: `DEA-Net-CR`
- Current: `DEA-Net-LF gate x0.5`
- Samples: `20`

## Summary

- `num_samples`: `20`
- `mean_input_psnr`: `16.957819`
- `mean_baseline_psnr`: `31.520396`
- `mean_current_psnr`: `31.251933`
- `mean_delta_psnr`: `-0.268463`
- `mean_input_ssim`: `0.848667`
- `mean_baseline_ssim`: `0.982670`
- `mean_current_ssim`: `0.982061`
- `mean_delta_ssim`: `-0.000609`
- `mean_delta_mae_improvement`: `-0.000217`
- `mean_delta_e_improvement`: `-0.040559`
- `mean_luma_abs_bias_improvement`: `-0.000086`
- `mean_saturation_abs_bias_improvement`: `-0.000983`
- `mean_dark_channel_abs_bias_improvement`: `-0.001243`
- `mean_edge_error_improvement`: `0.000166`

Tag counts:

- `current_better_objective`: `6`
- `current_worse_objective`: `9`
- `mixed_or_neutral`: `5`

## Top 8 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 195_0.61_1.47.png | 2.1953 | 0.0024 | 0.0095 | 0.8606 | current_better_objective |
| 241_0.85_0.72.png | 1.2460 | 0.0054 | 0.0039 | 0.4591 | current_better_objective |
| 80_0.74_1.76.png | 0.8337 | -0.0125 | 0.0036 | 0.1640 | current_better_objective |
| 620_0.56_0.87.png | 0.7427 | -0.0009 | 0.0002 | 0.0522 | current_better_objective |
| 858_0.8_1.72.png | 0.3992 | -0.0010 | 0.0011 | 0.1788 | current_better_objective |
| 526_0.73_0.69.png | 0.3133 | -0.0003 | 0.0004 | 0.0493 | current_better_objective |
| 336_0.94_1.46.png | 0.1140 | 0.0025 | -0.0035 | -0.2706 | mixed_or_neutral |
| 147_0.87_1.14.png | 0.1086 | 0.0006 | 0.0006 | 0.2766 | mixed_or_neutral |

## Top 8 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -3.0867 | -0.0015 | -0.0058 | -0.8179 | current_worse_objective |
| 9_0.85_1.67.png | -2.0477 | -0.0008 | -0.0046 | -0.6508 | current_worse_objective |
| 715_0.63_1.36.png | -2.0052 | -0.0008 | -0.0010 | -0.1027 | current_worse_objective |
| 952_0.97_1.33.png | -0.9709 | -0.0011 | -0.0005 | -0.1947 | current_worse_objective |
| 763_0.82_1.68.png | -0.8969 | -0.0002 | -0.0008 | -0.0890 | current_worse_objective |
| 904_0.82_1.3.png | -0.6828 | -0.0007 | -0.0013 | -0.1416 | current_worse_objective |
| 1000_0.73_1.8.png | -0.6242 | -0.0006 | -0.0022 | -0.1840 | current_worse_objective |
| 431_0.86_1.86.png | -0.4285 | -0.0010 | -0.0023 | -0.2047 | current_worse_objective |

## Top 8 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -3.0867 | -0.0015 | -0.0058 | -0.8179 | current_worse_objective |
| 9_0.85_1.67.png | -2.0477 | -0.0008 | -0.0046 | -0.6508 | current_worse_objective |
| 336_0.94_1.46.png | 0.1140 | 0.0025 | -0.0035 | -0.2706 | mixed_or_neutral |
| 479_0.79_0.62.png | -0.3812 | -0.0025 | -0.0027 | -0.2140 | current_worse_objective |
| 431_0.86_1.86.png | -0.4285 | -0.0010 | -0.0023 | -0.2047 | current_worse_objective |
| 952_0.97_1.33.png | -0.9709 | -0.0011 | -0.0005 | -0.1947 | current_worse_objective |
| 1000_0.73_1.8.png | -0.6242 | -0.0006 | -0.0022 | -0.1840 | current_worse_objective |
| 904_0.82_1.3.png | -0.6828 | -0.0007 | -0.0013 | -0.1416 | current_worse_objective |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
