# Objective Visual Comparison Analysis

- Compare dir: `/root/workspace/Dehaze-Net/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-gate-sweep-20260522/scale_0p75`
- Baseline: `DEA-Net-CR`
- Current: `DEA-Net-LF gate x0.75`
- Samples: `20`

## Summary

- `num_samples`: `20`
- `mean_input_psnr`: `16.957819`
- `mean_baseline_psnr`: `31.520396`
- `mean_current_psnr`: `31.248369`
- `mean_delta_psnr`: `-0.272027`
- `mean_input_ssim`: `0.848667`
- `mean_baseline_ssim`: `0.982670`
- `mean_current_ssim`: `0.982078`
- `mean_delta_ssim`: `-0.000592`
- `mean_delta_mae_improvement`: `-0.000214`
- `mean_delta_e_improvement`: `-0.040232`
- `mean_luma_abs_bias_improvement`: `-0.000183`
- `mean_saturation_abs_bias_improvement`: `-0.001014`
- `mean_dark_channel_abs_bias_improvement`: `-0.001351`
- `mean_edge_error_improvement`: `0.000160`

Tag counts:

- `current_better_objective`: `6`
- `current_worse_objective`: `9`
- `mixed_or_neutral`: `5`

## Top 8 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 195_0.61_1.47.png | 2.1793 | 0.0024 | 0.0095 | 0.8553 | current_better_objective |
| 241_0.85_0.72.png | 1.1782 | 0.0053 | 0.0037 | 0.4359 | current_better_objective |
| 80_0.74_1.76.png | 0.8888 | -0.0123 | 0.0041 | 0.2150 | current_better_objective |
| 620_0.56_0.87.png | 0.7550 | -0.0009 | 0.0002 | 0.0532 | current_better_objective |
| 858_0.8_1.72.png | 0.4904 | -0.0009 | 0.0013 | 0.1915 | current_better_objective |
| 526_0.73_0.69.png | 0.3043 | -0.0003 | 0.0004 | 0.0485 | current_better_objective |
| 147_0.87_1.14.png | 0.1555 | 0.0008 | 0.0010 | 0.3165 | mixed_or_neutral |
| 336_0.94_1.46.png | 0.0672 | 0.0024 | -0.0040 | -0.3198 | mixed_or_neutral |

## Top 8 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -3.1567 | -0.0015 | -0.0060 | -0.8381 | current_worse_objective |
| 715_0.63_1.36.png | -1.9686 | -0.0008 | -0.0009 | -0.1006 | current_worse_objective |
| 9_0.85_1.67.png | -1.9669 | -0.0007 | -0.0044 | -0.6255 | current_worse_objective |
| 952_0.97_1.33.png | -1.0082 | -0.0012 | -0.0006 | -0.2036 | current_worse_objective |
| 763_0.82_1.68.png | -0.9759 | -0.0002 | -0.0010 | -0.0990 | current_worse_objective |
| 904_0.82_1.3.png | -0.6960 | -0.0007 | -0.0013 | -0.1475 | current_worse_objective |
| 1000_0.73_1.8.png | -0.5992 | -0.0006 | -0.0021 | -0.1768 | current_worse_objective |
| 479_0.79_0.62.png | -0.4448 | -0.0026 | -0.0029 | -0.2357 | current_worse_objective |

## Top 8 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -3.1567 | -0.0015 | -0.0060 | -0.8381 | current_worse_objective |
| 9_0.85_1.67.png | -1.9669 | -0.0007 | -0.0044 | -0.6255 | current_worse_objective |
| 336_0.94_1.46.png | 0.0672 | 0.0024 | -0.0040 | -0.3198 | mixed_or_neutral |
| 479_0.79_0.62.png | -0.4448 | -0.0026 | -0.0029 | -0.2357 | current_worse_objective |
| 952_0.97_1.33.png | -1.0082 | -0.0012 | -0.0006 | -0.2036 | current_worse_objective |
| 431_0.86_1.86.png | -0.4264 | -0.0010 | -0.0021 | -0.1961 | current_worse_objective |
| 1000_0.73_1.8.png | -0.5992 | -0.0006 | -0.0021 | -0.1768 | current_worse_objective |
| 904_0.82_1.3.png | -0.6960 | -0.0007 | -0.0013 | -0.1475 | current_worse_objective |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
