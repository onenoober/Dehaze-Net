# Objective Visual Comparison Analysis

- Compare dir: `/root/workspace/Dehaze-Net/experiment/HAZE4K/visual_compare/DEA-Net-CR-vs-LF-20260522`
- Baseline: `DEA-Net-CR`
- Current: `DEA-Net-LF`
- Samples: `20`

## Summary

- `num_samples`: `20`
- `mean_input_psnr`: `16.957819`
- `mean_baseline_psnr`: `31.520396`
- `mean_current_psnr`: `31.242195`
- `mean_delta_psnr`: `-0.278201`
- `mean_input_ssim`: `0.848667`
- `mean_baseline_ssim`: `0.982670`
- `mean_current_ssim`: `0.982089`
- `mean_delta_ssim`: `-0.000581`
- `mean_delta_mae_improvement`: `-0.000218`
- `mean_delta_e_improvement`: `-0.040726`
- `mean_luma_abs_bias_improvement`: `-0.000281`
- `mean_saturation_abs_bias_improvement`: `-0.001047`
- `mean_dark_channel_abs_bias_improvement`: `-0.001463`
- `mean_edge_error_improvement`: `0.000150`

Tag counts:

- `current_better_objective`: `5`
- `current_worse_objective`: `9`
- `mixed_or_neutral`: `6`

## Top 8 Current Improvements By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 195_0.61_1.47.png | 2.1625 | 0.0024 | 0.0094 | 0.8493 | current_better_objective |
| 241_0.85_0.72.png | 1.1099 | 0.0052 | 0.0034 | 0.4121 | current_better_objective |
| 80_0.74_1.76.png | 0.9455 | -0.0121 | 0.0047 | 0.2671 | current_better_objective |
| 620_0.56_0.87.png | 0.7687 | -0.0009 | 0.0002 | 0.0547 | current_better_objective |
| 858_0.8_1.72.png | 0.5712 | -0.0008 | 0.0014 | 0.2012 | current_better_objective |
| 526_0.73_0.69.png | 0.2945 | -0.0003 | 0.0004 | 0.0479 | mixed_or_neutral |
| 147_0.87_1.14.png | 0.2029 | 0.0011 | 0.0014 | 0.3555 | mixed_or_neutral |
| 336_0.94_1.46.png | 0.0193 | 0.0022 | -0.0045 | -0.3702 | mixed_or_neutral |

## Top 8 Current Regressions By PSNR

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -3.2287 | -0.0016 | -0.0061 | -0.8587 | current_worse_objective |
| 715_0.63_1.36.png | -1.9336 | -0.0008 | -0.0009 | -0.0989 | current_worse_objective |
| 9_0.85_1.67.png | -1.8960 | -0.0007 | -0.0043 | -0.6029 | current_worse_objective |
| 763_0.82_1.68.png | -1.0667 | -0.0002 | -0.0011 | -0.1108 | current_worse_objective |
| 952_0.97_1.33.png | -1.0479 | -0.0012 | -0.0007 | -0.2133 | current_worse_objective |
| 904_0.82_1.3.png | -0.7103 | -0.0007 | -0.0014 | -0.1537 | current_worse_objective |
| 1000_0.73_1.8.png | -0.5733 | -0.0006 | -0.0020 | -0.1692 | current_worse_objective |
| 479_0.79_0.62.png | -0.5090 | -0.0027 | -0.0031 | -0.2579 | current_worse_objective |

## Top 8 Color Regressions

| filename | delta_psnr | delta_ssim | delta_mae_improvement | delta_e_improvement | tags |
| --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | -3.2287 | -0.0016 | -0.0061 | -0.8587 | current_worse_objective |
| 9_0.85_1.67.png | -1.8960 | -0.0007 | -0.0043 | -0.6029 | current_worse_objective |
| 336_0.94_1.46.png | 0.0193 | 0.0022 | -0.0045 | -0.3702 | mixed_or_neutral |
| 479_0.79_0.62.png | -0.5090 | -0.0027 | -0.0031 | -0.2579 | current_worse_objective |
| 952_0.97_1.33.png | -1.0479 | -0.0012 | -0.0007 | -0.2133 | current_worse_objective |
| 431_0.86_1.86.png | -0.4254 | -0.0010 | -0.0020 | -0.1891 | current_worse_objective |
| 1000_0.73_1.8.png | -0.5733 | -0.0006 | -0.0020 | -0.1692 | current_worse_objective |
| 904_0.82_1.3.png | -0.7103 | -0.0007 | -0.0014 | -0.1537 | current_worse_objective |

## Reading Notes

- Positive `delta_psnr`, `delta_ssim`, `delta_mae_improvement`, and `delta_e_improvement` mean the current output is closer to GT than baseline by that metric.
- Heatmaps use bright error colors for absolute error. The improvement map uses green where current improves over baseline and red where current is worse.
- Risk tags are objective hints for triage, not final visual judgments.
