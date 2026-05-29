# Three-Way HAZE4K Output Analysis

## Summary

- Images: `20`
- Baseline PSNR/SSIM: `31.5373` / `0.983465`
- LF-v1 PSNR/SSIM: `31.2580` / `0.982879`
- ResidualCalib PSNR/SSIM: `31.0085` / `0.982078`
- LF-v1 vs baseline mean delta: `-0.2793` dB
- ResidualCalib vs baseline mean delta: `-0.5288` dB
- ResidualCalib vs LF-v1 mean delta: `-0.2495` dB
- PSNR winner counts: `{'baseline': 7, 'lfv1': 6, 'residual': 7}`
- Pattern counts: `{'both_regress_baseline': 2, 'lost_lfv1_gain': 4, 'mitigates_lfv1_regression': 1, 'mixed_or_small': 1, 'residual_beats_both': 7, 'residual_worst': 5}`
- ResidualCalib wrong-direction vs baseline: `4`
- LF-v1 wrong-direction vs baseline: `4`

## Key Group Rows

- `baseline_strength_bin=baseline_weakest_25` n=`5` LF-v1 delta `0.5809`, Residual delta baseline `-0.1994`, Residual delta LF-v1 `-0.7803`
- `baseline_strength_bin=baseline_middle_50` n=`10` LF-v1 delta `-0.5721`, Residual delta baseline `-0.4569`, Residual delta LF-v1 `0.1151`
- `baseline_strength_bin=baseline_strongest_25` n=`5` LF-v1 delta `-0.5539`, Residual delta baseline `-1.0020`, Residual delta LF-v1 `-0.4481`
- `pattern=residual_beats_both` n=`7` LF-v1 delta `-0.1058`, Residual delta baseline `1.0383`, Residual delta LF-v1 `1.1441`
- `pattern=lost_lfv1_gain` n=`4` LF-v1 delta `1.2501`, Residual delta baseline `0.0991`, Residual delta LF-v1 `-1.1509`
- `pattern=mitigates_lfv1_regression` n=`1` LF-v1 delta `-3.2380`, Residual delta baseline `-0.6134`, Residual delta LF-v1 `2.6246`
- `pattern=residual_worst` n=`5` LF-v1 delta `-0.7774`, Residual delta baseline `-2.9995`, Residual delta LF-v1 `-2.2221`

## Hard Case Lists

### residual_beats_both

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 573_0.91_1.54.png | residual_beats_both | baseline_middle_50 | 30.2610 | -0.1104 | 2.4452 | 2.5556 | 0.6788 |
| 904_0.82_1.3.png | residual_beats_both | baseline_middle_50 | 32.4190 | -0.7107 | 0.6556 | 1.3664 | 0.4157 |
| 858_0.8_1.72.png | residual_beats_both | baseline_middle_50 | 34.0939 | 0.5750 | 1.7991 | 1.2242 | 0.6824 |
| 1000_0.73_1.8.png | residual_beats_both | baseline_middle_50 | 29.2765 | -0.5735 | 0.6276 | 1.2011 | 0.4168 |
| 147_0.87_1.14.png | residual_beats_both | baseline_weakest_25 | 21.6629 | 0.2026 | 0.9885 | 0.7860 | 0.6907 |
| 28_0.81_1.96.png | residual_beats_both | baseline_middle_50 | 30.7622 | -0.1421 | 0.4119 | 0.5540 | 0.3327 |
| 336_0.94_1.46.png | residual_beats_both | baseline_weakest_25 | 25.0246 | 0.0189 | 0.3404 | 0.3216 | 0.2850 |

### lost_lfv1_gain

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 241_0.85_0.72.png | lost_lfv1_gain | baseline_middle_50 | 28.4126 | 1.1115 | -0.8101 | -1.9216 | -0.1293 |
| 195_0.61_1.47.png | lost_lfv1_gain | baseline_weakest_25 | 26.8060 | 2.1630 | 0.3839 | -1.7791 | 0.2798 |
| 80_0.74_1.76.png | lost_lfv1_gain | baseline_weakest_25 | 19.3312 | 0.9457 | 0.3740 | -0.5717 | 0.3611 |
| 620_0.56_0.87.png | lost_lfv1_gain | baseline_strongest_25 | 39.9391 | 0.7801 | 0.4487 | -0.3314 | 0.5051 |

### mitigates_lfv1_regression

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 384_0.97_0.82.png | mitigates_lfv1_regression | baseline_middle_50 | 34.8945 | -3.2380 | -0.6134 | 2.6246 | 0.1467 |

### residual_worst

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 952_0.97_1.33.png | residual_worst | baseline_middle_50 | 33.2106 | -1.0516 | -4.6958 | -3.6442 | -0.0204 |
| 479_0.79_0.62.png | residual_worst | baseline_middle_50 | 30.9131 | -0.5095 | -3.2338 | -2.7243 | -0.6544 |
| 431_0.86_1.86.png | residual_worst | baseline_weakest_25 | 23.4472 | -0.4256 | -3.0839 | -2.6583 | -0.7405 |
| 9_0.85_1.67.png | residual_worst | baseline_strongest_25 | 35.7146 | -1.9043 | -2.9101 | -1.0058 | 0.0199 |
| 668_0.69_1.25.png | residual_worst | baseline_strongest_25 | 37.5273 | 0.0040 | -1.0741 | -1.0781 | 0.2128 |

### strong_baseline_residual_regressions

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9_0.85_1.67.png | residual_worst | baseline_strongest_25 | 35.7146 | -1.9043 | -2.9101 | -1.0058 | 0.0199 |
| 715_0.63_1.36.png | both_regress_baseline | baseline_strongest_25 | 40.1246 | -1.9517 | -1.7070 | 0.2447 | 0.0634 |
| 668_0.69_1.25.png | residual_worst | baseline_strongest_25 | 37.5273 | 0.0040 | -1.0741 | -1.0781 | 0.2128 |

### weak_baseline_residual_gains

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 147_0.87_1.14.png | residual_beats_both | baseline_weakest_25 | 21.6629 | 0.2026 | 0.9885 | 0.7860 | 0.6907 |
| 195_0.61_1.47.png | lost_lfv1_gain | baseline_weakest_25 | 26.8060 | 2.1630 | 0.3839 | -1.7791 | 0.2798 |
| 80_0.74_1.76.png | lost_lfv1_gain | baseline_weakest_25 | 19.3312 | 0.9457 | 0.3740 | -0.5717 | 0.3611 |
| 336_0.94_1.46.png | residual_beats_both | baseline_weakest_25 | 25.0246 | 0.0189 | 0.3404 | 0.3216 | 0.2850 |

### rescalib_wrong_direction

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 431_0.86_1.86.png | residual_worst | baseline_weakest_25 | 23.4472 | -0.4256 | -3.0839 | -2.6583 | -0.7405 |
| 479_0.79_0.62.png | residual_worst | baseline_middle_50 | 30.9131 | -0.5095 | -3.2338 | -2.7243 | -0.6544 |
| 241_0.85_0.72.png | lost_lfv1_gain | baseline_middle_50 | 28.4126 | 1.1115 | -0.8101 | -1.9216 | -0.1293 |
| 952_0.97_1.33.png | residual_worst | baseline_middle_50 | 33.2106 | -1.0516 | -4.6958 | -3.6442 | -0.0204 |

### largest_color_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 431_0.86_1.86.png | residual_worst | baseline_weakest_25 | 23.4472 | -0.4256 | -3.0839 | -2.6583 | -0.7405 |
| 479_0.79_0.62.png | residual_worst | baseline_middle_50 | 30.9131 | -0.5095 | -3.2338 | -2.7243 | -0.6544 |
| 952_0.97_1.33.png | residual_worst | baseline_middle_50 | 33.2106 | -1.0516 | -4.6958 | -3.6442 | -0.0204 |
| 241_0.85_0.72.png | lost_lfv1_gain | baseline_middle_50 | 28.4126 | 1.1115 | -0.8101 | -1.9216 | -0.1293 |
| 195_0.61_1.47.png | lost_lfv1_gain | baseline_weakest_25 | 26.8060 | 2.1630 | 0.3839 | -1.7791 | 0.2798 |
| 80_0.74_1.76.png | lost_lfv1_gain | baseline_weakest_25 | 19.3312 | 0.9457 | 0.3740 | -0.5717 | 0.3611 |
| 668_0.69_1.25.png | residual_worst | baseline_strongest_25 | 37.5273 | 0.0040 | -1.0741 | -1.0781 | 0.2128 |
| 763_0.82_1.68.png | both_regress_baseline | baseline_middle_50 | 35.6948 | -1.0712 | -1.1556 | -0.0844 | 0.1681 |
| 620_0.56_0.87.png | lost_lfv1_gain | baseline_strongest_25 | 39.9391 | 0.7801 | 0.4487 | -0.3314 | 0.5051 |
| 526_0.73_0.69.png | mixed_or_small | baseline_strongest_25 | 41.2303 | 0.3024 | 0.2325 | -0.0698 | 0.4522 |

### largest_tone_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 431_0.86_1.86.png | residual_worst | baseline_weakest_25 | 23.4472 | -0.4256 | -3.0839 | -2.6583 | -0.7405 |
| 479_0.79_0.62.png | residual_worst | baseline_middle_50 | 30.9131 | -0.5095 | -3.2338 | -2.7243 | -0.6544 |
| 1000_0.73_1.8.png | residual_beats_both | baseline_middle_50 | 29.2765 | -0.5735 | 0.6276 | 1.2011 | 0.4168 |
| 195_0.61_1.47.png | lost_lfv1_gain | baseline_weakest_25 | 26.8060 | 2.1630 | 0.3839 | -1.7791 | 0.2798 |
| 9_0.85_1.67.png | residual_worst | baseline_strongest_25 | 35.7146 | -1.9043 | -2.9101 | -1.0058 | 0.0199 |
| 80_0.74_1.76.png | lost_lfv1_gain | baseline_weakest_25 | 19.3312 | 0.9457 | 0.3740 | -0.5717 | 0.3611 |
| 952_0.97_1.33.png | residual_worst | baseline_middle_50 | 33.2106 | -1.0516 | -4.6958 | -3.6442 | -0.0204 |
| 241_0.85_0.72.png | lost_lfv1_gain | baseline_middle_50 | 28.4126 | 1.1115 | -0.8101 | -1.9216 | -0.1293 |
| 668_0.69_1.25.png | residual_worst | baseline_strongest_25 | 37.5273 | 0.0040 | -1.0741 | -1.0781 | 0.2128 |
| 620_0.56_0.87.png | lost_lfv1_gain | baseline_strongest_25 | 39.9391 | 0.7801 | 0.4487 | -0.3314 | 0.5051 |

### largest_dark_channel_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 431_0.86_1.86.png | residual_worst | baseline_weakest_25 | 23.4472 | -0.4256 | -3.0839 | -2.6583 | -0.7405 |
| 479_0.79_0.62.png | residual_worst | baseline_middle_50 | 30.9131 | -0.5095 | -3.2338 | -2.7243 | -0.6544 |
| 1000_0.73_1.8.png | residual_beats_both | baseline_middle_50 | 29.2765 | -0.5735 | 0.6276 | 1.2011 | 0.4168 |
| 195_0.61_1.47.png | lost_lfv1_gain | baseline_weakest_25 | 26.8060 | 2.1630 | 0.3839 | -1.7791 | 0.2798 |
| 952_0.97_1.33.png | residual_worst | baseline_middle_50 | 33.2106 | -1.0516 | -4.6958 | -3.6442 | -0.0204 |
| 80_0.74_1.76.png | lost_lfv1_gain | baseline_weakest_25 | 19.3312 | 0.9457 | 0.3740 | -0.5717 | 0.3611 |
| 241_0.85_0.72.png | lost_lfv1_gain | baseline_middle_50 | 28.4126 | 1.1115 | -0.8101 | -1.9216 | -0.1293 |
| 668_0.69_1.25.png | residual_worst | baseline_strongest_25 | 37.5273 | 0.0040 | -1.0741 | -1.0781 | 0.2128 |
| 715_0.63_1.36.png | both_regress_baseline | baseline_strongest_25 | 40.1246 | -1.9517 | -1.7070 | 0.2447 | 0.0634 |
| 526_0.73_0.69.png | mixed_or_small | baseline_strongest_25 | 41.2303 | 0.3024 | 0.2325 | -0.0698 | 0.4522 |

## Reading Notes

- `residual_delta_lfv1_psnr > 0` means ResidualCalib beats LF-v1 on that image.
- `lost_lfv1_gain` means LF-v1 has a meaningful gain over baseline but ResidualCalib gives a meaningful loss relative to LF-v1.
- `mitigates_lfv1_regression` means LF-v1 hurts baseline and ResidualCalib recovers at least part of that loss.
- Heatmaps use green where the later model reduces absolute error and red where it increases absolute error.
