# Three-Way HAZE4K Output Analysis

## Summary

- Images: `2`
- Baseline PSNR/SSIM: `30.0200` / `0.987972`
- LF-v1 PSNR/SSIM: `30.8540` / `0.987316`
- ResidualCalib PSNR/SSIM: `29.8126` / `0.987980`
- LF-v1 vs baseline mean delta: `0.8340` dB
- ResidualCalib vs baseline mean delta: `-0.2074` dB
- ResidualCalib vs LF-v1 mean delta: `-1.0414` dB
- PSNR winner counts: `{'lfv1': 1, 'residual': 1}`
- Pattern counts: `{'lost_lfv1_gain': 1, 'residual_beats_both': 1}`
- ResidualCalib wrong-direction vs baseline: `1`
- LF-v1 wrong-direction vs baseline: `1`

## Key Group Rows

- `baseline_strength_bin=baseline_weakest_25` n=`1` LF-v1 delta `-0.5735`, Residual delta baseline `0.6276`, Residual delta LF-v1 `1.2011`
- `baseline_strength_bin=baseline_strongest_25` n=`1` LF-v1 delta `2.2415`, Residual delta baseline `-1.0424`, Residual delta LF-v1 `-3.2839`
- `pattern=residual_beats_both` n=`1` LF-v1 delta `-0.5735`, Residual delta baseline `0.6276`, Residual delta LF-v1 `1.2011`
- `pattern=lost_lfv1_gain` n=`1` LF-v1 delta `2.2415`, Residual delta baseline `-1.0424`, Residual delta LF-v1 `-3.2839`

## Hard Case Lists

### residual_beats_both

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000_0.73_1.8.png | residual_beats_both | baseline_weakest_25 | 29.2765 | -0.5735 | 0.6276 | 1.2011 | 0.4168 |

### lost_lfv1_gain

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100_0.5_1.76.png | lost_lfv1_gain | baseline_strongest_25 | 30.7635 | 2.2415 | -1.0424 | -3.2839 | -0.0797 |

### mitigates_lfv1_regression

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |

### residual_worst

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |

### strong_baseline_residual_regressions

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100_0.5_1.76.png | lost_lfv1_gain | baseline_strongest_25 | 30.7635 | 2.2415 | -1.0424 | -3.2839 | -0.0797 |

### weak_baseline_residual_gains

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000_0.73_1.8.png | residual_beats_both | baseline_weakest_25 | 29.2765 | -0.5735 | 0.6276 | 1.2011 | 0.4168 |

### rescalib_wrong_direction

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100_0.5_1.76.png | lost_lfv1_gain | baseline_strongest_25 | 30.7635 | 2.2415 | -1.0424 | -3.2839 | -0.0797 |

### largest_color_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100_0.5_1.76.png | lost_lfv1_gain | baseline_strongest_25 | 30.7635 | 2.2415 | -1.0424 | -3.2839 | -0.0797 |
| 1000_0.73_1.8.png | residual_beats_both | baseline_weakest_25 | 29.2765 | -0.5735 | 0.6276 | 1.2011 | 0.4168 |

### largest_tone_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000_0.73_1.8.png | residual_beats_both | baseline_weakest_25 | 29.2765 | -0.5735 | 0.6276 | 1.2011 | 0.4168 |
| 100_0.5_1.76.png | lost_lfv1_gain | baseline_strongest_25 | 30.7635 | 2.2415 | -1.0424 | -3.2839 | -0.0797 |

### largest_dark_channel_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000_0.73_1.8.png | residual_beats_both | baseline_weakest_25 | 29.2765 | -0.5735 | 0.6276 | 1.2011 | 0.4168 |
| 100_0.5_1.76.png | lost_lfv1_gain | baseline_strongest_25 | 30.7635 | 2.2415 | -1.0424 | -3.2839 | -0.0797 |

## Reading Notes

- `residual_delta_lfv1_psnr > 0` means ResidualCalib beats LF-v1 on that image.
- `lost_lfv1_gain` means LF-v1 has a meaningful gain over baseline but ResidualCalib gives a meaningful loss relative to LF-v1.
- `mitigates_lfv1_regression` means LF-v1 hurts baseline and ResidualCalib recovers at least part of that loss.
- Heatmaps use green where the later model reduces absolute error and red where it increases absolute error.
