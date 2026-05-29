# Three-Way HAZE4K Output Analysis

## Summary

- Images: `1000`
- Baseline PSNR/SSIM: `32.2253` / `0.984417`
- LF-v1 PSNR/SSIM: `32.4283` / `0.984454`
- ResidualCalib PSNR/SSIM: `32.3936` / `0.984500`
- LF-v1 vs baseline mean delta: `0.2030` dB
- ResidualCalib vs baseline mean delta: `0.1682` dB
- ResidualCalib vs LF-v1 mean delta: `-0.0347` dB
- PSNR winner counts: `{'baseline': 295, 'lfv1': 322, 'residual': 383}`
- Pattern counts: `{'both_improve_baseline': 82, 'both_regress_baseline': 66, 'lost_lfv1_gain': 218, 'mitigates_lfv1_regression': 110, 'mixed_or_small': 59, 'residual_beats_both': 300, 'residual_worst': 165}`
- ResidualCalib wrong-direction vs baseline: `163`
- LF-v1 wrong-direction vs baseline: `160`

## Key Group Rows

- `baseline_strength_bin=baseline_weakest_25` n=`250` LF-v1 delta `0.4921`, Residual delta baseline `0.6354`, Residual delta LF-v1 `0.1433`
- `baseline_strength_bin=baseline_middle_50` n=`500` LF-v1 delta `0.1859`, Residual delta baseline `0.0773`, Residual delta LF-v1 `-0.1086`
- `baseline_strength_bin=baseline_strongest_25` n=`250` LF-v1 delta `-0.0520`, Residual delta baseline `-0.1171`, Residual delta LF-v1 `-0.0651`
- `pattern=residual_beats_both` n=`300` LF-v1 delta `0.4225`, Residual delta baseline `1.8307`, Residual delta LF-v1 `1.4082`
- `pattern=lost_lfv1_gain` n=`218` LF-v1 delta `1.5558`, Residual delta baseline `-0.1031`, Residual delta LF-v1 `-1.6589`
- `pattern=mitigates_lfv1_regression` n=`110` LF-v1 delta `-1.6903`, Residual delta baseline `-0.4550`, Residual delta LF-v1 `1.2354`
- `pattern=residual_worst` n=`165` LF-v1 delta `-0.7235`, Residual delta baseline `-2.1215`, Residual delta LF-v1 `-1.3979`

## Hard Case Lists

### residual_beats_both

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 510_0.71_1.55.png | residual_beats_both | baseline_middle_50 | 31.0928 | -3.0118 | 3.0266 | 6.0384 | 0.7430 |
| 401_0.7_0.6.png | residual_beats_both | baseline_middle_50 | 29.7915 | 4.1779 | 9.6409 | 5.4630 | 0.9598 |
| 553_0.51_1.76.png | residual_beats_both | baseline_middle_50 | 34.1980 | -1.6502 | 3.7154 | 5.3656 | 0.8392 |
| 960_0.78_0.97.png | residual_beats_both | baseline_strongest_25 | 36.5167 | -2.7994 | 2.2631 | 5.0625 | 0.6895 |
| 958_0.9_0.64.png | residual_beats_both | baseline_strongest_25 | 36.9363 | -3.5949 | 1.1683 | 4.7632 | 0.5369 |
| 556_0.61_0.91.png | residual_beats_both | baseline_strongest_25 | 40.9971 | -2.1211 | 2.1384 | 4.2595 | 0.7731 |
| 402_0.5_1.99.png | residual_beats_both | baseline_weakest_25 | 17.6416 | 5.5931 | 9.7377 | 4.1446 | 0.9862 |
| 51_0.6_1.79.png | residual_beats_both | baseline_middle_50 | 29.8018 | -1.4071 | 2.7030 | 4.1101 | 0.7207 |
| 785_0.54_0.53.png | residual_beats_both | baseline_strongest_25 | 38.1035 | -2.0598 | 2.0130 | 4.0728 | 0.6600 |
| 497_0.7_0.61.png | residual_beats_both | baseline_middle_50 | 32.4266 | -3.0967 | 0.7493 | 3.8461 | 0.4071 |

### lost_lfv1_gain

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 412_0.52_0.64.png | lost_lfv1_gain | baseline_middle_50 | 30.3727 | 6.1472 | -2.8325 | -8.9797 | -0.7711 |
| 411_0.76_1.73.png | lost_lfv1_gain | baseline_weakest_25 | 24.1999 | 5.3276 | -2.1555 | -7.4831 | -0.7395 |
| 343_0.83_1.07.png | lost_lfv1_gain | baseline_middle_50 | 30.0833 | 3.2963 | -4.0295 | -7.3258 | -0.7134 |
| 344_0.82_1.51.png | lost_lfv1_gain | baseline_weakest_25 | 28.4856 | 1.0483 | -4.5611 | -5.6094 | -0.6976 |
| 150_0.6_1.29.png | lost_lfv1_gain | baseline_middle_50 | 31.9685 | 3.7568 | -1.1705 | -4.9273 | 0.2845 |
| 410_0.98_1.98.png | lost_lfv1_gain | baseline_weakest_25 | 21.8971 | 3.1634 | -1.7062 | -4.8696 | -0.6390 |
| 972_0.99_0.62.png | lost_lfv1_gain | baseline_middle_50 | 31.2650 | 1.3624 | -3.4613 | -4.8237 | -0.2684 |
| 253_0.53_1.99.png | lost_lfv1_gain | baseline_middle_50 | 31.1950 | 0.6943 | -4.1026 | -4.7969 | -0.2385 |
| 415_0.56_1.96.png | lost_lfv1_gain | baseline_weakest_25 | 28.5258 | 4.6218 | -0.1429 | -4.7647 | 0.4432 |
| 342_0.96_1.08.png | lost_lfv1_gain | baseline_middle_50 | 29.1585 | 2.1401 | -2.5819 | -4.7220 | -0.5753 |

### mitigates_lfv1_regression

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 49_0.7_1.47.png | mitigates_lfv1_regression | baseline_middle_50 | 35.3095 | -4.4989 | -0.0547 | 4.4441 | 0.4360 |
| 381_0.63_1.38.png | mitigates_lfv1_regression | baseline_middle_50 | 33.6004 | -3.9622 | 0.1378 | 4.1000 | 0.2631 |
| 107_0.89_1.5.png | mitigates_lfv1_regression | baseline_weakest_25 | 27.8776 | -4.0195 | -0.3298 | 3.6897 | -0.0615 |
| 167_0.53_1.34.png | mitigates_lfv1_regression | baseline_middle_50 | 32.5293 | -4.3617 | -0.7890 | 3.5728 | 0.1149 |
| 382_0.53_1.12.png | mitigates_lfv1_regression | baseline_middle_50 | 31.9895 | -3.3871 | -0.5849 | 2.8022 | -0.1212 |
| 383_0.62_1.77.png | mitigates_lfv1_regression | baseline_middle_50 | 32.3498 | -3.4698 | -0.6919 | 2.7779 | -0.0584 |
| 511_0.79_0.83.png | mitigates_lfv1_regression | baseline_middle_50 | 34.9927 | -2.9150 | -0.2090 | 2.7060 | 0.3598 |
| 384_0.97_0.82.png | mitigates_lfv1_regression | baseline_middle_50 | 34.8945 | -3.2380 | -0.6134 | 2.6246 | 0.1467 |
| 512_0.94_1.48.png | mitigates_lfv1_regression | baseline_weakest_25 | 27.7310 | -2.4388 | 0.0535 | 2.4922 | 0.2391 |
| 222_0.76_0.62.png | mitigates_lfv1_regression | baseline_middle_50 | 33.4130 | -2.2510 | 0.2026 | 2.4537 | 0.3629 |

### residual_worst

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 219_0.62_1.27.png | residual_worst | baseline_middle_50 | 34.0196 | -1.7096 | -8.8306 | -7.1211 | -0.4642 |
| 390_0.55_1.07.png | residual_worst | baseline_weakest_25 | 26.8931 | -4.9448 | -8.0063 | -3.0614 | -0.7622 |
| 220_0.68_1.35.png | residual_worst | baseline_middle_50 | 34.9663 | -0.5207 | -7.5728 | -7.0521 | 0.4518 |
| 886_0.51_0.87.png | residual_worst | baseline_strongest_25 | 39.9371 | -0.9391 | -6.9119 | -5.9728 | -0.3349 |
| 88_0.63_0.94.png | residual_worst | baseline_middle_50 | 32.8294 | -4.0783 | -6.1948 | -2.1165 | -0.0826 |
| 218_0.53_1.18.png | residual_worst | baseline_weakest_25 | 28.0497 | -0.1971 | -6.0702 | -5.8732 | -0.8570 |
| 127_0.57_0.92.png | residual_worst | baseline_strongest_25 | 36.1922 | -4.0114 | -6.0580 | -2.0466 | -0.3399 |
| 191_0.86_1.88.png | residual_worst | baseline_middle_50 | 34.6567 | -3.3691 | -5.7245 | -2.3553 | 0.0471 |
| 949_0.8_0.6.png | residual_worst | baseline_strongest_25 | 40.9401 | -0.2645 | -5.1485 | -4.8840 | 0.3009 |
| 316_0.91_0.85.png | residual_worst | baseline_middle_50 | 33.0416 | -0.6944 | -5.0792 | -4.3848 | -0.5540 |

### strong_baseline_residual_regressions

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 886_0.51_0.87.png | residual_worst | baseline_strongest_25 | 39.9371 | -0.9391 | -6.9119 | -5.9728 | -0.3349 |
| 127_0.57_0.92.png | residual_worst | baseline_strongest_25 | 36.1922 | -4.0114 | -6.0580 | -2.0466 | -0.3399 |
| 949_0.8_0.6.png | residual_worst | baseline_strongest_25 | 40.9401 | -0.2645 | -5.1485 | -4.8840 | 0.3009 |
| 128_0.65_0.67.png | residual_worst | baseline_strongest_25 | 37.0246 | -2.3244 | -4.8527 | -2.5283 | -0.2909 |
| 946_0.65_0.58.png | mitigates_lfv1_regression | baseline_strongest_25 | 41.9177 | -6.2238 | -3.9348 | 2.2890 | -0.3047 |
| 887_0.61_1.36.png | residual_worst | baseline_strongest_25 | 36.5838 | 0.0679 | -3.7083 | -3.7762 | 0.3846 |
| 300_0.92_0.72.png | residual_worst | baseline_strongest_25 | 36.7709 | -1.5366 | -3.6763 | -2.1396 | -0.0041 |
| 951_0.57_0.85.png | both_regress_baseline | baseline_strongest_25 | 39.3161 | -3.1863 | -3.4809 | -0.2947 | -0.0730 |
| 287_0.77_1.14.png | residual_worst | baseline_strongest_25 | 36.2377 | -2.9732 | -3.3099 | -0.3367 | -0.1855 |
| 950_0.57_1.28.png | mitigates_lfv1_regression | baseline_strongest_25 | 37.1064 | -3.8846 | -3.0100 | 0.8745 | -0.1752 |

### weak_baseline_residual_gains

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 402_0.5_1.99.png | residual_beats_both | baseline_weakest_25 | 17.6416 | 5.5931 | 9.7377 | 4.1446 | 0.9862 |
| 800_0.55_1.84.png | residual_beats_both | baseline_weakest_25 | 28.8168 | 5.8964 | 7.5817 | 1.6853 | 0.9490 |
| 422_0.65_0.92.png | residual_beats_both | baseline_weakest_25 | 28.0346 | 3.6300 | 6.5840 | 2.9540 | 0.9438 |
| 799_0.52_1.97.png | residual_beats_both | baseline_weakest_25 | 27.2150 | 4.4193 | 6.5343 | 2.1149 | 0.9484 |
| 67_0.68_1.12.png | lost_lfv1_gain | baseline_weakest_25 | 26.1742 | 6.4500 | 5.2715 | -1.1785 | 0.8569 |
| 404_0.71_1.78.png | both_improve_baseline | baseline_weakest_25 | 27.3415 | 4.2127 | 4.4891 | 0.2764 | 0.9204 |
| 178_0.64_1.88.png | residual_beats_both | baseline_weakest_25 | 22.5965 | 3.1769 | 4.0992 | 0.9223 | 0.8576 |
| 585_0.75_1.97.png | residual_beats_both | baseline_weakest_25 | 28.4349 | 1.4174 | 4.0238 | 2.6063 | 0.8381 |
| 278_0.98_1.19.png | residual_beats_both | baseline_weakest_25 | 27.3436 | 0.9236 | 3.9094 | 2.9858 | 0.8754 |
| 587_0.87_1.84.png | residual_beats_both | baseline_weakest_25 | 26.7499 | 2.5772 | 3.8582 | 1.2810 | 0.8392 |

### rescalib_wrong_direction

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 218_0.53_1.18.png | residual_worst | baseline_weakest_25 | 28.0497 | -0.1971 | -6.0702 | -5.8732 | -0.8570 |
| 75_0.99_0.83.png | residual_worst | baseline_weakest_25 | 26.6793 | -0.1863 | -2.5600 | -2.3737 | -0.7747 |
| 412_0.52_0.64.png | lost_lfv1_gain | baseline_middle_50 | 30.3727 | 6.1472 | -2.8325 | -8.9797 | -0.7711 |
| 390_0.55_1.07.png | residual_worst | baseline_weakest_25 | 26.8931 | -4.9448 | -8.0063 | -3.0614 | -0.7622 |
| 431_0.86_1.86.png | residual_worst | baseline_weakest_25 | 23.4472 | -0.4256 | -3.0839 | -2.6583 | -0.7405 |
| 411_0.76_1.73.png | lost_lfv1_gain | baseline_weakest_25 | 24.1999 | 5.3276 | -2.1555 | -7.4831 | -0.7395 |
| 480_0.54_0.98.png | residual_worst | baseline_weakest_25 | 26.7938 | -1.6084 | -3.1380 | -1.5296 | -0.7156 |
| 343_0.83_1.07.png | lost_lfv1_gain | baseline_middle_50 | 30.0833 | 3.2963 | -4.0295 | -7.3258 | -0.7134 |
| 344_0.82_1.51.png | lost_lfv1_gain | baseline_weakest_25 | 28.4856 | 1.0483 | -4.5611 | -5.6094 | -0.6976 |
| 429_0.52_1.11.png | residual_worst | baseline_middle_50 | 28.9934 | -3.5449 | -4.9729 | -1.4280 | -0.6909 |

### largest_color_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 411_0.76_1.73.png | lost_lfv1_gain | baseline_weakest_25 | 24.1999 | 5.3276 | -2.1555 | -7.4831 | -0.7395 |
| 218_0.53_1.18.png | residual_worst | baseline_weakest_25 | 28.0497 | -0.1971 | -6.0702 | -5.8732 | -0.8570 |
| 410_0.98_1.98.png | lost_lfv1_gain | baseline_weakest_25 | 21.8971 | 3.1634 | -1.7062 | -4.8696 | -0.6390 |
| 390_0.55_1.07.png | residual_worst | baseline_weakest_25 | 26.8931 | -4.9448 | -8.0063 | -3.0614 | -0.7622 |
| 343_0.83_1.07.png | lost_lfv1_gain | baseline_middle_50 | 30.0833 | 3.2963 | -4.0295 | -7.3258 | -0.7134 |
| 219_0.62_1.27.png | residual_worst | baseline_middle_50 | 34.0196 | -1.7096 | -8.8306 | -7.1211 | -0.4642 |
| 409_0.54_1.68.png | lost_lfv1_gain | baseline_weakest_25 | 24.0846 | 2.7619 | -0.9307 | -3.6926 | -0.4844 |
| 412_0.52_0.64.png | lost_lfv1_gain | baseline_middle_50 | 30.3727 | 6.1472 | -2.8325 | -8.9797 | -0.7711 |
| 344_0.82_1.51.png | lost_lfv1_gain | baseline_weakest_25 | 28.4856 | 1.0483 | -4.5611 | -5.6094 | -0.6976 |
| 220_0.68_1.35.png | residual_worst | baseline_middle_50 | 34.9663 | -0.5207 | -7.5728 | -7.0521 | 0.4518 |

### largest_tone_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 411_0.76_1.73.png | lost_lfv1_gain | baseline_weakest_25 | 24.1999 | 5.3276 | -2.1555 | -7.4831 | -0.7395 |
| 410_0.98_1.98.png | lost_lfv1_gain | baseline_weakest_25 | 21.8971 | 3.1634 | -1.7062 | -4.8696 | -0.6390 |
| 218_0.53_1.18.png | residual_worst | baseline_weakest_25 | 28.0497 | -0.1971 | -6.0702 | -5.8732 | -0.8570 |
| 390_0.55_1.07.png | residual_worst | baseline_weakest_25 | 26.8931 | -4.9448 | -8.0063 | -3.0614 | -0.7622 |
| 412_0.52_0.64.png | lost_lfv1_gain | baseline_middle_50 | 30.3727 | 6.1472 | -2.8325 | -8.9797 | -0.7711 |
| 415_0.56_1.96.png | lost_lfv1_gain | baseline_weakest_25 | 28.5258 | 4.6218 | -0.1429 | -4.7647 | 0.4432 |
| 219_0.62_1.27.png | residual_worst | baseline_middle_50 | 34.0196 | -1.7096 | -8.8306 | -7.1211 | -0.4642 |
| 344_0.82_1.51.png | lost_lfv1_gain | baseline_weakest_25 | 28.4856 | 1.0483 | -4.5611 | -5.6094 | -0.6976 |
| 409_0.54_1.68.png | lost_lfv1_gain | baseline_weakest_25 | 24.0846 | 2.7619 | -0.9307 | -3.6926 | -0.4844 |
| 430_0.66_1.24.png | lost_lfv1_gain | baseline_weakest_25 | 25.9784 | 3.2560 | -0.4730 | -3.7290 | -0.0182 |

### largest_dark_channel_regression_vs_lfv1

| filename | pattern | baseline_strength_bin | baseline_psnr | lfv1_delta_baseline_psnr | residual_delta_baseline_psnr | residual_delta_lfv1_psnr | rescalib_from_baseline_residual_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 411_0.76_1.73.png | lost_lfv1_gain | baseline_weakest_25 | 24.1999 | 5.3276 | -2.1555 | -7.4831 | -0.7395 |
| 218_0.53_1.18.png | residual_worst | baseline_weakest_25 | 28.0497 | -0.1971 | -6.0702 | -5.8732 | -0.8570 |
| 410_0.98_1.98.png | lost_lfv1_gain | baseline_weakest_25 | 21.8971 | 3.1634 | -1.7062 | -4.8696 | -0.6390 |
| 430_0.66_1.24.png | lost_lfv1_gain | baseline_weakest_25 | 25.9784 | 3.2560 | -0.4730 | -3.7290 | -0.0182 |
| 343_0.83_1.07.png | lost_lfv1_gain | baseline_middle_50 | 30.0833 | 3.2963 | -4.0295 | -7.3258 | -0.7134 |
| 344_0.82_1.51.png | lost_lfv1_gain | baseline_weakest_25 | 28.4856 | 1.0483 | -4.5611 | -5.6094 | -0.6976 |
| 412_0.52_0.64.png | lost_lfv1_gain | baseline_middle_50 | 30.3727 | 6.1472 | -2.8325 | -8.9797 | -0.7711 |
| 429_0.52_1.11.png | residual_worst | baseline_middle_50 | 28.9934 | -3.5449 | -4.9729 | -1.4280 | -0.6909 |
| 219_0.62_1.27.png | residual_worst | baseline_middle_50 | 34.0196 | -1.7096 | -8.8306 | -7.1211 | -0.4642 |
| 137_0.99_1.98.png | lost_lfv1_gain | baseline_middle_50 | 29.3827 | 0.7157 | -2.5948 | -3.3105 | -0.2739 |

## Reading Notes

- `residual_delta_lfv1_psnr > 0` means ResidualCalib beats LF-v1 on that image.
- `lost_lfv1_gain` means LF-v1 has a meaningful gain over baseline but ResidualCalib gives a meaningful loss relative to LF-v1.
- `mitigates_lfv1_regression` means LF-v1 hurts baseline and ResidualCalib recovers at least part of that loss.
- Heatmaps use green where the later model reduces absolute error and red where it increases absolute error.
