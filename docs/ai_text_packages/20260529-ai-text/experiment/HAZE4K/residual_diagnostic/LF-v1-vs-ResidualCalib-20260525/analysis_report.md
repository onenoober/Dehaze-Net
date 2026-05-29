# LF Residual Direction Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `LF-v1` step `90000`
- Current: `ResidualCalib` step `90000`
- Low-frequency pool: `8`
- Mean delta PSNR: `-0.0347`
- Better / worse at 0.30 dB: `414` / `386`
- Mean residual cosine: `0.2592`
- Mean luma residual cosine: `0.2613`
- Mean residual norm ratio: `0.5804`
- Mean residual error ratio: `1.0259`
- Low-frequency MSE improved / regressed: `512` / `488`
- Luma low-frequency MSE improved / regressed: `509` / `491`
- Wrong-direction count (cosine < 0): `211`
- Possible overshoot / under-correction counts: `0` / `121`
- Corr(delta PSNR, residual cosine): `0.858046094629602`
- Corr(delta PSNR, residual norm ratio): `-0.4071914368470715`

## Baseline Strength Groups

- `baseline_middle_50` n=`500` delta `-0.1492`, cos `0.2416`, norm_ratio `0.6063`, LF MSE improved/regressed `250/250`
- `baseline_strongest_25` n=`250` delta `-0.2374`, cos `0.3096`, norm_ratio `0.7530`, LF MSE improved/regressed `112/138`
- `baseline_weakest_25` n=`250` delta `0.3970`, cos `0.2441`, norm_ratio `0.3559`, LF MSE improved/regressed `150/100`

## Reading Guide

- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.
- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.
- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.
- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.

## Hard Cases

### Worst Delta PSNR

- `412_0.52_0.64.png` delta `-8.9797`, cos `-0.4565`, norm_ratio `2.3195`, LF MSE delta `0.00152916`
- `411_0.76_1.73.png` delta `-7.4831`, cos `-0.8863`, norm_ratio `1.4617`, LF MSE delta `0.00511765`
- `343_0.83_1.07.png` delta `-7.3258`, cos `-0.7469`, norm_ratio `1.6378`, LF MSE delta `0.00201416`
- `219_0.62_1.27.png` delta `-7.1211`, cos `-0.6998`, norm_ratio `1.4887`, LF MSE delta `0.00243387`
- `220_0.68_1.35.png` delta `-7.0521`, cos `-0.1644`, norm_ratio `1.9265`, LF MSE delta `0.00145559`
- `886_0.51_0.87.png` delta `-5.9728`, cos `-0.4135`, norm_ratio `1.4817`, LF MSE delta `0.00036127`
- `218_0.53_1.18.png` delta `-5.8732`, cos `-0.8698`, norm_ratio `1.0419`, LF MSE delta `0.00469060`
- `344_0.82_1.51.png` delta `-5.6094`, cos `-0.7678`, norm_ratio `1.0925`, LF MSE delta `0.00293248`
- `150_0.6_1.29.png` delta `-4.9273`, cos `-0.2836`, norm_ratio `1.2996`, LF MSE delta `0.00050579`
- `949_0.8_0.6.png` delta `-4.8840`, cos `-0.0729`, norm_ratio `1.5057`, LF MSE delta `0.00017430`

### Worst Residual Cosine

- `390_0.55_1.07.png` cos `-0.9055`, delta `-3.0614`, norm_ratio `0.4508`
- `410_0.98_1.98.png` cos `-0.8950`, delta `-4.8696`, norm_ratio `0.8133`
- `411_0.76_1.73.png` cos `-0.8863`, delta `-7.4831`, norm_ratio `1.4617`
- `75_0.99_0.83.png` cos `-0.8860`, delta `-2.3737`, norm_ratio `0.3517`
- `218_0.53_1.18.png` cos `-0.8698`, delta `-5.8732`, norm_ratio `1.0419`
- `76_0.64_1.74.png` cos `-0.8681`, delta `-1.7973`, norm_ratio `0.2600`
- `409_0.54_1.68.png` cos `-0.8494`, delta `-3.6926`, norm_ratio `0.5929`
- `74_0.62_1.6.png` cos `-0.8484`, delta `-1.9048`, norm_ratio `0.2815`
- `362_0.82_0.88.png` cos `-0.8484`, delta `-1.7515`, norm_ratio `0.2597`
- `431_0.86_1.86.png` cos `-0.7914`, delta `-2.6583`, norm_ratio `0.4258`

### Largest Low-Frequency MSE Regressions

- `443_0.56_1.04.png` LF MSE delta `0.00905772`, delta `-0.7385`, cos `-0.7781`
- `390_0.55_1.07.png` LF MSE delta `0.00645571`, delta `-3.0614`, cos `-0.9055`
- `410_0.98_1.98.png` LF MSE delta `0.00643830`, delta `-4.8696`, cos `-0.8950`
- `411_0.76_1.73.png` LF MSE delta `0.00511765`, delta `-7.4831`, cos `-0.8863`
- `218_0.53_1.18.png` LF MSE delta `0.00469060`, delta `-5.8732`, cos `-0.8698`
- `431_0.86_1.86.png` LF MSE delta `0.00420639`, delta `-2.6583`, cos `-0.7914`
- `344_0.82_1.51.png` LF MSE delta `0.00293248`, delta `-5.6094`, cos `-0.7678`
- `409_0.54_1.68.png` LF MSE delta `0.00276303`, delta `-3.6926`, cos `-0.8494`
- `219_0.62_1.27.png` LF MSE delta `0.00243387`, delta `-7.1211`, cos `-0.6998`
- `368_0.79_1.95.png` LF MSE delta `0.00239406`, delta `-2.3412`, cos `-0.4171`
