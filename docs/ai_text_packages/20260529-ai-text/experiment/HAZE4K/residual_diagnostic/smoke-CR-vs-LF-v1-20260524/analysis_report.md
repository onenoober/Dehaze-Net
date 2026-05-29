# LF Residual Direction Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `2`
- Baseline: `DEA-Net-CR` step `90000`
- Current: `DEA-Net-LF-v1` step `90000`
- Low-frequency pool: `8`
- Mean delta PSNR: `0.8340`
- Better / worse at 0.30 dB: `1` / `1`
- Mean residual cosine: `0.3324`
- Mean luma residual cosine: `0.3235`
- Mean residual norm ratio: `0.4967`
- Mean residual error ratio: `0.9077`
- Low-frequency MSE improved / regressed: `1` / `1`
- Luma low-frequency MSE improved / regressed: `1` / `1`
- Wrong-direction count (cosine < 0): `1`
- Possible overshoot / under-correction counts: `0` / `0`
- Corr(delta PSNR, residual cosine): `1.0`
- Corr(delta PSNR, residual norm ratio): `0.9999999999999999`

## Baseline Strength Groups

- `baseline_strongest_25` n=`1` delta `2.2415`, cos `0.6691`, norm_ratio `0.6161`, LF MSE improved/regressed `1/0`
- `baseline_weakest_25` n=`1` delta `-0.5735`, cos `-0.0043`, norm_ratio `0.3773`, LF MSE improved/regressed `0/1`

## Reading Guide

- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.
- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.
- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.
- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.

## Hard Cases

### Worst Delta PSNR

- `1000_0.73_1.8.png` delta `-0.5735`, cos `-0.0043`, norm_ratio `0.3773`, LF MSE delta `0.00016162`
- `100_0.5_1.76.png` delta `2.2415`, cos `0.6691`, norm_ratio `0.6161`, LF MSE delta `-0.00033078`

### Worst Residual Cosine

- `1000_0.73_1.8.png` cos `-0.0043`, delta `-0.5735`, norm_ratio `0.3773`
- `100_0.5_1.76.png` cos `0.6691`, delta `2.2415`, norm_ratio `0.6161`

### Largest Low-Frequency MSE Regressions

- `1000_0.73_1.8.png` LF MSE delta `0.00016162`, delta `-0.5735`, cos `-0.0043`
- `100_0.5_1.76.png` LF MSE delta `-0.00033078`, delta `2.2415`, cos `0.6691`
