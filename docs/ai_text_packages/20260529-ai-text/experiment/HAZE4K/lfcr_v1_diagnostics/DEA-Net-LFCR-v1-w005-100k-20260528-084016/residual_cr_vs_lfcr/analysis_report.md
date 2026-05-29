# LF Residual Direction Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CR` step `90000`
- Current: `LFCR-v1-w005` step `100000`
- Low-frequency pool: `8`
- Mean delta PSNR: `-0.0148`
- Better / worse at 0.30 dB: `414` / `410`
- Mean residual cosine: `0.2584`
- Mean luma residual cosine: `0.2598`
- Mean residual norm ratio: `0.5760`
- Mean residual error ratio: `1.0232`
- Low-frequency MSE improved / regressed: `508` / `492`
- Luma low-frequency MSE improved / regressed: `507` / `493`
- Wrong-direction count (cosine < 0): `218`
- Possible overshoot / under-correction counts: `1` / `106`
- Corr(delta PSNR, residual cosine): `0.8329816276517086`
- Corr(delta PSNR, residual norm ratio): `-0.3705855095527215`

## Baseline Strength Groups

- `baseline_middle_50` n=`500` delta `-0.0469`, cos `0.2579`, norm_ratio `0.6096`, LF MSE improved/regressed `248/252`
- `baseline_strongest_25` n=`250` delta `-0.1789`, cos `0.3248`, norm_ratio `0.7534`, LF MSE improved/regressed `120/130`
- `baseline_weakest_25` n=`250` delta `0.2136`, cos `0.1929`, norm_ratio `0.3315`, LF MSE improved/regressed `140/110`

## Reading Guide

- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.
- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.
- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.
- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.

## Hard Cases

### Worst Delta PSNR

- `253_0.53_1.99.png` delta `-11.2479`, cos `-0.4189`, norm_ratio `3.4187`, LF MSE delta `0.00931823`
- `255_0.53_1.86.png` delta `-10.9301`, cos `-0.2530`, norm_ratio `3.5062`, LF MSE delta `0.00613277`
- `256_0.51_1.17.png` delta `-5.6627`, cos `-0.5491`, norm_ratio `1.2747`, LF MSE delta `0.00122475`
- `88_0.63_0.94.png` delta `-5.4605`, cos `-0.2162`, norm_ratio `1.4819`, LF MSE delta `0.00128378`
- `246_0.77_1.86.png` delta `-4.4560`, cos `-0.1732`, norm_ratio `1.3440`, LF MSE delta `0.00094742`
- `44_0.73_0.53.png` delta `-4.4454`, cos `-0.2375`, norm_ratio `1.1800`, LF MSE delta `0.00025406`
- `86_0.65_0.51.png` delta `-4.3388`, cos `-0.0415`, norm_ratio `1.3559`, LF MSE delta `0.00080981`
- `339_0.64_1.47.png` delta `-4.2439`, cos `-0.3871`, norm_ratio `0.9604`, LF MSE delta `0.00079392`
- `390_0.55_1.07.png` delta `-4.1487`, cos `-0.6360`, norm_ratio `0.7998`, LF MSE delta `0.00328480`
- `127_0.57_0.92.png` delta `-4.0587`, cos `-0.2883`, norm_ratio `1.1217`, LF MSE delta `0.00036065`

### Worst Residual Cosine

- `73_0.54_1.5.png` cos `-0.8315`, delta `-2.0098`, norm_ratio `0.3026`
- `74_0.62_1.6.png` cos `-0.7796`, delta `-2.1499`, norm_ratio `0.3401`
- `480_0.54_0.98.png` cos `-0.7252`, delta `-2.3131`, norm_ratio `0.3840`
- `33_0.55_1.75.png` cos `-0.7229`, delta `-1.7846`, norm_ratio `0.2939`
- `268_0.7_1.83.png` cos `-0.7164`, delta `-2.0827`, norm_ratio `0.3508`
- `357_0.86_1.37.png` cos `-0.6849`, delta `-0.6403`, norm_ratio `0.1096`
- `382_0.53_1.12.png` cos `-0.6805`, delta `-2.7092`, norm_ratio `0.4906`
- `429_0.52_1.11.png` cos `-0.6690`, delta `-2.6603`, norm_ratio `0.5026`
- `250_0.69_1.38.png` cos `-0.6596`, delta `-1.2137`, norm_ratio `0.2158`
- `383_0.62_1.77.png` cos `-0.6588`, delta `-3.3919`, norm_ratio `0.6605`

### Largest Low-Frequency MSE Regressions

- `253_0.53_1.99.png` LF MSE delta `0.00931823`, delta `-11.2479`, cos `-0.4189`
- `255_0.53_1.86.png` LF MSE delta `0.00613277`, delta `-10.9301`, cos `-0.2530`
- `33_0.55_1.75.png` LF MSE delta `0.00348376`, delta `-1.7846`, cos `-0.7229`
- `390_0.55_1.07.png` LF MSE delta `0.00328480`, delta `-4.1487`, cos `-0.6360`
- `73_0.54_1.5.png` LF MSE delta `0.00308975`, delta `-2.0098`, cos `-0.8315`
- `74_0.62_1.6.png` LF MSE delta `0.00229305`, delta `-2.1499`, cos `-0.7796`
- `358_0.68_1.38.png` LF MSE delta `0.00224523`, delta `-1.3657`, cos `-0.5081`
- `389_0.75_1.27.png` LF MSE delta `0.00217545`, delta `-1.9805`, cos `-0.5985`
- `218_0.53_1.18.png` LF MSE delta `0.00184078`, delta `-3.3598`, cos `-0.5558`
- `271_0.97_1.97.png` LF MSE delta `0.00180035`, delta `-1.2061`, cos `-0.5399`
