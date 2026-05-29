# LF Residual Direction Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CR-baseline-best` step `90000`
- Current: `LFCR-v2-decay-final` step `100000`
- Low-frequency pool: `8`
- Mean delta PSNR: `-0.0735`
- Better / worse at 0.30 dB: `391` / `431`
- Mean residual cosine: `0.2353`
- Mean luma residual cosine: `0.2354`
- Mean residual norm ratio: `0.5604`
- Mean residual error ratio: `1.0283`
- Low-frequency MSE improved / regressed: `491` / `509`
- Luma low-frequency MSE improved / regressed: `488` / `512`
- Wrong-direction count (cosine < 0): `222`
- Possible overshoot / under-correction counts: `1` / `101`
- Corr(delta PSNR, residual cosine): `0.8456770498725436`
- Corr(delta PSNR, residual norm ratio): `-0.3336457684956607`

## Baseline Strength Groups

- `baseline_middle_50` n=`500` delta `-0.1715`, cos `0.2184`, norm_ratio `0.5867`, LF MSE improved/regressed `236/264`
- `baseline_strongest_25` n=`250` delta `-0.1957`, cos `0.3117`, norm_ratio `0.7319`, LF MSE improved/regressed `115/135`
- `baseline_weakest_25` n=`250` delta `0.2447`, cos `0.1927`, norm_ratio `0.3363`, LF MSE improved/regressed `140/110`

## Reading Guide

- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.
- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.
- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.
- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.

## Hard Cases

### Worst Delta PSNR

- `951_0.57_0.85.png` delta `-7.7924`, cos `-0.3140`, norm_ratio `2.1190`, LF MSE delta `0.00057653`
- `950_0.57_1.28.png` delta `-7.7148`, cos `-0.5828`, norm_ratio `1.8266`, LF MSE delta `0.00093377`
- `222_0.76_0.62.png` delta `-5.9515`, cos `-0.6583`, norm_ratio `1.2261`, LF MSE delta `0.00132558`
- `255_0.53_1.86.png` delta `-5.6817`, cos `-0.1954`, norm_ratio `1.6452`, LF MSE delta `0.00146027`
- `429_0.52_1.11.png` delta `-5.4837`, cos `-0.7018`, norm_ratio `1.1078`, LF MSE delta `0.00321430`
- `221_0.63_0.68.png` delta `-5.2867`, cos `-0.6963`, norm_ratio `1.0265`, LF MSE delta `0.00168743`
- `253_0.53_1.99.png` delta `-5.2273`, cos `-0.3248`, norm_ratio `1.3682`, LF MSE delta `0.00176783`
- `344_0.82_1.51.png` delta `-5.1449`, cos `-0.7846`, norm_ratio `0.9642`, LF MSE delta `0.00321346`
- `125_0.57_1.94.png` delta `-5.1213`, cos `-0.1458`, norm_ratio `1.5309`, LF MSE delta `0.00129131`
- `126_0.64_1.73.png` delta `-4.7995`, cos `0.1061`, norm_ratio `1.6996`, LF MSE delta `0.00081085`

### Worst Residual Cosine

- `410_0.98_1.98.png` cos `-0.7949`, delta `-2.3337`, norm_ratio `0.3666`
- `344_0.82_1.51.png` cos `-0.7846`, delta `-5.1449`, norm_ratio `0.9642`
- `342_0.96_1.08.png` cos `-0.7683`, delta `-4.6791`, norm_ratio `0.8640`
- `343_0.83_1.07.png` cos `-0.7417`, delta `-4.0957`, norm_ratio `0.7570`
- `376_0.67_0.58.png` cos `-0.7361`, delta `-2.6113`, norm_ratio `0.4411`
- `79_0.63_1.89.png` cos `-0.7294`, delta `-0.6896`, norm_ratio `0.1108`
- `480_0.54_0.98.png` cos `-0.7081`, delta `-1.8720`, norm_ratio `0.3118`
- `429_0.52_1.11.png` cos `-0.7018`, delta `-5.4837`, norm_ratio `1.1078`
- `221_0.63_0.68.png` cos `-0.6963`, delta `-5.2867`, norm_ratio `1.0265`
- `147_0.87_1.14.png` cos `-0.6865`, delta `-0.9590`, norm_ratio `0.1617`

### Largest Low-Frequency MSE Regressions

- `410_0.98_1.98.png` LF MSE delta `0.00457866`, delta `-2.3337`, cos `-0.7949`
- `429_0.52_1.11.png` LF MSE delta `0.00321430`, delta `-5.4837`, cos `-0.7018`
- `344_0.82_1.51.png` LF MSE delta `0.00321346`, delta `-5.1449`, cos `-0.7846`
- `391_0.87_1.8.png` LF MSE delta `0.00309359`, delta `-2.8402`, cos `-0.3476`
- `392_0.97_1.81.png` LF MSE delta `0.00275325`, delta `-2.5718`, cos `-0.5993`
- `79_0.63_1.89.png` LF MSE delta `0.00271676`, delta `-0.6896`, cos `-0.7294`
- `342_0.96_1.08.png` LF MSE delta `0.00234770`, delta `-4.6791`, cos `-0.7683`
- `431_0.86_1.86.png` LF MSE delta `0.00207454`, delta `-1.6434`, cos `-0.6864`
- `463_0.97_1.52.png` LF MSE delta `0.00199966`, delta `-3.2737`, cos `-0.4259`
- `346_0.77_1.92.png` LF MSE delta `0.00188596`, delta `-2.3689`, cos `-0.4198`
