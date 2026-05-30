# LF Residual Direction Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `LF-v1-best` step `90000`
- Current: `LFCR-v2-decay-final` step `100000`
- Low-frequency pool: `8`
- Mean delta PSNR: `-0.2765`
- Better / worse at 0.30 dB: `332` / `493`
- Mean residual cosine: `0.1960`
- Mean luma residual cosine: `0.1972`
- Mean residual norm ratio: `0.5780`
- Mean residual error ratio: `1.0565`
- Low-frequency MSE improved / regressed: `413` / `587`
- Luma low-frequency MSE improved / regressed: `417` / `583`
- Wrong-direction count (cosine < 0): `264`
- Possible overshoot / under-correction counts: `2` / `102`
- Corr(delta PSNR, residual cosine): `0.853882219606973`
- Corr(delta PSNR, residual norm ratio): `-0.38294533064521236`

## Baseline Strength Groups

- `baseline_middle_50` n=`500` delta `-0.3918`, cos `0.1835`, norm_ratio `0.6043`, LF MSE improved/regressed `194/306`
- `baseline_strongest_25` n=`250` delta `-0.4078`, cos `0.2695`, norm_ratio `0.7412`, LF MSE improved/regressed `89/161`
- `baseline_weakest_25` n=`250` delta `0.0856`, cos `0.1475`, norm_ratio `0.3621`, LF MSE improved/regressed `130/120`

## Reading Guide

- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.
- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.
- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.
- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.

## Hard Cases

### Worst Delta PSNR

- `343_0.83_1.07.png` delta `-7.3920`, cos `-0.6753`, norm_ratio `1.7053`, LF MSE delta `0.00204650`
- `342_0.96_1.08.png` delta `-6.8192`, cos `-0.7124`, norm_ratio `1.4587`, LF MSE delta `0.00281135`
- `344_0.82_1.51.png` delta `-6.1933`, cos `-0.7947`, norm_ratio `1.2217`, LF MSE delta `0.00350759`
- `411_0.76_1.73.png` delta `-6.0370`, cos `-0.8512`, norm_ratio `1.1029`, LF MSE delta `0.00334904`
- `255_0.53_1.86.png` delta `-5.9608`, cos `-0.2481`, norm_ratio `1.6993`, LF MSE delta `0.00149540`
- `253_0.53_1.99.png` delta `-5.9216`, cos `-0.4693`, norm_ratio `1.4705`, LF MSE delta `0.00187803`
- `895_0.51_1.82.png` delta `-5.6827`, cos `-0.5833`, norm_ratio `1.2842`, LF MSE delta `0.00062455`
- `410_0.98_1.98.png` delta `-5.4972`, cos `-0.9141`, norm_ratio `0.9409`, LF MSE delta `0.00792291`
- `414_0.58_1.54.png` delta `-5.4793`, cos `-0.0886`, norm_ratio `1.6279`, LF MSE delta `0.00091939`
- `100_0.5_1.76.png` delta `-4.6712`, cos `-0.1252`, norm_ratio `1.3881`, LF MSE delta `0.00093855`

### Worst Residual Cosine

- `410_0.98_1.98.png` cos `-0.9141`, delta `-5.4972`, norm_ratio `0.9409`
- `374_0.52_1.1.png` cos `-0.8645`, delta `-2.6463`, norm_ratio `0.4037`
- `362_0.82_0.88.png` cos `-0.8558`, delta `-2.0400`, norm_ratio `0.3186`
- `411_0.76_1.73.png` cos `-0.8512`, delta `-6.0370`, norm_ratio `1.1029`
- `373_0.53_1.07.png` cos `-0.8461`, delta `-2.3807`, norm_ratio `0.3634`
- `376_0.67_0.58.png` cos `-0.8460`, delta `-3.2918`, norm_ratio `0.5232`
- `89_0.63_1.05.png` cos `-0.8251`, delta `-2.2243`, norm_ratio `0.3412`
- `79_0.63_1.89.png` cos `-0.8024`, delta `-1.2954`, norm_ratio `0.1936`
- `409_0.54_1.68.png` cos `-0.8002`, delta `-3.1560`, norm_ratio `0.5132`
- `344_0.82_1.51.png` cos `-0.7947`, delta `-6.1933`, norm_ratio `1.2217`

### Largest Low-Frequency MSE Regressions

- `361_0.51_1.62.png` LF MSE delta `0.00866456`, delta `-2.4917`, cos `-0.7321`
- `410_0.98_1.98.png` LF MSE delta `0.00792291`, delta `-5.4972`, cos `-0.9141`
- `359_0.5_1.58.png` LF MSE delta `0.00721388`, delta `-2.2316`, cos `-0.6350`
- `79_0.63_1.89.png` LF MSE delta `0.00473777`, delta `-1.2954`, cos `-0.8024`
- `402_0.5_1.99.png` LF MSE delta `0.00384441`, delta `-2.6103`, cos `-0.7231`
- `80_0.74_1.76.png` LF MSE delta `0.00379892`, delta `-1.4870`, cos `-0.6272`
- `344_0.82_1.51.png` LF MSE delta `0.00350759`, delta `-6.1933`, cos `-0.7947`
- `411_0.76_1.73.png` LF MSE delta `0.00334904`, delta `-6.0370`, cos `-0.8512`
- `392_0.97_1.81.png` LF MSE delta `0.00322631`, delta `-3.2520`, cos `-0.1104`
- `342_0.96_1.08.png` LF MSE delta `0.00281135`, delta `-6.8192`, cos `-0.7124`
