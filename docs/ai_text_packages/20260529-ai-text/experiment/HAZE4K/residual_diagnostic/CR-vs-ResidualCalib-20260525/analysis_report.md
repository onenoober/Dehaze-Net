# LF Residual Direction Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CR-baseline` step `90000`
- Current: `ResidualCalib` step `90000`
- Low-frequency pool: `8`
- Mean delta PSNR: `0.1682`
- Better / worse at 0.30 dB: `469` / `379`
- Mean residual cosine: `0.3150`
- Mean luma residual cosine: `0.3160`
- Mean residual norm ratio: `0.6122`
- Mean residual error ratio: `1.0046`
- Low-frequency MSE improved / regressed: `554` / `446`
- Luma low-frequency MSE improved / regressed: `557` / `443`
- Wrong-direction count (cosine < 0): `163`
- Possible overshoot / under-correction counts: `0` / `124`
- Corr(delta PSNR, residual cosine): `0.8489507447354444`
- Corr(delta PSNR, residual norm ratio): `-0.37531980480273514`

## Baseline Strength Groups

- `baseline_middle_50` n=`500` delta `0.0773`, cos `0.2997`, norm_ratio `0.6430`, LF MSE improved/regressed `267/233`
- `baseline_strongest_25` n=`250` delta `-0.1171`, cos `0.3488`, norm_ratio `0.7834`, LF MSE improved/regressed `118/132`
- `baseline_weakest_25` n=`250` delta `0.6354`, cos `0.3119`, norm_ratio `0.3794`, LF MSE improved/regressed `169/81`

## Reading Guide

- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.
- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.
- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.
- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.

## Hard Cases

### Worst Delta PSNR

- `219_0.62_1.27.png` delta `-8.8306`, cos `-0.4642`, norm_ratio `2.2722`, LF MSE delta `0.00263726`
- `390_0.55_1.07.png` delta `-8.0063`, cos `-0.7622`, norm_ratio `1.6938`, LF MSE delta `0.01080487`
- `220_0.68_1.35.png` delta `-7.5728`, cos `0.4518`, norm_ratio `2.8068`, LF MSE delta `0.00150827`
- `886_0.51_0.87.png` delta `-6.9119`, cos `-0.3349`, norm_ratio `1.8635`, LF MSE delta `0.00038527`
- `88_0.63_0.94.png` delta `-6.1948`, cos `-0.0826`, norm_ratio `1.8347`, LF MSE delta `0.00166040`
- `218_0.53_1.18.png` delta `-6.0702`, cos `-0.8570`, norm_ratio `1.1072`, LF MSE delta `0.00477920`
- `127_0.57_0.92.png` delta `-6.0580`, cos `-0.3399`, norm_ratio `1.6302`, LF MSE delta `0.00071296`
- `191_0.86_1.88.png` delta `-5.7245`, cos `0.0471`, norm_ratio `1.8057`, LF MSE delta `0.00092997`
- `949_0.8_0.6.png` delta `-5.1485`, cos `0.3009`, norm_ratio `1.9861`, LF MSE delta `0.00017921`
- `316_0.91_0.85.png` delta `-5.0792`, cos `-0.5540`, norm_ratio `1.1216`, LF MSE delta `0.00106491`

### Worst Residual Cosine

- `218_0.53_1.18.png` cos `-0.8570`, delta `-6.0702`, norm_ratio `1.1072`
- `75_0.99_0.83.png` cos `-0.7747`, delta `-2.5600`, norm_ratio `0.4206`
- `412_0.52_0.64.png` cos `-0.7711`, delta `-2.8325`, norm_ratio `0.4601`
- `390_0.55_1.07.png` cos `-0.7622`, delta `-8.0063`, norm_ratio `1.6938`
- `431_0.86_1.86.png` cos `-0.7405`, delta `-3.0839`, norm_ratio `0.5239`
- `411_0.76_1.73.png` cos `-0.7395`, delta `-2.1555`, norm_ratio `0.3538`
- `480_0.54_0.98.png` cos `-0.7156`, delta `-3.1380`, norm_ratio `0.5399`
- `343_0.83_1.07.png` cos `-0.7134`, delta `-4.0295`, norm_ratio `0.7594`
- `344_0.82_1.51.png` cos `-0.6976`, delta `-4.5611`, norm_ratio `0.8811`
- `429_0.52_1.11.png` cos `-0.6909`, delta `-4.9729`, norm_ratio `0.9877`

### Largest Low-Frequency MSE Regressions

- `390_0.55_1.07.png` LF MSE delta `0.01080487`, delta `-8.0063`, cos `-0.7622`
- `443_0.56_1.04.png` LF MSE delta `0.00521930`, delta `-0.4088`, cos `-0.6478`
- `218_0.53_1.18.png` LF MSE delta `0.00477920`, delta `-6.0702`, cos `-0.8570`
- `431_0.86_1.86.png` LF MSE delta `0.00467488`, delta `-3.0839`, cos `-0.7405`
- `389_0.75_1.27.png` LF MSE delta `0.00413741`, delta `-3.2129`, cos `-0.5192`
- `410_0.98_1.98.png` LF MSE delta `0.00309406`, delta `-1.7062`, cos `-0.6390`
- `429_0.52_1.11.png` LF MSE delta `0.00270392`, delta `-4.9729`, cos `-0.6909`
- `74_0.62_1.6.png` LF MSE delta `0.00264569`, delta `-2.4046`, cos `-0.6682`
- `344_0.82_1.51.png` LF MSE delta `0.00263835`, delta `-4.5611`, cos `-0.6976`
- `219_0.62_1.27.png` LF MSE delta `0.00263726`, delta `-8.8306`, cos `-0.4642`
