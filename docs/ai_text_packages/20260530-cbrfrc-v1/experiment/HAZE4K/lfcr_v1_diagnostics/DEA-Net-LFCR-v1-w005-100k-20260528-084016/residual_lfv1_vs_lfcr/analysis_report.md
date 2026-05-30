# LF Residual Direction Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `LF-v1` step `90000`
- Current: `LFCR-v1-w005` step `100000`
- Low-frequency pool: `8`
- Mean delta PSNR: `-0.2178`
- Better / worse at 0.30 dB: `362` / `438`
- Mean residual cosine: `0.2194`
- Mean luma residual cosine: `0.2217`
- Mean residual norm ratio: `0.5831`
- Mean residual error ratio: `1.0485`
- Low-frequency MSE improved / regressed: `463` / `537`
- Luma low-frequency MSE improved / regressed: `467` / `533`
- Wrong-direction count (cosine < 0): `245`
- Possible overshoot / under-correction counts: `4` / `83`
- Corr(delta PSNR, residual cosine): `0.8079895983958689`
- Corr(delta PSNR, residual norm ratio): `-0.452521488091785`

## Baseline Strength Groups

- `baseline_middle_50` n=`500` delta `-0.2523`, cos `0.2193`, norm_ratio `0.6072`, LF MSE improved/regressed `231/269`
- `baseline_strongest_25` n=`250` delta `-0.3944`, cos `0.2948`, norm_ratio `0.7745`, LF MSE improved/regressed `96/154`
- `baseline_weakest_25` n=`250` delta `0.0281`, cos `0.1440`, norm_ratio `0.3434`, LF MSE improved/regressed `136/114`

## Reading Guide

- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.
- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.
- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.
- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.

## Hard Cases

### Worst Delta PSNR

- `253_0.53_1.99.png` delta `-11.9422`, cos `-0.5646`, norm_ratio `3.6902`, LF MSE delta `0.00942843`
- `255_0.53_1.86.png` delta `-11.2092`, cos `-0.3355`, norm_ratio `3.6015`, LF MSE delta `0.00616789`
- `414_0.58_1.54.png` delta `-5.7675`, cos `0.1681`, norm_ratio `1.9936`, LF MSE delta `0.00103368`
- `312_0.61_0.79.png` delta `-5.6524`, cos `-0.2342`, norm_ratio `1.5938`, LF MSE delta `0.00039105`
- `415_0.56_1.96.png` delta `-5.6022`, cos `0.1012`, norm_ratio `1.8606`, LF MSE delta `0.00126647`
- `402_0.5_1.99.png` delta `-5.5132`, cos `-0.8162`, norm_ratio `0.9997`, LF MSE delta `0.01195542`
- `67_0.68_1.12.png` delta `-4.7949`, cos `-0.2606`, norm_ratio `1.2623`, LF MSE delta `0.00108524`
- `90_0.89_0.67.png` delta `-4.7579`, cos `-0.7615`, norm_ratio `0.8456`, LF MSE delta `0.00192745`
- `895_0.51_1.82.png` delta `-4.7503`, cos `-0.5445`, norm_ratio `1.0667`, LF MSE delta `0.00045636`
- `765_0.56_1.92.png` delta `-4.5994`, cos `-0.5995`, norm_ratio `0.9660`, LF MSE delta `0.00088281`

### Worst Residual Cosine

- `218_0.53_1.18.png` cos `-0.8749`, delta `-3.1627`, norm_ratio `0.4845`
- `89_0.63_1.05.png` cos `-0.8168`, delta `-2.8026`, norm_ratio `0.4418`
- `402_0.5_1.99.png` cos `-0.8162`, delta `-5.5132`, norm_ratio `0.9997`
- `357_0.86_1.37.png` cos `-0.8104`, delta `-1.1236`, norm_ratio `0.1671`
- `359_0.5_1.58.png` cos `-0.7909`, delta `-1.6483`, norm_ratio `0.2535`
- `421_0.58_1.77.png` cos `-0.7728`, delta `-2.8139`, norm_ratio `0.4636`
- `90_0.89_0.67.png` cos `-0.7615`, delta `-4.7579`, norm_ratio `0.8456`
- `73_0.54_1.5.png` cos `-0.7585`, delta `-2.0528`, norm_ratio `0.3343`
- `358_0.68_1.38.png` cos `-0.7395`, delta `-1.2578`, norm_ratio `0.2037`
- `242_0.62_1.18.png` cos `-0.7381`, delta `-3.1398`, norm_ratio `0.5329`

### Largest Low-Frequency MSE Regressions

- `402_0.5_1.99.png` LF MSE delta `0.01195542`, delta `-5.5132`, cos `-0.8162`
- `253_0.53_1.99.png` LF MSE delta `0.00942843`, delta `-11.9422`, cos `-0.5646`
- `361_0.51_1.62.png` LF MSE delta `0.00909463`, delta `-2.5759`, cos `-0.5721`
- `255_0.53_1.86.png` LF MSE delta `0.00616789`, delta `-11.2092`, cos `-0.3355`
- `33_0.55_1.75.png` LF MSE delta `0.00604524`, delta `-3.8184`, cos `-0.7102`
- `359_0.5_1.58.png` LF MSE delta `0.00497160`, delta `-1.6483`, cos `-0.7909`
- `73_0.54_1.5.png` LF MSE delta `0.00316733`, delta `-2.0528`, cos `-0.7585`
- `357_0.86_1.37.png` LF MSE delta `0.00286768`, delta `-1.1236`, cos `-0.8104`
- `79_0.63_1.89.png` LF MSE delta `0.00270816`, delta `-0.7979`, cos `-0.3273`
- `271_0.97_1.97.png` LF MSE delta `0.00263469`, delta `-1.9426`, cos `-0.6959`
