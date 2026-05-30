# LF Residual Direction Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `DEA-Net-CR` step `90000`
- Current: `DEA-Net-LF-v1` step `90000`
- Low-frequency pool: `8`
- Mean delta PSNR: `0.2030`
- Better / worse at 0.30 dB: `453` / `351`
- Mean residual cosine: `0.3028`
- Mean luma residual cosine: `0.3029`
- Mean residual norm ratio: `0.5638`
- Mean residual error ratio: `0.9914`
- Low-frequency MSE improved / regressed: `545` / `455`
- Luma low-frequency MSE improved / regressed: `547` / `453`
- Wrong-direction count (cosine < 0): `160`
- Possible overshoot / under-correction counts: `0` / `120`
- Corr(delta PSNR, residual cosine): `0.8774521339519118`
- Corr(delta PSNR, residual norm ratio): `-0.22735905563675574`

## Baseline Strength Groups

- `baseline_middle_50` n=`500` delta `0.1859`, cos `0.2879`, norm_ratio `0.5806`, LF MSE improved/regressed `268/232`
- `baseline_strongest_25` n=`250` delta `-0.0520`, cos `0.3490`, norm_ratio `0.7440`, LF MSE improved/regressed `124/126`
- `baseline_weakest_25` n=`250` delta `0.4921`, cos `0.2865`, norm_ratio `0.3501`, LF MSE improved/regressed `153/97`

## Reading Guide

- `lf_residual_cosine < 0` means the current model low-frequency correction points opposite to the target correction.
- `lf_residual_norm_ratio > 1` means the current correction is larger than the target correction; very high values are overshoot candidates.
- `lf_mse_delta > 0` means the current model made low-frequency MSE worse than the baseline.
- If PSNR regressions align with low cosine, high norm ratio, or positive LF MSE delta, the next route should calibrate residual direction/magnitude rather than add another spatial mask.

## Hard Cases

### Worst Delta PSNR

- `946_0.65_0.58.png` delta `-6.2238`, cos `-0.5088`, norm_ratio `1.5371`, LF MSE delta `0.00020041`
- `390_0.55_1.07.png` delta `-4.9448`, cos `-0.6894`, norm_ratio `0.9444`, LF MSE delta `0.00434916`
- `49_0.7_1.47.png` delta `-4.4989`, cos `-0.4811`, norm_ratio `1.0155`, LF MSE delta `0.00052973`
- `167_0.53_1.34.png` delta `-4.3617`, cos `-0.2592`, norm_ratio `1.1124`, LF MSE delta `0.00096928`
- `947_0.79_1.07.png` delta `-4.3226`, cos `-0.5642`, norm_ratio `0.9182`, LF MSE delta `0.00046619`
- `88_0.63_0.94.png` delta `-4.0783`, cos `-0.1729`, norm_ratio `1.1502`, LF MSE delta `0.00077866`
- `107_0.89_1.5.png` delta `-4.0195`, cos `-0.5920`, norm_ratio `0.8038`, LF MSE delta `0.00249552`
- `127_0.57_0.92.png` delta `-4.0114`, cos `-0.2192`, norm_ratio `1.1826`, LF MSE delta `0.00036294`
- `381_0.63_1.38.png` delta `-3.9622`, cos `-0.5871`, norm_ratio `0.8034`, LF MSE delta `0.00061750`
- `950_0.57_1.28.png` delta `-3.8846`, cos `-0.2699`, norm_ratio `1.0095`, LF MSE delta `0.00026722`

### Worst Residual Cosine

- `497_0.7_0.61.png` cos `-0.7058`, delta `-3.0967`, norm_ratio `0.5582`
- `390_0.55_1.07.png` cos `-0.6894`, delta `-4.9448`, norm_ratio `0.9444`
- `382_0.53_1.12.png` cos `-0.6640`, delta `-3.3871`, norm_ratio `0.6320`
- `168_0.8_1.19.png` cos `-0.6560`, delta `-1.9999`, norm_ratio `0.3553`
- `429_0.52_1.11.png` cos `-0.6517`, delta `-3.5449`, norm_ratio `0.7045`
- `948_0.89_0.6.png` cos `-0.6401`, delta `-3.1452`, norm_ratio `0.5991`
- `52_0.57_1.57.png` cos `-0.6368`, delta `-2.6461`, norm_ratio `0.4982`
- `268_0.7_1.83.png` cos `-0.6272`, delta `-1.3156`, norm_ratio `0.2428`
- `107_0.89_1.5.png` cos `-0.5920`, delta `-4.0195`, norm_ratio `0.8038`
- `251_0.55_1.12.png` cos `-0.5914`, delta `-1.5640`, norm_ratio `0.3098`

### Largest Low-Frequency MSE Regressions

- `390_0.55_1.07.png` LF MSE delta `0.00434916`, delta `-4.9448`, cos `-0.6894`
- `459_0.97_1.71.png` LF MSE delta `0.00361590`, delta `-0.6800`, cos `-0.5847`
- `389_0.75_1.27.png` LF MSE delta `0.00288214`, delta `-2.4694`, cos `-0.4530`
- `107_0.89_1.5.png` LF MSE delta `0.00249552`, delta `-4.0195`, cos `-0.5920`
- `391_0.87_1.8.png` LF MSE delta `0.00225959`, delta `-2.2452`, cos `-0.0725`
- `429_0.52_1.11.png` LF MSE delta `0.00163430`, delta `-3.5449`, cos `-0.6517`
- `346_0.77_1.92.png` LF MSE delta `0.00143530`, delta `-1.9278`, cos `-0.3081`
- `303_0.89_1.94.png` LF MSE delta `0.00135633`, delta `-1.7883`, cos `-0.2926`
- `509_0.95_1.52.png` LF MSE delta `0.00134878`, delta `-2.5735`, cos `-0.4233`
- `170_0.96_1.28.png` LF MSE delta `0.00123468`, delta `-1.5610`, cos `-0.2985`
