# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `DEA-Net-LF-v1` step `90000`
- Current: `CRPlus-v2` step `100000`
- Mean baseline: PSNR `32.4283`, SSIM `0.9845`
- Mean current: PSNR `32.3649`, SSIM `0.9847`
- Mean delta: PSNR `-0.0633`, SSIM `0.000235`
- Better / worse by PSNR: `488` / `512`
- Meaningful better / worse at 0.30 dB: `401` / `411`
- Baseline strongest 25% mean delta: `-0.1116`; regressions <= -0.30 dB: `99`
- Baseline weakest 25% mean delta: `0.2187`; gains >= +0.30 dB: `122`
- Pearson corr(A, delta PSNR): `0.021397493226876334`
- Pearson corr(beta, delta PSNR): `-0.06635931730712318`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.1802`, better/worse `224/276`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.1116`, better/worse `118/132`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.2187`, better/worse `146/104`
- `0.65<=A<0.75` n=`194` mean delta PSNR `-0.0159`, better/worse `95/99`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.0864`, better/worse `98/112`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.0863`, better/worse `76/99`
- `A<0.65` n=`295` mean delta PSNR `-0.0954`, better/worse `156/139`
- `A>=0.95` n=`126` mean delta PSNR `0.0088`, better/worse `63/63`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `0.0750`, better/worse `102/86`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.1388`, better/worse `91/108`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.0304`, better/worse `98/101`
- `beta<0.8` n=`196` mean delta PSNR `0.0402`, better/worse `99/97`
- `beta>=1.7` n=`218` mean delta PSNR `-0.2369`, better/worse `98/120`

## Hard Cases

### Worst Delta PSNR

- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `-6.5906`
- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `-6.5178`
- `150_0.6_1.29.png` A=`0.6` beta=`1.29` delta `-5.7333`
- `414_0.58_1.54.png` A=`0.58` beta=`1.54` delta `-5.6332`
- `100_0.5_1.76.png` A=`0.5` beta=`1.76` delta `-5.5167`
- `895_0.51_1.82.png` A=`0.51` beta=`1.82` delta `-5.3861`
- `152_0.61_1.16.png` A=`0.61` beta=`1.16` delta `-5.1028`
- `92_0.97_1.46.png` A=`0.97` beta=`1.46` delta `-4.5638`
- `949_0.8_0.6.png` A=`0.8` beta=`0.6` delta `-4.4378`
- `415_0.56_1.96.png` A=`0.56` beta=`1.96` delta `-4.2329`

### Best Delta PSNR

- `24_1.0_0.62.png` A=`1.0` beta=`0.62` delta `5.0114`
- `23_0.58_0.76.png` A=`0.58` beta=`0.76` delta `4.3610`
- `785_0.54_0.53.png` A=`0.54` beta=`0.53` delta `3.9694`
- `556_0.61_0.91.png` A=`0.61` beta=`0.91` delta `3.9196`
- `286_0.6_1.07.png` A=`0.6` beta=`1.07` delta `3.8107`
- `960_0.78_0.97.png` A=`0.78` beta=`0.97` delta `3.4768`
- `553_0.51_1.76.png` A=`0.51` beta=`1.76` delta `3.3182`
- `220_0.68_1.35.png` A=`0.68` beta=`1.35` delta `3.2447`
- `181_0.82_0.77.png` A=`0.82` beta=`0.77` delta `3.1349`
- `966_0.54_1.62.png` A=`0.54` beta=`1.62` delta `3.0226`
