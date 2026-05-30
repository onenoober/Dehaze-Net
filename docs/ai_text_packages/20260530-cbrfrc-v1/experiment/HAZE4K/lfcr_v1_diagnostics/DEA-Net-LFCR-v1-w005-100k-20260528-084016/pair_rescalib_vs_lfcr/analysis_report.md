# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `ResidualCalib` step `90000`
- Current: `LFCR-v1-w005` step `100000`
- Mean baseline: PSNR `32.3936`, SSIM `0.9845`
- Mean current: PSNR `32.2105`, SSIM `0.9844`
- Mean delta: PSNR `-0.1830`, SSIM `-0.000067`
- Better / worse by PSNR: `474` / `526`
- Meaningful better / worse at 0.30 dB: `380` / `448`
- Baseline strongest 25% mean delta: `-0.4291`; regressions <= -0.30 dB: `126`
- Baseline weakest 25% mean delta: `0.2658`; gains >= +0.30 dB: `126`
- Pearson corr(A, delta PSNR): `0.0359781395416173`
- Pearson corr(beta, delta PSNR): `-0.02960328483951`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.2844`, better/worse `227/273`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.4291`, better/worse `95/155`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.2658`, better/worse `152/98`
- `0.65<=A<0.75` n=`194` mean delta PSNR `-0.3511`, better/worse `87/107`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.1510`, better/worse `91/119`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.1054`, better/worse `92/83`
- `A<0.65` n=`295` mean delta PSNR `-0.1977`, better/worse `138/157`
- `A>=0.95` n=`126` mean delta PSNR `-0.0511`, better/worse `66/60`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.0823`, better/worse `93/95`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.1907`, better/worse `94/105`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.3270`, better/worse `86/113`
- `beta<0.8` n=`196` mean delta PSNR `-0.0990`, better/worse `93/103`
- `beta>=1.7` n=`218` mean delta PSNR `-0.2070`, better/worse `108/110`

## Hard Cases

### Worst Delta PSNR

- `402_0.5_1.99.png` A=`0.5` beta=`1.99` delta `-9.6577`
- `401_0.7_0.6.png` A=`0.7` beta=`0.6` delta `-7.6944`
- `403_0.91_0.7.png` A=`0.91` beta=`0.7` delta `-7.2658`
- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `-7.1453`
- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `-6.6848`
- `422_0.65_0.92.png` A=`0.65` beta=`0.92` delta `-6.2681`
- `766_0.58_1.2.png` A=`0.58` beta=`1.2` delta `-6.1931`
- `256_0.51_1.17.png` A=`0.51` beta=`1.17` delta `-5.9622`
- `11_0.98_0.72.png` A=`0.98` beta=`0.72` delta `-5.2068`
- `231_0.55_1.16.png` A=`0.55` beta=`1.16` delta `-4.9072`

### Best Delta PSNR

- `220_0.68_1.35.png` A=`0.68` beta=`1.35` delta `8.8914`
- `972_0.99_0.62.png` A=`0.99` beta=`0.62` delta `7.6861`
- `412_0.52_0.64.png` A=`0.52` beta=`0.64` delta `7.1203`
- `343_0.83_1.07.png` A=`0.83` beta=`1.07` delta `6.8349`
- `886_0.51_0.87.png` A=`0.51` beta=`0.87` delta `6.4561`
- `970_0.74_1.09.png` A=`0.74` beta=`1.09` delta `5.6070`
- `219_0.62_1.27.png` A=`0.62` beta=`1.27` delta `5.5637`
- `3_0.5_0.82.png` A=`0.5` beta=`0.82` delta `5.3225`
- `837_0.58_1.79.png` A=`0.58` beta=`1.79` delta `5.2070`
- `411_0.76_1.73.png` A=`0.76` beta=`1.73` delta `4.9811`
