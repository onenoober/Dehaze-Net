# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `ResidualCalib-best` step `90000`
- Current: `LFCR-v2-decay-final` step `100000`
- Mean baseline: PSNR `32.3936`, SSIM `0.9845`
- Mean current: PSNR `32.1518`, SSIM `0.9844`
- Mean delta: PSNR `-0.2417`, SSIM `-0.000133`
- Better / worse by PSNR: `431` / `569`
- Meaningful better / worse at 0.30 dB: `345` / `476`
- Baseline strongest 25% mean delta: `-0.4489`; regressions <= -0.30 dB: `136`
- Baseline weakest 25% mean delta: `0.1736`; gains >= +0.30 dB: `109`
- Pearson corr(A, delta PSNR): `-0.02972182702480518`
- Pearson corr(beta, delta PSNR): `-0.041591051616019455`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.3459`, better/worse `215/285`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.4489`, better/worse `91/159`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.1736`, better/worse `125/125`
- `0.65<=A<0.75` n=`194` mean delta PSNR `-0.2950`, better/worse `80/114`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.2704`, better/worse `85/125`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.1858`, better/worse `79/96`
- `A<0.65` n=`295` mean delta PSNR `-0.1514`, better/worse `141/154`
- `A>=0.95` n=`126` mean delta PSNR `-0.4010`, better/worse `46/80`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.2479`, better/worse `75/113`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.1413`, better/worse `90/109`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.3926`, better/worse `76/123`
- `beta<0.8` n=`196` mean delta PSNR `-0.1127`, better/worse `93/103`
- `beta>=1.7` n=`218` mean delta PSNR `-0.3063`, better/worse `97/121`

## Hard Cases

### Worst Delta PSNR

- `402_0.5_1.99.png` A=`0.5` beta=`1.99` delta `-6.7548`
- `222_0.76_0.62.png` A=`0.76` beta=`0.62` delta `-6.1541`
- `223_0.66_1.32.png` A=`0.66` beta=`1.32` delta `-5.6337`
- `510_0.71_1.55.png` A=`0.71` beta=`1.55` delta `-4.9060`
- `601_0.58_1.51.png` A=`0.58` beta=`1.51` delta `-4.8936`
- `224_0.7_1.92.png` A=`0.7` beta=`1.92` delta `-4.7549`
- `800_0.55_1.84.png` A=`0.55` beta=`1.84` delta `-4.7521`
- `950_0.57_1.28.png` A=`0.57` beta=`1.28` delta `-4.7048`
- `374_0.52_1.1.png` A=`0.52` beta=`1.1` delta `-4.6045`
- `221_0.63_0.68.png` A=`0.63` beta=`0.68` delta `-4.5611`

### Best Delta PSNR

- `220_0.68_1.35.png` A=`0.68` beta=`1.35` delta `10.6143`
- `219_0.62_1.27.png` A=`0.62` beta=`1.27` delta `9.9921`
- `390_0.55_1.07.png` A=`0.55` beta=`1.07` delta `9.4629`
- `972_0.99_0.62.png` A=`0.99` beta=`0.62` delta `6.7379`
- `886_0.51_0.87.png` A=`0.51` beta=`0.87` delta `6.5356`
- `218_0.53_1.18.png` A=`0.53` beta=`1.18` delta `5.6821`
- `412_0.52_0.64.png` A=`0.52` beta=`0.64` delta `4.3583`
- `887_0.61_1.36.png` A=`0.61` beta=`1.36` delta `4.1864`
- `996_0.71_1.66.png` A=`0.71` beta=`1.66` delta `4.1235`
- `212_0.54_1.78.png` A=`0.54` beta=`1.78` delta `3.5224`
