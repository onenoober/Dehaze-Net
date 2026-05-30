# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CRPlus-v2` step `100000`
- Current: `LFCR-v1-w005` step `100000`
- Mean baseline: PSNR `32.3649`, SSIM `0.9847`
- Mean current: PSNR `32.2105`, SSIM `0.9844`
- Mean delta: PSNR `-0.1544`, SSIM `-0.000257`
- Better / worse by PSNR: `462` / `538`
- Meaningful better / worse at 0.30 dB: `377` / `438`
- Baseline strongest 25% mean delta: `-0.4296`; regressions <= -0.30 dB: `124`
- Baseline weakest 25% mean delta: `0.1326`; gains >= +0.30 dB: `111`
- Pearson corr(A, delta PSNR): `0.09126939508904029`
- Pearson corr(beta, delta PSNR): `0.005831084831496969`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.1603`, better/worse `237/263`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.4296`, better/worse `97/153`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.1326`, better/worse `128/122`
- `0.65<=A<0.75` n=`194` mean delta PSNR `-0.2004`, better/worse `86/108`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.1110`, better/worse `97/113`
- `0.85<=A<0.95` n=`175` mean delta PSNR `0.0120`, better/worse `88/87`
- `A<0.65` n=`295` mean delta PSNR `-0.3292`, better/worse `126/169`
- `A>=0.95` n=`126` mean delta PSNR `0.0220`, better/worse `65/61`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.1629`, better/worse `81/107`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.0752`, better/worse `97/102`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.1788`, better/worse `97/102`
- `beta<0.8` n=`196` mean delta PSNR `-0.2188`, better/worse `91/105`
- `beta>=1.7` n=`218` mean delta PSNR `-0.1392`, better/worse `96/122`

## Hard Cases

### Worst Delta PSNR

- `66_0.62_1.57.png` A=`0.62` beta=`1.57` delta `-5.9581`
- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `-5.3516`
- `231_0.55_1.16.png` A=`0.55` beta=`1.16` delta `-4.7135`
- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `-4.6914`
- `457_0.61_0.57.png` A=`0.61` beta=`0.57` delta `-4.6394`
- `766_0.58_1.2.png` A=`0.58` beta=`1.2` delta `-4.5615`
- `11_0.98_0.72.png` A=`0.98` beta=`0.72` delta `-4.5282`
- `458_0.58_0.65.png` A=`0.58` beta=`0.65` delta `-4.5141`
- `422_0.65_0.92.png` A=`0.65` beta=`0.92` delta `-4.3773`
- `256_0.51_1.17.png` A=`0.51` beta=`1.17` delta `-4.2558`

### Best Delta PSNR

- `837_0.58_1.79.png` A=`0.58` beta=`1.79` delta `5.1361`
- `3_0.5_0.82.png` A=`0.5` beta=`0.82` delta `4.1632`
- `150_0.6_1.29.png` A=`0.6` beta=`1.29` delta `3.9432`
- `949_0.8_0.6.png` A=`0.8` beta=`0.6` delta `3.8506`
- `216_0.94_0.88.png` A=`0.94` beta=`0.88` delta `3.7305`
- `152_0.61_1.16.png` A=`0.61` beta=`1.16` delta `3.6953`
- `13_0.52_1.9.png` A=`0.52` beta=`1.9` delta `3.6073`
- `9_0.85_1.67.png` A=`0.85` beta=`1.67` delta `3.4893`
- `162_0.9_0.74.png` A=`0.9` beta=`0.74` delta `3.4146`
- `473_0.63_0.94.png` A=`0.63` beta=`0.94` delta `3.3324`
