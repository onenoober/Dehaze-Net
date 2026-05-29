# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `LF-v1` step `90000`
- Current: `ResidualCalib` step `90000`
- Mean baseline: PSNR `32.4283`, SSIM `0.9845`
- Mean current: PSNR `32.3936`, SSIM `0.9845`
- Mean delta: PSNR `-0.0347`, SSIM `0.000045`
- Better / worse by PSNR: `509` / `491`
- Meaningful better / worse at 0.30 dB: `414` / `386`
- Baseline strongest 25% mean delta: `-0.2374`; regressions <= -0.30 dB: `103`
- Baseline weakest 25% mean delta: `0.3970`; gains >= +0.30 dB: `127`
- Pearson corr(A, delta PSNR): `0.059930507056092985`
- Pearson corr(beta, delta PSNR): `-0.020684371338407812`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.1492`, better/worse `249/251`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.2374`, better/worse `110/140`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.3970`, better/worse `150/100`
- `0.65<=A<0.75` n=`194` mean delta PSNR `0.1349`, better/worse `106/88`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.0464`, better/worse `97/113`
- `0.85<=A<0.95` n=`175` mean delta PSNR `0.0311`, better/worse `90/85`
- `A<0.65` n=`295` mean delta PSNR `-0.2269`, better/worse `145/150`
- `A>=0.95` n=`126` mean delta PSNR `0.0820`, better/worse `71/55`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.0056`, better/worse `96/92`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.0233`, better/worse `103/96`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `0.1178`, better/worse `105/94`
- `beta<0.8` n=`196` mean delta PSNR `-0.0796`, better/worse `93/103`
- `beta>=1.7` n=`218` mean delta PSNR `-0.1691`, better/worse `112/106`

## Hard Cases

### Worst Delta PSNR

- `412_0.52_0.64.png` A=`0.52` beta=`0.64` delta `-8.9797`
- `411_0.76_1.73.png` A=`0.76` beta=`1.73` delta `-7.4831`
- `343_0.83_1.07.png` A=`0.83` beta=`1.07` delta `-7.3258`
- `219_0.62_1.27.png` A=`0.62` beta=`1.27` delta `-7.1211`
- `220_0.68_1.35.png` A=`0.68` beta=`1.35` delta `-7.0521`
- `886_0.51_0.87.png` A=`0.51` beta=`0.87` delta `-5.9728`
- `218_0.53_1.18.png` A=`0.53` beta=`1.18` delta `-5.8732`
- `344_0.82_1.51.png` A=`0.82` beta=`1.51` delta `-5.6094`
- `150_0.6_1.29.png` A=`0.6` beta=`1.29` delta `-4.9273`
- `949_0.8_0.6.png` A=`0.8` beta=`0.6` delta `-4.8840`

### Best Delta PSNR

- `510_0.71_1.55.png` A=`0.71` beta=`1.55` delta `6.0384`
- `401_0.7_0.6.png` A=`0.7` beta=`0.6` delta `5.4630`
- `553_0.51_1.76.png` A=`0.51` beta=`1.76` delta `5.3656`
- `960_0.78_0.97.png` A=`0.78` beta=`0.97` delta `5.0625`
- `958_0.9_0.64.png` A=`0.9` beta=`0.64` delta `4.7632`
- `49_0.7_1.47.png` A=`0.7` beta=`1.47` delta `4.4441`
- `556_0.61_0.91.png` A=`0.61` beta=`0.91` delta `4.2595`
- `402_0.5_1.99.png` A=`0.5` beta=`1.99` delta `4.1446`
- `51_0.6_1.79.png` A=`0.6` beta=`1.79` delta `4.1101`
- `381_0.63_1.38.png` A=`0.63` beta=`1.38` delta `4.1000`
