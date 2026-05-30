# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `LF-v1-best` step `90000`
- Current: `LFCR-v2-decay-final` step `100000`
- Mean baseline: PSNR `32.4283`, SSIM `0.9845`
- Mean current: PSNR `32.1518`, SSIM `0.9844`
- Mean delta: PSNR `-0.2765`, SSIM `-0.000087`
- Better / worse by PSNR: `415` / `585`
- Meaningful better / worse at 0.30 dB: `332` / `493`
- Baseline strongest 25% mean delta: `-0.4078`; regressions <= -0.30 dB: `129`
- Baseline weakest 25% mean delta: `0.0856`; gains >= +0.30 dB: `110`
- Pearson corr(A, delta PSNR): `0.031918880306373874`
- Pearson corr(beta, delta PSNR): `-0.06397964404960849`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.3918`, better/worse `196/304`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.4078`, better/worse `89/161`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.0856`, better/worse `130/120`
- `0.65<=A<0.75` n=`194` mean delta PSNR `-0.1602`, better/worse `89/105`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.3168`, better/worse `83/127`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.1546`, better/worse `74/101`
- `A<0.65` n=`295` mean delta PSNR `-0.3783`, better/worse `117/178`
- `A>=0.95` n=`126` mean delta PSNR `-0.3190`, better/worse `52/74`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.2536`, better/worse `78/110`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.1647`, better/worse `85/114`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.2748`, better/worse `81/118`
- `beta<0.8` n=`196` mean delta PSNR `-0.1924`, better/worse `89/107`
- `beta>=1.7` n=`218` mean delta PSNR `-0.4754`, better/worse `82/136`

## Hard Cases

### Worst Delta PSNR

- `343_0.83_1.07.png` A=`0.83` beta=`1.07` delta `-7.3920`
- `342_0.96_1.08.png` A=`0.96` beta=`1.08` delta `-6.8192`
- `344_0.82_1.51.png` A=`0.82` beta=`1.51` delta `-6.1933`
- `411_0.76_1.73.png` A=`0.76` beta=`1.73` delta `-6.0370`
- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `-5.9608`
- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `-5.9216`
- `895_0.51_1.82.png` A=`0.51` beta=`1.82` delta `-5.6827`
- `410_0.98_1.98.png` A=`0.98` beta=`1.98` delta `-5.4972`
- `414_0.58_1.54.png` A=`0.58` beta=`1.54` delta `-5.4793`
- `100_0.5_1.76.png` A=`0.5` beta=`1.76` delta `-4.6712`

### Best Delta PSNR

- `390_0.55_1.07.png` A=`0.55` beta=`1.07` delta `6.4015`
- `958_0.9_0.64.png` A=`0.9` beta=`0.64` delta `5.5067`
- `960_0.78_0.97.png` A=`0.78` beta=`0.97` delta `5.2484`
- `167_0.53_1.34.png` A=`0.53` beta=`1.34` delta `4.9216`
- `995_0.53_1.8.png` A=`0.53` beta=`1.8` delta `4.2097`
- `356_0.67_0.95.png` A=`0.67` beta=`0.95` delta `4.0403`
- `994_0.53_1.75.png` A=`0.53` beta=`1.75` delta `3.9616`
- `946_0.65_0.58.png` A=`0.65` beta=`0.58` delta `3.9435`
- `996_0.71_1.66.png` A=`0.71` beta=`1.66` delta `3.9088`
- `181_0.82_0.77.png` A=`0.82` beta=`0.77` delta `3.6816`
