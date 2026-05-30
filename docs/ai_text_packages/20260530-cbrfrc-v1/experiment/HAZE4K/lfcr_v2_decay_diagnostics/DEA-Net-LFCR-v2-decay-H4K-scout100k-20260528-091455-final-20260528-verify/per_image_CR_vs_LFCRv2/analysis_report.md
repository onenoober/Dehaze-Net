# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CR-baseline-best` step `90000`
- Current: `LFCR-v2-decay-final` step `100000`
- Mean baseline: PSNR `32.2253`, SSIM `0.9844`
- Mean current: PSNR `32.1518`, SSIM `0.9844`
- Mean delta: PSNR `-0.0735`, SSIM `-0.000051`
- Better / worse by PSNR: `492` / `508`
- Meaningful better / worse at 0.30 dB: `391` / `431`
- Baseline strongest 25% mean delta: `-0.1957`; regressions <= -0.30 dB: `114`
- Baseline weakest 25% mean delta: `0.2447`; gains >= +0.30 dB: `112`
- Pearson corr(A, delta PSNR): `-0.0347255631954272`
- Pearson corr(beta, delta PSNR): `-0.04252987784527211`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.1715`, better/worse `237/263`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.1957`, better/worse `114/136`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.2447`, better/worse `141/109`
- `0.65<=A<0.75` n=`194` mean delta PSNR `0.0597`, better/worse `100/94`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.2399`, better/worse `101/109`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.0211`, better/worse `85/90`
- `A<0.65` n=`295` mean delta PSNR `-0.0266`, better/worse `143/152`
- `A>=0.95` n=`126` mean delta PSNR `-0.1839`, better/worse `63/63`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.1216`, better/worse `95/93`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.0740`, better/worse `97/102`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.1416`, better/worse `93/106`
- `beta<0.8` n=`196` mean delta PSNR `0.0829`, better/worse `103/93`
- `beta>=1.7` n=`218` mean delta PSNR `-0.1100`, better/worse `104/114`

## Hard Cases

### Worst Delta PSNR

- `951_0.57_0.85.png` A=`0.57` beta=`0.85` delta `-7.7924`
- `950_0.57_1.28.png` A=`0.57` beta=`1.28` delta `-7.7148`
- `222_0.76_0.62.png` A=`0.76` beta=`0.62` delta `-5.9515`
- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `-5.6817`
- `429_0.52_1.11.png` A=`0.52` beta=`1.11` delta `-5.4837`
- `221_0.63_0.68.png` A=`0.63` beta=`0.68` delta `-5.2867`
- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `-5.2273`
- `344_0.82_1.51.png` A=`0.82` beta=`1.51` delta `-5.1449`
- `125_0.57_1.94.png` A=`0.57` beta=`1.94` delta `-5.1213`
- `126_0.64_1.73.png` A=`0.64` beta=`1.73` delta `-4.7995`

### Best Delta PSNR

- `401_0.7_0.6.png` A=`0.7` beta=`0.6` delta `6.7414`
- `404_0.71_1.78.png` A=`0.71` beta=`1.78` delta `5.8757`
- `996_0.71_1.66.png` A=`0.71` beta=`1.66` delta `5.8637`
- `403_0.91_0.7.png` A=`0.91` beta=`0.7` delta `5.5860`
- `994_0.53_1.75.png` A=`0.53` beta=`1.75` delta `5.2933`
- `67_0.68_1.12.png` A=`0.68` beta=`1.12` delta `5.2414`
- `212_0.54_1.78.png` A=`0.54` beta=`1.78` delta `4.7976`
- `995_0.53_1.8.png` A=`0.53` beta=`1.8` delta `4.5904`
- `330_0.6_0.81.png` A=`0.6` beta=`0.81` delta `3.8846`
- `331_0.72_0.9.png` A=`0.72` beta=`0.9` delta `3.8212`
