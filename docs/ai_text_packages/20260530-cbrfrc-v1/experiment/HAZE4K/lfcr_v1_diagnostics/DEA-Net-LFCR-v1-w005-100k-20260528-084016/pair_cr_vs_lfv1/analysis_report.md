# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CR` step `90000`
- Current: `LF-v1` step `90000`
- Mean baseline: PSNR `32.2253`, SSIM `0.9844`
- Mean current: PSNR `32.4283`, SSIM `0.9845`
- Mean delta: PSNR `0.2030`, SSIM `0.000037`
- Better / worse by PSNR: `549` / `451`
- Meaningful better / worse at 0.30 dB: `453` / `351`
- Baseline strongest 25% mean delta: `-0.0520`; regressions <= -0.30 dB: `100`
- Baseline weakest 25% mean delta: `0.4921`; gains >= +0.30 dB: `127`
- Pearson corr(A, delta PSNR): `-0.06866208227680139`
- Pearson corr(beta, delta PSNR): `0.02240958077379462`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `0.1859`, better/worse `273/227`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.0520`, better/worse `123/127`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.4921`, better/worse `153/97`
- `0.65<=A<0.75` n=`194` mean delta PSNR `0.2199`, better/worse `108/86`
- `0.75<=A<0.85` n=`210` mean delta PSNR `0.0769`, better/worse `106/104`
- `0.85<=A<0.95` n=`175` mean delta PSNR `0.1336`, better/worse `92/83`
- `A<0.65` n=`295` mean delta PSNR `0.3517`, better/worse `176/119`
- `A>=0.95` n=`126` mean delta PSNR `0.1351`, better/worse `67/59`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `0.1319`, better/worse `105/83`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `0.0907`, better/worse `106/93`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `0.1332`, better/worse `101/98`
- `beta<0.8` n=`196` mean delta PSNR `0.2753`, better/worse `111/85`
- `beta>=1.7` n=`218` mean delta PSNR `0.3654`, better/worse `126/92`

## Hard Cases

### Worst Delta PSNR

- `946_0.65_0.58.png` A=`0.65` beta=`0.58` delta `-6.2238`
- `390_0.55_1.07.png` A=`0.55` beta=`1.07` delta `-4.9448`
- `49_0.7_1.47.png` A=`0.7` beta=`1.47` delta `-4.4989`
- `167_0.53_1.34.png` A=`0.53` beta=`1.34` delta `-4.3617`
- `947_0.79_1.07.png` A=`0.79` beta=`1.07` delta `-4.3226`
- `88_0.63_0.94.png` A=`0.63` beta=`0.94` delta `-4.0783`
- `107_0.89_1.5.png` A=`0.89` beta=`1.5` delta `-4.0195`
- `127_0.57_0.92.png` A=`0.57` beta=`0.92` delta `-4.0114`
- `381_0.63_1.38.png` A=`0.63` beta=`1.38` delta `-3.9622`
- `950_0.57_1.28.png` A=`0.57` beta=`1.28` delta `-3.8846`

### Best Delta PSNR

- `67_0.68_1.12.png` A=`0.68` beta=`1.12` delta `6.4500`
- `412_0.52_0.64.png` A=`0.52` beta=`0.64` delta `6.1472`
- `800_0.55_1.84.png` A=`0.55` beta=`1.84` delta `5.8964`
- `65_0.89_0.51.png` A=`0.89` beta=`0.51` delta `5.6146`
- `402_0.5_1.99.png` A=`0.5` beta=`1.99` delta `5.5931`
- `411_0.76_1.73.png` A=`0.76` beta=`1.73` delta `5.3276`
- `90_0.89_0.67.png` A=`0.89` beta=`0.67` delta `4.7986`
- `415_0.56_1.96.png` A=`0.56` beta=`1.96` delta `4.6218`
- `403_0.91_0.7.png` A=`0.91` beta=`0.7` delta `4.6019`
- `212_0.54_1.78.png` A=`0.54` beta=`1.78` delta `4.5273`
