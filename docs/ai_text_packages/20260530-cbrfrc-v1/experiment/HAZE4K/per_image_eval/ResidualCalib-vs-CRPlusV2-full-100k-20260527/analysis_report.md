# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `DEA-Net-LF-ResidualCalib` step `90000`
- Current: `CRPlus-v2` step `100000`
- Mean baseline: PSNR `32.3936`, SSIM `0.9845`
- Mean current: PSNR `32.3649`, SSIM `0.9847`
- Mean delta: PSNR `-0.0286`, SSIM `0.000189`
- Better / worse by PSNR: `461` / `539`
- Meaningful better / worse at 0.30 dB: `361` / `441`
- Baseline strongest 25% mean delta: `-0.1092`; regressions <= -0.30 dB: `106`
- Baseline weakest 25% mean delta: `0.4367`; gains >= +0.30 dB: `120`
- Pearson corr(A, delta PSNR): `-0.04275668135722725`
- Pearson corr(beta, delta PSNR): `-0.03893939411786059`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.2210`, better/worse `208/292`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.1092`, better/worse `112/138`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.4367`, better/worse `141/109`
- `0.65<=A<0.75` n=`194` mean delta PSNR `-0.1507`, better/worse `85/109`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.0400`, better/worse `92/118`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.1174`, better/worse `79/96`
- `A<0.65` n=`295` mean delta PSNR `0.1315`, better/worse `153/142`
- `A>=0.95` n=`126` mean delta PSNR `-0.0731`, better/worse `52/74`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `0.0806`, better/worse `89/99`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.1155`, better/worse `77/122`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.1482`, better/worse `85/114`
- `beta<0.8` n=`196` mean delta PSNR `0.1198`, better/worse `105/91`
- `beta>=1.7` n=`218` mean delta PSNR `-0.0678`, better/worse `105/113`

## Hard Cases

### Worst Delta PSNR

- `402_0.5_1.99.png` A=`0.5` beta=`1.99` delta `-8.3603`
- `272_0.94_1.17.png` A=`0.94` beta=`1.17` delta `-5.1966`
- `401_0.7_0.6.png` A=`0.7` beta=`0.6` delta `-5.0916`
- `270_0.73_1.75.png` A=`0.73` beta=`1.75` delta `-4.8063`
- `222_0.76_0.62.png` A=`0.76` beta=`0.62` delta `-4.6852`
- `381_0.63_1.38.png` A=`0.63` beta=`1.38` delta `-3.9944`
- `403_0.91_0.7.png` A=`0.91` beta=`0.7` delta `-3.9614`
- `895_0.51_1.82.png` A=`0.51` beta=`1.82` delta `-3.9376`
- `339_0.64_1.47.png` A=`0.64` beta=`1.47` delta `-3.8905`
- `269_0.97_1.84.png` A=`0.97` beta=`1.84` delta `-3.7690`

### Best Delta PSNR

- `220_0.68_1.35.png` A=`0.68` beta=`1.35` delta `10.2968`
- `219_0.62_1.27.png` A=`0.62` beta=`1.27` delta `9.1777`
- `412_0.52_0.64.png` A=`0.52` beta=`0.64` delta `7.1038`
- `972_0.99_0.62.png` A=`0.99` beta=`0.62` delta `7.0470`
- `342_0.96_1.08.png` A=`0.96` beta=`1.08` delta `6.1389`
- `886_0.51_0.87.png` A=`0.51` beta=`0.87` delta `6.1000`
- `343_0.83_1.07.png` A=`0.83` beta=`1.07` delta `6.0366`
- `23_0.58_0.76.png` A=`0.58` beta=`0.76` delta `5.3638`
- `887_0.61_1.36.png` A=`0.61` beta=`1.36` delta `5.1764`
- `218_0.53_1.18.png` A=`0.53` beta=`1.18` delta `5.0139`
