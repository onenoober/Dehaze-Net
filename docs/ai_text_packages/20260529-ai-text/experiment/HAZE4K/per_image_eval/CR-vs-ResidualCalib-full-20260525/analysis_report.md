# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CR-baseline` step `90000`
- Current: `ResidualCalib` step `90000`
- Mean baseline: PSNR `32.2253`, SSIM `0.9844`
- Mean current: PSNR `32.3936`, SSIM `0.9845`
- Mean delta: PSNR `0.1682`, SSIM `0.000082`
- Better / worse by PSNR: `547` / `453`
- Meaningful better / worse at 0.30 dB: `469` / `379`
- Baseline strongest 25% mean delta: `-0.1171`; regressions <= -0.30 dB: `110`
- Baseline weakest 25% mean delta: `0.6354`; gains >= +0.30 dB: `136`
- Pearson corr(A, delta PSNR): `-0.0035166758267742193`
- Pearson corr(beta, delta PSNR): `0.00016243373719527593`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `0.0773`, better/worse `262/238`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.1171`, better/worse `117/133`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.6354`, better/worse `168/82`
- `0.65<=A<0.75` n=`194` mean delta PSNR `0.3547`, better/worse `114/80`
- `0.75<=A<0.85` n=`210` mean delta PSNR `0.0305`, better/worse `115/95`
- `0.85<=A<0.95` n=`175` mean delta PSNR `0.1647`, better/worse `94/81`
- `A<0.65` n=`295` mean delta PSNR `0.1249`, better/worse `151/144`
- `A>=0.95` n=`126` mean delta PSNR `0.2171`, better/worse `73/53`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `0.1263`, better/worse `106/82`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `0.0673`, better/worse `103/96`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `0.2510`, better/worse `118/81`
- `beta<0.8` n=`196` mean delta PSNR `0.1957`, better/worse `100/96`
- `beta>=1.7` n=`218` mean delta PSNR `0.1963`, better/worse `120/98`

## Hard Cases

### Worst Delta PSNR

- `219_0.62_1.27.png` A=`0.62` beta=`1.27` delta `-8.8306`
- `390_0.55_1.07.png` A=`0.55` beta=`1.07` delta `-8.0063`
- `220_0.68_1.35.png` A=`0.68` beta=`1.35` delta `-7.5728`
- `886_0.51_0.87.png` A=`0.51` beta=`0.87` delta `-6.9119`
- `88_0.63_0.94.png` A=`0.63` beta=`0.94` delta `-6.1948`
- `218_0.53_1.18.png` A=`0.53` beta=`1.18` delta `-6.0702`
- `127_0.57_0.92.png` A=`0.57` beta=`0.92` delta `-6.0580`
- `191_0.86_1.88.png` A=`0.86` beta=`1.88` delta `-5.7245`
- `949_0.8_0.6.png` A=`0.8` beta=`0.6` delta `-5.1485`
- `316_0.91_0.85.png` A=`0.91` beta=`0.85` delta `-5.0792`

### Best Delta PSNR

- `402_0.5_1.99.png` A=`0.5` beta=`1.99` delta `9.7377`
- `401_0.7_0.6.png` A=`0.7` beta=`0.6` delta `9.6409`
- `403_0.91_0.7.png` A=`0.91` beta=`0.7` delta `7.5849`
- `800_0.55_1.84.png` A=`0.55` beta=`1.84` delta `7.5817`
- `422_0.65_0.92.png` A=`0.65` beta=`0.92` delta `6.5840`
- `799_0.52_1.97.png` A=`0.52` beta=`1.97` delta `6.5343`
- `67_0.68_1.12.png` A=`0.68` beta=`1.12` delta `5.2715`
- `217_0.86_1.31.png` A=`0.86` beta=`1.31` delta `4.9730`
- `457_0.61_0.57.png` A=`0.61` beta=`0.57` delta `4.8849`
- `404_0.71_1.78.png` A=`0.71` beta=`1.78` delta `4.4891`
