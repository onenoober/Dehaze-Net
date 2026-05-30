# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CRPlus-v2-final` step `100000`
- Current: `LFCR-v2-decay-final` step `100000`
- Mean baseline: PSNR `32.3649`, SSIM `0.9847`
- Mean current: PSNR `32.1518`, SSIM `0.9844`
- Mean delta: PSNR `-0.2131`, SSIM `-0.000322`
- Better / worse by PSNR: `450` / `550`
- Meaningful better / worse at 0.30 dB: `343` / `461`
- Baseline strongest 25% mean delta: `-0.4115`; regressions <= -0.30 dB: `127`
- Baseline weakest 25% mean delta: `0.0790`; gains >= +0.30 dB: `106`
- Pearson corr(A, delta PSNR): `0.013713788875078105`
- Pearson corr(beta, delta PSNR): `-0.0038856947376453405`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.2600`, better/worse `230/270`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.4115`, better/worse `89/161`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.0790`, better/worse `131/119`
- `0.65<=A<0.75` n=`194` mean delta PSNR `-0.1443`, better/worse `88/106`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.2304`, better/worse `92/118`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.0684`, better/worse `92/83`
- `A<0.65` n=`295` mean delta PSNR `-0.2829`, better/worse `130/165`
- `A>=0.95` n=`126` mean delta PSNR `-0.3279`, better/worse `48/78`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.3286`, better/worse `76/112`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.0258`, better/worse `100/99`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.2443`, better/worse `93/106`
- `beta<0.8` n=`196` mean delta PSNR `-0.2326`, better/worse `91/105`
- `beta>=1.7` n=`218` mean delta PSNR `-0.2385`, better/worse `90/128`

## Hard Cases

### Worst Delta PSNR

- `342_0.96_1.08.png` A=`0.96` beta=`1.08` delta `-8.2361`
- `343_0.83_1.07.png` A=`0.83` beta=`1.07` delta `-6.1028`
- `344_0.82_1.51.png` A=`0.82` beta=`1.51` delta `-4.9435`
- `458_0.58_0.65.png` A=`0.58` beta=`0.65` delta `-4.5354`
- `429_0.52_1.11.png` A=`0.52` beta=`1.11` delta `-4.4164`
- `463_0.97_1.52.png` A=`0.97` beta=`1.52` delta `-4.3618`
- `23_0.58_0.76.png` A=`0.58` beta=`0.76` delta `-4.2744`
- `462_0.91_1.52.png` A=`0.91` beta=`1.52` delta `-4.2289`
- `285_0.71_1.94.png` A=`0.71` beta=`1.94` delta `-4.2251`
- `224_0.7_1.92.png` A=`0.7` beta=`1.92` delta `-4.1370`

### Best Delta PSNR

- `390_0.55_1.07.png` A=`0.55` beta=`1.07` delta `5.7903`
- `994_0.53_1.75.png` A=`0.53` beta=`1.75` delta `4.3954`
- `995_0.53_1.8.png` A=`0.53` beta=`1.8` delta `3.9775`
- `150_0.6_1.29.png` A=`0.6` beta=`1.29` delta `3.9313`
- `356_0.67_0.95.png` A=`0.67` beta=`0.95` delta `3.7514`
- `152_0.61_1.16.png` A=`0.61` beta=`1.16` delta `3.7494`
- `270_0.73_1.75.png` A=`0.73` beta=`1.75` delta `3.4816`
- `958_0.9_0.64.png` A=`0.9` beta=`0.64` delta `3.3672`
- `191_0.86_1.88.png` A=`0.86` beta=`1.88` delta `3.3575`
- `155_0.68_1.77.png` A=`0.68` beta=`1.77` delta `3.3438`
