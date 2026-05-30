# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `LF-v1` step `90000`
- Current: `LFCR-v1-w005` step `100000`
- Mean baseline: PSNR `32.4283`, SSIM `0.9845`
- Mean current: PSNR `32.2105`, SSIM `0.9844`
- Mean delta: PSNR `-0.2178`, SSIM `-0.000022`
- Better / worse by PSNR: `461` / `539`
- Meaningful better / worse at 0.30 dB: `362` / `438`
- Baseline strongest 25% mean delta: `-0.3944`; regressions <= -0.30 dB: `119`
- Baseline weakest 25% mean delta: `0.0281`; gains >= +0.30 dB: `110`
- Pearson corr(A, delta PSNR): `0.10652375180132007`
- Pearson corr(beta, delta PSNR): `-0.05692387739460908`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.2523`, better/worse `229/271`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.3944`, better/worse `96/154`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.0281`, better/worse `136/114`
- `0.65<=A<0.75` n=`194` mean delta PSNR `-0.2163`, better/worse `84/110`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.1974`, better/worse `94/116`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.0743`, better/worse `84/91`
- `A<0.65` n=`295` mean delta PSNR `-0.4246`, better/worse `132/163`
- `A>=0.95` n=`126` mean delta PSNR `0.0309`, better/worse `67/59`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.0879`, better/worse `89/99`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.2140`, better/worse `89/110`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.2093`, better/worse `95/104`
- `beta<0.8` n=`196` mean delta PSNR `-0.1786`, better/worse `90/106`
- `beta>=1.7` n=`218` mean delta PSNR `-0.3761`, better/worse `98/120`

## Hard Cases

### Worst Delta PSNR

- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `-11.9422`
- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `-11.2092`
- `414_0.58_1.54.png` A=`0.58` beta=`1.54` delta `-5.7675`
- `312_0.61_0.79.png` A=`0.61` beta=`0.79` delta `-5.6524`
- `415_0.56_1.96.png` A=`0.56` beta=`1.96` delta `-5.6022`
- `402_0.5_1.99.png` A=`0.5` beta=`1.99` delta `-5.5132`
- `67_0.68_1.12.png` A=`0.68` beta=`1.12` delta `-4.7949`
- `90_0.89_0.67.png` A=`0.89` beta=`0.67` delta `-4.7579`
- `895_0.51_1.82.png` A=`0.51` beta=`1.82` delta `-4.7503`
- `765_0.56_1.92.png` A=`0.56` beta=`1.92` delta `-4.5994`

### Best Delta PSNR

- `954_0.75_1.19.png` A=`0.75` beta=`1.19` delta `5.0998`
- `955_0.87_0.65.png` A=`0.87` beta=`0.65` delta `4.7749`
- `497_0.7_0.61.png` A=`0.7` beta=`0.61` delta `4.2609`
- `3_0.5_0.82.png` A=`0.5` beta=`0.82` delta `3.6299`
- `107_0.89_1.5.png` A=`0.89` beta=`1.5` delta `3.5050`
- `333_0.76_0.55.png` A=`0.76` beta=`0.55` delta `3.4974`
- `324_0.66_0.57.png` A=`0.66` beta=`0.57` delta `3.4145`
- `553_0.51_1.76.png` A=`0.51` beta=`1.76` delta `3.3728`
- `970_0.74_1.09.png` A=`0.74` beta=`1.09` delta `3.2314`
- `161_0.56_1.13.png` A=`0.56` beta=`1.13` delta `3.1756`
