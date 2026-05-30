# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `LFCR-v1-final` step `100000`
- Current: `LFCR-v2-decay-final` step `100000`
- Mean baseline: PSNR `32.2105`, SSIM `0.9844`
- Mean current: PSNR `32.1518`, SSIM `0.9844`
- Mean delta: PSNR `-0.0587`, SSIM `-0.000065`
- Better / worse by PSNR: `456` / `544`
- Meaningful better / worse at 0.30 dB: `384` / `430`
- Baseline strongest 25% mean delta: `-0.3360`; regressions <= -0.30 dB: `121`
- Baseline weakest 25% mean delta: `0.3818`; gains >= +0.30 dB: `117`
- Pearson corr(A, delta PSNR): `-0.07165588048552213`
- Pearson corr(beta, delta PSNR): `-0.008918278583580401`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.1403`, better/worse `226/274`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.3360`, better/worse `96/154`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.3818`, better/worse `134/116`
- `0.65<=A<0.75` n=`194` mean delta PSNR `0.0561`, better/worse `90/104`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.1194`, better/worse `95/115`
- `0.85<=A<0.95` n=`175` mean delta PSNR `-0.0803`, better/worse `81/94`
- `A<0.65` n=`295` mean delta PSNR `0.0463`, better/worse `144/151`
- `A>=0.95` n=`126` mean delta PSNR `-0.3499`, better/worse `46/80`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.1657`, better/worse `91/97`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `0.0494`, better/worse `94/105`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.0655`, better/worse `87/112`
- `beta<0.8` n=`196` mean delta PSNR `-0.0137`, better/worse `87/109`
- `beta>=1.7` n=`218` mean delta PSNR `-0.0993`, better/worse `97/121`

## Hard Cases

### Worst Delta PSNR

- `343_0.83_1.07.png` A=`0.83` beta=`1.07` delta `-6.9011`
- `342_0.96_1.08.png` A=`0.96` beta=`1.08` delta `-5.7069`
- `950_0.57_1.28.png` A=`0.57` beta=`1.28` delta `-5.5116`
- `344_0.82_1.51.png` A=`0.82` beta=`1.51` delta `-4.7186`
- `601_0.58_1.51.png` A=`0.58` beta=`1.51` delta `-4.5517`
- `163_0.97_1.57.png` A=`0.97` beta=`1.57` delta `-4.4820`
- `951_0.57_0.85.png` A=`0.57` beta=`0.85` delta `-4.4420`
- `615_0.52_0.98.png` A=`0.52` beta=`0.98` delta `-4.3360`
- `162_0.9_0.74.png` A=`0.9` beta=`0.74` delta `-4.1914`
- `161_0.56_1.13.png` A=`0.56` beta=`1.13` delta `-4.1193`

### Best Delta PSNR

- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `6.0206`
- `256_0.51_1.17.png` A=`0.51` beta=`1.17` delta `5.8489`
- `390_0.55_1.07.png` A=`0.55` beta=`1.07` delta `5.6054`
- `470_0.53_1.34.png` A=`0.53` beta=`1.34` delta `5.5506`
- `231_0.55_1.16.png` A=`0.55` beta=`1.16` delta `5.3902`
- `403_0.91_0.7.png` A=`0.91` beta=`0.7` delta `5.2669`
- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `5.2484`
- `66_0.62_1.57.png` A=`0.62` beta=`1.57` delta `5.2194`
- `401_0.7_0.6.png` A=`0.7` beta=`0.6` delta `4.7949`
- `958_0.9_0.64.png` A=`0.9` beta=`0.64` delta `4.6031`
