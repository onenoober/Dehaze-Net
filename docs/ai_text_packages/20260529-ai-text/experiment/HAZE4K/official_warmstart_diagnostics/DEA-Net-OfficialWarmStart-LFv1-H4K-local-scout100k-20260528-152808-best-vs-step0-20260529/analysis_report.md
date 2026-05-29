# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `OfficialWarmStart-step0` step `0`
- Current: `OfficialWarmStart-LFv1-best90k` step `90000`
- Mean baseline: PSNR `34.2548`, SSIM `0.9885`
- Mean current: PSNR `34.2661`, SSIM `0.9886`
- Mean delta: PSNR `0.0114`, SSIM `0.000044`
- Better / worse by PSNR: `484` / `516`
- Meaningful better / worse at 0.30 dB: `193` / `172`
- Baseline strongest 25% mean delta: `0.0455`; regressions <= -0.30 dB: `44`
- Baseline weakest 25% mean delta: `-0.0139`; gains >= +0.30 dB: `22`
- Pearson corr(A, delta PSNR): `-0.017584172324184275`
- Pearson corr(beta, delta PSNR): `-0.024204109278730974`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `0.0069`, better/worse `233/267`
- `baseline_strongest_25` n=`250` mean delta PSNR `0.0455`, better/worse `137/113`
- `baseline_weakest_25` n=`250` mean delta PSNR `-0.0139`, better/worse `114/136`
- `0.65<=A<0.75` n=`194` mean delta PSNR `0.0223`, better/worse `89/105`
- `0.75<=A<0.85` n=`210` mean delta PSNR `0.0381`, better/worse `106/104`
- `0.85<=A<0.95` n=`175` mean delta PSNR `0.0012`, better/worse `86/89`
- `A<0.65` n=`295` mean delta PSNR `0.0077`, better/worse `144/151`
- `A>=0.95` n=`126` mean delta PSNR `-0.0274`, better/worse `59/67`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `-0.0602`, better/worse `78/110`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `0.0069`, better/worse `97/102`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `0.0253`, better/worse `94/105`
- `beta<0.8` n=`196` mean delta PSNR `0.0729`, better/worse `112/84`
- `beta>=1.7` n=`218` mean delta PSNR `0.0091`, better/worse `103/115`

## Hard Cases

### Worst Delta PSNR

- `951_0.57_0.85.png` A=`0.57` beta=`0.85` delta `-1.8677`
- `544_0.53_0.73.png` A=`0.53` beta=`0.73` delta `-1.6229`
- `542_0.78_0.72.png` A=`0.78` beta=`0.72` delta `-1.3974`
- `949_0.8_0.6.png` A=`0.8` beta=`0.6` delta `-1.1359`
- `950_0.57_1.28.png` A=`0.57` beta=`1.28` delta `-1.1083`
- `167_0.53_1.34.png` A=`0.53` beta=`1.34` delta `-1.1001`
- `615_0.52_0.98.png` A=`0.52` beta=`0.98` delta `-1.0322`
- `541_0.58_1.83.png` A=`0.58` beta=`1.83` delta `-0.9974`
- `818_0.52_0.76.png` A=`0.52` beta=`0.76` delta `-0.9731`
- `690_0.62_0.99.png` A=`0.62` beta=`0.99` delta `-0.9388`

### Best Delta PSNR

- `946_0.65_0.58.png` A=`0.65` beta=`0.58` delta `1.8668`
- `828_0.63_0.64.png` A=`0.63` beta=`0.64` delta `1.4486`
- `830_0.58_1.52.png` A=`0.58` beta=`1.52` delta `1.3455`
- `993_0.73_0.6.png` A=`0.73` beta=`0.6` delta `1.2669`
- `498_0.81_1.42.png` A=`0.81` beta=`1.42` delta `1.2666`
- `935_0.58_0.55.png` A=`0.58` beta=`0.55` delta `1.1328`
- `499_0.85_1.47.png` A=`0.85` beta=`1.47` delta `1.1297`
- `556_0.61_0.91.png` A=`0.61` beta=`0.91` delta `1.1141`
- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `1.0856`
- `948_0.89_0.6.png` A=`0.89` beta=`0.6` delta `1.0308`
