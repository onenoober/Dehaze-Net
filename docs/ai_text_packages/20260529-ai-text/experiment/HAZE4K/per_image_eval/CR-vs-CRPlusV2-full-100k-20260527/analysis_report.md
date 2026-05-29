# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `DEA-Net-CR` step `90000`
- Current: `CRPlus-v2` step `100000`
- Mean baseline: PSNR `32.2253`, SSIM `0.9844`
- Mean current: PSNR `32.3649`, SSIM `0.9847`
- Mean delta: PSNR `0.1396`, SSIM `0.000271`
- Better / worse by PSNR: `552` / `448`
- Meaningful better / worse at 0.30 dB: `457` / `359`
- Baseline strongest 25% mean delta: `0.0921`; regressions <= -0.30 dB: `92`
- Baseline weakest 25% mean delta: `0.4462`; gains >= +0.30 dB: `132`
- Pearson corr(A, delta PSNR): `-0.047780496413400084`
- Pearson corr(beta, delta PSNR): `-0.03945333130925879`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `0.0101`, better/worse `262/238`
- `baseline_strongest_25` n=`250` mean delta PSNR `0.0921`, better/worse `137/113`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.4462`, better/worse `153/97`
- `0.65<=A<0.75` n=`194` mean delta PSNR `0.2040`, better/worse `109/85`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.0095`, better/worse `102/108`
- `0.85<=A<0.95` n=`175` mean delta PSNR `0.0473`, better/worse `93/82`
- `A<0.65` n=`295` mean delta PSNR `0.2564`, better/worse `179/116`
- `A>=0.95` n=`126` mean delta PSNR `0.1439`, better/worse `69/57`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `0.2069`, better/worse `115/73`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.0482`, better/worse `94/105`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `0.1028`, better/worse `112/87`
- `beta<0.8` n=`196` mean delta PSNR `0.3155`, better/worse `102/94`
- `beta>=1.7` n=`218` mean delta PSNR `0.1285`, better/worse `129/89`

## Hard Cases

### Worst Delta PSNR

- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `-6.2388`
- `191_0.86_1.88.png` A=`0.86` beta=`1.88` delta `-6.1515`
- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `-5.8963`
- `88_0.63_0.94.png` A=`0.63` beta=`0.94` delta `-5.8357`
- `951_0.57_0.85.png` A=`0.57` beta=`0.85` delta `-5.5062`
- `85_0.81_1.32.png` A=`0.81` beta=`1.32` delta `-5.4977`
- `946_0.65_0.58.png` A=`0.65` beta=`0.58` delta `-4.8765`
- `87_0.87_0.93.png` A=`0.87` beta=`0.93` delta `-4.7167`
- `949_0.8_0.6.png` A=`0.8` beta=`0.6` delta `-4.7023`
- `339_0.64_1.47.png` A=`0.64` beta=`1.47` delta `-4.6108`

### Best Delta PSNR

- `457_0.61_0.57.png` A=`0.61` beta=`0.57` delta `5.1194`
- `67_0.68_1.12.png` A=`0.68` beta=`1.12` delta `5.0684`
- `422_0.65_0.92.png` A=`0.65` beta=`0.92` delta `4.6932`
- `800_0.55_1.84.png` A=`0.55` beta=`1.84` delta `4.6872`
- `212_0.54_1.78.png` A=`0.54` beta=`1.78` delta `4.6231`
- `401_0.7_0.6.png` A=`0.7` beta=`0.6` delta `4.5492`
- `65_0.89_0.51.png` A=`0.89` beta=`0.51` delta `4.4779`
- `412_0.52_0.64.png` A=`0.52` beta=`0.64` delta `4.2713`
- `818_0.52_0.76.png` A=`0.52` beta=`0.76` delta `4.2269`
- `624_0.58_0.68.png` A=`0.58` beta=`0.68` delta `4.0296`
