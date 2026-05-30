# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `1000`
- Baseline: `CR` step `90000`
- Current: `LFCR-v1-w005` step `100000`
- Mean baseline: PSNR `32.2253`, SSIM `0.9844`
- Mean current: PSNR `32.2105`, SSIM `0.9844`
- Mean delta: PSNR `-0.0148`, SSIM `0.000015`
- Better / worse by PSNR: `513` / `487`
- Meaningful better / worse at 0.30 dB: `414` / `410`
- Baseline strongest 25% mean delta: `-0.1789`; regressions <= -0.30 dB: `108`
- Baseline weakest 25% mean delta: `0.2136`; gains >= +0.30 dB: `110`
- Pearson corr(A, delta PSNR): `0.03517363393816347`
- Pearson corr(beta, delta PSNR): `-0.032046729152008846`

## Group Highlights

- `baseline_middle_50` n=`500` mean delta PSNR `-0.0469`, better/worse `250/250`
- `baseline_strongest_25` n=`250` mean delta PSNR `-0.1789`, better/worse `123/127`
- `baseline_weakest_25` n=`250` mean delta PSNR `0.2136`, better/worse `140/110`
- `0.65<=A<0.75` n=`194` mean delta PSNR `0.0036`, better/worse `102/92`
- `0.75<=A<0.85` n=`210` mean delta PSNR `-0.1205`, better/worse `95/115`
- `0.85<=A<0.95` n=`175` mean delta PSNR `0.0593`, better/worse `90/85`
- `A<0.65` n=`295` mean delta PSNR `-0.0728`, better/worse `151/144`
- `A>=0.95` n=`126` mean delta PSNR `0.1660`, better/worse `75/51`
- `0.8<=beta<1.1` n=`188` mean delta PSNR `0.0440`, better/worse `98/90`
- `1.1<=beta<1.4` n=`199` mean delta PSNR `-0.1234`, better/worse `105/94`
- `1.4<=beta<1.7` n=`199` mean delta PSNR `-0.0760`, better/worse `94/105`
- `beta<0.8` n=`196` mean delta PSNR `0.0967`, better/worse `102/94`
- `beta>=1.7` n=`218` mean delta PSNR `-0.0107`, better/worse `114/104`

## Hard Cases

### Worst Delta PSNR

- `253_0.53_1.99.png` A=`0.53` beta=`1.99` delta `-11.2479`
- `255_0.53_1.86.png` A=`0.53` beta=`1.86` delta `-10.9301`
- `256_0.51_1.17.png` A=`0.51` beta=`1.17` delta `-5.6627`
- `88_0.63_0.94.png` A=`0.63` beta=`0.94` delta `-5.4605`
- `246_0.77_1.86.png` A=`0.77` beta=`1.86` delta `-4.4560`
- `44_0.73_0.53.png` A=`0.73` beta=`0.53` delta `-4.4454`
- `86_0.65_0.51.png` A=`0.65` beta=`0.51` delta `-4.3388`
- `339_0.64_1.47.png` A=`0.64` beta=`1.47` delta `-4.2439`
- `390_0.55_1.07.png` A=`0.55` beta=`1.07` delta `-4.1487`
- `127_0.57_0.92.png` A=`0.57` beta=`0.92` delta `-4.0587`

### Best Delta PSNR

- `800_0.55_1.84.png` A=`0.55` beta=`1.84` delta `6.3713`
- `3_0.5_0.82.png` A=`0.5` beta=`0.82` delta `5.2182`
- `178_0.64_1.88.png` A=`0.64` beta=`1.88` delta `5.0276`
- `13_0.52_1.9.png` A=`0.52` beta=`1.9` delta `4.9523`
- `996_0.71_1.66.png` A=`0.71` beta=`1.66` delta `4.8508`
- `799_0.52_1.97.png` A=`0.52` beta=`1.97` delta `4.8247`
- `404_0.71_1.78.png` A=`0.71` beta=`1.78` delta `4.7459`
- `614_0.73_0.93.png` A=`0.73` beta=`0.93` delta `4.6650`
- `615_0.52_0.98.png` A=`0.52` beta=`0.98` delta `4.5437`
- `412_0.52_0.64.png` A=`0.52` beta=`0.64` delta `4.2878`
