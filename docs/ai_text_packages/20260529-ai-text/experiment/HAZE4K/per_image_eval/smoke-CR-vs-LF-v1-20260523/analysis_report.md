# Train Checkpoint Per-Image Comparison

## Summary

- Dataset: `HAZE4K` / `test`
- Images: `5`
- Baseline: `DEA-Net-CR` step `90000`
- Current: `DEA-Net-LF-v1` step `90000`
- Mean baseline: PSNR `27.7197`, SSIM `0.9781`
- Mean current: PSNR `28.2743`, SSIM `0.9769`
- Mean delta: PSNR `0.5546`, SSIM `-0.001167`
- Better / worse by PSNR: `2` / `3`
- Meaningful better / worse at 0.30 dB: `2` / `1`
- Pearson corr(A, delta PSNR): `-0.5421745968814781`
- Pearson corr(beta, delta PSNR): `0.3046274659432429`

## Group Highlights

- `0.65<=A<0.75` n=`2` mean delta PSNR `-0.3117`, better/worse `0/2`
- `0.75<=A<0.85` n=`1` mean delta PSNR `1.2250`, better/worse `1/0`
- `0.85<=A<0.95` n=`1` mean delta PSNR `-0.0702`, better/worse `0/1`
- `A<0.65` n=`1` mean delta PSNR `2.2415`, better/worse `1/0`
- `0.8<=beta<1.1` n=`1` mean delta PSNR `-0.0702`, better/worse `0/1`
- `1.1<=beta<1.4` n=`1` mean delta PSNR `-0.0499`, better/worse `0/1`
- `1.4<=beta<1.7` n=`1` mean delta PSNR `1.2250`, better/worse `1/0`
- `beta>=1.7` n=`2` mean delta PSNR `0.8340`, better/worse `1/1`

## Hard Cases

### Worst Delta PSNR

- `1000_0.73_1.8.png` A=`0.73` beta=`1.8` delta `-0.5735`
- `103_0.94_0.98.png` A=`0.94` beta=`0.98` delta `-0.0702`
- `102_0.67_1.36.png` A=`0.67` beta=`1.36` delta `-0.0499`
- `101_0.83_1.44.png` A=`0.83` beta=`1.44` delta `1.2250`
- `100_0.5_1.76.png` A=`0.5` beta=`1.76` delta `2.2415`

### Best Delta PSNR

- `100_0.5_1.76.png` A=`0.5` beta=`1.76` delta `2.2415`
- `101_0.83_1.44.png` A=`0.83` beta=`1.44` delta `1.2250`
- `102_0.67_1.36.png` A=`0.67` beta=`1.36` delta `-0.0499`
- `103_0.94_0.98.png` A=`0.94` beta=`0.98` delta `-0.0702`
- `1000_0.73_1.8.png` A=`0.73` beta=`1.8` delta `-0.5735`
