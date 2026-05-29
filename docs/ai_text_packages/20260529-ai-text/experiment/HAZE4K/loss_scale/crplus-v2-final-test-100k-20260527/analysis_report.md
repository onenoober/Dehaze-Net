# CRPlus-v2 Loss Scale Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Model: `DEA-Net-CRPlusV2`
- Images: `1000`
- Checkpoint step: `100000`
- Patch size: `0`
- Low-pass pool: `8`
- Mean PSNR: `32.3649`
- Mean L1: `0.021967`
- Mean VGG positive distance: `0.047680`
- Mean frequency positive distance: `0.002921`
- Mean low-frequency positive distance: `0.021258`
- Selected combined margin loss: `0.001901`
- Selected combined ratio loss: `0.357142`
- Candidate objective: `ratio`

## Negative Candidates

### hazy

- Combined mean negative distance: `0.179561`
- Combined mean ratio: `0.315121`
- Combined mean gap: `0.129463`
- Combined p10 gap: `0.045582`
- Margin `0.02` mean loss: `0.000414`
- Margin `0.02` active fraction: `0.024000`

### hazy_lowpass

- Combined mean negative distance: `0.568014`
- Combined mean ratio: `0.089960`
- Combined mean gap: `0.517916`
- Combined p10 gap: `0.389247`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### output_lowpass

- Combined mean negative distance: `0.548124`
- Combined mean ratio: `0.093835`
- Combined mean gap: `0.498027`
- Combined p10 gap: `0.369541`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### under_dehazed_mix

- Combined mean negative distance: `0.083918`
- Combined mean ratio: `0.662470`
- Combined mean gap: `0.033821`
- Combined p10 gap: `0.002791`
- Margin `0.02` mean loss: `0.005289`
- Margin `0.02` active fraction: `0.345000`

## Candidate Weights

- `w=0.005`: weighted CRPlus-v2 loss `0.001786`, ratio to L1 `0.081291`
- `w=0.01`: weighted CRPlus-v2 loss `0.003571`, ratio to L1 `0.162581`
- `w=0.03`: weighted CRPlus-v2 loss `0.010714`, ratio to L1 `0.487743`
- `w=0.05`: weighted CRPlus-v2 loss `0.017857`, ratio to L1 `0.812905`
- `w=0.1`: weighted CRPlus-v2 loss `0.035714`, ratio to L1 `1.625810`

## Reading Guide

- This is a read-only scale check; it does not prove a training run will improve PSNR.
- Good candidates have a usable ratio signal without the weighted loss dominating L1.
- Margin values are diagnostic; if the combined margin loss is zero, use bounded ratio or harder staged negatives.
- If a negative has tiny or negative gaps on most samples, do not use it as an early hard negative.
- If the selected weighted loss exceeds the predeclared ratio limit, change margins or curriculum before training.
