# CRPlus-v2 Loss Scale Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Model: `LFCR-v1-w005-100k`
- Images: `1000`
- Checkpoint step: `100000`
- Patch size: `0`
- Low-pass pool: `8`
- Mean PSNR: `32.2105`
- Mean L1: `0.022410`
- Mean VGG positive distance: `0.047921`
- Mean frequency positive distance: `0.002935`
- Mean low-frequency positive distance: `0.021714`
- Selected combined margin loss: `0.001964`
- Selected combined ratio loss: `0.360615`
- Candidate objective: `ratio`

## Negative Candidates

### hazy

- Combined mean negative distance: `0.179786`
- Combined mean ratio: `0.318368`
- Combined mean gap: `0.129400`
- Combined p10 gap: `0.045557`
- Margin `0.02` mean loss: `0.000447`
- Margin `0.02` active fraction: `0.023000`

### hazy_lowpass

- Combined mean negative distance: `0.568105`
- Combined mean ratio: `0.090435`
- Combined mean gap: `0.517719`
- Combined p10 gap: `0.391326`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### output_lowpass

- Combined mean negative distance: `0.548218`
- Combined mean ratio: `0.094334`
- Combined mean gap: `0.497833`
- Combined p10 gap: `0.369536`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### under_dehazed_mix

- Combined mean negative distance: `0.083999`
- Combined mean ratio: `0.669144`
- Combined mean gap: `0.033614`
- Combined p10 gap: `0.002698`
- Margin `0.02` mean loss: `0.005444`
- Margin `0.02` active fraction: `0.343000`

## Candidate Weights

- `w=0.001`: weighted CRPlus-v2 loss `0.000361`, ratio to L1 `0.016091`
- `w=0.003`: weighted CRPlus-v2 loss `0.001082`, ratio to L1 `0.048274`
- `w=0.005`: weighted CRPlus-v2 loss `0.001803`, ratio to L1 `0.080457`
- `w=0.01`: weighted CRPlus-v2 loss `0.003606`, ratio to L1 `0.160914`

## Reading Guide

- This is a read-only scale check; it does not prove a training run will improve PSNR.
- Good candidates have a usable ratio signal without the weighted loss dominating L1.
- Margin values are diagnostic; if the combined margin loss is zero, use bounded ratio or harder staged negatives.
- If a negative has tiny or negative gaps on most samples, do not use it as an early hard negative.
- If the selected weighted loss exceeds the predeclared ratio limit, change margins or curriculum before training.
