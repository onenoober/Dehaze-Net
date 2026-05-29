# CRPlus-v2 Loss Scale Diagnostic

## Summary

- Dataset: `HAZE4K` / `train`
- Model: `DEA-Net-CR-baseline`
- Images: `64`
- Checkpoint step: `90000`
- Patch size: `256`
- Low-pass pool: `8`
- Mean PSNR: `38.1450`
- Mean L1: `0.009603`
- Mean VGG positive distance: `0.041498`
- Mean frequency positive distance: `0.002804`
- Mean low-frequency positive distance: `0.008559`
- Selected combined margin loss: `0.000141`
- Selected combined ratio loss: `0.189611`
- Candidate objective: `ratio`

## Negative Candidates

### hazy

- Combined mean negative distance: `0.291179`
- Combined mean ratio: `0.160558`
- Combined mean gap: `0.248545`
- Combined p10 gap: `0.119329`
- Margin `0.02` mean loss: `0.000064`
- Margin `0.02` active fraction: `0.015625`

### hazy_lowpass

- Combined mean negative distance: `0.720759`
- Combined mean ratio: `0.059205`
- Combined mean gap: `0.678125`
- Combined p10 gap: `0.488947`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### output_lowpass

- Combined mean negative distance: `0.693917`
- Combined mean ratio: `0.061876`
- Combined mean gap: `0.651284`
- Combined p10 gap: `0.454471`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### under_dehazed_mix

- Combined mean negative distance: `0.132811`
- Combined mean ratio: `0.346400`
- Combined mean gap: `0.090177`
- Combined p10 gap: `0.040888`
- Margin `0.02` mean loss: `0.000358`
- Margin `0.02` active fraction: `0.015625`

## Candidate Weights

- `w=0.001`: weighted CRPlus-v2 loss `0.000190`, ratio to L1 `0.019746`
- `w=0.003`: weighted CRPlus-v2 loss `0.000569`, ratio to L1 `0.059237`
- `w=0.005`: weighted CRPlus-v2 loss `0.000948`, ratio to L1 `0.098728`
- `w=0.01`: weighted CRPlus-v2 loss `0.001896`, ratio to L1 `0.197455`

## Reading Guide

- This is a read-only scale check; it does not prove a training run will improve PSNR.
- Good candidates have a usable ratio signal without the weighted loss dominating L1.
- Margin values are diagnostic; if the combined margin loss is zero, use bounded ratio or harder staged negatives.
- If a negative has tiny or negative gaps on most samples, do not use it as an early hard negative.
- If the selected weighted loss exceeds the predeclared ratio limit, change margins or curriculum before training.
