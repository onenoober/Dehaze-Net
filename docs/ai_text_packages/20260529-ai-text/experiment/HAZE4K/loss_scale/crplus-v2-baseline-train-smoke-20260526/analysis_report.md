# CRPlus-v2 Loss Scale Diagnostic

## Summary

- Dataset: `HAZE4K` / `train`
- Model: `DEA-Net-CR-baseline`
- Images: `2`
- Checkpoint step: `90000`
- Patch size: `256`
- Low-pass pool: `8`
- Mean PSNR: `37.6052`
- Mean L1: `0.009393`
- Mean VGG positive distance: `0.034167`
- Mean frequency positive distance: `0.002355`
- Mean low-frequency positive distance: `0.008633`
- Selected combined margin loss: `0.000000`
- Selected combined ratio loss: `0.181594`
- Candidate objective: `ratio`

## Negative Candidates

### hazy

- Combined mean negative distance: `0.234072`
- Combined mean ratio: `0.150654`
- Combined mean gap: `0.198805`
- Combined p10 gap: `0.196147`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### hazy_lowpass

- Combined mean negative distance: `0.646730`
- Combined mean ratio: `0.054661`
- Combined mean gap: `0.611464`
- Combined p10 gap: `0.590487`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### output_lowpass

- Combined mean negative distance: `0.614655`
- Combined mean ratio: `0.057568`
- Combined mean gap: `0.579389`
- Combined p10 gap: `0.555104`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### under_dehazed_mix

- Combined mean negative distance: `0.104838`
- Combined mean ratio: `0.336561`
- Combined mean gap: `0.069572`
- Combined p10 gap: `0.067757`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

## Candidate Weights

- `w=0.005`: weighted CRPlus-v2 loss `0.000908`, ratio to L1 `0.096664`
- `w=0.01`: weighted CRPlus-v2 loss `0.001816`, ratio to L1 `0.193327`

## Reading Guide

- This is a read-only scale check; it does not prove a training run will improve PSNR.
- Good candidates have a usable ratio signal without the weighted loss dominating L1.
- Margin values are diagnostic; if the combined margin loss is zero, use bounded ratio or harder staged negatives.
- If a negative has tiny or negative gaps on most samples, do not use it as an early hard negative.
- If the selected weighted loss exceeds the predeclared ratio limit, change margins or curriculum before training.
