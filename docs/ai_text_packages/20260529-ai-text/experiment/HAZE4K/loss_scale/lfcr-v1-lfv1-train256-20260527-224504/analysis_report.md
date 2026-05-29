# CRPlus-v2 Loss Scale Diagnostic

## Summary

- Dataset: `HAZE4K` / `train`
- Model: `DEA-Net-LF-v1`
- Images: `256`
- Checkpoint step: `90000`
- Patch size: `256`
- Low-pass pool: `8`
- Mean PSNR: `38.7278`
- Mean L1: `0.009002`
- Mean VGG positive distance: `0.037805`
- Mean frequency positive distance: `0.002655`
- Mean low-frequency positive distance: `0.008010`
- Selected combined margin loss: `0.000141`
- Selected combined ratio loss: `0.184433`
- Candidate objective: `ratio`

## Negative Candidates

### hazy

- Combined mean negative distance: `0.279304`
- Combined mean ratio: `0.157680`
- Combined mean gap: `0.240432`
- Combined p10 gap: `0.101314`
- Margin `0.02` mean loss: `0.000009`
- Margin `0.02` active fraction: `0.003906`

### hazy_lowpass

- Combined mean negative distance: `0.702327`
- Combined mean ratio: `0.056568`
- Combined mean gap: `0.663455`
- Combined p10 gap: `0.491474`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### output_lowpass

- Combined mean negative distance: `0.675594`
- Combined mean ratio: `0.059278`
- Combined mean gap: `0.636722`
- Combined p10 gap: `0.456624`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### under_dehazed_mix

- Combined mean negative distance: `0.128000`
- Combined mean ratio: `0.336340`
- Combined mean gap: `0.089128`
- Combined p10 gap: `0.035867`
- Margin `0.02` mean loss: `0.000413`
- Margin `0.02` active fraction: `0.035156`

## Candidate Weights

- `w=0.001`: weighted CRPlus-v2 loss `0.000184`, ratio to L1 `0.020487`
- `w=0.003`: weighted CRPlus-v2 loss `0.000553`, ratio to L1 `0.061462`
- `w=0.005`: weighted CRPlus-v2 loss `0.000922`, ratio to L1 `0.102437`
- `w=0.01`: weighted CRPlus-v2 loss `0.001844`, ratio to L1 `0.204874`

## Reading Guide

- This is a read-only scale check; it does not prove a training run will improve PSNR.
- Good candidates have a usable ratio signal without the weighted loss dominating L1.
- Margin values are diagnostic; if the combined margin loss is zero, use bounded ratio or harder staged negatives.
- If a negative has tiny or negative gaps on most samples, do not use it as an early hard negative.
- If the selected weighted loss exceeds the predeclared ratio limit, change margins or curriculum before training.
