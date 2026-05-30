# CRPlus-v2 Loss Scale Diagnostic

## Summary

- Dataset: `HAZE4K` / `test`
- Model: `LFCR-v2-decay-final`
- Images: `1000`
- Checkpoint step: `100000`
- Patch size: `0`
- Low-pass pool: `8`
- Mean PSNR: `32.1518`
- Mean L1: `0.022418`
- Mean VGG positive distance: `0.047906`
- Mean frequency positive distance: `0.002936`
- Mean low-frequency positive distance: `0.021725`
- Selected combined margin loss: `0.001967`
- Selected combined ratio loss: `0.362525`
- Candidate objective: `ratio`

## Negative Candidates

### hazy

- Combined mean negative distance: `0.178647`
- Combined mean ratio: `0.320229`
- Combined mean gap: `0.128274`
- Combined p10 gap: `0.044462`
- Margin `0.02` mean loss: `0.000314`
- Margin `0.02` active fraction: `0.023000`

### hazy_lowpass

- Combined mean negative distance: `0.567277`
- Combined mean ratio: `0.090708`
- Combined mean gap: `0.516904`
- Combined p10 gap: `0.389408`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### output_lowpass

- Combined mean negative distance: `0.547575`
- Combined mean ratio: `0.094587`
- Combined mean gap: `0.497203`
- Combined p10 gap: `0.369701`
- Margin `0.02` mean loss: `0.000000`
- Margin `0.02` active fraction: `0.000000`

### under_dehazed_mix

- Combined mean negative distance: `0.083507`
- Combined mean ratio: `0.672759`
- Combined mean gap: `0.033134`
- Combined p10 gap: `0.001260`
- Margin `0.02` mean loss: `0.005588`
- Margin `0.02` active fraction: `0.359000`

## Candidate Weights

- `w=0.003`: weighted CRPlus-v2 loss `0.001088`, ratio to L1 `0.048514`
- `w=0.005`: weighted CRPlus-v2 loss `0.001813`, ratio to L1 `0.080857`
- `w=0.01`: weighted CRPlus-v2 loss `0.003625`, ratio to L1 `0.161713`

## Reading Guide

- This is a read-only scale check; it does not prove a training run will improve PSNR.
- Good candidates have a usable ratio signal without the weighted loss dominating L1.
- Margin values are diagnostic; if the combined margin loss is zero, use bounded ratio or harder staged negatives.
- If a negative has tiny or negative gaps on most samples, do not use it as an early hard negative.
- If the selected weighted loss exceeds the predeclared ratio limit, change margins or curriculum before training.
