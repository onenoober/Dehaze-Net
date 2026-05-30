# HAZE4K Route Evidence Review Artifact

Generated locally from existing full-test per-image CSVs; no model forward pass or training was run.

## Model Means

| Model | Mean PSNR | Mean SSIM | Delta vs CR | Winner Count | Weak CR Delta | Strong CR Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CR baseline | 32.2253 | 0.9844 | 0.0000 | 187 | 0.0000 | 0.0000 |
| LF-v1 | 32.4283 | 0.9845 | 0.2030 | 204 | 0.4921 | -0.0520 |
| ResidualCalib | 32.3936 | 0.9845 | 0.1682 | 250 | 0.6354 | -0.1171 |
| CRPlus-v2 | 32.3649 | 0.9847 | 0.1396 | 176 | 0.4462 | 0.0921 |
| LFCR-v1 w0.005 | 32.2105 | 0.9844 | -0.0148 | 183 | 0.2136 | -0.1789 |

Best mean PSNR in this matrix: LF-v1 (32.4283).

## LF-v1 Complementarity

LF-v1 gain set (>= +0.30 dB vs CR): 453 images; LF-v1 regression set (<= -0.30 dB vs CR): 351 images.

| Candidate | Rescue >=0.30 on LF-v1 regressions | Full rescue vs CR | Preserve within 0.10 on LF-v1 gains | Lose >=0.30 on LF-v1 gains |
| --- | ---: | ---: | ---: | ---: |
| ResidualCalib | 188 | 108 | 200 | 218 |
| CRPlus-v2 | 200 | 108 | 157 | 267 |
| LFCR-v1 w0.005 | 182 | 84 | 163 | 264 |

## Oracle Ceilings

| Oracle | Mean PSNR | Gain vs CR | Gain vs LF-v1 | Best single in set | Gain vs best single | Winner counts |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| LF-v1_or_ResidualCalib | 33.0034 | 0.7781 | 0.5751 | lfv1 | 0.5751 | {"lfv1": 491, "rescalib": 509} |
| LF-v1_or_CRPlus-v2 | 32.9373 | 0.7120 | 0.5090 | lfv1 | 0.5090 | {"crplusv2": 488, "lfv1": 512} |
| LF-v1_or_ResidualCalib_or_CRPlus-v2 | 33.2308 | 1.0055 | 0.8026 | lfv1 | 0.8026 | {"crplusv2": 292, "lfv1": 347, "rescalib": 361} |
| CR_LFv1_ResidualCalib_CRPlus | 33.3975 | 1.1722 | 0.9692 | lfv1 | 0.9692 | {"cr": 224, "crplusv2": 224, "lfv1": 255, "rescalib": 297} |
| all_current_five | 33.5098 | 1.2845 | 1.0815 | lfv1 | 1.0815 | {"cr": 187, "crplusv2": 176, "lfcr_v1_w005": 183, "lfv1": 204, "rescalib": 250} |

## Files

- `model_per_image_matrix.csv`
- `model_summary.csv`
- `pairwise_delta_matrix.csv`
- `lfv1_complementarity_summary.csv`
- `oracle_summary.json`
- `pattern_counts.json`
- `group_breakdown.csv`
- `top_route_conflict_cases.csv`
