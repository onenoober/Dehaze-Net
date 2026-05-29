# LFCR-v1 w0.005 100k Diagnostic Summary

- Output dir: `/root/workspace/Dehaze-Net-audit-sync/experiment/HAZE4K/lfcr_v1_diagnostics/DEA-Net-LFCR-v1-w005-100k-20260528-084016`
- Images: `1000`
- cr_vs_lfv1: mean delta PSNR `0.20295903587607567`; better/worse `549/451`; >=0.30/<=-0.30 `453/351`
- cr_vs_lfcr: mean delta PSNR `-0.014796048782325101`; better/worse `513/487`; >=0.30/<=-0.30 `414/410`
- lfv1_vs_lfcr: mean delta PSNR `-0.21775508465840074`; better/worse `461/539`; >=0.30/<=-0.30 `362/438`
- crplusv2_vs_lfcr: mean delta PSNR `-0.15441999971571227`; better/worse `462/538`; >=0.30/<=-0.30 `377/438`
- rescalib_vs_lfcr: mean delta PSNR `-0.1830337378002708`; better/worse `474/526`; >=0.30/<=-0.30 `380/448`

## Derived Gates
- cr_q25_psnr: `28.823246292754938`
- cr_q75_psnr: `35.793551546872266`
- lfcr_vs_lfv1_mean_delta: `-0.21775508465840063`
- lfcr_vs_lfv1_better_count: `461`
- lfcr_vs_lfv1_worse_count: `539`
- lfcr_vs_lfv1_better_030_count: `362`
- lfcr_vs_lfv1_worse_030_count: `438`
- lfv1_regress_030_count: `351`
- lfv1_regress_lfcr_improves_010_count: `203`
- lfv1_regress_lfcr_improves_030_count: `182`
- lfv1_regress_lfcr_full_rescue_count: `84`
- lfv1_gain_030_count: `453`
- lfv1_gain_lfcr_preserves_count: `323`
- lfv1_gain_lfcr_loses_030_vs_lfv1_count: `264`
- weak_cr_mean_lfv1_delta: `0.4891488310222166`
- weak_cr_mean_lfcr_delta: `0.2172385260389918`
- weak_cr_mean_lfcr_delta_lfv1: `-0.2719103049832251`
- strong_cr_mean_lfv1_delta: `-0.052003885342475485`
- strong_cr_mean_lfcr_delta: `-0.1789401701953449`
- strong_cr_mean_lfcr_delta_lfv1: `-0.1269362848528694`
- strong_cr_lfcr_regress_030_count: `108`
- strong_cr_lfv1_regress_030_count: `100`

## Residual Diagnostics
- cr_vs_lfcr: mean current delta baseline PSNR `None`, wrong direction `218`, mean residual cosine `0.25841822655341823`, mean LF MSE delta `1.2571579711220693e-05`
- lfv1_vs_lfcr: mean current delta baseline PSNR `None`, wrong direction `245`, mean residual cosine `0.21935565588835743`, mean LF MSE delta `9.575519023928791e-05`

## Loss Scale
- selected_ratio_objective: `None`
- mean_l1_loss: `None`
- mean_selected_ratio_loss: `None`
- training_margin: `0.02`

```json
{
  "cr_q25_psnr": 28.823246292754938,
  "cr_q75_psnr": 35.793551546872266,
  "lfcr_vs_lfv1_mean_delta": -0.21775508465840063,
  "lfcr_vs_lfv1_better_count": 461,
  "lfcr_vs_lfv1_worse_count": 539,
  "lfcr_vs_lfv1_better_030_count": 362,
  "lfcr_vs_lfv1_worse_030_count": 438,
  "lfv1_regress_030_count": 351,
  "lfv1_regress_lfcr_improves_010_count": 203,
  "lfv1_regress_lfcr_improves_030_count": 182,
  "lfv1_regress_lfcr_full_rescue_count": 84,
  "lfv1_gain_030_count": 453,
  "lfv1_gain_lfcr_preserves_count": 323,
  "lfv1_gain_lfcr_loses_030_vs_lfv1_count": 264,
  "weak_cr_mean_lfv1_delta": 0.4891488310222166,
  "weak_cr_mean_lfcr_delta": 0.2172385260389918,
  "weak_cr_mean_lfcr_delta_lfv1": -0.2719103049832251,
  "strong_cr_mean_lfv1_delta": -0.052003885342475485,
  "strong_cr_mean_lfcr_delta": -0.1789401701953449,
  "strong_cr_mean_lfcr_delta_lfv1": -0.1269362848528694,
  "strong_cr_lfcr_regress_030_count": 108,
  "strong_cr_lfv1_regress_030_count": 100
}
```
