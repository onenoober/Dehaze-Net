# LFCR-v2 Decay Final Diagnostic Summary

- Run: `DEA-Net-LFCR-v2-decay-H4K-scout100k-20260528-091455`
- Verified mean: `32.1518 / 0.9844`
- LF gate: `0.019099`
- Decision: `negative_or_neutral_fair_ablation_mechanism_partly_valid_quality_insufficient`

## Pairwise Mean Delta PSNR
- vs CR-baseline-best: `-0.0735` dB, better/worse `492/508`, >=0.30 `391/431`
- vs LF-v1-best: `-0.2765` dB, better/worse `415/585`, >=0.30 `332/493`
- vs ResidualCalib-best: `-0.2417` dB, better/worse `431/569`, >=0.30 `345/476`
- vs CRPlus-v2-final: `-0.2131` dB, better/worse `450/550`, >=0.30 `343/461`
- vs LFCR-v1-final: `-0.0587` dB, better/worse `456/544`, >=0.30 `384/430`

## LF-v1 Regression Rescue
- lfv1_regression_cases_le_minus_030: `351`
- improved_over_lfv1_by_010: `208`
- improved_over_lfv1_by_030: `186`
- fully_rescued_vs_cr_delta_ge_0: `96`
- mean_lfv1_delta_on_regressions: `-1.305851750715984`
- mean_lfcrv2_delta_on_lfv1_regressions: `-0.8438841542864634`
- mean_rescue_gain: `0.46196759642951934`
- lfv1_gain_cases_ge_030: `453`
- lfcrv2_loses_030_on_lfv1_gain_cases: `316`
- mean_lfv1_delta_on_gain_cases: `1.460109813284404`
- mean_lfcrv2_delta_on_lfv1_gain_cases: `0.5296695129360109`
- mean_loss_on_lfv1_gain_cases: `-0.9304403003483926`

## Residual
- LF-v1 -> LFCR-v2 cosine `0.1960`, norm ratio `0.5780`, wrong direction `264/1000`, LF MSE improved/regressed `413/587`
- CR -> LFCR-v2 cosine `0.2353`, norm ratio `0.5604`, wrong direction `222/1000`, LF MSE improved/regressed `491/509`

## Loss Scale
- selected ratio `0.3625`, selected margin `0.001967`
- w=0.003: ratio_to_l1 `0.0485`
- w=0.005: ratio_to_l1 `0.0809`
- w=0.01: ratio_to_l1 `0.1617`
- active margin 0.02: under_dehazed_mix `0.359`, hazy `0.023`, output_lowpass `0.0`
