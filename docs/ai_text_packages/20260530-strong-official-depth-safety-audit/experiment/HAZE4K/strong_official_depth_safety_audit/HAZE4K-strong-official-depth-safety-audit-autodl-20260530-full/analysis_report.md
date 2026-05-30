# HAZE4K Strong-Official Depth Safety Audit

## Decision

- `stop_depth_route`
- Do not launch depth-aware DEA training from this audit.

## Oracle And Labels

- Images: `1000`
- Mean official PSNR: `34.2556`
- Mean best-candidate gain vs official: `0.5532`
- No-change / strong-official no-change / intervention-worthy counts: `411` / `125` / `493`
- Best-candidate counts: `{'warm': 665, 'cold_cr': 61, 'lfv1': 94, 'residualcalib': 85, 'crplus': 67, 'cbrfrc': 28}`

## Depth Source

- Depth models: `[{'alias': 'depthanything', 'model_id': 'depth-anything/Depth-Anything-V2-Small-hf'}, {'alias': 'midas', 'model_id': 'Intel/dpt-hybrid-midas'}]`
- Depth generated: `{'depthanything': 6986, 'midas': 6986}`
- Depth loaded: `{'depthanything': 14, 'midas': 14}`
- Depth seconds: `{'depthanything': 338.40961718559265, 'midas': 331.3160674571991}`

## C2 vs C1

- strong-official FI relative reduction: `0.0`
- all no-change FI relative reduction: `0.0`
- precision delta: `0.0`
- simulated gain delta: `0.0`
- bootstrap gain p05 delta: `0.0`

## Heldout And Controls

- Heldout: `{'official_strength': 'stable', 'CR_strength': 'stable', 'residual_low_energy': 'stable'}`
- Controls: `{'shuffled_depth': 'fail_or_close', 'label_permutation': 'pass', 'basic_stat_control': 'fail', 'depth_estimator_consistency': 'fail'}`

## Summary

| Group | Feature Set | Head | Split | Precision | Recall | Strong FI | No-change FI | Gain | p05 | Preserve | Low Preserve |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | C0_deployable | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C0 | C0_deployable | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C0 | C0_deployable | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C0 | C0_deployable | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C0 | C0_deployable | hgb | test_strong_cr | 0.7500 | 0.3830 | 0.0990 | 0.1053 | 0.1507 | 0.0934 | 0.8048 | 0.8333 |
| C0 | C0_deployable | hgb | test_strong_official | 0.6552 | 0.2235 | 0.0800 | 0.0800 | 0.0259 | 0.0099 | 0.8640 | 0.8618 |
| C0 | C0_deployable | hgb | random | 0.7449 | 0.8340 | 0.2000 | 0.3494 | 0.6194 | 0.4947 | 0.6320 | 0.3901 |
| C0 | C0_deployable | hgb | test_residual_low | 0.7431 | 0.6585 | 0.1765 | 0.2979 | 0.2751 | 0.2131 | 0.7500 | 0.5360 |
| C0 | C0_deployable | logistic | test_strong_cr | 0.6092 | 0.5638 | 0.3069 | 0.2982 | 0.1895 | 0.1315 | 0.6190 | 0.5802 |
| C0 | C0_deployable | logistic | test_strong_official | 0.5735 | 0.4588 | 0.2320 | 0.2320 | 0.0625 | 0.0310 | 0.6640 | 0.6579 |
| C0 | C0_deployable | logistic | random | 0.7559 | 0.8473 | 0.2520 | 0.3360 | 0.6115 | 0.4864 | 0.5332 | 0.3856 |
| C0 | C0_deployable | logistic | test_residual_low | 0.6750 | 0.6585 | 0.3971 | 0.4149 | 0.2679 | 0.2064 | 0.5197 | 0.4520 |
| C1 | C1_basic | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C1 | C1_basic | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C1 | C1_basic | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C1 | C1_basic | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C1 | C1_basic | hgb | test_strong_cr | 0.7714 | 0.2872 | 0.0594 | 0.0702 | 0.1334 | 0.0770 | 0.8714 | 0.8765 |
| C1 | C1_basic | hgb | test_strong_official | 0.6389 | 0.2706 | 0.1040 | 0.1040 | 0.0315 | 0.0141 | 0.8240 | 0.8158 |
| C1 | C1_basic | hgb | random | 0.7469 | 0.8543 | 0.2202 | 0.3535 | 0.6248 | 0.4991 | 0.6262 | 0.3844 |
| C1 | C1_basic | hgb | test_residual_low | 0.7500 | 0.6585 | 0.1471 | 0.2872 | 0.2791 | 0.2182 | 0.7697 | 0.5440 |
| C1 | C1_basic | logistic | test_strong_cr | 0.6136 | 0.5745 | 0.3069 | 0.2982 | 0.1914 | 0.1325 | 0.6143 | 0.5741 |
| C1 | C1_basic | logistic | test_strong_official | 0.5735 | 0.4588 | 0.2320 | 0.2320 | 0.0625 | 0.0310 | 0.6640 | 0.6579 |
| C1 | C1_basic | logistic | random | 0.7578 | 0.8473 | 0.2520 | 0.3328 | 0.6121 | 0.4874 | 0.5332 | 0.3856 |
| C1 | C1_basic | logistic | test_residual_low | 0.6750 | 0.6585 | 0.3971 | 0.4149 | 0.2679 | 0.2064 | 0.5197 | 0.4520 |
| C2 | C2_both | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C2 | C2_both | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C2 | C2_both | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C2 | C2_both | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C2 | C2_both | hgb | test_strong_cr | 0.6818 | 0.3191 | 0.1089 | 0.1228 | 0.1373 | 0.0829 | 0.8238 | 0.8457 |
| C2 | C2_both | hgb | test_strong_official | 0.7000 | 0.2471 | 0.0720 | 0.0720 | 0.0304 | 0.0156 | 0.8640 | 0.8618 |
| C2 | C2_both | hgb | random | 0.7497 | 0.8469 | 0.2101 | 0.3465 | 0.6165 | 0.4915 | 0.6278 | 0.3732 |
| C2 | C2_both | hgb | test_residual_low | 0.7692 | 0.6504 | 0.1324 | 0.2553 | 0.2812 | 0.2206 | 0.7895 | 0.5640 |
| C2 | C2_both | logistic | test_strong_cr | 0.5806 | 0.5745 | 0.3564 | 0.3421 | 0.1829 | 0.1227 | 0.5952 | 0.5556 |
| C2 | C2_both | logistic | test_strong_official | 0.5652 | 0.4588 | 0.2400 | 0.2400 | 0.0588 | 0.0256 | 0.6560 | 0.6447 |
| C2 | C2_both | logistic | random | 0.7530 | 0.8417 | 0.2520 | 0.3393 | 0.6085 | 0.4841 | 0.5396 | 0.3856 |
| C2 | C2_both | logistic | test_residual_low | 0.6891 | 0.6667 | 0.3676 | 0.3936 | 0.2742 | 0.2110 | 0.5329 | 0.4560 |
| C2 | C2_both_shuffled_depth | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C2 | C2_both_shuffled_depth | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C2 | C2_both_shuffled_depth | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C2 | C2_both_shuffled_depth | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C2 | C2_both_shuffled_depth | hgb | test_strong_cr | 0.7714 | 0.2872 | 0.0594 | 0.0702 | 0.1334 | 0.0770 | 0.8714 | 0.8765 |
| C2 | C2_both_shuffled_depth | hgb | test_strong_official | 0.5588 | 0.2235 | 0.1200 | 0.1200 | 0.0242 | 0.0082 | 0.8360 | 0.8355 |
| C2 | C2_both_shuffled_depth | hgb | random | 0.7480 | 0.8540 | 0.2101 | 0.3500 | 0.6266 | 0.5013 | 0.6278 | 0.3951 |
| C2 | C2_both_shuffled_depth | hgb | test_residual_low | 0.7500 | 0.6585 | 0.1471 | 0.2872 | 0.2791 | 0.2182 | 0.7697 | 0.5440 |
| C2 | C2_both_shuffled_depth | logistic | test_strong_cr | 0.5652 | 0.5532 | 0.3564 | 0.3509 | 0.1776 | 0.1133 | 0.5905 | 0.5556 |
| C2 | C2_both_shuffled_depth | logistic | test_strong_official | 0.5441 | 0.4353 | 0.2480 | 0.2480 | 0.0551 | 0.0236 | 0.6640 | 0.6513 |
| C2 | C2_both_shuffled_depth | logistic | random | 0.7457 | 0.8446 | 0.2990 | 0.3524 | 0.6078 | 0.4829 | 0.5114 | 0.3694 |
| C2 | C2_both_shuffled_depth | logistic | test_residual_low | 0.6885 | 0.6829 | 0.3824 | 0.4043 | 0.2760 | 0.2136 | 0.5263 | 0.4440 |
| C2 | C2_depthanything | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C2 | C2_depthanything | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C2 | C2_depthanything | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C2 | C2_depthanything | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C2 | C2_depthanything | hgb | test_strong_cr | 0.8049 | 0.3511 | 0.0594 | 0.0702 | 0.1514 | 0.0972 | 0.8571 | 0.8704 |
| C2 | C2_depthanything | hgb | test_strong_official | 0.6176 | 0.2471 | 0.1040 | 0.1040 | 0.0274 | 0.0108 | 0.8400 | 0.8487 |
| C2 | C2_depthanything | hgb | random | 0.7448 | 0.8494 | 0.2202 | 0.3566 | 0.6170 | 0.4922 | 0.6169 | 0.3845 |
| C2 | C2_depthanything | hgb | test_residual_low | 0.7619 | 0.6504 | 0.1618 | 0.2660 | 0.2815 | 0.2214 | 0.7697 | 0.5520 |
| C2 | C2_depthanything | logistic | test_strong_cr | 0.5761 | 0.5638 | 0.3564 | 0.3421 | 0.1809 | 0.1204 | 0.6048 | 0.5679 |
| C2 | C2_depthanything | logistic | test_strong_official | 0.5571 | 0.4588 | 0.2480 | 0.2480 | 0.0579 | 0.0252 | 0.6600 | 0.6447 |
| C2 | C2_depthanything | logistic | random | 0.7567 | 0.8444 | 0.2520 | 0.3327 | 0.6101 | 0.4861 | 0.5396 | 0.3856 |
| C2 | C2_depthanything | logistic | test_residual_low | 0.6807 | 0.6585 | 0.3824 | 0.4043 | 0.2679 | 0.2065 | 0.5263 | 0.4560 |
| C2 | C2_depthanything_shuffled_depth | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C2 | C2_depthanything_shuffled_depth | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C2 | C2_depthanything_shuffled_depth | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C2 | C2_depthanything_shuffled_depth | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C2 | C2_depthanything_shuffled_depth | hgb | test_strong_cr | 0.7714 | 0.2872 | 0.0594 | 0.0702 | 0.1334 | 0.0770 | 0.8714 | 0.8765 |
| C2 | C2_depthanything_shuffled_depth | hgb | test_strong_official | 0.5588 | 0.2235 | 0.1200 | 0.1200 | 0.0242 | 0.0082 | 0.8360 | 0.8355 |
| C2 | C2_depthanything_shuffled_depth | hgb | random | 0.7452 | 0.8461 | 0.2202 | 0.3535 | 0.6203 | 0.4944 | 0.6262 | 0.3844 |
| C2 | C2_depthanything_shuffled_depth | hgb | test_residual_low | 0.7500 | 0.6585 | 0.1471 | 0.2872 | 0.2791 | 0.2182 | 0.7697 | 0.5440 |
| C2 | C2_depthanything_shuffled_depth | logistic | test_strong_cr | 0.5618 | 0.5319 | 0.3564 | 0.3421 | 0.1753 | 0.1130 | 0.6048 | 0.5617 |
| C2 | C2_depthanything_shuffled_depth | logistic | test_strong_official | 0.5455 | 0.4235 | 0.2400 | 0.2400 | 0.0563 | 0.0258 | 0.6760 | 0.6711 |
| C2 | C2_depthanything_shuffled_depth | logistic | random | 0.7437 | 0.8421 | 0.2815 | 0.3557 | 0.6080 | 0.4833 | 0.5233 | 0.3800 |
| C2 | C2_depthanything_shuffled_depth | logistic | test_residual_low | 0.6860 | 0.6748 | 0.3824 | 0.4043 | 0.2720 | 0.2087 | 0.5263 | 0.4480 |
| C2 | C2_midas | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C2 | C2_midas | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C2 | C2_midas | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C2 | C2_midas | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C2 | C2_midas | hgb | test_strong_cr | 0.7292 | 0.3723 | 0.1089 | 0.1140 | 0.1489 | 0.0906 | 0.8095 | 0.8395 |
| C2 | C2_midas | hgb | test_strong_official | 0.6000 | 0.2471 | 0.1120 | 0.1120 | 0.0304 | 0.0126 | 0.8400 | 0.8487 |
| C2 | C2_midas | hgb | random | 0.7506 | 0.8468 | 0.2101 | 0.3434 | 0.6182 | 0.4937 | 0.6262 | 0.3957 |
| C2 | C2_midas | hgb | test_residual_low | 0.7500 | 0.6585 | 0.1471 | 0.2872 | 0.2794 | 0.2185 | 0.7697 | 0.5400 |
| C2 | C2_midas | logistic | test_strong_cr | 0.5978 | 0.5851 | 0.3366 | 0.3246 | 0.1888 | 0.1282 | 0.6000 | 0.5617 |
| C2 | C2_midas | logistic | test_strong_official | 0.5735 | 0.4588 | 0.2320 | 0.2320 | 0.0631 | 0.0311 | 0.6600 | 0.6513 |
| C2 | C2_midas | logistic | random | 0.7578 | 0.8416 | 0.2520 | 0.3295 | 0.6095 | 0.4852 | 0.5396 | 0.3856 |
| C2 | C2_midas | logistic | test_residual_low | 0.6891 | 0.6667 | 0.3676 | 0.3936 | 0.2755 | 0.2123 | 0.5197 | 0.4480 |
| C2 | C2_midas_shuffled_depth | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C2 | C2_midas_shuffled_depth | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C2 | C2_midas_shuffled_depth | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C2 | C2_midas_shuffled_depth | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C2 | C2_midas_shuffled_depth | hgb | test_strong_cr | 0.7714 | 0.2872 | 0.0594 | 0.0702 | 0.1334 | 0.0770 | 0.8714 | 0.8765 |
| C2 | C2_midas_shuffled_depth | hgb | test_strong_official | 0.6389 | 0.2706 | 0.1040 | 0.1040 | 0.0315 | 0.0141 | 0.8240 | 0.8158 |
| C2 | C2_midas_shuffled_depth | hgb | random | 0.7491 | 0.8595 | 0.2101 | 0.3500 | 0.6291 | 0.5034 | 0.6227 | 0.3895 |
| C2 | C2_midas_shuffled_depth | hgb | test_residual_low | 0.7500 | 0.6585 | 0.1471 | 0.2872 | 0.2791 | 0.2182 | 0.7697 | 0.5440 |
| C2 | C2_midas_shuffled_depth | logistic | test_strong_cr | 0.5789 | 0.5851 | 0.3663 | 0.3509 | 0.1838 | 0.1223 | 0.5810 | 0.5494 |
| C2 | C2_midas_shuffled_depth | logistic | test_strong_official | 0.5588 | 0.4471 | 0.2400 | 0.2400 | 0.0617 | 0.0301 | 0.6600 | 0.6447 |
| C2 | C2_midas_shuffled_depth | logistic | random | 0.7477 | 0.8472 | 0.2990 | 0.3520 | 0.6094 | 0.4861 | 0.5178 | 0.3694 |
| C2 | C2_midas_shuffled_depth | logistic | test_residual_low | 0.6803 | 0.6748 | 0.3971 | 0.4149 | 0.2734 | 0.2091 | 0.5197 | 0.4440 |
| C3 | C3_both | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C3 | C3_both | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C3 | C3_both | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C3 | C3_both | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C3 | C3_both | hgb | test_strong_cr | 0.7111 | 0.3404 | 0.0990 | 0.1140 | 0.1365 | 0.0788 | 0.8238 | 0.8395 |
| C3 | C3_both | hgb | test_strong_official | 0.5833 | 0.2471 | 0.1200 | 0.1200 | 0.0254 | 0.0088 | 0.8240 | 0.8158 |
| C3 | C3_both | hgb | random | 0.7477 | 0.8530 | 0.2000 | 0.3526 | 0.6252 | 0.4989 | 0.6022 | 0.3644 |
| C3 | C3_both | hgb | test_residual_low | 0.7664 | 0.6667 | 0.1324 | 0.2660 | 0.2829 | 0.2221 | 0.7697 | 0.5480 |
| C3 | C3_both | logistic | test_strong_cr | 0.5870 | 0.5745 | 0.3465 | 0.3333 | 0.1852 | 0.1243 | 0.5952 | 0.5556 |
| C3 | C3_both | logistic | test_strong_official | 0.5455 | 0.4235 | 0.2400 | 0.2400 | 0.0547 | 0.0239 | 0.6760 | 0.6711 |
| C3 | C3_both | logistic | random | 0.7542 | 0.8477 | 0.2621 | 0.3394 | 0.6065 | 0.4839 | 0.5290 | 0.3744 |
| C3 | C3_both | logistic | test_residual_low | 0.6923 | 0.6585 | 0.3529 | 0.3830 | 0.2742 | 0.2122 | 0.5263 | 0.4600 |
| C3 | C3_both_shuffled_depth | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C3 | C3_both_shuffled_depth | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C3 | C3_both_shuffled_depth | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C3 | C3_both_shuffled_depth | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C3 | C3_both_shuffled_depth | hgb | test_strong_cr | 0.6429 | 0.3830 | 0.1683 | 0.1754 | 0.1489 | 0.0907 | 0.7524 | 0.7901 |
| C3 | C3_both_shuffled_depth | hgb | test_strong_official | 0.6087 | 0.3294 | 0.1440 | 0.1440 | 0.0428 | 0.0237 | 0.7720 | 0.7434 |
| C3 | C3_both_shuffled_depth | hgb | random | 0.7587 | 0.8366 | 0.1899 | 0.3263 | 0.6155 | 0.4893 | 0.6327 | 0.4064 |
| C3 | C3_both_shuffled_depth | hgb | test_residual_low | 0.7333 | 0.6260 | 0.1765 | 0.2979 | 0.2698 | 0.2072 | 0.7566 | 0.5480 |
| C3 | C3_both_shuffled_depth | logistic | test_strong_cr | 0.5900 | 0.6277 | 0.3663 | 0.3596 | 0.1885 | 0.1238 | 0.5619 | 0.5309 |
| C3 | C3_both_shuffled_depth | logistic | test_strong_official | 0.5195 | 0.4706 | 0.2960 | 0.2960 | 0.0571 | 0.0252 | 0.6320 | 0.6250 |
| C3 | C3_both_shuffled_depth | logistic | random | 0.7417 | 0.8444 | 0.2895 | 0.3594 | 0.6072 | 0.4839 | 0.5120 | 0.3638 |
| C3 | C3_both_shuffled_depth | logistic | test_residual_low | 0.6829 | 0.6829 | 0.3824 | 0.4149 | 0.2793 | 0.2161 | 0.5263 | 0.4400 |
| C3 | C3_depthanything | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C3 | C3_depthanything | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C3 | C3_depthanything | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C3 | C3_depthanything | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C3 | C3_depthanything | hgb | test_strong_cr | 0.6604 | 0.3723 | 0.1485 | 0.1579 | 0.1512 | 0.0949 | 0.7857 | 0.8025 |
| C3 | C3_depthanything | hgb | test_strong_official | 0.5897 | 0.2706 | 0.1280 | 0.1280 | 0.0326 | 0.0148 | 0.8080 | 0.7961 |
| C3 | C3_depthanything | hgb | random | 0.7390 | 0.8657 | 0.2199 | 0.3724 | 0.6267 | 0.5011 | 0.5942 | 0.3633 |
| C3 | C3_depthanything | hgb | test_residual_low | 0.7524 | 0.6423 | 0.1471 | 0.2766 | 0.2775 | 0.2190 | 0.7763 | 0.5560 |
| C3 | C3_depthanything | logistic | test_strong_cr | 0.5824 | 0.5638 | 0.3465 | 0.3333 | 0.1831 | 0.1219 | 0.6000 | 0.5617 |
| C3 | C3_depthanything | logistic | test_strong_official | 0.5692 | 0.4353 | 0.2240 | 0.2240 | 0.0587 | 0.0293 | 0.6800 | 0.6776 |
| C3 | C3_depthanything | logistic | random | 0.7459 | 0.8504 | 0.2621 | 0.3560 | 0.6064 | 0.4835 | 0.5290 | 0.3687 |
| C3 | C3_depthanything | logistic | test_residual_low | 0.6833 | 0.6667 | 0.3824 | 0.4043 | 0.2728 | 0.2101 | 0.5263 | 0.4560 |
| C3 | C3_depthanything_shuffled_depth | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C3 | C3_depthanything_shuffled_depth | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C3 | C3_depthanything_shuffled_depth | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C3 | C3_depthanything_shuffled_depth | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C3 | C3_depthanything_shuffled_depth | hgb | test_strong_cr | 0.6429 | 0.3830 | 0.1683 | 0.1754 | 0.1489 | 0.0907 | 0.7524 | 0.7901 |
| C3 | C3_depthanything_shuffled_depth | hgb | test_strong_official | 0.6222 | 0.3294 | 0.1360 | 0.1360 | 0.0430 | 0.0239 | 0.7760 | 0.7500 |
| C3 | C3_depthanything_shuffled_depth | hgb | random | 0.7614 | 0.8466 | 0.2101 | 0.3271 | 0.6221 | 0.4943 | 0.6179 | 0.3698 |
| C3 | C3_depthanything_shuffled_depth | hgb | test_residual_low | 0.7315 | 0.6423 | 0.1618 | 0.3085 | 0.2701 | 0.2089 | 0.7697 | 0.5400 |
| C3 | C3_depthanything_shuffled_depth | logistic | test_strong_cr | 0.5632 | 0.5213 | 0.3465 | 0.3333 | 0.1749 | 0.1116 | 0.6143 | 0.5741 |
| C3 | C3_depthanything_shuffled_depth | logistic | test_strong_official | 0.5143 | 0.4235 | 0.2720 | 0.2720 | 0.0553 | 0.0250 | 0.6600 | 0.6645 |
| C3 | C3_depthanything_shuffled_depth | logistic | random | 0.7369 | 0.8421 | 0.3091 | 0.3694 | 0.6086 | 0.4844 | 0.5060 | 0.3693 |
| C3 | C3_depthanything_shuffled_depth | logistic | test_residual_low | 0.6860 | 0.6748 | 0.3676 | 0.4043 | 0.2774 | 0.2110 | 0.5263 | 0.4440 |
| C3 | C3_midas | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C3 | C3_midas | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C3 | C3_midas | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C3 | C3_midas | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C3 | C3_midas | hgb | test_strong_cr | 0.7021 | 0.3511 | 0.1089 | 0.1228 | 0.1404 | 0.0862 | 0.8143 | 0.8333 |
| C3 | C3_midas | hgb | test_strong_official | 0.6279 | 0.3176 | 0.1280 | 0.1280 | 0.0416 | 0.0227 | 0.7880 | 0.7829 |
| C3 | C3_midas | hgb | random | 0.7550 | 0.8253 | 0.2104 | 0.3305 | 0.6109 | 0.4863 | 0.6195 | 0.3956 |
| C3 | C3_midas | hgb | test_residual_low | 0.7453 | 0.6423 | 0.1471 | 0.2872 | 0.2765 | 0.2134 | 0.7829 | 0.5480 |
| C3 | C3_midas | logistic | test_strong_cr | 0.6111 | 0.5851 | 0.3168 | 0.3070 | 0.1924 | 0.1318 | 0.6048 | 0.5679 |
| C3 | C3_midas | logistic | test_strong_official | 0.5634 | 0.4706 | 0.2480 | 0.2480 | 0.0635 | 0.0309 | 0.6440 | 0.6316 |
| C3 | C3_midas | logistic | random | 0.7570 | 0.8389 | 0.2520 | 0.3295 | 0.6091 | 0.4853 | 0.5396 | 0.3856 |
| C3 | C3_midas | logistic | test_residual_low | 0.6923 | 0.6585 | 0.3676 | 0.3830 | 0.2713 | 0.2089 | 0.5197 | 0.4560 |
| C3 | C3_midas_shuffled_depth | decision_tree | test_strong_cr | 1.0000 | 0.0213 | 0.0000 | 0.0000 | 0.0182 | 0.0000 | 1.0000 | 0.9938 |
| C3 | C3_midas_shuffled_depth | decision_tree | test_strong_official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| C3 | C3_midas_shuffled_depth | decision_tree | random | 0.6553 | 0.0992 | 0.0550 | 0.0397 | 0.0952 | 0.0539 | 0.9574 | 0.9414 |
| C3 | C3_midas_shuffled_depth | decision_tree | test_residual_low | 0.7500 | 0.1951 | 0.0147 | 0.0851 | 0.1020 | 0.0665 | 0.9934 | 0.8680 |
| C3 | C3_midas_shuffled_depth | hgb | test_strong_cr | 0.7317 | 0.3191 | 0.0891 | 0.0965 | 0.1353 | 0.0776 | 0.8333 | 0.8580 |
| C3 | C3_midas_shuffled_depth | hgb | test_strong_official | 0.6279 | 0.3176 | 0.1280 | 0.1280 | 0.0455 | 0.0261 | 0.7760 | 0.7632 |
| C3 | C3_midas_shuffled_depth | hgb | random | 0.7519 | 0.8358 | 0.1902 | 0.3368 | 0.6124 | 0.4867 | 0.6317 | 0.3956 |
| C3 | C3_midas_shuffled_depth | hgb | test_residual_low | 0.7576 | 0.6098 | 0.1176 | 0.2553 | 0.2680 | 0.2089 | 0.8092 | 0.5880 |
| C3 | C3_midas_shuffled_depth | logistic | test_strong_cr | 0.5816 | 0.6064 | 0.3663 | 0.3596 | 0.1866 | 0.1239 | 0.5667 | 0.5432 |
| C3 | C3_midas_shuffled_depth | logistic | test_strong_official | 0.5588 | 0.4471 | 0.2400 | 0.2400 | 0.0607 | 0.0332 | 0.6680 | 0.6513 |
| C3 | C3_midas_shuffled_depth | logistic | random | 0.7499 | 0.8452 | 0.2794 | 0.3489 | 0.6044 | 0.4814 | 0.5217 | 0.3699 |
| C3 | C3_midas_shuffled_depth | logistic | test_residual_low | 0.6833 | 0.6667 | 0.3824 | 0.4043 | 0.2732 | 0.2101 | 0.5263 | 0.4520 |
