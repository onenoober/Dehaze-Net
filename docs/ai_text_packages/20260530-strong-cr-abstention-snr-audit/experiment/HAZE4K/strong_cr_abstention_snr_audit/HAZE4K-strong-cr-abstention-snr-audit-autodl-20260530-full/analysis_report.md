# HAZE4K Strong-CR Abstention / Residual-SNR Audit

## Recommendation

- `do_not_train_abstention_brf_v3_yet`

## Source

- Images: `1000`
- CR checkpoint step: `90000`
- LF-v1 checkpoint step: `90000`
- ResidualCalib checkpoint step: `90000`
- CRPlus-v2 checkpoint step: ``
- CBRFRC-v1 checkpoint step: `10000`

## Residual-SNR Key Readouts

- strong_q4 target residual norm / weak_q1 target residual norm: `0.2583`
- strong_q4 sign flip rate: `0.0570`

## Probe Summary

| Feature Set | Head | Split | Strong Preserve | Strong False Int | Precision | LF-v1 Gain Keep | LF-v1 Regr Rescue | Corr | Sim-CR | Sim-LFv1 | Shuffle Gap | Random Pass | Heldout Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A_output | calibrated_logistic | airlight_bin | 0.7704 | 0.2296 | 0.4388 | 0.2175 | 0.7738 | -0.0106 | 0.0653 | -0.1179 | 0.0730 | no | no |
| A_output | calibrated_logistic | beta_bin | 0.3721 | 0.6279 | 0.4248 | 0.6406 | 0.3775 | 0.0088 | 0.1775 | -0.0215 | 0.0789 | no | no |
| A_output | calibrated_logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.5556 | 0.0481 | 0.9789 | 0.1347 | 0.0124 | 0.0354 | 0.1886 | no | no |
| A_output | calibrated_logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.5192 | 0.2362 | 0.8409 | 0.0651 | 0.1467 | -0.2456 | 0.2500 | no | no |
| A_output | calibrated_logistic | cr_strength_bin=strong_cr_q4 | 0.0000 | 1.0000 | 0.3240 | 1.0000 | 0.0000 | -0.1917 | -0.0521 | 0.0000 | -0.0037 | no | no |
| A_output | calibrated_logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5159 | 0.5433 | 0.5942 | 0.1329 | 0.3568 | -0.1370 | 0.0300 | no | no |
| A_output | calibrated_logistic | random | 0.3333 | 0.6667 | 0.2512 | 0.6697 | 0.3333 | -0.0320 | 0.1181 | -0.0940 | -0.1339 | no | no |
| A_output | hgb | airlight_bin | 0.4734 | 0.5266 | 0.4820 | 0.7383 | 0.4365 | 0.3252 | 0.2588 | 0.0756 | 0.0725 | no | no |
| A_output | hgb | beta_bin | 0.4364 | 0.5636 | 0.4909 | 0.8189 | 0.4300 | 0.3591 | 0.3114 | 0.1123 | 0.0784 | no | no |
| A_output | hgb | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.4335 | 0.8077 | 0.3895 | 0.2957 | 0.1274 | 0.1504 | 0.0458 | no | no |
| A_output | hgb | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4197 | 0.7795 | 0.2727 | 0.1651 | 0.3788 | -0.0134 | 0.0066 | no | no |
| A_output | hgb | cr_strength_bin=strong_cr_q4 | 0.0651 | 0.9349 | 0.3389 | 0.9579 | 0.0600 | 0.2816 | -0.0289 | 0.0232 | 0.0083 | no | no |
| A_output | hgb | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5798 | 0.5354 | 0.6232 | 0.1102 | 0.3274 | -0.1664 | 0.0581 | no | no |
| A_output | hgb | random | 0.3599 | 0.6401 | 0.4922 | 0.7731 | 0.4274 | 0.3351 | 0.2902 | 0.0782 | 0.0722 | no | no |
| A_output | logistic | airlight_bin | 0.7255 | 0.2745 | 0.4673 | 0.3291 | 0.7626 | 0.1064 | 0.1051 | -0.0781 | 0.0821 | no | no |
| A_output | logistic | beta_bin | 0.6590 | 0.3410 | 0.4965 | 0.3652 | 0.7534 | 0.1422 | 0.1432 | -0.0559 | 0.0772 | no | no |
| A_output | logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.4632 | 0.4519 | 0.6632 | 0.1549 | 0.0507 | 0.0737 | 0.0703 | no | no |
| A_output | logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4286 | 0.2126 | 0.7273 | -0.0241 | 0.0810 | -0.3112 | 0.1319 | no | no |
| A_output | logistic | cr_strength_bin=strong_cr_q4 | 0.0828 | 0.9172 | 0.3404 | 0.9684 | 0.0900 | 0.2122 | -0.0003 | 0.0518 | -0.0280 | no | no |
| A_output | logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.6111 | 0.2047 | 0.8841 | 0.0884 | 0.1257 | -0.3681 | 0.0165 | no | no |
| A_output | logistic | random | 0.7297 | 0.2703 | 0.5215 | 0.3512 | 0.7515 | 0.1418 | 0.1417 | -0.0703 | 0.1161 | no | no |
| A_output | ridge_classifier | airlight_bin | 0.7060 | 0.2940 | 0.4604 | 0.3321 | 0.7652 | 0.1053 | 0.1019 | -0.0813 | 0.0580 | no | no |
| A_output | ridge_classifier | beta_bin | 0.7734 | 0.2266 | 0.5221 | 0.3012 | 0.8249 | 0.1387 | 0.1127 | -0.0863 | 0.1053 | no | no |
| A_output | ridge_classifier | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.4796 | 0.4615 | 0.6421 | 0.1465 | 0.0397 | 0.0627 | 0.0582 | no | no |
| A_output | ridge_classifier | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4902 | 0.2047 | 0.7955 | -0.0038 | 0.1173 | -0.2750 | 0.0868 | no | no |
| A_output | ridge_classifier | cr_strength_bin=strong_cr_q4 | 0.1834 | 0.8166 | 0.3521 | 0.9053 | 0.2100 | 0.1253 | 0.0246 | 0.0767 | -0.0291 | no | no |
| A_output | ridge_classifier | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.4848 | 0.1417 | 0.8406 | 0.0146 | 0.0458 | -0.4480 | -0.0410 | no | no |
| A_output | ridge_classifier | random | 0.7850 | 0.2150 | 0.5373 | 0.3241 | 0.8023 | 0.1427 | 0.1423 | -0.0698 | 0.1085 | no | no |
| D_feature_contrast | calibrated_logistic | airlight_bin | 0.4886 | 0.5114 | 0.4295 | 0.5628 | 0.5032 | 0.0133 | 0.1166 | -0.0666 | 0.0637 | no | no |
| D_feature_contrast | calibrated_logistic | beta_bin | 0.5539 | 0.4461 | 0.4905 | 0.4681 | 0.5564 | 0.0095 | 0.1096 | -0.0894 | 0.1447 | no | no |
| D_feature_contrast | calibrated_logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.3669 | 1.0000 | 0.0000 | -0.2138 | -0.0230 | 0.0000 | 0.0000 | no | no |
| D_feature_contrast | calibrated_logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.5000 | 0.4409 | 0.6705 | 0.1475 | 0.2744 | -0.1178 | 0.2308 | no | no |
| D_feature_contrast | calibrated_logistic | cr_strength_bin=strong_cr_q4 | 0.0000 | 1.0000 | 0.3240 | 1.0000 | 0.0000 | -0.2409 | -0.0521 | 0.0000 | -0.0037 | no | no |
| D_feature_contrast | calibrated_logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5833 | 0.0551 | 0.9710 | -0.0534 | 0.0513 | -0.4425 | 0.0974 | no | no |
| D_feature_contrast | calibrated_logistic | random | 0.3070 | 0.6930 | 0.4027 | 0.6817 | 0.3289 | 0.0425 | 0.1252 | -0.0868 | 0.0176 | no | no |
| D_feature_contrast | hgb | airlight_bin | 0.4311 | 0.5689 | 0.4833 | 0.7700 | 0.4418 | 0.2992 | 0.2674 | 0.0842 | 0.0738 | no | no |
| D_feature_contrast | hgb | beta_bin | 0.3927 | 0.6073 | 0.5051 | 0.8033 | 0.4841 | 0.3470 | 0.3242 | 0.1251 | 0.0926 | no | no |
| D_feature_contrast | hgb | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.4393 | 0.7981 | 0.4211 | 0.2829 | 0.1059 | 0.1289 | 0.0516 | no | no |
| D_feature_contrast | hgb | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4393 | 0.7402 | 0.3977 | 0.1853 | 0.3816 | -0.0107 | 0.0263 | no | no |
| D_feature_contrast | hgb | cr_strength_bin=strong_cr_q4 | 0.0769 | 0.9231 | 0.3418 | 0.9684 | 0.0900 | 0.3071 | -0.0096 | 0.0425 | 0.0112 | no | no |
| D_feature_contrast | hgb | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5755 | 0.4488 | 0.5942 | 0.0540 | 0.2450 | -0.2488 | 0.0537 | no | no |
| D_feature_contrast | hgb | random | 0.4041 | 0.5959 | 0.5153 | 0.7702 | 0.4967 | 0.3656 | 0.3256 | 0.1136 | 0.0953 | no | no |
| D_feature_contrast | logistic | airlight_bin | 0.7227 | 0.2773 | 0.5988 | 0.5817 | 0.7384 | 0.2994 | 0.2714 | 0.0882 | 0.2136 | no | no |
| D_feature_contrast | logistic | beta_bin | 0.7016 | 0.2984 | 0.6145 | 0.6148 | 0.7662 | 0.3477 | 0.3201 | 0.1211 | 0.1951 | no | no |
| D_feature_contrast | logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.5431 | 0.6442 | 0.7368 | 0.3442 | 0.2419 | 0.2649 | 0.1502 | no | no |
| D_feature_contrast | logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.5586 | 0.5276 | 0.6932 | 0.2236 | 0.3545 | -0.0378 | 0.2619 | no | no |
| D_feature_contrast | logistic | cr_strength_bin=strong_cr_q4 | 0.3432 | 0.6568 | 0.3833 | 0.8211 | 0.4400 | 0.3408 | 0.1386 | 0.1907 | 0.0149 | no | no |
| D_feature_contrast | logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5096 | 0.4409 | 0.6232 | -0.0489 | 0.1883 | -0.3055 | -0.0850 | no | no |
| D_feature_contrast | logistic | random | 0.7121 | 0.2879 | 0.6333 | 0.5949 | 0.7831 | 0.3099 | 0.3040 | 0.0919 | 0.2279 | no | no |
| D_feature_contrast | ridge_classifier | airlight_bin | 0.6883 | 0.3117 | 0.5625 | 0.6098 | 0.6927 | 0.2581 | 0.2705 | 0.0873 | 0.1600 | no | no |
| D_feature_contrast | ridge_classifier | beta_bin | 0.7005 | 0.2995 | 0.5342 | 0.6764 | 0.6428 | 0.2819 | 0.2988 | 0.0998 | 0.1174 | no | no |
| D_feature_contrast | ridge_classifier | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.5333 | 0.5096 | 0.8000 | 0.2526 | 0.1531 | 0.1761 | 0.1119 | no | no |
| D_feature_contrast | ridge_classifier | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4777 | 0.7087 | 0.4886 | 0.2682 | 0.4303 | 0.0380 | 0.0743 | no | no |
| D_feature_contrast | ridge_classifier | cr_strength_bin=strong_cr_q4 | 0.4615 | 0.5385 | 0.4277 | 0.7474 | 0.5000 | 0.3037 | 0.1630 | 0.2151 | 0.0465 | no | no |
| D_feature_contrast | ridge_classifier | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5000 | 0.5433 | 0.5217 | 0.0466 | 0.2880 | -0.2058 | -0.0259 | no | no |
| D_feature_contrast | ridge_classifier | random | 0.7483 | 0.2517 | 0.6083 | 0.5809 | 0.7422 | 0.2586 | 0.2770 | 0.0650 | 0.1796 | no | no |
| E_abstention_risk | calibrated_logistic | airlight_bin | 0.3602 | 0.6398 | 0.4610 | 0.6640 | 0.4356 | 0.0387 | 0.1535 | -0.0297 | 0.0953 | no | no |
| E_abstention_risk | calibrated_logistic | beta_bin | 0.6762 | 0.3238 | 0.4739 | 0.3808 | 0.6924 | 0.1028 | 0.1117 | -0.0874 | 0.1281 | no | no |
| E_abstention_risk | calibrated_logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.3669 | 1.0000 | 0.0000 | -0.1663 | -0.0230 | 0.0000 | 0.0000 | no | no |
| E_abstention_risk | calibrated_logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.5439 | 0.5669 | 0.6705 | 0.2008 | 0.3601 | -0.0321 | 0.2746 | no | no |
| E_abstention_risk | calibrated_logistic | cr_strength_bin=strong_cr_q4 | 0.0000 | 1.0000 | 0.3240 | 1.0000 | 0.0000 | -0.2491 | -0.0521 | 0.0000 | -0.0037 | no | no |
| E_abstention_risk | calibrated_logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.3333 | 0.0315 | 0.9855 | -0.0020 | 0.0209 | -0.4729 | -0.1526 | no | no |
| E_abstention_risk | calibrated_logistic | random | 0.6053 | 0.3947 | 0.6423 | 0.3937 | 0.6579 | 0.0119 | 0.1311 | -0.0809 | 0.2573 | no | no |
| E_abstention_risk | hgb | airlight_bin | 0.4356 | 0.5644 | 0.4921 | 0.7443 | 0.4462 | 0.3207 | 0.2453 | 0.0620 | 0.0826 | no | no |
| E_abstention_risk | hgb | beta_bin | 0.4054 | 0.5946 | 0.5053 | 0.7988 | 0.4947 | 0.3401 | 0.3175 | 0.1184 | 0.0928 | no | no |
| E_abstention_risk | hgb | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.4286 | 0.7788 | 0.3789 | 0.2443 | 0.0731 | 0.0961 | 0.0408 | no | no |
| E_abstention_risk | hgb | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4419 | 0.7165 | 0.3636 | 0.1246 | 0.3361 | -0.0561 | 0.0288 | no | no |
| E_abstention_risk | hgb | cr_strength_bin=strong_cr_q4 | 0.1065 | 0.8935 | 0.3491 | 0.9789 | 0.1500 | 0.3086 | 0.0605 | 0.1126 | 0.0186 | no | no |
| E_abstention_risk | hgb | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.6044 | 0.4173 | 0.7246 | 0.0328 | 0.2558 | -0.2380 | 0.0827 | no | no |
| E_abstention_risk | hgb | random | 0.4871 | 0.5129 | 0.5319 | 0.7425 | 0.5717 | 0.3495 | 0.3222 | 0.1102 | 0.1119 | no | no |
| E_abstention_risk | logistic | airlight_bin | 0.6930 | 0.3070 | 0.6051 | 0.6117 | 0.7526 | 0.3351 | 0.3101 | 0.1269 | 0.2199 | no | no |
| E_abstention_risk | logistic | beta_bin | 0.7505 | 0.2495 | 0.6028 | 0.6500 | 0.7369 | 0.3752 | 0.3351 | 0.1361 | 0.1834 | no | yes |
| E_abstention_risk | logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.5652 | 0.5192 | 0.8000 | 0.3226 | 0.1929 | 0.2159 | 0.1724 | no | no |
| E_abstention_risk | logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.5645 | 0.6063 | 0.6932 | 0.2598 | 0.4001 | 0.0078 | 0.2678 | no | no |
| E_abstention_risk | logistic | cr_strength_bin=strong_cr_q4 | 0.2781 | 0.7219 | 0.3744 | 0.8842 | 0.3600 | 0.3374 | 0.1368 | 0.1889 | 0.0059 | no | no |
| E_abstention_risk | logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.4732 | 0.4567 | 0.5507 | -0.0055 | 0.2317 | -0.2621 | -0.1214 | no | no |
| E_abstention_risk | logistic | random | 0.7385 | 0.2615 | 0.6631 | 0.5952 | 0.8169 | 0.3498 | 0.3294 | 0.1174 | 0.2578 | no | no |
| E_abstention_risk | ridge_classifier | airlight_bin | 0.6024 | 0.3976 | 0.5341 | 0.6447 | 0.6231 | 0.2350 | 0.2651 | 0.0819 | 0.1316 | no | no |
| E_abstention_risk | ridge_classifier | beta_bin | 0.6353 | 0.3647 | 0.5186 | 0.6775 | 0.5981 | 0.2479 | 0.2630 | 0.0640 | 0.1018 | no | no |
| E_abstention_risk | ridge_classifier | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.5225 | 0.6058 | 0.7158 | 0.3080 | 0.1979 | 0.2209 | 0.1011 | no | no |
| E_abstention_risk | ridge_classifier | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4762 | 0.6535 | 0.5000 | 0.2200 | 0.3624 | -0.0299 | 0.0728 | no | no |
| E_abstention_risk | ridge_classifier | cr_strength_bin=strong_cr_q4 | 0.4734 | 0.5266 | 0.4295 | 0.6947 | 0.4500 | 0.2684 | 0.1273 | 0.1794 | 0.0483 | no | no |
| E_abstention_risk | ridge_classifier | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5000 | 0.4252 | 0.6812 | 0.0796 | 0.2583 | -0.2355 | -0.0259 | no | no |
| E_abstention_risk | ridge_classifier | random | 0.6179 | 0.3821 | 0.5481 | 0.6528 | 0.6493 | 0.3008 | 0.2969 | 0.0848 | 0.1193 | no | no |
| E_abstention_risk_shuffled | calibrated_logistic | airlight_bin | 0.3465 | 0.6535 | 0.3657 | 0.6364 | 0.3730 | 0.0375 | 0.1301 | -0.0531 | 0.0000 | no | no |
| E_abstention_risk_shuffled | calibrated_logistic | beta_bin | 0.3699 | 0.6301 | 0.3458 | 0.6269 | 0.3700 | -0.0265 | 0.1168 | -0.0822 | 0.0000 | no | no |
| E_abstention_risk_shuffled | calibrated_logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.3669 | 1.0000 | 0.0000 | 0.0404 | -0.0230 | 0.0000 | 0.0000 | no | no |
| E_abstention_risk_shuffled | calibrated_logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.2692 | 0.2047 | 0.7500 | 0.1296 | 0.1191 | -0.2732 | 0.0000 | no | no |
| E_abstention_risk_shuffled | calibrated_logistic | cr_strength_bin=strong_cr_q4 | 0.0533 | 0.9467 | 0.3277 | 0.9474 | 0.0500 | 0.1065 | -0.0401 | 0.0120 | 0.0000 | no | no |
| E_abstention_risk_shuffled | calibrated_logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.4859 | 0.5433 | 0.4058 | 0.1437 | 0.2902 | -0.2036 | 0.0000 | no | no |
| E_abstention_risk_shuffled | calibrated_logistic | random | 0.0000 | 1.0000 | 0.3850 | 1.0000 | 0.0000 | -0.0025 | 0.2120 | 0.0000 | 0.0000 | no | no |
| E_abstention_risk_shuffled | hgb | airlight_bin | 0.2319 | 0.7681 | 0.4095 | 0.8321 | 0.2873 | 0.1218 | 0.1988 | 0.0156 | 0.0000 | no | no |
| E_abstention_risk_shuffled | hgb | beta_bin | 0.2293 | 0.7707 | 0.4125 | 0.8227 | 0.2415 | 0.1264 | 0.2033 | 0.0042 | 0.0000 | no | no |
| E_abstention_risk_shuffled | hgb | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.3878 | 0.8462 | 0.2211 | 0.0982 | 0.0299 | 0.0529 | 0.0000 | no | no |
| E_abstention_risk_shuffled | hgb | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4130 | 0.7717 | 0.2841 | 0.0363 | 0.3606 | -0.0317 | 0.0000 | no | no |
| E_abstention_risk_shuffled | hgb | cr_strength_bin=strong_cr_q4 | 0.0414 | 0.9586 | 0.3306 | 0.9684 | 0.0400 | 0.1692 | -0.0308 | 0.0213 | 0.0000 | no | no |
| E_abstention_risk_shuffled | hgb | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5217 | 0.5118 | 0.5507 | 0.0797 | 0.2570 | -0.2368 | 0.0000 | no | no |
| E_abstention_risk_shuffled | hgb | random | 0.1742 | 0.8258 | 0.4200 | 0.8421 | 0.2213 | 0.1486 | 0.2254 | 0.0133 | 0.0000 | no | no |
| E_abstention_risk_shuffled | logistic | airlight_bin | 0.5658 | 0.4342 | 0.3852 | 0.4284 | 0.6113 | 0.0474 | 0.1104 | -0.0729 | 0.0000 | no | no |
| E_abstention_risk_shuffled | logistic | beta_bin | 0.6277 | 0.3723 | 0.4194 | 0.4271 | 0.6511 | 0.0858 | 0.1395 | -0.0595 | 0.0000 | no | no |
| E_abstention_risk_shuffled | logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.3929 | 0.4615 | 0.5368 | 0.0222 | -0.0113 | 0.0117 | 0.0000 | no | no |
| E_abstention_risk_shuffled | logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.2967 | 0.3622 | 0.6364 | 0.0376 | 0.1718 | -0.2204 | 0.0000 | no | no |
| E_abstention_risk_shuffled | logistic | cr_strength_bin=strong_cr_q4 | 0.4320 | 0.5680 | 0.3684 | 0.6737 | 0.4500 | 0.0788 | 0.0171 | 0.0692 | 0.0000 | no | no |
| E_abstention_risk_shuffled | logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5946 | 0.3622 | 0.7681 | 0.1599 | 0.2445 | -0.2493 | 0.0000 | no | no |
| E_abstention_risk_shuffled | logistic | random | 0.5100 | 0.4900 | 0.4054 | 0.4832 | 0.5399 | 0.0270 | 0.1149 | -0.0972 | 0.0000 | no | no |
| E_abstention_risk_shuffled | ridge_classifier | airlight_bin | 0.4324 | 0.5676 | 0.4025 | 0.6198 | 0.4546 | 0.0851 | 0.1437 | -0.0395 | 0.0000 | no | no |
| E_abstention_risk_shuffled | ridge_classifier | beta_bin | 0.4823 | 0.5177 | 0.4168 | 0.6227 | 0.4372 | 0.1012 | 0.1644 | -0.0347 | 0.0000 | no | no |
| E_abstention_risk_shuffled | ridge_classifier | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.4214 | 0.6250 | 0.5263 | 0.1084 | 0.0616 | 0.0846 | 0.0000 | no | no |
| E_abstention_risk_shuffled | ridge_classifier | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.4034 | 0.4646 | 0.5341 | 0.0184 | 0.1898 | -0.2025 | 0.0000 | no | no |
| E_abstention_risk_shuffled | ridge_classifier | cr_strength_bin=strong_cr_q4 | 0.3373 | 0.6627 | 0.3812 | 0.8000 | 0.3600 | 0.1984 | 0.0672 | 0.1193 | 0.0000 | no | no |
| E_abstention_risk_shuffled | ridge_classifier | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.5259 | 0.4882 | 0.5072 | 0.0489 | 0.2370 | -0.2568 | 0.0000 | no | no |
| E_abstention_risk_shuffled | ridge_classifier | random | 0.3771 | 0.6229 | 0.4288 | 0.6451 | 0.4051 | 0.0439 | 0.1530 | -0.0590 | 0.0000 | no | no |
| F_diagnostic_leakage | calibrated_logistic | airlight_bin | 0.9378 | 0.0622 | 0.8983 | 0.7446 | 0.9904 | 0.5816 | 0.5137 | 0.3305 | 0.5325 | no | no |
| F_diagnostic_leakage | calibrated_logistic | beta_bin | 0.9321 | 0.0679 | 0.9001 | 0.7480 | 0.9695 | 0.5893 | 0.5283 | 0.3292 | 0.5543 | no | no |
| F_diagnostic_leakage | calibrated_logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.9479 | 0.7788 | 0.9895 | 0.6268 | 0.4521 | 0.4751 | 0.5810 | no | no |
| F_diagnostic_leakage | calibrated_logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.8785 | 0.7244 | 0.9886 | 0.5669 | 0.6455 | 0.2533 | 0.6093 | no | no |
| F_diagnostic_leakage | calibrated_logistic | cr_strength_bin=strong_cr_q4 | 0.5799 | 0.4201 | 0.5329 | 0.8842 | 0.7300 | 0.6393 | 0.3335 | 0.3856 | 0.2052 | no | no |
| F_diagnostic_leakage | calibrated_logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.8054 | 0.9055 | 0.9275 | 0.5786 | 0.7581 | 0.2643 | 0.3195 | no | no |
| F_diagnostic_leakage | calibrated_logistic | random | 0.9275 | 0.0725 | 0.9021 | 0.7293 | 0.9781 | 0.5466 | 0.5061 | 0.2941 | 0.5171 | no | no |
| F_diagnostic_leakage | hgb | airlight_bin | 1.0000 | 0.0000 | 1.0000 | 0.6845 | 1.0000 | 0.5587 | 0.4729 | 0.2897 | 0.5905 | no | no |
| F_diagnostic_leakage | hgb | beta_bin | 1.0000 | 0.0000 | 1.0000 | 0.7002 | 1.0000 | 0.5663 | 0.4956 | 0.2966 | 0.5875 | no | no |
| F_diagnostic_leakage | hgb | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 1.0000 | 0.7500 | 1.0000 | 0.6125 | 0.4390 | 0.4620 | 0.6122 | no | no |
| F_diagnostic_leakage | hgb | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 1.0000 | 0.6299 | 1.0000 | 0.5255 | 0.5600 | 0.1678 | 0.5870 | no | no |
| F_diagnostic_leakage | hgb | cr_strength_bin=strong_cr_q4 | 1.0000 | 0.0000 | 1.0000 | 0.6737 | 1.0000 | 0.5937 | 0.3507 | 0.4028 | 0.6694 | no | no |
| F_diagnostic_leakage | hgb | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 1.0000 | 0.7480 | 1.0000 | 0.5141 | 0.6338 | 0.1400 | 0.4783 | no | no |
| F_diagnostic_leakage | hgb | random | 1.0000 | 0.0000 | 1.0000 | 0.6804 | 1.0000 | 0.5271 | 0.4741 | 0.2620 | 0.5800 | no | no |
| F_diagnostic_leakage | logistic | airlight_bin | 0.9678 | 0.0322 | 0.9479 | 0.7112 | 0.9976 | 0.5695 | 0.4922 | 0.3090 | 0.5627 | no | no |
| F_diagnostic_leakage | logistic | beta_bin | 0.9764 | 0.0236 | 0.9479 | 0.7298 | 0.9892 | 0.5802 | 0.5212 | 0.3221 | 0.5286 | no | no |
| F_diagnostic_leakage | logistic | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.9785 | 0.7596 | 1.0000 | 0.6205 | 0.4455 | 0.4684 | 0.5856 | no | no |
| F_diagnostic_leakage | logistic | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.9307 | 0.6850 | 1.0000 | 0.5482 | 0.6277 | 0.2354 | 0.6340 | no | no |
| F_diagnostic_leakage | logistic | cr_strength_bin=strong_cr_q4 | 0.9112 | 0.0888 | 0.8438 | 0.7263 | 1.0000 | 0.6090 | 0.3701 | 0.4222 | 0.4753 | no | no |
| F_diagnostic_leakage | logistic | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.8759 | 0.8268 | 0.9420 | 0.5528 | 0.6750 | 0.1812 | 0.2813 | no | no |
| F_diagnostic_leakage | logistic | random | 0.9599 | 0.0401 | 0.9573 | 0.6957 | 0.9891 | 0.5357 | 0.4882 | 0.2761 | 0.5519 | no | no |
| F_diagnostic_leakage | ridge_classifier | airlight_bin | 0.8113 | 0.1887 | 0.7402 | 0.7769 | 0.8178 | 0.5614 | 0.4436 | 0.2604 | 0.3377 | no | no |
| F_diagnostic_leakage | ridge_classifier | beta_bin | 0.8359 | 0.1641 | 0.8012 | 0.7523 | 0.8357 | 0.5681 | 0.4481 | 0.2490 | 0.3844 | no | no |
| F_diagnostic_leakage | ridge_classifier | cr_strength_bin=midhigh_cr_q3 | 0.0000 | 0.0000 | 0.8053 | 0.7788 | 0.8947 | 0.6138 | 0.4183 | 0.4413 | 0.3839 | no | no |
| F_diagnostic_leakage | ridge_classifier | cr_strength_bin=midlow_cr_q2 | 0.0000 | 0.0000 | 0.6267 | 0.7953 | 0.6818 | 0.5339 | 0.5619 | 0.1696 | 0.2233 | no | no |
| F_diagnostic_leakage | ridge_classifier | cr_strength_bin=strong_cr_q4 | 0.6923 | 0.3077 | 0.6090 | 0.7368 | 0.7500 | 0.5984 | 0.2637 | 0.3158 | 0.2278 | no | no |
| F_diagnostic_leakage | ridge_classifier | cr_strength_bin=weak_cr_q1 | 0.0000 | 0.0000 | 0.7895 | 0.8583 | 0.8116 | 0.5248 | 0.6618 | 0.1680 | 0.2636 | no | no |
| F_diagnostic_leakage | ridge_classifier | random | 0.8285 | 0.1715 | 0.7819 | 0.7456 | 0.8499 | 0.5290 | 0.4350 | 0.2230 | 0.3531 | no | no |

## Pass Line

- strong_q4 preserve recall >= `0.75`.
- strong_q4 false-intervention rate <= `0.20`.
- intervention precision >= `0.60`.
- LF-v1 gain preservation >= `0.70`.
- confidence correlation >= `0.45`.
- simulated PSNR close to LF-v1, minimum delta `-0.02`.
