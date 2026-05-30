# HAZE4K BRFRC-v2 Representation Audit

## Recommendation

- `do_not_train_brf_v2_representation_yet`

## Source

- Images: `3000`
- Baseline checkpoint step: `90000`
- LF-v1 checkpoint step: `90000`
- Target grid: `8`

## Summary

| Feature Set | Head | Split | Cos | Wrong | LF Improve | LF-v1 Preserve | Strong-CR Preserve | Precision | Conf Corr | Gap Output | Gap Shuffled | Random Pass | Heldout Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A_output | ridge | airlight_bin | 0.3106 | 0.1724 | 0.5408 | 0.6237 | 0.2001 | 0.5824 | 0.3710 | 0.0000 | 0.0000 | no | no |
| A_output | ridge | beta_bin | 0.3237 | 0.1620 | 0.5526 | 0.6256 | 0.2069 | 0.6067 | 0.4451 | 0.0000 | 0.0000 | no | no |
| A_output | ridge | cr_strength_bin | 0.2791 | 0.1823 | 0.5110 | 0.5391 | 0.0423 | 0.5251 | 0.0446 | 0.0000 | 0.0000 | no | no |
| A_output | ridge | random | 0.3271 | 0.1538 | 0.5738 | 0.6485 | 0.2708 | 0.6285 | 0.4530 | 0.0000 | 0.0000 | no | no |
| A_output | tiny_mlp | airlight_bin | 0.0745 | 0.2993 | 0.1332 | 0.1782 | 0.0258 | 0.2793 | 0.2643 | 0.0000 | 0.0000 | no | no |
| A_output | tiny_mlp | beta_bin | 0.0778 | 0.2814 | 0.1203 | 0.1597 | 0.0137 | 0.2947 | 0.4078 | 0.0000 | 0.0000 | no | no |
| A_output | tiny_mlp | cr_strength_bin | 0.0559 | 0.3297 | 0.0527 | 0.0649 | 0.0010 | 0.0984 | 0.2783 | 0.0000 | 0.0000 | no | no |
| A_output | tiny_mlp | random | 0.0637 | 0.2978 | 0.0689 | 0.1081 | 0.0036 | 0.2248 | 0.4412 | 0.0000 | 0.0000 | no | no |
| B_cr_features | ridge | airlight_bin | 0.5682 | 0.0487 | 0.6203 | 0.7210 | 0.2664 | 0.6684 | 0.4322 | 0.2576 | 0.3625 | no | no |
| B_cr_features | ridge | beta_bin | 0.5989 | 0.0342 | 0.6565 | 0.7353 | 0.3706 | 0.7015 | 0.5470 | 0.2752 | 0.3902 | no | no |
| B_cr_features | ridge | cr_strength_bin | 0.5003 | 0.0767 | 0.5607 | 0.5821 | 0.0427 | 0.5785 | -0.0857 | 0.2212 | 0.3301 | no | no |
| B_cr_features | ridge | random | 0.6176 | 0.0302 | 0.6933 | 0.7739 | 0.4081 | 0.7479 | 0.5371 | 0.2905 | 0.4124 | no | no |
| B_cr_features | tiny_mlp | airlight_bin | 0.0629 | 0.2999 | 0.0319 | 0.0371 | 0.0069 | 0.0456 | 0.3694 | -0.0116 | 0.0242 | no | no |
| B_cr_features | tiny_mlp | beta_bin | 0.0640 | 0.3070 | 0.0369 | 0.0500 | 0.0080 | 0.0494 | 0.4161 | -0.0138 | 0.0318 | no | no |
| B_cr_features | tiny_mlp | cr_strength_bin | 0.0497 | 0.3230 | 0.0097 | 0.0094 | 0.0010 | 0.0135 | 0.2965 | -0.0063 | 0.0227 | no | no |
| B_cr_features | tiny_mlp | random | 0.0729 | 0.2827 | 0.0529 | 0.0737 | 0.0109 | 0.0766 | 0.4040 | 0.0092 | 0.0379 | no | no |
| B_cr_features_shuffled | ridge | airlight_bin | 0.2057 | 0.2608 | 0.2634 | 0.3306 | 0.0132 | 0.2979 | 0.1014 | -0.1049 | 0.0000 | no | no |
| B_cr_features_shuffled | ridge | beta_bin | 0.2087 | 0.2567 | 0.2744 | 0.3533 | 0.0220 | 0.3102 | 0.1078 | -0.1150 | 0.0000 | no | no |
| B_cr_features_shuffled | ridge | cr_strength_bin | 0.1703 | 0.2887 | 0.2477 | 0.2783 | 0.0030 | 0.2559 | -0.0834 | -0.1088 | 0.0000 | no | no |
| B_cr_features_shuffled | ridge | random | 0.2051 | 0.2476 | 0.2560 | 0.3277 | 0.0157 | 0.2894 | 0.1129 | -0.1219 | 0.0000 | no | no |
| B_cr_features_shuffled | tiny_mlp | airlight_bin | 0.0387 | 0.3550 | 0.0118 | 0.0203 | 0.0007 | 0.0175 | 0.5336 | -0.0358 | 0.0000 | no | no |
| B_cr_features_shuffled | tiny_mlp | beta_bin | 0.0322 | 0.3836 | 0.0156 | 0.0192 | 0.0069 | 0.0205 | 0.2558 | -0.0456 | 0.0000 | no | no |
| B_cr_features_shuffled | tiny_mlp | cr_strength_bin | 0.0270 | 0.3910 | 0.0050 | 0.0037 | 0.0007 | 0.0055 | 0.3598 | -0.0290 | 0.0000 | no | no |
| B_cr_features_shuffled | tiny_mlp | random | 0.0350 | 0.3720 | 0.0129 | 0.0208 | 0.0019 | 0.0149 | 0.3388 | -0.0287 | 0.0000 | no | no |
| C_lfv1_features | ridge | airlight_bin | 0.5889 | 0.0387 | 0.6589 | 0.7685 | 0.2923 | 0.7009 | 0.4633 | 0.2783 | 0.3836 | no | no |
| C_lfv1_features | ridge | beta_bin | 0.6265 | 0.0216 | 0.6908 | 0.7828 | 0.3647 | 0.7433 | 0.6115 | 0.3028 | 0.4200 | no | no |
| C_lfv1_features | ridge | cr_strength_bin | 0.5289 | 0.0627 | 0.5867 | 0.6287 | 0.0433 | 0.6047 | -0.0058 | 0.2498 | 0.3593 | no | no |
| C_lfv1_features | ridge | random | 0.6411 | 0.0262 | 0.7267 | 0.8310 | 0.4136 | 0.7773 | 0.5859 | 0.3140 | 0.4376 | no | no |
| C_lfv1_features | tiny_mlp | airlight_bin | 0.0721 | 0.2795 | 0.0442 | 0.0600 | 0.0092 | 0.0659 | 0.3742 | -0.0024 | 0.0368 | no | no |
| C_lfv1_features | tiny_mlp | beta_bin | 0.0664 | 0.2885 | 0.0368 | 0.0567 | 0.0049 | 0.0424 | 0.4370 | -0.0114 | 0.0359 | no | no |
| C_lfv1_features | tiny_mlp | cr_strength_bin | 0.0502 | 0.3130 | 0.0093 | 0.0097 | 0.0003 | 0.0121 | 0.2205 | -0.0057 | 0.0252 | no | no |
| C_lfv1_features | tiny_mlp | random | 0.0758 | 0.2813 | 0.0622 | 0.0805 | 0.0128 | 0.0817 | 0.3987 | 0.0121 | 0.0446 | no | no |
| C_lfv1_features_shuffled | ridge | airlight_bin | 0.2053 | 0.2607 | 0.2734 | 0.3453 | 0.0144 | 0.3061 | 0.0989 | -0.1053 | 0.0000 | no | no |
| C_lfv1_features_shuffled | ridge | beta_bin | 0.2065 | 0.2595 | 0.2796 | 0.3581 | 0.0225 | 0.3194 | 0.1100 | -0.1172 | 0.0000 | no | no |
| C_lfv1_features_shuffled | ridge | cr_strength_bin | 0.1696 | 0.2950 | 0.2433 | 0.2657 | 0.0020 | 0.2481 | -0.1151 | -0.1095 | 0.0000 | no | no |
| C_lfv1_features_shuffled | ridge | random | 0.2034 | 0.2618 | 0.2662 | 0.3551 | 0.0177 | 0.3040 | 0.0808 | -0.1236 | 0.0000 | no | no |
| C_lfv1_features_shuffled | tiny_mlp | airlight_bin | 0.0353 | 0.3723 | 0.0210 | 0.0140 | 0.0020 | 0.0314 | 0.4473 | -0.0393 | 0.0000 | no | no |
| C_lfv1_features_shuffled | tiny_mlp | beta_bin | 0.0305 | 0.3906 | 0.0146 | 0.0210 | 0.0000 | 0.0200 | 0.4709 | -0.0473 | 0.0000 | no | no |
| C_lfv1_features_shuffled | tiny_mlp | cr_strength_bin | 0.0250 | 0.3987 | 0.0060 | 0.0044 | 0.0000 | 0.0086 | 0.3953 | -0.0309 | 0.0000 | no | no |
| C_lfv1_features_shuffled | tiny_mlp | random | 0.0312 | 0.3729 | 0.0089 | 0.0113 | 0.0000 | 0.0142 | 0.5256 | -0.0325 | 0.0000 | no | no |
| D_feature_contrast | ridge | airlight_bin | 0.6015 | 0.0276 | 0.6585 | 0.7616 | 0.2931 | 0.7013 | 0.4639 | 0.2909 | 0.4022 | no | no |
| D_feature_contrast | ridge | beta_bin | 0.6282 | 0.0249 | 0.6694 | 0.7649 | 0.3633 | 0.7174 | 0.5792 | 0.3045 | 0.4284 | no | no |
| D_feature_contrast | ridge | cr_strength_bin | 0.5316 | 0.0673 | 0.5820 | 0.6224 | 0.0430 | 0.6014 | -0.0587 | 0.2525 | 0.3598 | no | no |
| D_feature_contrast | ridge | random | 0.6488 | 0.0222 | 0.7284 | 0.8321 | 0.4473 | 0.7702 | 0.5604 | 0.3217 | 0.4520 | no | no |
| D_feature_contrast | tiny_mlp | airlight_bin | 0.0660 | 0.3029 | 0.0275 | 0.0381 | 0.0029 | 0.0399 | 0.3675 | -0.0085 | 0.0357 | no | no |
| D_feature_contrast | tiny_mlp | beta_bin | 0.0682 | 0.2633 | 0.0211 | 0.0295 | 0.0020 | 0.0308 | 0.3509 | -0.0096 | 0.0370 | no | no |
| D_feature_contrast | tiny_mlp | cr_strength_bin | 0.0474 | 0.3217 | 0.0053 | 0.0033 | 0.0013 | 0.0055 | 0.3288 | -0.0086 | 0.0247 | no | no |
| D_feature_contrast | tiny_mlp | random | 0.0680 | 0.2800 | 0.0360 | 0.0539 | 0.0053 | 0.0469 | 0.4943 | 0.0043 | 0.0395 | no | no |
| D_feature_contrast_shuffled | ridge | airlight_bin | 0.1993 | 0.2591 | 0.2501 | 0.3211 | 0.0105 | 0.2867 | 0.0855 | -0.1113 | 0.0000 | no | no |
| D_feature_contrast_shuffled | ridge | beta_bin | 0.1999 | 0.2637 | 0.2491 | 0.3174 | 0.0090 | 0.2841 | 0.1005 | -0.1238 | 0.0000 | no | no |
| D_feature_contrast_shuffled | ridge | cr_strength_bin | 0.1717 | 0.2817 | 0.2260 | 0.2331 | 0.0013 | 0.2343 | -0.1058 | -0.1074 | 0.0000 | no | no |
| D_feature_contrast_shuffled | ridge | random | 0.1968 | 0.2684 | 0.2489 | 0.3113 | 0.0141 | 0.2859 | 0.0806 | -0.1303 | 0.0000 | no | no |
| D_feature_contrast_shuffled | tiny_mlp | airlight_bin | 0.0303 | 0.3979 | 0.0069 | 0.0120 | 0.0000 | 0.0112 | 0.4696 | -0.0442 | 0.0000 | no | no |
| D_feature_contrast_shuffled | tiny_mlp | beta_bin | 0.0312 | 0.3853 | 0.0110 | 0.0168 | 0.0000 | 0.0194 | 0.4340 | -0.0466 | 0.0000 | no | no |
| D_feature_contrast_shuffled | tiny_mlp | cr_strength_bin | 0.0226 | 0.4150 | 0.0010 | 0.0015 | 0.0000 | 0.0008 | 0.6251 | -0.0333 | 0.0000 | no | no |
| D_feature_contrast_shuffled | tiny_mlp | random | 0.0285 | 0.3876 | 0.0053 | 0.0079 | 0.0000 | 0.0084 | 0.6325 | -0.0351 | 0.0000 | no | no |

## Pass Line

- Random residual cosine >= `0.20` and output/shuffled cosine gap >= `0.05`.
- Wrong-direction rate <= `0.35`.
- LF MSE improved rate >= `0.55`.
- LF-v1 gain and strong-CR preservation recall >= `0.70`.
- Intervention precision >= `0.60` and confidence correlation >= `0.45`.
