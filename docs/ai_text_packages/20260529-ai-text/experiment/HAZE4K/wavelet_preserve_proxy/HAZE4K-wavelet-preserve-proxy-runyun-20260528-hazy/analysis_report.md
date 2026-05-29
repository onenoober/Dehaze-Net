# HAZE4K Wavelet Preserve Proxy Audit

## Recommendation

- `do_not_train_wavelet_preserve_yet`

## Task Counts

| Task | Samples | Preserve | Intervene |
| --- | ---: | ---: | ---: |
| extreme_preserve_vs_intervene | 804 | 453 | 351 |
| oracle_lfv1_vs_best_alt | 1000 | 204 | 796 |

## Summary

| Task | Feature Set | Split Family | Gain vs LF-v1 | Recovery | Balanced Acc | Preserve Recall | Intervene Precision | Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| extreme_preserve_vs_intervene | hazy_basic | airlight_bin | 0.5445 | 0.4940 | 0.5443 | 0.4926 | 0.4841 | no |
| extreme_preserve_vs_intervene | hazy_basic | beta_bin | 0.5258 | 0.4732 | 0.5379 | 0.5077 | 0.4737 | no |
| extreme_preserve_vs_intervene | hazy_basic | random | 0.4158 | 0.3842 | 0.5340 | 0.5767 | 0.4536 | no |
| extreme_preserve_vs_intervene | hazy_wavelet | airlight_bin | 0.5697 | 0.5154 | 0.5834 | 0.5625 | 0.5233 | no |
| extreme_preserve_vs_intervene | hazy_wavelet | beta_bin | 0.5464 | 0.4909 | 0.5804 | 0.5814 | 0.5200 | no |
| extreme_preserve_vs_intervene | hazy_wavelet | random | 0.5693 | 0.5185 | 0.6090 | 0.6247 | 0.5541 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_metadata | airlight_bin | 0.5706 | 0.5196 | 0.5960 | 0.5631 | 0.5328 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_metadata | beta_bin | 0.5398 | 0.4835 | 0.5775 | 0.5596 | 0.5109 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_metadata | random | 0.5530 | 0.4965 | 0.5924 | 0.5889 | 0.5321 | no |
| extreme_preserve_vs_intervene | metadata_diagnostic | airlight_bin | 0.5970 | 0.5618 | 0.4949 | 0.3746 | 0.3391 | no |
| extreme_preserve_vs_intervene | metadata_diagnostic | beta_bin | 0.5756 | 0.5117 | 0.5358 | 0.4105 | 0.4746 | no |
| extreme_preserve_vs_intervene | metadata_diagnostic | random | 0.4676 | 0.4254 | 0.5486 | 0.5628 | 0.4931 | no |
| oracle_lfv1_vs_best_alt | hazy_basic | airlight_bin | 0.5643 | 0.5222 | 0.5444 | 0.4270 | 0.8228 | no |
| oracle_lfv1_vs_best_alt | hazy_basic | beta_bin | 0.6094 | 0.5639 | 0.5507 | 0.3995 | 0.8241 | no |
| oracle_lfv1_vs_best_alt | hazy_basic | random | 0.6256 | 0.5844 | 0.5621 | 0.4138 | 0.8266 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet | airlight_bin | 0.6410 | 0.5935 | 0.5637 | 0.4190 | 0.8312 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet | beta_bin | 0.6500 | 0.6004 | 0.5840 | 0.4534 | 0.8428 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet | random | 0.6428 | 0.6126 | 0.5779 | 0.4426 | 0.8362 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_metadata | airlight_bin | 0.6073 | 0.5630 | 0.5663 | 0.4486 | 0.8345 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_metadata | beta_bin | 0.6673 | 0.6167 | 0.5886 | 0.4488 | 0.8437 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_metadata | random | 0.6679 | 0.6039 | 0.5805 | 0.4165 | 0.8389 | no |
| oracle_lfv1_vs_best_alt | metadata_diagnostic | airlight_bin | 0.3812 | 0.3534 | 0.4881 | 0.6069 | 0.4610 | no |
| oracle_lfv1_vs_best_alt | metadata_diagnostic | beta_bin | 0.5986 | 0.5477 | 0.4792 | 0.3493 | 0.6242 | no |
| oracle_lfv1_vs_best_alt | metadata_diagnostic | random | 0.4185 | 0.3992 | 0.4705 | 0.4808 | 0.7693 | no |
