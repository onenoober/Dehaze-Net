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
| extreme_preserve_vs_intervene | activation_only | airlight_bin | 0.5606 | 0.5113 | 0.5409 | 0.4557 | 0.4822 | no |
| extreme_preserve_vs_intervene | activation_only | beta_bin | 0.5128 | 0.4597 | 0.5500 | 0.5080 | 0.4870 | no |
| extreme_preserve_vs_intervene | activation_only | random | 0.4612 | 0.4271 | 0.5546 | 0.5563 | 0.4703 | no |
| extreme_preserve_vs_intervene | hazy_basic | airlight_bin | 0.5445 | 0.4940 | 0.5443 | 0.4926 | 0.4841 | no |
| extreme_preserve_vs_intervene | hazy_basic | beta_bin | 0.5258 | 0.4732 | 0.5379 | 0.5077 | 0.4737 | no |
| extreme_preserve_vs_intervene | hazy_basic | random | 0.4158 | 0.3842 | 0.5340 | 0.5767 | 0.4536 | no |
| extreme_preserve_vs_intervene | hazy_wavelet | airlight_bin | 0.5697 | 0.5154 | 0.5834 | 0.5625 | 0.5233 | no |
| extreme_preserve_vs_intervene | hazy_wavelet | beta_bin | 0.5464 | 0.4909 | 0.5804 | 0.5814 | 0.5200 | no |
| extreme_preserve_vs_intervene | hazy_wavelet | random | 0.5693 | 0.5185 | 0.6090 | 0.6247 | 0.5541 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_activation_plus_metadata | airlight_bin | 0.5144 | 0.4636 | 0.5355 | 0.5162 | 0.4770 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_activation_plus_metadata | beta_bin | 0.5320 | 0.4693 | 0.5251 | 0.4899 | 0.4609 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_activation_plus_metadata | random | 0.4605 | 0.4123 | 0.5214 | 0.5295 | 0.4526 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_activation | airlight_bin | 0.5225 | 0.4696 | 0.5357 | 0.5156 | 0.4783 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_activation | beta_bin | 0.5369 | 0.4737 | 0.5217 | 0.4831 | 0.4565 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_activation | random | 0.5022 | 0.4600 | 0.5360 | 0.5187 | 0.4757 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_metadata | airlight_bin | 0.5706 | 0.5196 | 0.5960 | 0.5631 | 0.5328 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_metadata | beta_bin | 0.5398 | 0.4835 | 0.5775 | 0.5596 | 0.5109 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_metadata | random | 0.5530 | 0.4965 | 0.5924 | 0.5889 | 0.5321 | no |
| extreme_preserve_vs_intervene | metadata_diagnostic | airlight_bin | 0.5970 | 0.5618 | 0.4949 | 0.3746 | 0.3391 | no |
| extreme_preserve_vs_intervene | metadata_diagnostic | beta_bin | 0.5756 | 0.5117 | 0.5358 | 0.4105 | 0.4746 | no |
| extreme_preserve_vs_intervene | metadata_diagnostic | random | 0.4676 | 0.4254 | 0.5486 | 0.5628 | 0.4931 | no |
| oracle_lfv1_vs_best_alt | activation_only | airlight_bin | 0.5415 | 0.5014 | 0.5444 | 0.4783 | 0.8225 | no |
| oracle_lfv1_vs_best_alt | activation_only | beta_bin | 0.5661 | 0.5245 | 0.5028 | 0.4061 | 0.7943 | no |
| oracle_lfv1_vs_best_alt | activation_only | random | 0.5775 | 0.5295 | 0.4878 | 0.3714 | 0.7848 | no |
| oracle_lfv1_vs_best_alt | hazy_basic | airlight_bin | 0.5643 | 0.5222 | 0.5444 | 0.4270 | 0.8228 | no |
| oracle_lfv1_vs_best_alt | hazy_basic | beta_bin | 0.6094 | 0.5639 | 0.5507 | 0.3995 | 0.8241 | no |
| oracle_lfv1_vs_best_alt | hazy_basic | random | 0.5655 | 0.5240 | 0.5409 | 0.4655 | 0.8292 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet | airlight_bin | 0.6410 | 0.5935 | 0.5637 | 0.4190 | 0.8312 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet | beta_bin | 0.6500 | 0.6004 | 0.5840 | 0.4534 | 0.8428 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet | random | 0.6125 | 0.5804 | 0.5712 | 0.4277 | 0.8360 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_activation_plus_metadata | airlight_bin | 0.6346 | 0.5888 | 0.5630 | 0.3810 | 0.8306 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_activation_plus_metadata | beta_bin | 0.4548 | 0.4214 | 0.5447 | 0.5858 | 0.8280 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_activation_plus_metadata | random | 0.5608 | 0.5038 | 0.5004 | 0.4812 | 0.8022 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_activation | airlight_bin | 0.6497 | 0.6027 | 0.5681 | 0.3839 | 0.8329 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_activation | beta_bin | 0.4495 | 0.4168 | 0.5409 | 0.5900 | 0.8261 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_activation | random | 0.5548 | 0.5055 | 0.5197 | 0.4915 | 0.8128 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_metadata | airlight_bin | 0.6073 | 0.5630 | 0.5663 | 0.4486 | 0.8345 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_metadata | beta_bin | 0.6673 | 0.6167 | 0.5886 | 0.4488 | 0.8437 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_metadata | random | 0.6543 | 0.6082 | 0.6057 | 0.4260 | 0.8490 | no |
| oracle_lfv1_vs_best_alt | metadata_diagnostic | airlight_bin | 0.3812 | 0.3534 | 0.4881 | 0.6069 | 0.4610 | no |
| oracle_lfv1_vs_best_alt | metadata_diagnostic | beta_bin | 0.5986 | 0.5477 | 0.4792 | 0.3493 | 0.6242 | no |
| oracle_lfv1_vs_best_alt | metadata_diagnostic | random | 0.4700 | 0.4429 | 0.4724 | 0.4659 | 0.7681 | no |
