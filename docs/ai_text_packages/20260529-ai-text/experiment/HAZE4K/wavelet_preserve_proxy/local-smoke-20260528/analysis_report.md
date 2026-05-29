# HAZE4K Wavelet Preserve Proxy Audit

## Recommendation

- `proceed_to_wavelet_preserve_route_card`

## Task Counts

| Task | Samples | Preserve | Intervene |
| --- | ---: | ---: | ---: |
| extreme_preserve_vs_intervene | 36 | 18 | 18 |
| oracle_lfv1_vs_best_alt | 40 | 7 | 33 |

## Summary

| Task | Feature Set | Split Family | Gain vs LF-v1 | Recovery | Balanced Acc | Preserve Recall | Intervene Precision | Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| extreme_preserve_vs_intervene | hazy_basic | airlight_bin | 0.7139 | 0.5276 | 0.6012 | 0.5000 | 0.7778 | no |
| extreme_preserve_vs_intervene | hazy_basic | random | 0.7867 | 0.4442 | 0.7952 | 0.8333 | 0.8333 | no |
| extreme_preserve_vs_intervene | hazy_wavelet | airlight_bin | 0.8128 | 0.6004 | 0.7887 | 0.8750 | 0.9167 | no |
| extreme_preserve_vs_intervene | hazy_wavelet | random | 0.8038 | 0.5123 | 0.8286 | 1.0000 | 1.0000 | yes |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_metadata | airlight_bin | 0.8128 | 0.6004 | 0.7887 | 0.8750 | 0.9167 | no |
| extreme_preserve_vs_intervene | hazy_wavelet_plus_metadata | random | 0.7239 | 0.4759 | 0.5833 | 0.5000 | 0.5750 | no |
| extreme_preserve_vs_intervene | metadata_diagnostic | airlight_bin | 1.0511 | 0.7768 | 0.5625 | 0.1250 | 0.6500 | no |
| extreme_preserve_vs_intervene | metadata_diagnostic | random | 0.6986 | 0.5035 | 0.5446 | 0.3750 | 0.6696 | no |
| oracle_lfv1_vs_best_alt | hazy_basic | airlight_bin | 0.7071 | 0.5573 | 0.7054 | 0.7500 | 0.9000 | no |
| oracle_lfv1_vs_best_alt | hazy_basic | random | 0.6268 | 0.5151 | 0.6364 | 0.4500 | 0.9167 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet | airlight_bin | 0.6968 | 0.5544 | 0.6637 | 0.6250 | 0.8333 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet | random | 0.5731 | 0.5697 | 0.6861 | 0.6667 | 0.9375 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_metadata | airlight_bin | 0.7613 | 0.6068 | 0.6845 | 0.6250 | 0.8333 | no |
| oracle_lfv1_vs_best_alt | hazy_wavelet_plus_metadata | random | 0.3549 | 0.3734 | 0.7222 | 0.5000 | 1.0000 | no |
| oracle_lfv1_vs_best_alt | metadata_diagnostic | airlight_bin | 0.3455 | 0.2649 | 0.5387 | 0.6250 | 0.8333 | no |
| oracle_lfv1_vs_best_alt | metadata_diagnostic | random | 0.7403 | 0.4713 | 0.2273 | 0.4000 | 0.8333 | no |
