# HAZE4K Supervised Preserve Proxy Audit

## Recommendation

- `do_not_train_supervised_preserve_yet`

## Teacher Label Source

- Images: `3000`
- Patches: `12000`
- Baseline checkpoint step: `reused_features`
- LF-v1 checkpoint step: `reused_features`

## Task Counts

| Task | Samples | Preserve | Intervene |
| --- | ---: | ---: | ---: |
| patch_preserve_vs_intervene | 10055 | 4947 | 5108 |

## Summary

| Task | Feature Set | Head | Split | Gain | Recovery | Bal Acc | Preserve Recall | Intervene Precision | Strong CR Recall | Main | Stable |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| patch_preserve_vs_intervene | hazy_wavelet_plus_teacher_outputs | sklearn_logistic | airlight_bin | 0.1806 | 0.2129 | 0.5815 | 0.5847 | 0.5905 | 0.5339 | no | no |
| patch_preserve_vs_intervene | hazy_wavelet_plus_teacher_outputs | sklearn_logistic | beta_bin | 0.1881 | 0.2131 | 0.5845 | 0.5878 | 0.5927 | 0.5813 | no | no |
| patch_preserve_vs_intervene | hazy_wavelet_plus_teacher_outputs | sklearn_logistic | random_image | 0.1955 | 0.2288 | 0.5889 | 0.5951 | 0.5934 | 0.5347 | no | no |
