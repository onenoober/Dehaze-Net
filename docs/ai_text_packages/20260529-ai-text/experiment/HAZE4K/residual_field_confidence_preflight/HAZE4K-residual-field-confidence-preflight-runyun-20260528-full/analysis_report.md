# HAZE4K Residual Field Confidence Preflight

## Recommendation

- `do_not_train_residual_field_confidence_yet`

## Source

- Images: `3000`
- Patches: `12000`
- Baseline checkpoint step: `90000`
- LF-v1 checkpoint step: `90000`
- Low-frequency pool: `8`

## Summary

| Feature Set | Head | Split | Gain | Recovery | Preserve Recall | Regression Improve | Strong CR Improve | Intervene Precision | c Corr | Main | Stable |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hazy_wavelet | sklearn_hgb | airlight_bin | 0.7883 | 0.6929 | 0.6003 | 1.0000 | 1.0000 | 0.4917 | 0.2217 | no | no |
| hazy_wavelet | sklearn_hgb | beta_bin | 0.7847 | 0.6834 | 0.5932 | 1.0000 | 1.0000 | 0.4847 | 0.2012 | no | no |
| hazy_wavelet | sklearn_hgb | random_image | 0.7785 | 0.6868 | 0.5882 | 1.0000 | 1.0000 | 0.4894 | 0.2460 | no | no |
| hazy_wavelet | sklearn_ridge | airlight_bin | 0.7669 | 0.6743 | 0.5911 | 1.0000 | 1.0000 | 0.4512 | 0.0886 | no | no |
| hazy_wavelet | sklearn_ridge | beta_bin | 0.7643 | 0.6658 | 0.5835 | 1.0000 | 1.0000 | 0.4470 | 0.0741 | no | no |
| hazy_wavelet | sklearn_ridge | random_image | 0.7611 | 0.6760 | 0.5760 | 1.0000 | 1.0000 | 0.4485 | 0.1061 | no | no |
| hazy_wavelet_plus_teacher_outputs | sklearn_hgb | airlight_bin | 0.8156 | 0.7170 | 0.6083 | 1.0000 | 1.0000 | 0.5113 | 0.3553 | no | no |
| hazy_wavelet_plus_teacher_outputs | sklearn_hgb | beta_bin | 0.8179 | 0.7124 | 0.6060 | 1.0000 | 1.0000 | 0.5290 | 0.3718 | no | no |
| hazy_wavelet_plus_teacher_outputs | sklearn_hgb | random_image | 0.8356 | 0.7304 | 0.6275 | 1.0000 | 1.0000 | 0.5320 | 0.3751 | no | no |
| hazy_wavelet_plus_teacher_outputs | sklearn_ridge | airlight_bin | 0.7824 | 0.6879 | 0.5972 | 0.9995 | 0.9993 | 0.4886 | 0.2363 | no | no |
| hazy_wavelet_plus_teacher_outputs | sklearn_ridge | beta_bin | 0.7812 | 0.6803 | 0.5916 | 0.9996 | 0.9979 | 0.4943 | 0.2260 | no | no |
| hazy_wavelet_plus_teacher_outputs | sklearn_ridge | random_image | 0.7971 | 0.6923 | 0.5946 | 0.9997 | 0.9991 | 0.5172 | 0.2466 | no | no |
| teacher_output_proxy | sklearn_hgb | airlight_bin | 0.8154 | 0.7169 | 0.6055 | 1.0000 | 1.0000 | 0.5059 | 0.3524 | no | no |
| teacher_output_proxy | sklearn_hgb | beta_bin | 0.8176 | 0.7121 | 0.6077 | 1.0000 | 1.0000 | 0.5266 | 0.3692 | no | no |
| teacher_output_proxy | sklearn_hgb | random_image | 0.8346 | 0.7233 | 0.6114 | 1.0000 | 1.0000 | 0.5292 | 0.4050 | no | no |
| teacher_output_proxy | sklearn_ridge | airlight_bin | 0.7823 | 0.6878 | 0.5957 | 0.9995 | 0.9993 | 0.4761 | 0.2241 | no | no |
| teacher_output_proxy | sklearn_ridge | beta_bin | 0.7824 | 0.6814 | 0.5911 | 0.9998 | 1.0000 | 0.4878 | 0.2243 | no | no |
| teacher_output_proxy | sklearn_ridge | random_image | 0.8000 | 0.6986 | 0.5987 | 0.9995 | 0.9992 | 0.5045 | 0.2422 | no | no |
