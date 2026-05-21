# Experiment Log Template

Use one row per run.

| Date | Branch | Dataset | Model | Change | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-11 | reproduce/deanet | ITS | DEA-Net-CR | baseline | TBD | official reproduction |
| 2026-05-19 | codex/haze4k-training-plan | HAZE4K | DEA-Net-CR | prepare HAZE4K baseline settings | remote smoke passed | commit `8a96abb`; train smoke `smoke-H4K-settings-20260519-230502` passed on RTX 5090 with 1 train step and 2 eval images; official eval smoke `eval-H4K-official-smoke-20260519-230527` passed on 2 images; single-GPU DataParallel disabled |
| 2026-05-20 | codex/haze4k-training-plan | HAZE4K | DEA-Net-CR | official checkpoint full eval | PSNR 34.2556 / SSIM 0.9885 | checkpoint `PSNR3426_SSIM9885.pth`; run `eval-H4K-official-full-20260520-095415`; 1000 test images; remote log under `experiment/HAZE4K/` |
| 2026-05-20 | codex/haze4k-training-plan | HAZE4K | DEA-Net-CR | baseline scouting train | cancelled at ~18k steps; step 10000 PSNR 27.1101 / SSIM 0.9615 | run `DEA-Net-CR-H4K-Baseline-scout-20260520-101334`; target was 100k steps; `bs=16`, `patch_size=256`, `w_loss_CR=0.1`; stopped by request; `best.pk`/`latest.pk` from step 10000 retained |
| 2026-05-20 | codex/haze4k-training-plan | HAZE4K | DEA-Net-CR | bs32 baseline scout | stopped for speed check | run `DEA-Net-CR-H4K-Baseline-scout-bs32-20260520-165148`; target 100k steps; stopped around step 1112 before checkpoint; `bs=32`, `patch_size=256`, `w_loss_CR=0.1`; smoke `smoke-H4K-bs32-vram-20260520-164745` passed; observed GPU memory about 24.8GB / 32.6GB |
| 2026-05-20 | codex/haze4k-training-plan | HAZE4K | DEA-Net-CR | batch-size speed benchmark | `bs=16` fastest by step/s and slightly fastest by images/s | remote dir `experiment/HAZE4K/bs_speed_benchmark_20260520-170841`; 600 steps each; `bs=16`: 4.2868 step/s, 68.59 img/s; `bs=24`: 2.8391 step/s, 68.14 img/s; `bs=32`: 2.1166 step/s, 67.73 img/s; observed peak VRAM about 13.0/19.1/24.8 GiB; keep `bs=16` for next HAZE4K baseline/scout runs |
| 2026-05-20 | codex/haze4k-training-plan | HAZE4K | DEA-Net-CR | baseline scout resume | completed 100k steps; best step 90000 PSNR 32.2255 / SSIM 0.9844; final step 100000 PSNR 32.0952 / SSIM 0.9844 | resumed `DEA-Net-CR-H4K-Baseline-scout-20260520-101334` from `saved_model/latest.pk`; tmux `h4k_resume_bs16_20260520_174712`; log `resume_20260520-174716.log`; `bs=16`, `patch_size=256`, `w_loss_CR=0.1`; `best.pk` at step 90000 / epoch 18; `latest.pk` at step 100000 / epoch 20; remote path `experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/` |
| 2026-05-21 | codex/haze4k-lf-prior | HAZE4K | DEA-Net-LF | lightweight LF prior scout | resumed after server restart; latest checkpoint at step 80000 PSNR 32.1721 / SSIM 0.9841 | run `DEA-Net-LF-H4K-scout-20260521-003100`; original tmux `h4k_lf_scout_20260521_003100`; resumed tmux `h4k_lf_resume_20260521_084700`; code commit `857661d`; `bs=16`, `patch_size=256`, `epochs=20`, `iters_per_epoch=5000`, `use_lf_prior=true`, `lf_prior_channels=8`, `lf_prior_pool=8`, `lf_prior_gate_init=0.0`, `w_loss_CR=0.1`; restart interrupted after about 86856 loss entries, but `best.pk`/`latest.pk` were at step 80000; resume log `experiment/HAZE4K/_run_logs/DEA-Net-LF-H4K-scout-20260521-003100-resume-20260521-084700.log` |

## Suggested notes
- Dataset split
- Checkpoint name
- Learning rate
- Batch size
- Patch size
- Hardware
- Runtime
