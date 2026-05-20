# Experiment Log Template

Use one row per run.

| Date | Branch | Dataset | Model | Change | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-11 | reproduce/deanet | ITS | DEA-Net-CR | baseline | TBD | official reproduction |
| 2026-05-19 | codex/haze4k-training-plan | HAZE4K | DEA-Net-CR | prepare HAZE4K baseline settings | remote smoke passed | commit `8a96abb`; train smoke `smoke-H4K-settings-20260519-230502` passed on RTX 5090 with 1 train step and 2 eval images; official eval smoke `eval-H4K-official-smoke-20260519-230527` passed on 2 images; single-GPU DataParallel disabled |
| 2026-05-20 | codex/haze4k-training-plan | HAZE4K | DEA-Net-CR | official checkpoint full eval | PSNR 34.2556 / SSIM 0.9885 | checkpoint `PSNR3426_SSIM9885.pth`; run `eval-H4K-official-full-20260520-095415`; 1000 test images; remote log under `experiment/HAZE4K/` |

## Suggested notes
- Dataset split
- Checkpoint name
- Learning rate
- Batch size
- Patch size
- Hardware
- Runtime
