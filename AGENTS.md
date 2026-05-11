# Repository Working Notes

This repository is a private research fork of DEA-Net.

## What to keep stable
- Keep the official training and evaluation entrypoints in `code/`.
- Avoid rewriting the upstream README unless a change affects onboarding.
- Do not commit datasets, checkpoints, experiment outputs, or temporary logs.

## Working conventions
- Use one feature branch per task.
- Prefer small, reviewable commits.
- Check `git status` before and after edits.
- Use `apply_patch` for file edits.
- Keep new docs and scripts ASCII unless there is a strong reason not to.
- Put generated artifacts under `experiment/`, `trained_models/`, or external storage.

## Common commands
From the repository root:

```powershell
cd code
python train.py
python eval.py --dataset ITS --model_name DEA-Net-CR --pre_trained_model PSNR4131_SSIM9945.pth
```

## Data and weights
- Dataset layout is documented in `dataset/README.md`.
- Pretrained weight layout is documented in `trained_models/README.md`.
- The upstream HAZE4K checkpoint filename is inconsistent across README files; use the actual downloaded filename when running evaluation.
- OTS evaluation may require Pillow 8.3.2 to match the official decoding behavior.

## Collaboration style
- Preserve existing behavior unless the task explicitly asks for a refactor.
- When introducing a new idea, document the change and add the smallest useful experiment note.
- If a change touches training behavior, record the branch, dataset, and checkpoint in the experiment log.
