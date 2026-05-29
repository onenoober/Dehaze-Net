# Repository Working Notes

This repository is a private research fork of DEA-Net.

## What to keep stable
- Keep the official training and evaluation entrypoints in `code/`.
- Avoid rewriting the upstream README unless a change affects onboarding.
- Do not commit datasets, checkpoints, experiment outputs, or temporary logs.
- Exception: curated text-only AI project packages under
  `docs/ai_text_packages/` must be committed and pushed to GitHub when created
  or updated, so external AI review can use GitHub links. These packages may
  include copied compact logs/CSV/JSON/MD text, but never images, datasets,
  checkpoints, arrays, or model weights.

## Working conventions
- For continuity across new conversations, read `docs/CURRENT_CONTEXT.md` before changing server, GitHub, dataset, or training workflow assumptions.
- Before modifying documentation, read `docs/README.md` and choose the target document by its authority table.
- Put each new fact in one authoritative document only: current state in `docs/CURRENT_CONTEXT.md`, run facts in `docs/EXPERIMENT_LOG.md`, artifact retention/path decisions in `docs/HAZE4K_RUN_MANIFEST.md`, server facts in `docs/CORE_SERVER_RUNBOOK.md`, operational templates in `docs/WORKFLOW.md`, analysis/evaluation commands in `docs/ANALYSIS_COMMANDS.md`, and route hypotheses/gates in a dated route card plus `docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md`.
- When a cloud server is missing a required dependency for the requested task, install it directly without asking for confirmation, then record durable server facts in `docs/CORE_SERVER_RUNBOOK.md` when they matter for future runs.
- Do not put full run history in current context, server environment facts in workflow, long command templates in route cards, or route conclusions in the artifact manifest.
- Use one feature branch per task.
- Prefer small, reviewable commits.
- Check `git status` before and after edits.
- Use `apply_patch` for file edits.
- Keep new docs and scripts ASCII unless there is a strong reason not to.
- Put generated artifacts under `experiment/`, `trained_models/`, or external storage.
- Put GitHub-readable AI text package copies under `docs/ai_text_packages/`.

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
