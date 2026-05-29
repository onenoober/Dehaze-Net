# AI Text Packages

This directory stores GitHub-readable, text-only project packages prepared for
external AI analysis.

Policy:

- Commit and push every created or updated AI text package to GitHub.
- Keep package copies under `docs/ai_text_packages/<date-or-id>/`.
- Include only readable text evidence: source, docs, scripts, compact logs,
  args, JSON/JSONL, CSV, and Markdown diagnostics.
- Do not include datasets, images, checkpoints, model weights, `.npy`/`.npz`
  arrays, inference image outputs, or large feature matrices.
- Keep raw generated artifacts in ignored `experiment/`; only the curated
  text-only package copy belongs here.
- After pushing, audit local package parity, remote GitHub parity, and forbidden
  extensions; record the result in `docs/HAZE4K_RUN_MANIFEST.md`.

Current packages:

| Package | Purpose |
| --- | --- |
| `20260529-ai-text/` | Text-only context package for AI model-architecture analysis of the current DEA-Net/HAZE4K fork. Start at `20260529-ai-text/PACKAGE_README.md`. |
