# AI Text Packages

This directory stores GitHub-readable, text-only project packages prepared for
external AI analysis. It is the only approved path for committing
experiment-derived small text evidence to GitHub.

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
- Do not call a package synchronized until the source package, committed copy,
  and remote branch have the same intended file set and the manifest records
  that audit.

Current packages:

| Package | Purpose |
| --- | --- |
| `20260529-ai-text/` | Text-only context package for AI model-architecture analysis of the current DEA-Net/HAZE4K fork. Start at `20260529-ai-text/PACKAGE_README.md`. |
| `20260530-cbrfrc-v1/` | Focused text-only CBRFRC-v1 evidence package with implementation files, route docs, preflight logs, diagnostic CSV/JSON, compact training log, and run args. Start at `20260530-cbrfrc-v1/PACKAGE_README.md`. |
| `20260530-brfrc-v2-representation-audit/` | Focused text-only BRFRC-v2 Stage 0 representation audit package with audit code, launch script, route docs, compact log, and small JSON/CSV/MD outputs. Start at `20260530-brfrc-v2-representation-audit/PACKAGE_README.md`. |
| `20260530-official-marginal-gain-audit/` | Focused text-only Official-centric Marginal Gain Audit package with audit code, launch script, route docs, compact log, and JSON/CSV/MD outputs. Start at `20260530-official-marginal-gain-audit/PACKAGE_README.md`. |
