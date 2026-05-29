# Dehaze-Net AI Text Project Package

Purpose: compact text-only context package for another AI to analyze the current DEA-Net/HAZE4K model architecture and propose next optimization points.

GitHub location: `docs/ai_text_packages/20260529-ai-text/`.

Recommended read order:
1. AGENTS.md
2. docs/CURRENT_CONTEXT.md
3. docs/README.md
4. code/model/backbone_train.py
5. code/model/modules/deablock_train.py, deconv.py, cga.py, fusion.py
6. code/option_train.py, code/train.py, code/loss/cr.py
7. docs/HAZE4K_ROUTE_EVIDENCE_REVIEW_20260528.md
8. docs/HAZE4K_MODEL_CHANGE_PROTOCOL.md
9. route-specific docs for LF-v1, CRPlus-v2, LFCR, LF-v2 MBR, and failed proxy routes
10. experiment/HAZE4K summaries, args files, compact logs, and CSV/JSON metrics as needed

Included:
- Root onboarding/config text files
- docs/*.md including current context, experiment log, route cards, workflow, runbook, analysis commands
- code/**/*.py model, train, eval, loss, data, and analysis scripts
- scripts/**/*.sh and run templates
- dataset/trained_models README files
- Text experiment evidence under experiment/HAZE4K up to 1 MB per file, excluding large feature matrices

Excluded by design:
- Images and figures
- Dataset images
- Checkpoints and model weights
- Inference image outputs
- Large feature CSVs and very large raw logs

Generated: 2026-05-29
