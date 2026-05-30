# DEA-Net CBRFRC-v1 Text Evidence Package

Purpose: focused text-only package for reviewing the DEA-Net-CBRFRC-v1
baseline-relative frequency residual corrector implementation, preflight,
100k scout result, and mechanism diagnostics.

GitHub location after sync:
`docs/ai_text_packages/20260530-cbrfrc-v1/`.

Recommended read order:

1. `docs/HAZE4K_CBRFRC_V1_PLAN_20260530.md`
2. `docs/CURRENT_CONTEXT.md`
3. `docs/EXPERIMENT_LOG.md`
4. `code/model/brfrc.py`
5. `code/preflight_brf_cbrfrc.py`
6. `code/diagnose_brf_residual_direction.py`
7. `experiment/HAZE4K/brf_diagnostics/DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811-best-20260530-0830/summary.json`
8. `experiment/HAZE4K/brf_diagnostics/DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811-best-20260530-0830/per_image_metrics.csv`
9. `experiment/HAZE4K/brf_preflight/*/summary.json`
10. `experiment/HAZE4K/_run_logs/DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811.compact.log`

Included:

- CBRFRC route card, current context, experiment log, manifest, and protocol.
- CBRFRC implementation and train/eval/diagnostic/preflight scripts.
- Full BRF per-image diagnostic CSV for the best checkpoint.
- BRF summary JSON/CSV and route evidence summary.
- Preflight summary JSON and CSV logs, including identity/oracle and
  micro-overfit checks.
- Small CBRFRC preflight logs and a compact training log excerpt.
- Run args text and validation curve log.

Excluded by design:

- Checkpoints and model weights.
- TensorBoard event files.
- `.npy` / `.npz` arrays.
- Images, datasets, inference outputs, and binary artifacts.
- The full raw 100k training log, because it is repetitive step-by-step text;
  a compact log with configuration, eval lines, and final tail is included.

Generated: 2026-05-30
