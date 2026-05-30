# DEA-Net BRFRC-v2 Representation Audit Text Evidence Package

Purpose: focused text-only package for reviewing the BRFRC-v2 Stage 0
representation audit on AutoDL. The package captures the audit route card,
implementation script, launch script, compact log, and the small JSON/CSV/MD
result files needed to understand why BRFRC-v2-Rep training is not authorized.

GitHub location after sync:
`docs/ai_text_packages/20260530-brfrc-v2-representation-audit/`.

Recommended read order:

1. `docs/HAZE4K_BRFRC_V2_REPRESENTATION_AUDIT_PLAN_20260530.md`
2. `docs/CURRENT_CONTEXT.md`
3. `docs/EXPERIMENT_LOG.md`
4. `code/analyze_brf_representation_audit.py`
5. `scripts/autodl-haze4k-brfrc-v2-representation-audit.sh`
6. `experiment/HAZE4K/brfrc_v2_representation_audit/HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028/analysis_report.md`
7. `experiment/HAZE4K/brfrc_v2_representation_audit/HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028/summary.json`
8. `experiment/HAZE4K/brfrc_v2_representation_audit/HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028/split_results.csv`
9. `experiment/HAZE4K/_run_logs/HAZE4K-brfrc-v2-representation-audit-autodl-20260530-111028.log`
10. `docs/HAZE4K_CBRFRC_V1_PLAN_20260530.md`
11. `docs/HAZE4K_ROUTE_EVIDENCE_REVIEW_20260528.md`

Included:

- BRFRC-v2 representation audit route card and current project context.
- Experiment log, run manifest, model-change protocol, and documentation map.
- Audit implementation code, supporting data/model utilities, and AutoDL launch
  script.
- Compact Stage 0 audit outputs: `analysis_report.md`, `summary.json`,
  `summary.csv`, and `split_results.csv`.
- The AutoDL audit log, small enough to include directly.
- The CBRFRC-v1 route card and route-evidence review needed to interpret the
  failed output-level residual corrector that motivated this audit.

Excluded by design:

- Checkpoints and model weights.
- TensorBoard event files.
- `.npy` / `.npz` arrays, including `feature_matrix_targets.npz`.
- Large feature tables such as `feature_rows_compact.csv`.
- Images, datasets, inference outputs, and binary artifacts.

Generated: 2026-05-30
