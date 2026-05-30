# DEA-Net Strong-CR Abstention / Residual-SNR Audit Text Evidence Package

Purpose: focused text-only package for reviewing the Strong-CR Abstention /
Residual-SNR audit on AutoDL. The package captures the route card,
implementation script, launch script, compact log, small CSV/JSON/MD outputs,
and the route-evidence CSV used to supply CRPlus-v2 per-image metrics on
AutoDL, where the CRPlus-v2 checkpoint was not present.

GitHub location after sync:
`docs/ai_text_packages/20260530-strong-cr-abstention-snr-audit/`.

Recommended read order:

1. `docs/HAZE4K_STRONG_CR_ABSTENTION_RESIDUAL_SNR_AUDIT_20260530.md`
2. `experiment/HAZE4K/strong_cr_abstention_snr_audit/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full/analysis_report.md`
3. `experiment/HAZE4K/strong_cr_abstention_snr_audit/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full/summary.json`
4. `experiment/HAZE4K/strong_cr_abstention_snr_audit/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full/split_results.csv`
5. `experiment/HAZE4K/strong_cr_abstention_snr_audit/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full/snr_by_cr_strength.csv`
6. `experiment/HAZE4K/strong_cr_abstention_snr_audit/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full/label_rows.csv`
7. `code/analyze_strong_cr_abstention_residual_snr_audit.py`
8. `scripts/autodl-haze4k-strong-cr-abstention-snr-audit.sh`
9. `experiment/HAZE4K/_run_logs/HAZE4K-strong-cr-abstention-snr-audit-autodl-20260530-full.log`
10. `docs/CURRENT_CONTEXT.md`
11. `docs/EXPERIMENT_LOG.md`
12. `docs/HAZE4K_RUN_MANIFEST.md`

Included:

- Strong-CR Abstention / Residual-SNR audit route card and current project
  context.
- Experiment log, run manifest, model-change protocol, and documentation map.
- Audit implementation code, the BRFRC-v2 audit helper imported by the audit,
  supporting data/model utilities, and the AutoDL launch script.
- Compact audit outputs: `analysis_report.md`, `summary.json`,
  `summary.csv`, `split_results.csv`, `snr_by_cr_strength.csv`, and
  `label_rows.csv`.
- The AutoDL audit log.
- The CRPlus-v2 route-evidence `model_per_image_matrix.csv` used because the
  CRPlus-v2 checkpoint was unavailable on AutoDL.
- Prior CBRFRC-v1 and BRFRC-v2 representation route cards needed to interpret
  why this abstention audit was the next cheap gate.

Excluded by design:

- Checkpoints and model weights.
- TensorBoard event files.
- `.npy` / `.npz` arrays and feature-matrix dumps.
- Images, datasets, inference outputs, and binary artifacts.
- Any full raw experiment directory outside the compact text files listed
  above.

Key decision:

- Recommendation: `do_not_train_abstention_brf_v3_yet`.
- Deployable rows did not satisfy the full gate, especially strong-CR
  preservation / false-intervention, LF-v1 gain preservation, and confidence
  correlation together.
- Diagnostic leakage rows show the target is separable when GT-derived
  information is allowed, but that signal is not deployable.

Generated: 2026-05-30
