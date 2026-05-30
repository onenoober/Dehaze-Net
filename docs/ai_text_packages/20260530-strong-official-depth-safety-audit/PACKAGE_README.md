# HAZE4K Strong-Official Depth Safety Audit Text Package

Date: 2026-05-30

Purpose: GitHub-readable, text-only evidence package for the completed
Strong-Official Depth Safety Audit. This package is intended for AI review and
cross-conversation handoff without committing datasets, images, checkpoints,
arrays, depth caches, or model weights.

## Start Here

- Route card and conclusion:
  `docs/HAZE4K_STRONG_OFFICIAL_DEPTH_SAFETY_AUDIT_20260530.md`
- Current project context:
  `docs/CURRENT_CONTEXT.md`
- Run ledger:
  `docs/EXPERIMENT_LOG.md`
- Artifact retention policy:
  `docs/HAZE4K_RUN_MANIFEST.md`
- Generated audit report:
  `experiment/HAZE4K/strong_official_depth_safety_audit/HAZE4K-strong-official-depth-safety-audit-autodl-20260530-full/analysis_report.md`

## Key Result

The audit decision is `stop_depth_route`.

C2 candidate-free depth did not improve the primary safety line beyond C1 basic
image statistics. C2-vs-C1 strong-official false-intervention relative
reduction, all no-change false-intervention relative reduction, precision
delta, simulated gain delta, and bootstrap p05 delta were all `0.0`.

Best C1, C2, and C3 rows were identical shallow decision-tree behavior:
strong-official false intervention `0.05499`, all no-change false intervention
`0.03972`, intervention precision `0.65526`, simulated mean PSNR gain
`0.09518`, and bootstrap p05 `0.05393`. Higher-gain HGB rows did not satisfy
the depth-specific safety claim, and shuffled-depth, basic-stat, and estimator
consistency controls failed.

## Contents

- `code/analyze_strong_official_depth_safety_audit.py`: audit implementation.
- `scripts/autodl-haze4k-strong-official-depth-safety-audit.sh`: AutoDL launch
  wrapper used for the full audit.
- `experiment/HAZE4K/_run_logs/`: compact AutoDL run log.
- `experiment/HAZE4K/strong_official_depth_safety_audit/.../`: generated JSON,
  CSV, and Markdown audit outputs, including the small 1000-row
  official-candidate-depth join for traceability.
- `docs/`: relevant route, command, context, experiment log, and manifest
  documents.
- `PACKAGE_FILE_LIST.txt`: complete package file list.

## Exclusions

This package intentionally excludes images, datasets, checkpoints, model
weights, `.npy`/`.npz` arrays, raw tensor dumps, and depth cache files. The
remote AutoDL experiment directory remains the source artifact for this run.
