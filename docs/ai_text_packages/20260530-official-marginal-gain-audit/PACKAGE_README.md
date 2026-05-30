# HAZE4K Official-Centric Marginal Gain Audit Text Package

Date: 2026-05-30

Purpose: GitHub-readable, text-only evidence package for the completed
Official-centric Marginal Gain Audit. This package is intended for AI review and
cross-conversation handoff without committing datasets, images, checkpoints,
arrays, or model weights.

## Start Here

- Route card and conclusion:
  `docs/HAZE4K_OFFICIAL_MARGINAL_GAIN_AUDIT_20260530.md`
- Current project context:
  `docs/CURRENT_CONTEXT.md`
- Run ledger:
  `docs/EXPERIMENT_LOG.md`
- Artifact retention policy:
  `docs/HAZE4K_RUN_MANIFEST.md`
- Generated audit report:
  `experiment/HAZE4K/official_marginal_gain_audit/HAZE4K-official-marginal-gain-audit-autodl-20260530-hgb-full/analysis_report.md`

## Key Result

The audit recommendation is `do_not_train_official_adapter_yet`.

The official oracle headroom is real: mean official PSNR was `34.2556`, while
mean oracle-with-no-change PSNR was `34.9015` (`+0.6459 dB`). Candidate gains
were at least `0.15 / 0.20 dB` on `523 / 493` of the 1000 HAZE4K test images.

The deployability gate failed: `0 / 448` HGB summary rows passed. The best
eligible random deployable row at margin `0.20` reached intervention precision
`0.6597`, strong-official no-change false intervention `0.4499`, all no-change
false intervention `0.6050`, bootstrap p05 gain `0.5158 dB`, and shuffled
precision gap `0.0462`. Leakage rows separated the target but are explicitly
not deployable.

## Contents

- `code/analyze_official_marginal_gain_audit.py`: audit implementation.
- `scripts/autodl-haze4k-official-marginal-gain-audit.sh`: AutoDL launch
  wrapper used for the full HGB run.
- `experiment/HAZE4K/_run_logs/`: compact run log.
- `experiment/HAZE4K/official_marginal_gain_audit/.../`: generated JSON, CSV,
  and Markdown audit outputs.
- `docs/`: relevant route, context, experiment log, and manifest documents.
- `PACKAGE_FILE_LIST.txt`: complete package file list.

## Exclusions

This package intentionally excludes images, datasets, checkpoints, model
weights, `.npy`/`.npz` arrays, and raw feature arrays. The remote AutoDL
experiment directory remains the source artifact for this run.
