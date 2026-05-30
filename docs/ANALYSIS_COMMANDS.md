# Analysis Commands

Purpose: reusable read-only evaluation, visualization, and diagnostic commands.
These commands help inspect checkpoints and route mechanisms; they are not
training launch templates and should not be used as run-status records.

For cloud/server setup, SSH, Git sync, tmux, and training launch templates, use
`docs/WORKFLOW.md`. For metrics and run decisions, use
`docs/EXPERIMENT_LOG.md`.

## Train Checkpoint Visual Compare

For CR baseline versus an LF variant, the default baseline model is non-LF:

```bash
/root/miniconda3/envs/py310/bin/python visual_compare_train_ckpt.py \
  --baseline_checkpoint <baseline-best.pk> \
  --lf_checkpoint <lf-variant-best.pk> \
  --output_dir <compare-dir> \
  --lf_label <label>
```

For LF-v1 versus another LF variant, explicitly enable the baseline LF
architecture so the baseline checkpoint is loaded with the correct module
shape:

```bash
/root/miniconda3/envs/py310/bin/python visual_compare_train_ckpt.py \
  --baseline_checkpoint <lf-v1-best.pk> \
  --baseline_use_lf_prior \
  --lf_checkpoint <lf-v2-best.pk> \
  --lf_conditional_mask \
  --lf_haze_aware_mask \
  --output_dir <compare-dir> \
  --lf_label <label>
```

## LF Residual Direction Diagnostic

Before designing another LF variant, run the residual diagnostic to check
whether the LF candidate is correcting low-frequency content in the right
direction and with the right magnitude. This is a read-only evaluation; it does
not start training.

Baseline CR versus LF-v1:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net/code
/root/miniconda3/envs/py310/bin/python diagnose_lf_residual_direction.py \
  --dataset HAZE4K \
  --split test \
  --baseline_checkpoint ../experiment/HAZE4K/DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk \
  --current_checkpoint ../experiment/HAZE4K/DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk \
  --current_use_lf_prior \
  --baseline_label DEA-Net-CR \
  --current_label DEA-Net-LF-v1 \
  --lowfreq_pool 8 \
  --output_dir ../experiment/HAZE4K/residual_diagnostic/CR-vs-LF-v1-20260524
```

For LF-v1 versus a later LF variant, load the baseline checkpoint with the LF
architecture and add the feature flags for the variant:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net/code
/root/miniconda3/envs/py310/bin/python diagnose_lf_residual_direction.py \
  --dataset HAZE4K \
  --split test \
  --baseline_checkpoint <lf-v1-best.pk> \
  --baseline_use_lf_prior \
  --current_checkpoint <lf-variant-best.pk> \
  --current_use_lf_prior \
  --lf_conditional_mask \
  --lf_haze_aware_mask \
  --baseline_label DEA-Net-LF-v1 \
  --current_label <label> \
  --output_dir <diagnostic-dir>
```

The script writes:

- `per_image_residual_metrics.csv`
- `group_summary.csv`
- `summary.json`
- `hard_cases.json`
- `analysis_report.md`

Important columns:

- `lf_residual_cosine`: direction match between
  `LP(current)-LP(baseline)` and `LP(GT)-LP(baseline)`.
- `lf_residual_norm_ratio`: correction magnitude relative to the target
  low-frequency correction.
- `lf_residual_error_ratio`: low-frequency residual error size relative to the
  target correction.
- `lf_mse_delta`: positive values mean the candidate made low-frequency MSE
  worse than the baseline.

Use this output to decide whether the next LF route should calibrate residual
direction/magnitude rather than add another spatial mask.

## Strong-Official Depth Safety Audit

Run this only as a diagnostic audit for the official-centric no-change
bottleneck; it does not authorize training by itself.

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net
bash scripts/autodl-haze4k-strong-official-depth-safety-audit.sh
```

The script writes compact text outputs under
`experiment/HAZE4K/strong_official_depth_safety_audit/<run-id>/`, including:

- `decision_summary.json`
- `analysis_report.md`
- `strong_nochange_safety_summary.csv`
- `heldout_safety_summary.csv`
- `shuffle_depth_control.csv`
- `label_permutation_control.csv`
- `depth_estimator_consistency.csv`
- `candidate_free_vs_candidate_aware.csv`
