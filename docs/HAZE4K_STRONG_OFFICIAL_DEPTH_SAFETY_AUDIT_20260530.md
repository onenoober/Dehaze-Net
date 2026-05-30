# HAZE4K Strong-Official Depth Safety Audit

Date: 2026-05-30

Status: Stage 0 audit plan. This audit is diagnostic only and does not
authorize HAZE4K training. Its purpose is to test whether predicted depth is a
deployable no-change variable for the paused official-centric marginal gain
route.

Execution boundary: run extraction, depth inference, and probe fitting on
`autodl-dehaze` in `/root/autodl-tmp/workspace/Dehaze-Net`. Local WSL work is
limited to source, documentation, Git, packaging, and compact sync.

## Most Valuable Attempt

- Current bottleneck:
  official oracle headroom exists, but previous deployable output/internal
  features did not keep strong-official no-change false intervention low.
- New variable:
  candidate-free predicted depth from the hazy input and official output.
- Main question:
  can predicted depth reduce strong-official/no-change false intervention
  beyond C1 basic image statistics?
- Decision value:
  a pass reopens only a tiny depth-aware safety gate card; a fail stops this
  depth route without launching a 100k scout.

## Data

Use the existing 1000-image HAZE4K official marginal gain audit join:

- official step0 output;
- official warm-start best output;
- cold-start CR, LF-v1, ResidualCalib, CRPlus-v2 if checkpoint is present,
  and CBRFRC-v1 candidates;
- candidate PSNR list, best candidate, official strength bin, CR strength bin,
  and residual-energy bin.

If CRPlus-v2 checkpoint is absent on AutoDL, use the committed route-evidence
CSV for CRPlus-v2 PSNR and exclude CRPlus-v2 from candidate-aware depth
features.

## Labels

- `Y_strong_nochange = 1` when `official_strength_bin == strong_official_q4`
  and `best_candidate_psnr - official_psnr <= 0.05`.
- `Y_nochange = 1` when `best_candidate_psnr - official_psnr <= 0.05`.
- `Y_intervene = 1` when `best_candidate_psnr - official_psnr >= 0.20`.

The primary metric is strong-official no-change false intervention rate.

## Feature Groups

- C0: deployable official/candidate output, residual, and DEA activation
  features.
- C1: C0 plus basic image statistics such as brightness, contrast, saturation,
  color cast, dark channel, edge density, entropy, low-texture ratio, and
  gradient energy.
- C2: C1 plus candidate-free depth from `depth(hazy)`, `depth(official)`, and
  hazy-official depth consistency.
- C3: C1 plus candidate-free and candidate-aware depth deltas from available
  candidate outputs.

Depth estimators:

- `depth-anything/Depth-Anything-V2-Small-hf`;
- `Intel/dpt-hybrid-midas`.

Do not use DCMPNet depth in this audit.

## Models

Only lightweight probes are allowed:

- logistic regression;
- decision tree with `max_depth <= 3` and `min_samples_leaf >= 20`;
- histogram gradient boosting with `max_depth <= 3` and
  `min_samples_leaf >= 20`.

The probes are not a selector training result. They only test whether depth
adds discriminative information.

## Splits And Controls

Splits:

- random image split for sanity;
- train non-strong official, test strong official;
- train weak/mid CR, test strong CR;
- train normal/high residual energy, test low residual energy.

Controls:

- compare C2 against C1, not just C0;
- shuffled-depth image-id control;
- label-permutation control.

## Pass Line

C2 can reopen the route only if it satisfies all of:

- strong-official false intervention relative reduction vs C1 is at least
  `30%`;
- all no-change false intervention relative reduction vs C1 is at least `20%`;
- intervention precision does not drop by more than `0.03`;
- simulated mean PSNR gain does not drop;
- bootstrap gain p05 is non-negative;
- official-strength, CR-strength, and residual-low-energy held-outs do not
  collapse;
- real depth is clearly better than shuffled depth;
- both depth estimators show the same direction of effect.

The stronger absolute readout is:

- strong-official FI `<= 0.10`;
- all no-change FI `<= 0.15`;
- intervention precision `>= 0.70`.

## Outputs

The audit writes only compact text artifacts:

- `depth_safety_audit_protocol.md`;
- `feature_group_summary.csv`;
- `strong_nochange_safety_summary.csv`;
- `heldout_safety_summary.csv`;
- `shuffle_depth_control.csv`;
- `label_permutation_control.csv`;
- `depth_estimator_consistency.csv`;
- `candidate_free_vs_candidate_aware.csv`;
- `decision_summary.json`;
- `analysis_report.md`.
