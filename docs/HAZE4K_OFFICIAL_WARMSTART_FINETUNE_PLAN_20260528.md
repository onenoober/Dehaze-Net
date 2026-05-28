# HAZE4K Official Warm-Start Fine-Tune Route

Status: active route card, isolated from cold-start HAZE4K training claims.

## Most Valuable Attempt

- Why this is the most valuable current attempt:
  Official HAZE4K weights are already a high-quality DEA-Net reference, while
  the current cold-start route spends large compute before deciding whether an
  architecture change is useful. A warm-start route can test whether LF-v1
  style additions improve from a strong starting point and whether useful
  quality is reached earlier.
- Cheap preflight evidence:
  The official HAZE4K checkpoint exists under `trained_models/HAZE4K/`, and
  local key inspection showed shape-compatible weights but non-identical key
  names between inference `Backbone` and training `DEANet`.
- Earliest decisive gate:
  Step 10000 is the first useful quality/speed gate because step 0 should be
  an official-checkpoint-equivalent state. Step 20000 is the first hard gate
  for whether fine-tuning is actually improving beyond the official starting
  point.
- Expected training-time or attempt-count saving:
  If the route works, it can reduce expensive cold-start scouting by using
  official weights as a deployment-oriented initialization and testing only
  small isolated architecture additions.
- What success decides:
  A successful run establishes a separate warm-start optimization route for
  practical fine-tuning and may justify converting the best cold-start
  candidate into an official-weight-initialized variant.
- What failure decides:
  Failure means official-weight warm-start does not rescue the tested
  architecture addition under the written LR/gate contract; do not infer that
  the same change fails as a cold-start fair candidate.
- Why a cheaper diagnostic is not enough:
  A state_dict/equivalence check can prove that initialization is valid, but
  only a short gated fine-tune can show whether the strong starting point keeps
  or improves image quality.

## Route Boundary

This is not the existing cold-start HAZE4K fair-candidate route. It is an
isolated fine-tuning route with its own labels, launcher, warm-start checkpoint
preparation, and comparison rules.

Allowed claims:

- time-to-quality from official HAZE4K initialization;
- whether an architecture addition preserves or improves the official
  checkpoint after fine-tuning;
- whether warm-start is useful as a practical deployment route.

Disallowed claims:

- do not use this route to claim that an architecture is better when trained
  from scratch;
- do not mix these metrics into cold-start fair-candidate tables;
- do not compare low-LR warm-start steps against cold-start steps as if they
  shared the same optimizer history.

## Hypothesis

- Prior evidence:
  LF-v1 remains the best standalone cold-start route, while CRPlus-v2 and
  LFCR variants show that training dynamics and late loss pressure can decide
  whether a mechanism helps or harms.
- Target failure mode:
  Cold-start candidate exploration spends many steps relearning a competent
  dehazing baseline before testing a small architecture addition.
- Mechanism hypothesis:
  If official DEA-Net weights are mapped into the training architecture with
  zero-impact added branches, LF-style modules can start from a strong output
  and learn only useful residual behavior, reducing training time and avoiding
  early collapse.

## Change

- Code branch:
  `codex/haze4k-official-warmstart-finetune`
- Primary variable:
  Initialization and run contract: start from official HAZE4K checkpoint
  converted into a step-0 training checkpoint.
- Architecture/loss definition:
  The first route uses training `DEANet` with optional LF-v1 prior enabled by
  the launcher. The official inference `Backbone` conv weights are transferred
  into matching training keys. For blocks whose training `conv1` is a `DEConv`,
  the official ordinary conv is copied into `conv1_5`, while `conv1_1` through
  `conv1_4` are zeroed so the initial effective conv matches the official
  checkpoint as closely as possible.
- Enabled flags:
  Default launcher: `--use_lf_prior`, `lf_prior_gate_init=0.0`,
  `lf_prior_injection=pre_mix`, `w_loss_L1=1.0`, `w_loss_CR=0.1`,
  `start_lr=0.00002`, `end_lr=0.000001`,
  `trainable_schedule=0:lf_prior;10001:lf_prior,bottleneck,fusion;30001:all`.
- Explicitly disabled related mechanisms:
  CRPlus-v2, ResidualDirLoss, teacher guard, selectors, and masks are disabled
  by default. They can be tested later only as separate warm-start route cards
  or clearly named sub-routes.

## Isolated Implementation

- `code/prepare_official_warmstart_checkpoint.py`
  Creates `official_warmstart_step0.pk` with `model`, fresh Adam `optimizer`,
  step `0`, empty metric histories, and `warmstart_meta`. It does not modify
  `train.py`, `eval.py`, or `option_train.py`.
- `scripts/haze4k-official-warmstart-finetune.sh`
  Creates the warm-start checkpoint, then launches existing `train.py` through
  `--resume true --pre_trained_model official_warmstart_step0.pk`. The run id
  is prefixed with `DEA-Net-OfficialWarmStart-...`.
- `code/warmstart_freeze.py`
  Provides the staged trainable-parameter schedule. The default training route
  is unaffected because `trainable_schedule=none` unless the warm-start
  launcher passes a schedule.
- Artifacts:
  The converted checkpoint and JSON transfer report live under
  `experiment/HAZE4K/<run-id>/`, which stays ignored by Git.

## References

- Official HAZE4K checkpoint:
  `trained_models/HAZE4K/PSNR3426_SSIM9885.pth`
- Official evaluation command:
  use `docs/WORKFLOW.md` and the actual downloaded checkpoint filename.
- Direct cold-start reference:
  LF-v1 remains the current best standalone cold-start evidence, but it is not
  a matched optimizer-history reference for this route.
- Matched warm-start reference:
  Step 0 official-equivalent output and the same run's later checkpoints.

## Reproduction Pitfalls To Avoid

- Do not pass the official `.pth` directly to `train.py --resume`; `train.py`
  expects a training checkpoint containing both `model` and `optimizer`.
- Do not use naive `strict=False` loading. It hides the fact that official
  `conv1.weight` keys do not map directly onto training `DEConv` branches.
- Do not leave `DEConv` side branches randomly initialized when warm-starting
  from official weights. That changes the step-0 function and can destroy the
  high starting point.
- Do not rename the HAZE4K official checkpoint by guessing from upstream docs.
  Use the actual file under `trained_models/HAZE4K/`.
- Run from `code/` or use the launcher, because existing option paths are
  relative to that directory.
- Keep the dataset/checkpoint/log outputs out of Git. Generated warm-start
  checkpoints belong under `experiment/`.
- If running on multiple GPUs, ensure the warm-start checkpoint uses
  `module.`-prefixed model keys. The preparation script uses auto detection
  for this case.
- Treat short warm-start smoke runs as diagnostic only. They are invalid for
  cold-start comparison and invalid for final warm-start route decisions.
- Keep the 100000-step LR horizon on resume. Stopping at gates is allowed, but
  changing `epochs * iters_per_epoch` changes the schedule.
- Do not use staged freezing without checking that each stage selects at least
  one parameter. The training script now fails fast if a stage has zero
  trainable parameters.
- If `USE_LF_PRIOR=0`, the launcher defaults to `TRAINABLE_SCHEDULE=none`;
  otherwise `0:lf_prior` would freeze every parameter and make backward invalid.

## Mechanism Metrics

| Metric | Why it matches this route | Gate subset | Full-test artifact |
| --- | --- | --- | --- |
| Forward equivalence max-abs diff | Proves the converted training model starts from the official function, except explicitly enabled zero-gated modules. | Step-0 preparation report | `official_warmstart_report.json` |
| Loaded/mapped/zeroed key counts | Catches accidental partial loading or random DEConv side branches. | Step-0 preparation report | `official_warmstart_report.json` |
| Steps-to-official-quality | Measures whether fine-tuning preserves the high starting point and recovers after optimizer updates. | 10k and 20k validation | `saved_data/log.txt` |
| Delta from official baseline | Main route quality signal, separate from cold-start references. | Each 10k validation | full-test per-image CSV when promoted |
| LF gate and LF branch stats | Confirms whether the added LF prior stays inactive, learns, or collapses. | 10k/20k/30k logs | saved loss logs and diagnostics |
| Trainable stage and parameter count | Confirms staged freezing/unfreezing happened at the intended steps. | `trainable_schedule.jsonl` and TensorBoard | `saved_data/trainable_schedule.jsonl` |
| Iteration speed and wall time | Measures whether warm-start saves practical training time. | Every gate | run log |

## Fine-Tuning Contract

- Dataset:
  HAZE4K.
- Total target:
  Default `epochs=20`, `iters_per_epoch=5000`, total `100000`, even if the
  run is stopped earlier at a written gate.
- Batch/patch:
  `bs=16`, `patch_size=256`.
- Loss weights:
  `w_loss_L1=1.0`, `w_loss_CR=0.1`; CRPlus-v2 disabled by default.
- LR:
  `start_lr=0.00002`, `end_lr=0.000001`. This is intentionally a fine-tune
  LR, not the cold-start fair-candidate LR.
- Eval/checkpoint cadence:
  `checkpoint_interval_steps=10000`, `eval_interval_steps=10000`,
  `save_epoch_checkpoints=false`.
- Staged freezing:
  Stage A `0-10000`: train `lf_prior` only. Stage B starts at step `10001`:
  train `lf_prior`, bottleneck (`down3`, `fe_level_3`, `level3_block*`,
  `mix1`), and fusion (`mix1`, `mix2`). Stage C starts at step `30001`:
  train all parameters with the fine-tune LR. The boundary is `10001` rather
  than `10000` so the 10k gate reflects the LF-only stage.

## Gates

| Step | Image metric rule | Mechanism metric rule | Stop/continue rule |
| ---: | --- | --- | --- |
| 0 | Official checkpoint eval is the route's starting reference. | Forward equivalence check should pass or be explicitly skipped only because CUDA is unavailable. | Do not launch training if mapping counts are wrong or DEConv side branches are not zeroed. |
| 10000 | Should remain close to official baseline; a small dip is acceptable only with clear recovery signs. | LF-only stage must be active, finite, and non-degenerate. | Stop if quality collapses or mapping/equivalence evidence is invalid. |
| 20000 | Must be recovering toward or improving over the official baseline. | Stage B should be active and bounded; bottleneck/fusion unfreeze must not destabilize the model. | Stop if still below official by a large margin with no mechanism upside. |
| 30000 | First hard gate for route value. | Stage B must show time-to-quality or branch-learning evidence that justifies full-model tiny-LR unfreeze. | Continue only if the route is close to or above official baseline, or gives decisive evidence for a narrower next warm-start variant. |
| 50000 | Promotion gate. | Regression control and per-image splits should be prepared if quality is competitive. | Continue to later gates only if there is a practical warm-start advantage. |
| 100000 | Final warm-start scout point. | Full-test mechanism and regression analysis required before promotion. | Record as positive warm-start route, negative warm-start route, or diagnostic only. |

## Analysis Plan

- If stopped:
  Record the failed gate, official-baseline delta, transfer report path, and
  which future warm-start variants are deprioritized.
- If promoted:
  Run full-test per-image analysis against the official baseline and the
  current best cold-start output, but keep the conclusions separated by route
  type.
- Required docs to update:
  Run metrics go to `docs/EXPERIMENT_LOG.md`; artifact retention decisions go
  to `docs/HAZE4K_RUN_MANIFEST.md`; only the current actionable pointer belongs
  in `docs/CURRENT_CONTEXT.md`.

## Launch Template

Use this only after syncing the branch to the intended CUDA server:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net
bash scripts/haze4k-official-warmstart-finetune.sh
```

Override the staged-freeze policy only for a named sub-route:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net
TRAINABLE_SCHEDULE='0:lf_prior;5001:lf_prior,bottleneck;20001:all' \
  MODEL_NAME=DEA-Net-OfficialWarmStart-LFv1-custom-stage-H4K-finetune100k-$(date +%Y%m%d-%H%M%S) \
  bash scripts/haze4k-official-warmstart-finetune.sh
```

Optional pure-CR warm-start baseline:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net
USE_LF_PRIOR=0 MODEL_NAME=DEA-Net-OfficialWarmStart-CR-H4K-finetune100k-$(date +%Y%m%d-%H%M%S) \
  bash scripts/haze4k-official-warmstart-finetune.sh
```

Short smoke is allowed only as diagnostic:

```bash
cd /root/autodl-tmp/workspace/Dehaze-Net
ALLOW_SHORT_WARMSTART=1 EPOCHS=1 ITERS_PER_EPOCH=20 MAX_TEST_BATCHES=2 \
  MODEL_NAME=smoke-OfficialWarmStart-H4K-$(date +%Y%m%d-%H%M%S) \
  bash scripts/haze4k-official-warmstart-finetune.sh
```
