# HAZE4K Official-Centric Marginal Gain Audit

Date: 2026-05-30

Status: Completed Stage 0 audit. Recommendation
`do_not_train_official_adapter_yet`. This card authorized only the
official-centric join, oracle, and tiny probe audit on `autodl-dehaze`; the
completed audit did not authorize a 100k HAZE4K model scout or a tiny
official-default adapter card.

Execution boundary: run extraction and probe fitting on `autodl-dehaze` in
`/root/autodl-tmp/workspace/Dehaze-Net`. Local WSL work is limited to source,
documentation, Git, packaging, and compact sync.

## Most Valuable Attempt

- Why this is the most valuable current attempt:
  The latest strong-CR abstention/SNR audit failed because deployable features
  did not stably preserve strong cold-start CR no-change samples. The user has
  reframed the bottleneck around official/strong-CR low-SNR no-change behavior,
  so the next useful question is whether official failures are separable at all.
- Cheap preflight evidence:
  Reuse the same 1000 HAZE4K test images and existing outputs/checkpoints:
  official step0, official warm-start best, cold-start CR, LF-v1,
  ResidualCalib, CRPlus-v2, and CBRFRC-v1.
- Earliest decisive gate:
  This audit. If the official oracle has little marginal headroom, or if
  official-failure labels fail held-out deployability, do not launch another
  HAZE4K PSNR-gain training run.
- Expected training-time or attempt-count saving:
  A failed audit blocks another large model/loss search and shifts work away
  from HAZE4K mean PSNR until a stable deployment variable appears.
- What success decides:
  A tiny official-default adapter route card is worth writing, with the first
  gate limited to 0-train or <=10k per-image safety.
- What failure decides:
  The official failure subset is either too small/noisy or not deployably
  separable under CR-strength, A/beta, content, and low-residual-energy
  held-outs.

## Hypothesis

- Prior evidence:
  Official warm-start LF-v1 improved only `+0.0114 dB` on average over matched
  official step0, but per-image candidate oracles over cold-start routes showed
  large local complementarity. Strong-CR abstention/SNR and representation
  audits found signal on random splits but collapsed under CR-strength
  held-out tests.
- Target failure mode:
  Official has a subset of low-SNR no-change images where intervention must be
  extremely rare, mixed with a failure subset where an existing candidate gives
  a meaningful `>=0.15/0.20 dB` gain.
- Mechanism hypothesis:
  If official failures are real deployment targets, small probes using
  official-output and candidate-output features should identify them with high
  precision while keeping strong-official no-change false interventions near
  zero across held-out conditions.

## Stage 0 Audit

- Code branch:
  `codex/haze4k-official-marginal-gain-audit`.
- Primary variable:
  The reference changes from cold-start CR to official step0.
- Explicitly disabled related mechanisms:
  No 100k model training, no BRFRC-v3, no selector-v2, no new deep adapter,
  no joint fine-tune, no CRPlus schedule, no TeacherGuard, no Mamba/DWT branch.
- Tiny probe family:
  Logistic regression, ridge classifier, and histogram gradient boosting.
  These probes are diagnostics, not hidden deployment models.

## Inputs

Use the same HAZE4K 1000-image test join:

| Output | Source |
| --- | --- |
| official step0 | `official_warmstart_step0.pk` from the isolated warm-start run |
| official warm-start best | `DEA-Net-OfficialWarmStart-LFv1-H4K-local-scout100k-20260528-152808/saved_model/best.pk` |
| cold-start CR | `DEA-Net-CR-H4K-Baseline-scout-20260520-101334/saved_model/best.pk` |
| LF-v1 | `DEA-Net-LF-H4K-scout-20260521-003100/saved_model/best.pk` |
| ResidualCalib | `DEA-Net-LF-ResidualCalib-H4K-scout100k-20260525-122654/saved_model/best.pk` |
| CRPlus-v2 | checkpoint if present, otherwise committed route-evidence CSV metrics |
| CBRFRC-v1 | `DEA-Net-CBRFRC-v1-H4K-scout100k-20260530-012811/saved_model/best.pk` |

## Labels And Splits

Positive labels:

- official can be improved by at least one candidate by `>=0.15 dB`;
- official can be improved by at least one candidate by `>=0.20 dB`.

No-change labels:

- best candidate gain is `<=0.05 dB`;
- strong-official/no-change is official-strength q4 and best candidate gain
  `<=0.05 dB`.

Held-out families:

- random image split;
- official-strength and cold-start CR-strength held-out;
- airlight and beta held-out;
- content-cluster held-out from deployable no-reference official-output
  features;
- residual-low-energy held-out from deployable candidate-vs-official
  residual energy bins.

## Pass Line

All criteria must pass for the same deployable feature row and head:

| Metric | Pass line |
| --- | ---: |
| intervention precision | `>= 0.75` |
| strong-official no-change false intervention | `<= 0.05` |
| all no-change false intervention | `<= 0.10` |
| bootstrap gain p05 | `> 0.00 dB` |
| mean oracle-gated gain | `>= 0.02 dB` |
| shuffled precision gap | `>= 0.05` |
| held-out families | no material collapse in every required family |

Simulated/oracle-gated PSNR is diagnostic only. It decides whether a route card
is worth writing; it is not a publishable model result.

## If Stage 0 Passes

Only then write a separate tiny adapter route card:

```text
Official-default tiny adapter

Default:
    J = J_official

First gate:
    0-train or <=10k per-image official-vs-adapter safety

Allowed correction:
    bounded low-frequency/color correction only

Hard constraint:
    if not selected, output is bitwise/functionally official no-op
```

Do not launch a 100k large model from this audit alone.

## Completed Result

- Run:
  `HAZE4K-official-marginal-gain-audit-autodl-20260530-hgb-full` completed on
  `autodl-dehaze` at `2026-05-30 18:51 CST`.
- Oracle over official:
  mean official PSNR `34.2556`; mean oracle-with-no-change PSNR `34.9015` and
  mean gain `+0.6459 dB`; mean best-candidate gain `+0.5532 dB`; candidate
  gains `>=0.15 / 0.20 dB` on `523 / 493` images.
- Best-candidate counts:
  official warm-start best `665`, cold-start CR `61`, LF-v1 `94`,
  ResidualCalib `85`, CRPlus-v2 `67`, CBRFRC-v1 `28`.
- Deployability verdict:
  `0 / 448` HGB summary rows passed the written line. The best eligible random
  deployable rows were `A_official_output` and `C_official_internal` at margin
  `0.20`, each with intervention precision `0.6597`, strong-official
  no-change false intervention `0.4499`, all no-change false intervention
  `0.6050`, bootstrap gain p05 `0.5158 dB`, mean oracle-gated gain
  `0.6440 dB`, and shuffled precision gap `0.0462`.
- Leakage control:
  `Z_diagnostic_leakage` rows reached intervention precision `1.0` and zero
  no-change false intervention on many held-outs, confirming the label is real,
  but those rows are explicitly ineligible and therefore do not provide a
  deployment variable.
- Decision:
  official failures are real, but they are not deployably separable with the
  tested official/candidate output features across official-strength,
  CR-strength, airlight, beta, content-cluster, and residual-low-energy
  held-outs. Do not write the tiny official-default adapter route card; do not
  launch new HAZE4K official-PSNR training from this audit alone.
