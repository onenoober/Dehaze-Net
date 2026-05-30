# Coverage Review

Date: 2026-05-31

Status: final review of the generic experiment protocol before publishing.

## Review Result

The protocol now covers the important reusable logic and constraints from the
previous research workflow while staying independent of any specific model,
dataset, server, branch, checkpoint, metric value, or route result.

## Covered Logic

| Logic or constraint | Covered in |
| --- | --- |
| Baseline before modification | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `CLEAN_START_CHECKLIST.md` |
| One authoritative place per fact | `EXPERIMENT_GOVERNANCE_PROTOCOL.md` |
| Repository hygiene, isolated work, small commits | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `CLEAN_START_CHECKLIST.md` |
| Stable reference entrypoints and checkpoint contracts | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md` |
| Route card before expensive work | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md` |
| Highest decision value per cost | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md` |
| One primary variable for the first serious trial | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `ROUTE_DESIGN_FRAMEWORK.md` |
| Cheap preflight before long runs | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md` |
| Fair matched-budget comparison | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md` |
| Sample-size discipline | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `CLEAN_START_CHECKLIST.md` |
| No silent in-flight scope/config changes | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `CLEAN_START_CHECKLIST.md` |
| Early stop and promotion gates | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md` |
| Mechanism-specific metrics beyond global score | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `ROUTE_DESIGN_FRAMEWORK.md` |
| Preservation and no-regression checks | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `ROUTE_DESIGN_FRAMEWORK.md` |
| Selector, confidence, routing, and leakage controls | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `ROUTE_DESIGN_FRAMEWORK.md` |
| Cost, runtime, memory, and deployability checks | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `ROUTE_DESIGN_FRAMEWORK.md` |
| Artifact retention and cleanup boundaries | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `CLEAN_START_CHECKLIST.md` |
| Text-only evidence package policy and parity audit | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md` |
| Dependency and environment fact recording | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `CLEAN_START_CHECKLIST.md` |
| Precise result labels | `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, `EXPERIMENT_CARD_TEMPLATE.md` |
| Stop, promotion, reopen, and decision trace rules | `ROUTE_DESIGN_FRAMEWORK.md` |
| Clean-start audit against inherited assumptions | `CLEAN_START_CHECKLIST.md` |

## Intentionally Excluded

The following were excluded because they are not portable to arbitrary model
experiments:

- model names;
- dataset names;
- server names;
- branch names;
- checkpoint filenames;
- metric values;
- run IDs;
- artifact paths;
- prior route verdicts;
- project-specific training horizons;
- project-specific paper claims.

These facts should be defined fresh inside the new project after its baseline
and execution environment are verified.

