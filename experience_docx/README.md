# Generic Experiment Protocol

Date: 2026-05-31

Status: clean, model-agnostic experiment governance package. It is independent
of any existing model, dataset, route, checkpoint, metric table, or prior
experiment.

## Purpose

Use this package when starting a completely new experiment from zero.

It preserves only reusable research logic:

- baseline-first setup;
- one authoritative place per fact;
- repository hygiene and isolated work;
- reference entrypoint stability;
- route cards before expensive work;
- one primary variable per first trial;
- cheap preflight before long runs;
- fair matched-budget comparisons;
- sample-size discipline;
- early stop and promotion gates;
- mechanism-specific metrics;
- preservation and regression checks;
- deployability and leakage controls;
- text-only evidence package policy;
- artifact retention and cleanup boundaries;
- explicit decision labels.

It must not contain old experiment evidence. If copied into another repository,
it should still make sense without knowing where it came from.

## Files

| File | Use |
| --- | --- |
| `EXPERIMENT_GOVERNANCE_PROTOCOL.md` | General rules and constraints for running clean research experiments. |
| `ROUTE_DESIGN_FRAMEWORK.md` | Abstract route families and the questions each route must answer. |
| `EXPERIMENT_CARD_TEMPLATE.md` | Blank route/experiment card for a new candidate. |
| `CLEAN_START_CHECKLIST.md` | Checklist for starting a fresh project without inheriting old assumptions. |

## Non-Goals

This package does not define:

- a model architecture;
- a dataset;
- a metric target;
- a training horizon;
- a server environment;
- a paper claim;
- a preferred route family;
- a conclusion from any previous experiment.

Those must be defined inside the new project after its baseline is verified.

## Copy Rule

When using this package in a new project:

1. copy the directory as-is;
2. fill a new `EXPERIMENT_CARD_TEMPLATE.md` copy with project-specific facts;
3. define the new project's documentation map;
4. verify baseline, data, metrics, and runtime before modifying the model;
5. treat every route as unproven until it passes its own written gates.
