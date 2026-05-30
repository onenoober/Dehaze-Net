# Strong-Official No-Change Depth Safety Audit Protocol

Main question: Can predicted depth reduce false intervention on strong-official/no-change samples beyond C1 basic image stats?

Labels:

- strong-official no-change: official_strength_bin == strong_official_q4 and best_candidate_gain_vs_official <= 0.05
- all no-change: best_candidate_gain_vs_official <= 0.05
- intervention-worthy: best_candidate_gain_vs_official >= 0.20

Feature groups:

- C0: deployable official/candidate output, residual, and DEA activation features.
- C1: C0 plus basic image statistics.
- C2: C1 plus candidate-free depth from hazy and official output.
- C3: C1 plus candidate-free and candidate-aware depth features.

Splits:

- random image split for sanity check.
- train non-strong official, test strong official.
- train weak/mid CR, test strong CR.
- train normal/high residual energy, test low residual energy.

Controls:

- shuffled-depth image-id control.
- C2 vs C1 basic-stat control.
- label-permutation control.

Primary metric: strong-official no-change false intervention rate.
