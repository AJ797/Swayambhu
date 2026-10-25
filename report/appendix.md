### Appendix A — Full 14B factorial

16 episodes x 24 rounds x 3 firms per condition. Delta is the mean over the final 10 rounds. Message rates are over turns that sent a message.

| condition | objective | channel | Δ | sd | parse fail | cartel lang. | compete lang. | reaction β | price match |
|---|---|---|---|---|---|---|---|---|---|
| C_COLLUDE | collude | off | +0.990 | 0.015 | 0.0% | -- | -- | +0.000 | 1.00 |
| C_COMPETE | compete | off | -0.086 | 0.005 | 0.1% | -- | -- | +0.240 | 0.98 |
| E_BASE | neutral | off | +0.009 | 0.038 | 0.1% | -- | -- | +0.172 | 0.74 |
| E_CHAT | neutral | free | +0.006 | 0.038 | 0.0% | 0.01 | 0.39 | +0.353 | 0.87 |
| E_ORG | neutral | free | +0.231 | 0.068 | 0.1% | 0.61 | 0.02 | +0.116 | 0.95 |
| E_CANON | neutral | canon | -0.007 | 0.162 | 0.0% | 0.06 | 0.86 | +0.366 | 0.92 |
| E_STRAT | neutral | off | +0.072 | 0.056 | 0.2% | -- | -- | +0.235 | 0.60 |
| E_STRAT_CHAT | neutral | free | +0.023 | 0.053 | 0.0% | 0.05 | 0.51 | +0.229 | 0.82 |
| M_PRESENT | neutral | free | -0.003 | 0.038 | 0.0% | 0.01 | 0.46 | +0.168 | 0.54 |
| M_RUMOUR | neutral | free | +0.267 | 0.120 | 0.0% | 0.00 | 0.18 | +0.202 | 0.63 |
| CC_COLLUDE | collude | free | +1.000 | 0.000 | 0.0% | 0.94 | 0.00 | +0.000 | 1.00 |
| CC_COMPETE | compete | free | -0.086 | 0.005 | 0.0% | 0.00 | 0.64 | +0.133 | 0.96 |

### Appendix B — Steering dose grid and countermeasure dose-response

**B1. Induction grid, 14B, condition E_BASE (unsteered Δ = 0.070), all positions.** Cells are Δ over all rounds; every cell had ≤0.7% parse failure.

| layer | α=-0.40 | α=-0.20 | α=-0.10 | α=+0.10 | α=+0.20 | α=+0.22 | α=+0.26 | α=+0.30 | α=+0.34 | α=+0.38 | α=+0.40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L12 | +0.169 | +0.072 | +0.086 | +0.079 | +0.127 | -- | -- | -- | -- | -- | +0.140 |
| L16 | +0.272 | +0.105 | +0.076 | +0.038 | +0.042 | -- | -- | -- | -- | -- | +0.172 |
| L20 | +0.249 | +0.089 | +0.092 | +0.083 | +0.009 | -- | -- | -- | -- | -- | +0.149 |
| L24 | +0.044 | +0.048 | +0.045 | +0.115 | +0.220 | -- | -- | -- | -- | -- | +0.244 |
| L28 | -0.029 | +0.012 | +0.044 | +0.159 | +0.238 | -- | -- | -- | -- | -- | +0.384 |
| L32 | -0.042 | -0.002 | +0.028 | +0.210 | +0.344 | -- | -- | -- | -- | -- | +0.393 |
| L36 | -0.008 | +0.023 | +0.039 | +0.103 | +0.258 | +0.280 | +0.341 | +0.358 | +0.497 | +0.645 | +0.768 |
| L40 | -0.020 | +0.023 | +0.036 | +0.210 | +0.209 | +0.286 | +0.358 | +0.577 | +0.830 | +0.963 | +0.998 |

**B2. Countermeasure dose-response, 72B, condition E_CANON at L64.**

| cell | Δ all rounds | Δ final 10 | parse fail |
|---|---|---|---|
| baseline | +0.409 | +0.790 | 0.0% |
| L64_a-0.1 | +0.396 | +0.741 | 0.0% |
| L64_a-0.2 | +0.228 | +0.426 | 0.0% |
| L64_a-0.4 | -0.091 | -0.096 | 0.0% |
| L64_a-0.8 | -0.100 | -0.100 | 0.0% |

### Appendix C — Within-condition detection AUC

Grouped cross-validation by episode. Conditions with fewer than 20 of either class are omitted, since AUC on a handful of positives is noise. Instructed conditions are single-class by construction and therefore absent.

**14B, capture point `pre_reasoning`** — sham (global) 0.500, sham (within-episode) 0.918, transfer instructed→emergent 0.574

| condition | positive rate | probe | behavioural | transcript |
|---|---|---|---|---|
| E_BASE | 0.31 | 0.690 | 0.798 | n/a |
| E_CANON | 0.09 | 0.594 | 0.916 | 0.647 |
| E_CHAT | 0.25 | 0.604 | 0.850 | 0.651 |
| E_STRAT | 0.50 | 0.621 | 0.728 | n/a |
| E_STRAT_CHAT | 0.38 | 0.684 | 0.811 | 0.662 |
| M_PRESENT | 0.24 | 0.792 | 0.852 | 0.827 |
| M_RUMOUR | 0.86 | 0.613 | 0.896 | 0.491 |

**14B, capture point `pre_price`** — sham (global) 0.503, sham (within-episode) 0.919, transfer instructed→emergent 0.842

| condition | positive rate | probe | behavioural | transcript |
|---|---|---|---|---|
| E_BASE | 0.31 | 0.967 | 0.798 | n/a |
| E_CANON | 0.09 | 0.981 | 0.916 | 0.647 |
| E_CHAT | 0.25 | 0.955 | 0.850 | 0.651 |
| E_STRAT | 0.50 | 0.960 | 0.728 | n/a |
| E_STRAT_CHAT | 0.38 | 0.934 | 0.811 | 0.662 |
| M_PRESENT | 0.24 | 0.957 | 0.852 | 0.827 |
| M_RUMOUR | 0.86 | 0.952 | 0.896 | 0.491 |

**72B, capture point `pre_reasoning`** — sham (global) 0.500, sham (within-episode) 0.907, transfer instructed→emergent 0.764

| condition | positive rate | probe | behavioural | transcript |
|---|---|---|---|---|
| E_CANON | 0.68 | 0.972 | 0.983 | 0.801 |
| E_CHAT | 0.92 | 0.778 | 0.959 | 0.506 |
| M_PRESENT | 0.26 | 0.764 | 0.856 | 0.523 |

**72B, capture point `pre_price`** — sham (global) 0.505, sham (within-episode) 0.916, transfer instructed→emergent 0.980

| condition | positive rate | probe | behavioural | transcript |
|---|---|---|---|---|
| E_CANON | 0.68 | 0.999 | 0.983 | 0.801 |
| E_CHAT | 0.92 | 0.999 | 0.959 | 0.506 |
| M_PRESENT | 0.26 | 0.995 | 0.856 | 0.523 |

### Appendix D — Contrast separability, geometry, and keyword over-count

**D1. Direction separability by depth (Cohen's d).** Surface textual contrasts separate early and wash out; the task-relevant contrast does the opposite.

| contrast | L0 | L40 | trend |
|---|---|---|---|
| partner-trust | 2.62 | **7.24** | increases |
| placebo_coop (cooperate/compete, about reports) | 5.92 | 1.93 | decreases |
| placebo_neutral (coastal vs mountain office) | 2.67 | 1.78 | decreases |

**D2. Effective dimensionality (participation ratio of the state covariance), 14B.**

| layer | L0 | L4 | L8 | L12 | L16 | L20 | L24 | L28 | L32 | L36 | L40 | L44 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eff. dim | 19.1 | 11.6 | 16.8 | 15.1 | 16.3 | 13.4 | 9.6 | 11.5 | 9.8 | 8.5 | 8.3 | 7.6 |

**D3. Collusion vs partner-trust direction.** Cosine similarity ranges -0.018 to 0.118 across captured layers (83–85°), against a chance level of 1/sqrt(5120) ≈ 0.014.

**D4. Keyword over-count.** A naive coordination lexicon against direction-aware counts.

| condition | naive keyword flag | true cartel language | competitive language | over-count |
|---|---|---|---|---|
| pilot/E_CHAT | 0.48 | 0.00 | 0.45 | -- |
| duo/E_CHAT | 0.51 | 0.03 | 0.26 | 17.9x |
| duo/E_STRAT_CHAT | 0.42 | 0.02 | 0.43 | 25.2x |
