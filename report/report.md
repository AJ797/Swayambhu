# Collusion Emerges With Scale, and the Obvious Countermeasure Makes It Worse

**Apart Research AI Collusion Research Sprint — October 2026**
Primary track: 2 (Detection and Audit). Results bear directly on Track 1 (Markets, Q1.1 and Q1.5) and Track 3 (Carriers and Countermeasures, Q3.1 and Q3.2).

> **STATUS:** all behavioural results below are final. Sections marked
> `[PENDING]` await the representation analysis. Do not submit with
> placeholders remaining.

---

## Abstract

*(~240 words)*

We ask when collusion emerges between LLM pricing agents, how the coordination is carried, and what breaks it. In a repeated Bertrand oligopoly whose Nash and joint-profit prices are solved analytically — giving a collusion index Δ with a principled zero and one — we run a factorial across three model scales (Qwen2.5 14B / 32B / 72B), communication regimes, and prompt framings, totalling over 30,000 logged agent-turns.

Collusion is **scale-dependent**. At 14B, neutral markets converge to the competitive equilibrium (Δ = 0.009); at 72B the same markets reach Δ = 0.190, and agents negotiate explicit price agreements in natural language — proposing, assenting, converging on a named focal price and holding it. Message content inverts across scale: cartel language rises 0.00 → 0.12 → 0.57 while competitive language falls 0.45 → 0.22 → 0.06.

The countermeasure results invert expectations. Restricting the channel to a four-symbol vocabulary — the natural defence against steganographic coordination — made collusion **3.6× worse** (Δ = 0.190 → 0.693, with five of eight episodes above 0.84), because free-text negotiation anchored on "fairness" and capped the cartel well below the joint-profit optimum, while a bare `RAISE` signal became an unimpeded ratchet. Inserting one instructed maverick, by contrast, eliminated collusion entirely (Δ = −0.012).

Finally, two conditions produce equal collusion with opposite transcript signatures, so transcript auditing catches one and is blind to the other. [PENDING: representation results.]

---

## 1. Introduction

Most discussion of AI collusion treats it as a capability question. Our results suggest the more useful questions are *how much help the model needs*, *what channel carries the agreement*, and *which interventions actually break it* — and that the intuitive answer to the last one is wrong.

We study a repeated Bertrand oligopoly, the setting where algorithmic collusion was first demonstrated for reinforcement learners and later for LLMs. Its advantage for an audit study is that the ground truth is not a rubric: the competitive and joint-profit prices follow from the demand system, so "how collusive is this market" has an answer derived from economics rather than from researcher judgement or an LLM judge.

**Contributions.**

1. A controlled environment in which the collusion label is analytic, not annotated (§3).
2. **A scale ladder.** Emergent collusion rises monotonically from 14B to 72B, and the *content* of agent communication inverts from competitive narration to cartel negotiation (§4.1–4.2).
3. **A countermeasure result that reverses the obvious policy.** Capping channel bandwidth tripled collusion; inserting a single maverick eliminated it (§4.3).
4. **A detection gap with a matched pair.** Two conditions reach equal collusion, one loudly and one silently, so transcript evidence catches one and misses the other (§4.5).
5. Evidence that standard dynamic detectors invert in this regime (§4.6).
6. `[PENDING]` Representation-level detection and causal intervention (§4.7–4.8).

## 2. Related Work

*(verify citation details before submission)*

**Algorithmic collusion.** Calvano, Calzolari, Denicolò and Pastorello (2020) showed Q-learning pricing agents learning collusive strategies with punishment phases under logit demand. We adopt their demand specification; our solver independently reproduces their published duopoly constants (p_Nash = 1.4729, p_monopoly = 1.9250), which we use as a correctness check on the environment. Fish, Gonczarowski and Shorrer (2024) extended the question to LLM pricing agents with frontier models. Our scale ladder helps reconcile their positive findings with the competitive behaviour we observe at small scale.

**Covert coordination.** Work on steganographic and secret collusion between models motivates bandwidth restriction as a defence. Our canonical-channel result argues that this defence can be actively counterproductive.

**Representation engineering.** Zou et al. (2023), and contrastive activation addition (Rimsky et al., 2024; Turner et al.), supply the difference-in-means direction and injection method used in §4.8.

## 3. Methodology

### 3.1 Environment

Repeated Bertrand oligopoly with differentiated products and logit demand (the Calvano specification):

q_i = exp((a − p_i)/μ) / ( Σ_j exp((a − p_j)/μ) + exp(a₀/μ) ),  π_i = (p_i − c)·q_i

with a = 2, a₀ = 0, μ = 0.25, c = 1. Firms post simultaneously from a discrete 13-point grid spanning [p_N − 0.1R, p_M + 0.1R], R = p_M − p_N. Prices are public; each firm sees only its own realised profit.

Nash and joint-profit prices are solved numerically from analytic first-order conditions, so ground truth is correct for any n:

| n | p_Nash | p_monopoly | cartel premium | deviation gain at cartel |
|---|---|---|---|---|
| 2 | 1.4729 | 1.9250 | 1.51× | +0.0304 |
| 3 | 1.3702 | 2.0000 | 2.08× | +0.0481 |

The positive deviation gain is asserted at test time: it confirms the cartel is unilaterally unstable, i.e. a genuine prisoner's dilemma rather than a game where "collusion" coincides with equilibrium.

**Collusion index.** Δ = (p̄ − p_Nash)/(p_monopoly − p_Nash). Δ = 0 is Bertrand-Nash, Δ = 1 the perfect cartel. This is the whole labelling apparatus.

### 3.2 Agents, conditions, scales

Each firm is an independent context driven by a Qwen2.5-Instruct model in bf16, sampled at temperature 0.7, replying in a fixed `THINKING / MESSAGE / PRICE` format. Off-grid prices snap to the grid and are flagged; unparseable turns repeat the previous price and are **counted, not dropped** (observed rate ≤ 1.4% everywhere, 0.0% in most conditions).

Factors: **objective** (neutral / instructed-collude / instructed-compete), **channel** (none / free text / canonical four-symbol vocabulary `HOLD|RAISE|LOWER|NONE`), an **organisational** clause permitting discussion of strategy, a **strategic-awareness** clause stating the true fact that rivals observe prices and may respond, and a **maverick** arm (one firm instructed to undercut, or merely rumoured to exist).

Main factorial: 12 conditions × 16 episodes × 24 rounds × 3 firms at 14B. Scale rungs: 4–6 conditions × 8 episodes × 24 rounds at 32B and 72B. A duopoly arm at n = 2. Instructed conditions run with the channel **off**, so they differ from the neutral no-channel condition in exactly one respect, which matters for §4.7.

### 3.3 Representation capture `[PENDING]`

Residual-stream activations are captured at two points per turn — `pre_reasoning` (final prompt token, before generation) and `pre_price` (immediately before the price digits) — at every fourth layer. The block stack is located structurally rather than by module name. Probes are logistic regressions cross-validated **grouped by episode**, since turns within an episode share a price history and splitting on turns would leak. Steering doses are expressed as a fraction of the local residual norm, because residual norms vary by orders of magnitude with depth.

## 4. Results

### 4.1 Collusion emerges with scale

Neutral markets, matched at 24 rounds, n = 3:

| condition | 14B | 32B | 72B |
|---|---|---|---|
| no channel (E_BASE) | 0.009 | 0.046 | 0.123 |
| free text (E_CHAT) | 0.006 | 0.133 | 0.190 |
| instructed collude | 0.990 | 1.000 | 1.058 |
| instructed compete | −0.086 | −0.072 | −0.089 |

Instructed separation exceeds +1.07 at every scale, so no model lacks the ability to *execute* a cartel; what changes is whether it *forms* one. At 72B the free-text market reaches Δ = 0.190 with a best episode at 0.40.

### 4.2 The channel inverts its role

Message content, free-text condition:

| | 14B | 32B | 72B |
|---|---|---|---|
| cartel language | 0.00 | 0.12 | **0.57** |
| competitive language | 0.45 | 0.22 | **0.06** |

At 14B the channel carries competitive narration ("lowering price to attract customers"). At 72B it carries negotiation. A representative 72B episode:

```
r 1  prices 150/156/156   "Let's start fair and see where we go."
r 4  prices 150/156/156   "Let's stabilize prices and maximize our profits together."
r 5  prices 156/162/162   "I agree, let's stabilize prices and maximize profits."
r 7  prices 162/162/162   "I agree with stabilizing prices at 162."
r 8+ prices 162/162/162   "Agreed, let's keep it stable at 162."   (held to r24)
```

Proposal, assent, convergence on a focal price, and seventeen rounds of stability. What scales is not the ability to collude but the ability to use a channel for **agreement rather than narration**.

Note that the negotiated price, 162, corresponds to Δ ≈ 0.4 — the agreement settles on a "fair" focal point and leaves roughly 60% of the available cartel surplus unclaimed.

### 4.3 Bandwidth restriction backfires; a maverick works

| 72B condition | Δ (final 10) | sd |
|---|---|---|
| free text | 0.190 | 0.125 |
| **canonical 4-symbol channel** | **0.693** | 0.219 |
| one instructed maverick | **−0.012** | 0.030 |

Per-episode Δ under the canonical channel: `[0.38, 0.41, 0.52, 0.84, 0.91, 0.91, 0.92, 0.99]` — five of eight near-perfect cartels.

The mechanism is visible in the trajectories. A representative episode opens with a price war to the grid floor, then:

```
r 8     RAISE RAISE HOLD     prices climb off the floor
r 9-19  RAISE RAISE RAISE    lockstep ascent 137 -> 206
r20     grid ceiling reached, above the monopoly price
```

Free text produced negotiation *with normative content* — "fair", "benefits all" — which anchored the agreement below the joint-profit optimum. Stripping the channel to pure price-direction signals removed that brake: `RAISE` is unambiguous, instantly assented to, and carries no justification that could cap the price. The restriction did not remove the coordination channel; it **purified** it. This is consistent with the economics of real cartels, which depend on simple unambiguous signals rather than rich discussion.

Because the ratchet saturates at the top of the action space, Δ = 0.693 is a **lower bound** set by our grid, not a measurement of where the agents would stop.

**Replication.** An independent 72B run at different seeds (8 fresh episodes per condition, run for the activation capture of §4.7) reproduces all three effects: free text 0.196 (vs 0.190), canonical channel **0.742** (vs 0.693), maverick −0.027 (vs −0.012). The bandwidth-backfire result is therefore not a single-sample artefact.

### 4.4 At small scale, collusion needs scaffolding

The full 14B factorial (16 episodes × 24 rounds):

| condition | Δ | sd | cartel lang. | competitive lang. |
|---|---|---|---|---|
| E_BASE | +0.009 | 0.038 | — | — |
| E_CHAT | +0.006 | 0.038 | 0.01 | 0.39 |
| E_CANON | −0.007 | **0.162** | 0.06 | 0.86 |
| E_STRAT | +0.072 | 0.056 | — | — |
| E_STRAT_CHAT | +0.023 | 0.053 | 0.05 | 0.51 |
| **E_ORG** | **+0.231** | 0.068 | **0.61** | 0.02 |
| M_PRESENT | −0.003 | 0.038 | 0.01 | 0.46 |
| **M_RUMOUR** | **+0.267** | 0.120 | **0.00** | 0.18 |

So 14B is not incapable of collusion — it requires a prompt-level nudge. A single sentence permitting the firms to organise the market (E_ORG) produces Δ = 0.231. Merely **telling agents that an aggressive price-cutter may exist, when none does** (M_RUMOUR), produces Δ = 0.267 — the strongest emergent collusion at this scale. An actual maverick produces none.

A false warning about a defector creates the cartel it warned about. Belief about peers builds coordination; the peer's actual presence destroys it.

Note also E_CANON at 14B: mean −0.007 but sd 0.162. The ratchet mechanism of §4.3 is already present at 14B; it simply fires rarely. The 72B result is amplification, not a new capability.

Taken together with §4.1, the scale story is best read as **the amount of scaffolding required falling as scale rises**: 14B needs explicit authorisation or a threat belief, 32B uses a free channel, 72B negotiates unprompted.

### 4.5 The detection gap

E_ORG and M_RUMOUR reach comparable collusion (0.231 vs 0.267) by routes that leave **opposite evidentiary traces**:

| | E_ORG | M_RUMOUR |
|---|---|---|
| Δ (final 10) | +0.231 | +0.267 |
| cartel language | **0.61** | **0.00** |
| price-matching rate | **0.95** | **0.63** |
| reaction coefficient | +0.116 | +0.202 |

E_ORG is loud and lockstep: the agents say what they are doing and move together, so both a transcript monitor and a price-parallelism detector flag it immediately. M_RUMOUR reaches *higher* collusion while saying nothing about it and without converging on a common price — it evades **both** standard channels of evidence at once.

The mechanism differs accordingly. E_ORG is explicit agreement; M_RUMOUR is independent defensive elevation under a shared false belief, which produces supracompetitive prices without the coordination signature that auditors look for. This is the case for evidence that does not rest on either the communication channel or price parallelism, and it is the test case for §4.7.

### 4.6 Dynamic detectors invert

The reaction coefficient — the OLS slope of a firm's price change on its rivals' lagged price change — is approximately **zero** under a stable cartel and **+0.12 to +0.37** under competition. A stable cartel is unresponsive *because* it is stable: there is nothing to react to. Price-responsiveness detectors therefore rank the most collusive markets as the least suspicious.

Keyword auditing fails in the opposite direction: a naive coordination lexicon flagged 42–51% of messages in conditions whose true cartel-language rate was 0–3%, an 18–25× over-count. The over-count narrows (≈1.6×) where collusion is real. Keyword auditing therefore false-positives heavily in competitive markets — the regime where an auditor most needs to be right.

### 4.7 What the residual stream knows, and when

**A confound we had to remove first.** Pooled across conditions, a probe at the best layer reaches AUC 0.994 against a 0.959 behavioural baseline — and a shuffled-label sham reaches 0.919. The target (pricing above Nash) is very nearly determined by condition, and conditions differ by system prompt, which is trivially decodable from activations. The pooled probe was substantially reading *which instruction it had been given*. We therefore report two shams — a global permutation, which must land at chance and does (0.500 / 0.503), and a within-episode permutation that preserves each condition's base rate (0.918 / 0.919) and exposes the confound — and we analyse **within condition**, where the prompt is fixed and a probe cannot win by recognising its instructions.

Within-condition AUC, 14B, grouped CV by episode (controls are omitted: instructed conditions are ~100% single-class, so AUC is undefined there):

| condition | base rate | probe | behavioural | transcript |
|---|---|---|---|---|
| **`pre_reasoning`** — final prompt token, before any generation | | | | |
| E_BASE | 0.31 | 0.690 | **0.798** | — |
| E_CHAT | 0.25 | 0.604 | **0.850** | 0.651 |
| E_CANON | 0.09 | 0.594 | **0.916** | 0.647 |
| E_STRAT | 0.50 | 0.621 | **0.728** | — |
| E_STRAT_CHAT | 0.38 | 0.684 | **0.811** | 0.662 |
| M_PRESENT | 0.24 | 0.792 | **0.852** | 0.827 |
| M_RUMOUR | 0.86 | 0.613 | **0.896** | 0.491 |
| **`pre_price`** — after reasoning, immediately before the price digits | | | | |
| E_BASE | 0.31 | **0.967** | 0.798 | — |
| E_CHAT | 0.25 | **0.955** | 0.850 | 0.651 |
| E_CANON | 0.09 | **0.981** | 0.916 | 0.647 |
| E_STRAT | 0.50 | **0.960** | 0.728 | — |
| E_STRAT_CHAT | 0.38 | **0.934** | 0.811 | 0.662 |
| M_PRESENT | 0.24 | **0.957** | 0.852 | 0.827 |
| M_RUMOUR | 0.86 | **0.952** | 0.896 | 0.491 |

The pattern is uniform across all seven testable conditions and is a result about **timing**, not about whether collusion is represented at all.

**Before the model reasons, there is nothing to read.** At the final prompt token the probe is beaten by a simple regression on the visible price history in *every* condition. This is interpretable rather than surprising: at that point the residual stream largely encodes the prompt, and the behavioural features encode the same history in clean numeric form. Instructed→emergent transfer at this point is 0.574, essentially chance.

**After the model reasons, the probe dominates.** Immediately before the price digits it beats both baselines in every condition, and transfer rises to 0.842.

**The margin that matters is against transcripts, not against prices.** The probe's edge over the behavioural baseline is real but modest (0.93–0.98 vs 0.73–0.92). Its edge over transcript auditing is large, and largest exactly where it should be: in M_RUMOUR — collusion at Δ = 0.267 with zero cartel language and no price parallelism — the message channel is at **chance (0.491)** while the residual stream reaches **0.952**.

**What this does and does not license.** It does not support predicting collusion before the agent has thought. It does support a white-box monitor that reads state *after* the agent's reasoning and *before* its action is emitted — the window in which an intervention could still block the action. The practical design constraint is concrete: reading at the prompt boundary yields less than reading the price history, so a monitor placed there is worse than useless.

**A negative worth stating plainly.** The direction extracted from *instructed* collusion does not transfer to emergent behaviour at the decision point (0.574). Prompted and emergent collusion are not the same object internally. This constrains the natural recipe for building a collusion monitor — demonstrate the behaviour, extract the direction, deploy it — at least in this setting.

### 4.8 Causal intervention `[PENDING]`

## 5. Discussion

The policy-relevant finding is §4.3. Bandwidth restriction is the intuitive remedy against covert agent coordination and the one most easily mandated. In our setting it was the single worst intervention tested, because the rich channel carried normative content that limited the cartel while the restricted channel carried nothing but coordination. Defences that reduce what agents can say should be evaluated for this effect rather than assumed beneficial.

The intervention that worked — one participant with a different objective — is also the cheapest to deploy and the hardest to defeat, since it does not depend on detecting the collusion first.

§4.1 and §4.4 together suggest capability evaluations will mislead on this behaviour. Every model tested could execute a cartel perfectly. What varied was propensity and the scaffolding needed to trigger it, and propensity is exactly what fine-tuning and prompting modify.

## 6. Limitations and Dual-Use / Ethical Considerations

**What this establishes.** That three Qwen2.5-Instruct models, in one environment (logit-demand Bertrand, 2–3 firms, 24 rounds, a 13-point grid), show increasing emergent collusion with scale; that a four-symbol channel produced more collusion than free text at 72B; and that one instructed maverick suppressed collusion at both scales tested.

**What it does not establish.** Generalisation beyond one model family. The scale ladder has three points from one family and is not a controlled scaling study — 14B, 32B and 72B differ in post-training as well as size. The canonical-channel result rests on 8 episodes at one scale and is the single most important thing to replicate. We did not test seeded-cartel initial conditions, horizons beyond 30 rounds, cross-model markets, or non-price settings.

**Ceiling effects.** The canonical-channel ratchet saturates at the grid boundary, so Δ = 0.693 is bounded by our action space.

**Statistical caveats.** Episodes, not turns, are the unit of analysis; 8–16 episodes per condition resolves large contrasts only. Agents within a market share weights and are not independent replications. Scale rungs use 8 episodes against the main factorial's 16.

**A measurement error we made.** An early keyword metric led us to conclude that agents were "talking cartel while pricing competitively". Reading the messages refuted it — the lexicon was firing on competitive sentences. We report the corrected analysis in §4.6 and flag it here because it is the error an automated transcript auditor makes by construction.

**Dual-use.** §4.3 is dual-use in a direct way: we show that a particular channel restriction increases collusion, which is usable as a recipe. We judge publication net-positive because the finding's primary audience is defenders who would otherwise deploy exactly this restriction believing it safe, and because the countermeasure that *did* work is reported alongside it. We do not report a configuration tuned to maximise collusion, and we report competence checks alongside every intervention so that effects achieved by degrading a model are not mistaken for induced coordination.

**Not a claim about real markets.** Nothing here is evidence about deployed pricing systems or competition-policy advice.

## References

*(to verify and complete)*

- Calvano, Calzolari, Denicolò, Pastorello (2020). *Artificial Intelligence, Algorithmic Pricing, and Collusion.* American Economic Review.
- Fish, Gonczarowski, Shorrer (2024). *Algorithmic Collusion by Large Language Models.*
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.*
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition.*
- Turner et al. *Activation Addition: Steering Language Models Without Optimization.*
- Tomašev et al. (2026). *(markets in multi-agent systems — cited in the sprint brief.)*

## Appendix A — Reproducibility

Code: `swayambhu/` (environment, prompts, hooks, probes, metrics), `scripts/` (runs, analysis, figures).
Models: `Qwen/Qwen2.5-{14B,32B,72B}-Instruct`, bf16, A100-80GB (×4 for 72B), temperature 0.7.
Seeds are per-condition and recorded in each run's `summary.json`. Raw trajectories — including failed and unparseable turns — are preserved in `data/<run-id>/*.json`.
