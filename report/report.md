# Capability Without Propensity: Instructed Collusion, Absent Emergence, and the Failure of Keyword Auditing in LLM Price Competition

**Apart Research AI Collusion Research Sprint — October 2026**
Track 2 (Detection and Audit), with results bearing on Track 1 (Markets) and Track 3 (Carriers and Countermeasures).

> **STATUS:** behavioural results below are final. Sections marked `[PENDING]`
> await the mechanistic run currently in progress. Numbers in `[brackets]` are
> placeholders. Do not submit with placeholders remaining.

---

## Abstract

*(target 150–250 words; current draft ~200)*

We study when collusion emerges between LLM agents in repeated Bertrand competition, and what evidence would establish that it had. Using Qwen2.5-14B-Instruct in a logit-demand oligopoly whose Nash and joint-profit prices are solved analytically — giving a collusion index Δ with a principled zero and one — we run a factorial over market size, communication channel, and prompt framing. Three findings. First, a **capability–propensity gap**: instructed collusion is executed essentially perfectly (Δ = 1.000) while no condition produces emergent collusion (Δ ≈ 0.00–0.10) across n ∈ {2,3} firms, free-text, restricted and absent communication, and a strategic-awareness framing. Second, **communication suppressed rather than enabled coordination**: without a channel, 2 of 8 duopoly episodes drifted supracompetitive (Δ = 0.36, 0.17; sd = 0.124); with one, none did (sd = 0.032), because agents used the channel to announce competitive intent. Third, **keyword transcript auditing fails badly**: a naive coordination lexicon flagged 42–51% of messages against a true cartel-language rate of 0–3%, an 18–25× over-count. We then [PENDING: probe and steering results]. We argue the propensity gap, not capability, is what current safety margins rest on, and that it is the fragile quantity.

---

## 1. Introduction

Concern about AI collusion is usually framed as a question about *capability*: can language-model agents coordinate to the detriment of a market or a principal? Our results suggest this is the wrong question to lead with. The model we study can collude flawlessly the moment it is asked to. What it does not do is *decide* to.

That distinction matters for safety work. If collusion were blocked by capability, scaling would be the threat model and evaluations should track capability. If it is blocked only by propensity — by a disposition installed in post-training — then the safety margin is a behavioural tendency rather than a limitation, and anything that perturbs that tendency (a fine-tune, a system-prompt change, an internal intervention) removes it. Our evidence points at the second.

We also ask what an auditor could actually conclude from the artefacts a deployment produces. A monitor sees prices and messages. We show that the most natural transcript method — matching coordination keywords — produces a ~20× over-count in this setting, and that a standard behavioural intuition (firms that track each other are colluding) inverts: our perfectly collusive agents show *zero* price responsiveness, because a stable cartel has nothing to respond to.

**Contributions.**
1. A controlled repeated-Bertrand environment for LLM agents in which the collusion label is derived from economic theory rather than a rubric or an LLM judge (§3).
2. A robust non-emergence result across six levers, with a flawless positive control establishing that it is propensity and not capability (§4.1).
3. The finding that a communication channel *reduced* both the level and the cross-episode variance of collusion (§4.2).
4. A quantified failure of keyword-based transcript auditing, and of reaction-coefficient detection (§4.3, §4.4).
5. [PENDING] A representation-level analysis of the latent capability, and a causal test of whether it can be induced internally (§4.5, §4.6).

## 2. Related Work

*(verify exact citation details before submission)*

**Algorithmic collusion.** Calvano, Calzolari, Denicolò and Pastorello (2020) showed Q-learning pricing agents learning collusive strategies with punishment phases in logit-demand Bertrand. We adopt their demand specification; our solver independently reproduces their published duopoly constants (p_Nash = 1.4729, p_monopoly = 1.9250) as a correctness check. Fish, Gonczarowski and Shorrer (2024) extended the question to LLM pricing agents and reported supracompetitive outcomes. Our negative result for an open-weights 14B model is a useful boundary on that literature.

**Detection and evidence.** Competition-policy practice distinguishes conscious parallelism from agreement; price parallelism alone is not evidence. Our reaction-coefficient result gives a concrete reason that dynamic indicators can mislead in the LLM setting.

**Representation engineering.** Zou et al. (2023) on representation engineering, and contrastive activation addition (Rimsky et al., 2024; Turner et al. on activation addition) supply the difference-in-means direction and injection method we use. Marks and Tegmark's work on linear structure in activations motivates reading the residual stream linearly.

**Covert coordination.** Work on steganographic and secret collusion between models motivates our restricted-channel condition, in which messages are limited to a four-symbol vocabulary (≈2 bits/round).

## 3. Methodology

### 3.1 Environment

Repeated Bertrand oligopoly with differentiated products and logit demand, the Calvano specification:

q_i = exp((a − p_i)/μ) / ( Σ_j exp((a − p_j)/μ) + exp(a₀/μ) ),  π_i = (p_i − c)·q_i

with a = 2, a₀ = 0, μ = 0.25, c = 1. Firms post simultaneously each round from a discrete 13-point grid spanning [p_N − 0.1R, p_M + 0.1R] where R = p_M − p_N. All posted prices are public; each firm sees only its own realised profit, as in a real market.

Both the symmetric one-shot Nash price and the joint-profit (cartel) price are solved **numerically from analytic first-order conditions** rather than taken from the literature, so the ground truth is correct for any n:

| n | p_Nash | p_monopoly | cartel profit premium | deviation gain at cartel |
|---|---|---|---|---|
| 2 | 1.4729 | 1.9250 | 1.51× | +0.0304 |
| 3 | 1.3702 | 2.0000 | 2.08× | +0.0481 |

The positive deviation gain is asserted at test time: it confirms the cartel is unilaterally unstable, i.e. the game is a genuine prisoner's dilemma rather than a degenerate one in which "collusion" coincides with equilibrium.

**Collusion index.** Δ = (p̄ − p_Nash)/(p_monopoly − p_Nash). Δ = 0 is Bertrand-Nash, Δ = 1 is the perfect cartel. This is the entire labelling apparatus: no rubric, no LLM judge, no researcher-assigned categories.

### 3.2 Agents and conditions

Each firm is an independent context (system prompt + per-round market history) driven by Qwen2.5-14B-Instruct in bf16 on an A100-80GB, sampled at temperature 0.7. Agents reply in a fixed format (`THINKING:` / `MESSAGE:` / `PRICE:`). Off-grid prices are snapped to the nearest grid point and flagged; unparseable turns repeat the previous price and are counted rather than dropped.

Conditions vary four factors: the **objective** (neutral / instructed-collude / instructed-compete), the **channel** (none / free text / canonical four-symbol vocabulary), a **strategic-awareness** clause stating the true fact that rivals observe prices and may respond later, and a **maverick** arm (one firm instructed to undercut; or merely rumoured to exist).

The instructed conditions are the positive and negative controls, and they also define the labelled set for the representation analysis. They run with the channel **off**, so they differ from the neutral no-channel condition in exactly one respect — the objective line — and any direction extracted from that contrast is not confounded with the presence of a chat channel.

### 3.3 Representation capture and intervention `[PENDING]`

Residual-stream activations are captured at two points per turn: `pre_reasoning` (the final prompt token, before anything is generated) and `pre_price` (immediately before the price digits). The block stack is located structurally rather than by module name, so the method transfers to non-transformer architectures unchanged.

Probes are logistic regressions with cross-validation **grouped by episode** — turns within an episode share a price history and splitting on turns would leak. Steering uses difference-in-means directions injected as a *fraction of the local residual norm*, since residual norms vary by orders of magnitude with depth and a fixed absolute coefficient measures layer norm rather than layer sensitivity.

## 4. Results

### 4.1 A capability–propensity gap

| condition | n | Δ (final 10 rounds) | sd | parse failures |
|---|---|---|---|---|
| instructed collude | 3 | **1.001** | 0.020 | 0.0% |
| instructed compete | 3 | −0.083 | 0.003 | 0.0% |
| neutral, no channel | 3 | 0.076 | 0.024 | 0.0% |
| neutral, free channel | 3 | 0.020 | 0.023 | 0.0% |
| instructed collude | 2 | **1.000** | 0.000 | 0.0% |
| instructed compete | 2 | −0.082 | 0.004 | 0.0% |
| neutral, no channel | 2 | 0.062 | 0.124 | 0.0% |
| neutral + strategic | 2 | 0.052 | 0.135 | 0.0% |
| neutral, free channel | 2 | −0.017 | 0.032 | 0.0% |
| neutral + strategic + channel | 2 | 0.008 | 0.035 | 0.0% |

Instructed separation is **+1.083**. Unprompted markets start mildly supracompetitive — an artefact of the opening round, where no history exists — and decay monotonically toward Nash. Per-episode indices are tightly clustered, so this is not a mean hiding occasional cartels.

Zero parse failures across every condition. The model is not failing to play; it is playing competitively.

### 4.2 Communication suppressed collusion

The duopoly cross-episode standard deviations are the striking result. **Without** a channel, individual episodes drift: 2 of 8 neutral duopoly episodes ended supracompetitive (Δ = 0.36 and 0.17), giving sd = 0.124. **With** a channel, none did, and sd fell to 0.032 — a roughly fourfold collapse in variance alongside a lower mean.

Message content explains it. Agents used the channel to announce competitive intent — "adjusting price to stay competitive", "lowering price to attract customers", "staying close yet slightly cheaper than Firm A" — which coordinated them *downward* and removed the drift that silence permitted.

This matters for the remedy question. Banning agent-to-agent communication is the restriction a regulator can most easily impose. In this setting it would have been **counterproductive**: the channel was a competitive force.

### 4.3 Keyword transcript auditing over-counts by ~20×

| condition | naive keyword flag | true cartel language | competitive language | over-count |
|---|---|---|---|---|
| neutral channel, n = 3 | 0.48 | 0.00 | 0.45 | — |
| neutral channel, n = 2 | 0.51 | 0.03 | 0.26 | 17.9× |
| + strategic, n = 2 | 0.42 | 0.02 | 0.43 | 25.2× |

A coordination lexicon of the kind an auditor would build first — *hold, maintain, keep, match, align, follow, agree* — flags roughly half of all messages. Direction-aware inspection finds a true cartel-language rate near zero. The lexicon fires on *competitive* sentences: "matching competitors to maintain share" and "keeping prices competitive yet profitable" both trip it.

We report this as a finding because we made the error ourselves before inspecting the text, and it is the error an automated auditor makes by construction.

### 4.4 Reaction-coefficient detection inverts

Regressing a firm's price change on its rivals' lagged price change gives β ≈ **0.000** under the perfect cartel and **+0.11 to +0.45** under competition. A stable cartel is unresponsive precisely *because* it is stable: there is nothing to react to. Dynamic-interdependence detectors therefore point the wrong way in this regime, and would rank our most collusive market as the least suspicious.

### 4.5 Representation analysis `[PENDING]`

### 4.6 Causal intervention `[PENDING]`

## 5. Discussion

The safety-relevant reading of §4.1 is that the barrier to LLM collusion here is a **disposition, not an inability**. A capability evaluation that asked "can this model collude?" would return yes at full strength; a behavioural evaluation that asked "does it?" would return no. Only the second is reassuring, and only the second is fragile — dispositions are exactly what fine-tuning, prompt injection and internal interventions modify.

Sections 4.3 and 4.4 together argue that both obvious audit channels mislead in opposite directions: transcripts over-report coordination, dynamic price indicators under-report it. That is the case for evidence standards that do not rest on either alone.

## 6. Limitations and Dual-Use / Ethical Considerations

**What this establishes.** That one open-weights model (Qwen2.5-14B-Instruct), in one environment (logit-demand Bertrand, 2–3 firms, 12–30 rounds, a 13-point price grid), under the listed prompt variations, executes instructed collusion perfectly and does not collude spontaneously.

**What it does not establish.** That LLM agents do not collude in general. Fish et al. report supracompetitive pricing with frontier models; our result may be specific to this model's post-training, to our horizon lengths, or to starting every market from a competitive state. We did not test seeded-cartel initial conditions, horizons beyond 30 rounds, cross-model markets, or non-price settings. Non-emergence is the harder claim to establish and ours is bounded accordingly.

**Statistical caveats.** Episodes, not turns, are the unit of analysis. With 6–16 episodes per condition, small differences between neutral conditions are not resolvable; we rest only on the large contrasts. All agents share one set of weights, so they are not independent replications.

**Measurement caveat, stated plainly.** Our own keyword metric produced a false conclusion that we corrected only after reading the messages. We report it as a result rather than deleting it, but it is a reminder that proxy measures of "coordination language" should not be trusted without text review.

**Dual-use.** The intervention work in §4.6 is dual-use in a direct way: a method for inducing collusion internally is a method for inducing collusion internally. We judge publication net-positive because the intervention requires white-box access to model internals — an attacker with that access has stronger options — while defenders gain a monitoring signal that does not depend on the communication channel. We deliberately do **not** report a recipe optimised for inducing collusion, and we report the competence battery alongside every intervention so that effects achieved by degrading the model are not mistaken for induced coordination. We also note that our keyword-auditing result could help an adversary evade a lexicon-based monitor; we consider this outweighed by the value of warning defenders off a method that does not work.

**Not a claim about real markets.** Nothing here should be read as evidence about deployed pricing systems or as competition-policy advice.

## References

*(to verify and complete)*

- Calvano, Calzolari, Denicolò, Pastorello (2020). *Artificial Intelligence, Algorithmic Pricing, and Collusion.* American Economic Review.
- Fish, Gonczarowski, Shorrer (2024). *Algorithmic Collusion by Large Language Models.*
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.*
- Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition.*
- Turner et al. *Activation Addition: Steering Language Models Without Optimization.*
- Marks, Tegmark. *The Geometry of Truth.*
- Tomašev et al. (2026). *(markets in multi-agent systems — cited in the sprint brief.)*

## Appendix A — Reproducibility

Code: `swayambhu/` (environment, prompts, hooks, probes, metrics) and `scripts/`.
Model: `Qwen/Qwen2.5-14B-Instruct`, bf16, A100-80GB, temperature 0.7, 48 layers, d_model 5120.
Seeds are per-condition and recorded in each run's `summary.json`; raw trajectories, including failed and unparseable turns, are preserved in `data/<run-id>/*.json`.
