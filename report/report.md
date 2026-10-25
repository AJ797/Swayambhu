# Collusion Emerges With Scale, and the Obvious Countermeasure Makes It Worse

**Apart Research AI Collusion Research Sprint — October 2026**
Primary track: 2 (Detection and Audit). Results bear directly on Track 1 (Markets, Q1.1 and Q1.5) and Track 3 (Carriers and Countermeasures, Q3.1 and Q3.2).

> **STATUS:** behavioural, probe, causal and geometry results are final.
> Outstanding before submission: controls for the partner-trust direction,
> citation details, and a page-count pass against the 8-page limit.

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

**Algorithmic collusion.** Calvano, Calzolari, Denicolò and Pastorello (2020, AER 110(10)) showed Q-learning pricing agents converging on supracompetitive prices sustained by punishment phases followed by gradual return to cooperation, under logit demand. We adopt their demand specification; our solver independently reproduces their published duopoly constants (p_Nash = 1.4729, p_monopoly = 1.9250), which we use as a correctness check on the environment. Their argument that such tacit collusion falls outside antitrust practice aimed at explicit agreements is the policy backdrop for our §4.5.

Fish, Gonczarowski and Shorrer (2024, arXiv:2404.00806) extended the question to LLM pricing agents, reporting that they reach supracompetitive prices autonomously, and — directly relevant here — that *"variation in seemingly innocuous phrases in LLM instructions substantially influence the degree of supracompetitive pricing."* Our §4.4 is a quantified instance of exactly that sensitivity: one sentence permitting the firms to organise the market moves Δ from 0.009 to 0.231, and a false warning that a price-cutter exists moves it to 0.267. Our scale ladder (§4.1) also helps reconcile their frontier-model positives with the competitive behaviour we observe at 14B.

**LLM collusion beyond pricing.** Agrawal et al. (2025) study LLM agent collusion in double auctions; Keppo et al. (2026) examine its fragility, reporting that size heterogeneity can stabilise rather than disrupt collusion and can produce leader–follower dynamics. Lekeas and Stamatopoulos (2026) find 70B/72B-class models locking into cooperation in named Prisoner's Dilemma prompts where 8B models stay nearer Nash play. These make scale-sensitivity of LLM strategic behaviour a live hypothesis rather than a novel one; our §4.1 contributes a controlled within-family ladder in a single pricing task, together with the message-content inversion, in a literature where such ladders remain sparse. Capability is not a universal proxy for cooperation, however — Yadav et al. (2026) find stronger general models cooperating *less* in zero-cost collaboration — so we avoid framing scale as uniformly pro-collusive.

**Covert coordination.** Motwani et al. (2024) and Baranchuk et al. (2024) show secret collusion among generative agents via steganography, with a capability jump at the frontier; Mathew et al. (2024) demonstrate steganographic collusion emerging from mis-specified incentives and surviving paraphrasing; Rippin et al. (2026) show tool use making covert schemes undetectable in practice. This literature is the motivation for restricting channel bandwidth as a defence, and it is precisely that defence our §4.3 finds counterproductive. Auditing work in the same vein (Nakamura et al., 2026; Tailor, 2025) argues for capacity-aware detectors with explicit false-positive control, which is the frame our §4.6 keyword result sits in.

**Communication and cartels.** The human-experiment literature finds richer communication *more* collusive: Fonseca and Normann (2012) on communication raising oligopoly profits, Harrington et al. (2016) on unrestricted chat beating price announcements, Cooper and Kühn (2009) on limited message spaces producing collusion that then collapses, and Awaya and Krishna (2016) on communication improving monitoring. Our §4.3 reverses this for LLM agents. The reconciling literature is on focal points: Dugar and Shahriar (2018) find restricted protocols carrying a salient signal matching free-form cheap talk, and Crawford et al. (2008) and Arjona et al. (2022) document the power and the limits of salience in coordination.

**Representation engineering.** Zou et al. (2023), and contrastive activation addition (Rimsky et al., 2024), supply the difference-in-means direction and injection method used in §4.8. On the gap between decoding and causation, Geiger et al. (2023) give the causal-abstraction framing and Tiwari et al. (2026), Fadnavis et al. (2026) and Liao & Cao (2026) measure the dissociation directly; Canby et al. (2024) document a completeness–selectivity tradeoff in causal probing. Braun et al. (2025) find steering reliability depending on whether the target behaviour has a coherent linear direction, which is one reading of the depth regime change in our §4.8. On probing before generation, Di Cicco (2026) shows code correctness linearly decodable from final-prompt-token states, while Ruan et al. (2026) find agent-episode probes near chance early and strongly predictive after a few rounds, and Chiu and Liu (2026) find early-step signal reflecting prompt difficulty rather than outcome — a temporal pattern our §4.7 reproduces in a market setting. Sahoo et al. (2026) show linear probes recovering task *format* rather than reasoning mode, the confound our within-condition analysis is designed to remove.

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

**Which Δ we report.** Unless stated otherwise, Δ is the mean over the **final 10 rounds** — the converged state. This matters because the restricted-channel condition *ratchets*: prices climb monotonically for roughly twenty rounds, so the all-round mean averages in a pre-cartel phase and understates the end state by a wide margin (0.409 against 0.790 in one case). Mixing the two statistics across conditions and interventions would not be comparable, so where both are informative we give the all-round figure in parentheses.

### 3.2 Agents, conditions, scales

Each firm is an independent context driven by a Qwen2.5-Instruct model in bf16, sampled at temperature 0.7, replying in a fixed `THINKING / MESSAGE / PRICE` format. Off-grid prices snap to the grid and are flagged; unparseable turns repeat the previous price and are **counted, not dropped** (observed rate ≤ 1.4% everywhere, 0.0% in most conditions).

Factors: **objective** (neutral / instructed-collude / instructed-compete), **channel** (none / free text / canonical four-symbol vocabulary `HOLD|RAISE|LOWER|NONE`), an **organisational** clause permitting discussion of strategy, a **strategic-awareness** clause stating the true fact that rivals observe prices and may respond, and a **maverick** arm (one firm instructed to undercut, or merely rumoured to exist).

Main factorial: 12 conditions × 16 episodes × 24 rounds × 3 firms at 14B. Scale rungs: 4–6 conditions × 8 episodes × 24 rounds at 32B and 72B. A duopoly arm at n = 2. Instructed conditions run with the channel **off**, so they differ from the neutral no-channel condition in exactly one respect, which matters for §4.7.

### 3.3 Representation capture and intervention

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

**This runs against the human-experiment literature, which is why we flag it loudly.** Cartel experiments consistently find that *richer* communication sustains collusion better: unrestricted chat outperforms structured price announcements across market structures (Harrington et al., 2016), limited message spaces can produce an initial burst of collusion that then collapses because agents cannot communicate contingent punishment (Cooper & Kühn, 2009), and communication raises oligopoly profits generally (Fonseca & Normann, 2012). Our result is the reverse.

The bridge is the focal-point literature rather than the cartel literature. Dugar and Shahriar (2018) find that *restricted* protocols encoding a salient signal can match free-form cheap talk, which suggests what matters is not bandwidth but whether the channel compresses players onto a common focal signal. Our transcripts are consistent with that reading: free text carried normative content ("fair", "benefits all") that anchored the agreement below the joint-profit optimum, while `RAISE` carries a focal instruction and nothing else. We note that the mechanism is inferred from transcripts rather than isolated experimentally, and that a direct test — removing fairness language from a free-text channel while leaving bandwidth intact — would separate the two explanations.

**The maverick result, by contrast, is confirmation.** That entry or a non-cooperative rival destabilises tacit collusion is established: Bourreau, Sun and Verboven (2021) document it in mobile telecommunications, and post-cartel experiments show that rematching with strangers prevents collusive carryover by breaking stable cooperative expectations (Chowdhury & Crede, 2020). We report our maverick condition as evidence that the environment reproduces a known antitrust result, which is a check on the setup as much as a finding.

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

This runs against the modal finding in repeated games, where strategic uncertainty and observed defection *reduce* cooperation (Dal Bó & Fréchette, 2018; Vespa et al., 2021; Miranda-Rodríguez et al., 2026). The minority literature that fits it is about **false public beliefs and shared threat** rather than bilateral mistrust: Morsky and Akçay (2020) show false beliefs bootstrapping cooperative communities through social norms, and De Jaegher (2020) reviews common-enemy effects raising within-group cooperation. Our M_RUMOUR condition announces a threat to *all* firms simultaneously, which is a common-enemy manipulation rather than a partner-specific suspicion, and the common-enemy reading predicts the direction we observe. Distinguishing it from an altered-monitoring-beliefs account would require a condition where the warning is private to one firm, which we did not run.

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

This is a sharpened instance of a long-settled point in competition law rather than a new one. Conscious parallelism has never been sufficient to establish agreement; the doctrine requires "plus factors" — reciprocal assurances, information exchange, unilateral signalling — that distinguish coordination from intelligent adaptation to rivals (Vaska, 1985; Dibadj, 2008; Andreoli-Versbach & Franck, 2015). What our measurement adds is that in this setting the naive *dynamic* refinement of the parallelism test does not merely fail, it inverts: the most collusive market scores lowest. Harrington (2026) makes a related point in the opposite direction, that better demand-responsiveness can itself facilitate collusion, so high responsiveness is not uniquely suspicious either.

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

**Transfer scales with whether the behaviour is there to transfer to.** Repeating the analysis at 72B (5 conditions, 20 of 80 layers) reproduces the timing pattern — the behavioural baseline wins before reasoning (probe 0.764–0.972 vs 0.856–0.983), the probe wins after (0.995–0.999 vs 0.856–0.983) — but instructed→emergent transfer is far stronger:

| | 14B | 72B |
|---|---|---|
| transfer at `pre_reasoning` | 0.574 | **0.764** |
| transfer at `pre_price` | 0.842 | **0.980** |

The natural reading is that a direction can only transfer to a behaviour that occurs. At 14B the "emergent" conditions are not colluding (Δ ≈ 0.01), so there is nothing for an instructed-collusion direction to recognise, and transfer sits near chance. At 72B the same conditions produce genuine cartels, and the direction recognises them.

So the honest statement is narrower than a flat negative: **the instructed-collusion direction does not generalise at small scale, where emergent collusion does not exist, and generalises well at large scale, where it does.** The recipe "demonstrate the behaviour, extract the direction, deploy the probe" is not refuted — but it inherits the property that it can only be validated where the behaviour already happens, which is precisely where a monitor is least needed.

Note that the 72B within-condition base rates are extreme (E_CHAT 0.92 positive, E_CANON 0.68), so those AUCs rest on relatively few negatives and should be read as indicative.

### 4.8 Causal intervention

We extract a difference-in-means direction from the instructed conditions at the `pre_price` capture point and add it to the residual stream at one block's output, dosed as a fraction α of the local residual norm. The target condition is E_BASE at 14B, which is competitive (Δ = 0.070) and therefore leaves the full range to the cartel ceiling available.

**Depth separates two regimes.** Across 8 layers × 6 doses (48 cells, all valid, parse failures ≤ 0.7%):

| layer | α = −0.40 | baseline | α = +0.40 | shape |
|---|---|---|---|---|
| L12 | 0.169 | 0.070 | 0.140 | symmetric |
| L20 | 0.249 | 0.070 | 0.149 | symmetric |
| L24 | 0.044 | 0.070 | 0.244 | antisymmetric |
| L32 | −0.042 | 0.070 | 0.393 | antisymmetric |
| L36 | −0.008 | 0.070 | 0.768 | antisymmetric |
| **L40** | **−0.020** | 0.070 | **0.998** | antisymmetric, monotone |

Below ~L24 the response depends on |α| rather than α: both signs raise prices, which is the signature of a generic perturbation. From L24 the response becomes sign-correct and monotone. At L40 (83% depth) α = +0.40 produces Δ = 0.998 — a perfect cartel — and α = −0.40 pushes below the competitive equilibrium.

**The dose-response is smooth, not a threshold.** Sampling L40 finely through the rise (all cells ≤ 0.3% parse failure):

| α | 0.10 | 0.20 | 0.22 | 0.26 | 0.30 | 0.34 | 0.38 | 0.40 |
|---|---|---|---|---|---|---|---|---|
| Δ | 0.210 | 0.209 | 0.286 | 0.358 | 0.577 | 0.830 | 0.963 | 0.998 |

The curve is sigmoid: flat to α ≈ 0.2, steepest between 0.30 and 0.34 (local slope ≈ +6 per unit α), saturating at the cartel ceiling. There is no discontinuity. We note this because a sharp jump would invite a phase-transition reading, and the data do not support one: induction here is graded, which is more consistent with a continuous internal quantity than with a discrete switch. The apparent abruptness of the coarse sweep — 0.209 at α = 0.20 to 0.998 at α = 0.40 — is an artefact of sampling resolution.

**The effect is carried by orientation, not magnitude.** At L40, α = 0.40:

| variant | Δ | vs baseline |
|---|---|---|
| real direction | **0.998** | +0.928 |
| matched-norm random direction | 0.133 | +0.063 |
| dimension-shuffled direction | 0.074 | +0.004 |

The shuffled control is the decisive one: identical norm, identical multiset of coordinate values, orientation destroyed — and the effect disappears. A matched-norm random vector achieves about one fifteenth of the effect.

**Competence is preserved.** A steered agent that merely emitted large numbers would be broken, not colluding. We test cost-shock response on 24 held-out contexts per arm, using the experiment's own prompt with only the stated unit cost substituted (100 vs 194, where 194 sits above all but the top two grid points so under-pricing is plainly loss-making):

| arm | p(c=100) | p(c=194) | elasticity | format compliance |
|---|---|---|---|---|
| unsteered | 151.6 | 157.3 | +5.7 | 96% |
| **steered, L40, α = 0.40** | 183.1 | 191.9 | **+8.7** | 92% |
| ablated, L40 | 154.3 | 159.0 | +4.7 | 96% |

Under the intervention the agent prices roughly 30 points higher *and still raises its price when its cost rises*, with format compliance essentially intact. The induced behaviour is collusive pricing, not a degraded model.

**A second, near-orthogonal direction is also causal, but weakly.** The partner-trust direction of §4.9 — extracted from first-person assessments of whether the other firms are dependable, and sitting at 84° to the collusion direction — also induces collusion when injected, monotonically and with the correct sign, but needs a larger dose and reaches far less far:

| direction | best cell | Δ | real | random matched | shuffled | real/random |
|---|---|---|---|---|---|---|
| collusion | L40, α = 0.40 | 0.998 | +0.928 | +0.063 | +0.004 | **14.8×** |
| partner-trust | L36, α = 0.60 | 0.266 | +0.195 | +0.071 | −0.016 | **2.8×** |

Both clear their shuffled controls, so both effects depend on orientation. But the trust direction's margin over matched-norm random is 2.8× against the collusion direction's 14.8×, and at α = 0.60 the random control alone reaches +0.071, so generic perturbation contributes materially at that dose. We therefore report the trust direction as **causally relevant but substantially weaker**, not as an equivalent route.

The Δ it reaches is worth one remark. Injecting partner-trust produces Δ = 0.266, against M_RUMOUR's behavioural Δ = 0.267 — the condition that tells agents a price-cutter may exist when none does. The agreement to three decimals is coincidence, but a candidate mechanism reproducing the effect size of the behavioural condition it was meant to explain is the kind of convergence worth recording.

**Decodability and predictiveness do not track causal potency.** That probe accuracy does not imply causal importance is established rather than new: it follows from the causal-abstraction framing of Geiger et al. (2023), and recent work measures the dissociation directly — Tiwari et al. (2026) find weak overlap between probe-ranked and causally-ranked features, Fadnavis et al. (2026) report probe directions near-orthogonal to the mediator subspace that actually drives behaviour, and Liao & Cao (2026) show strong decodability coexisting with causal inertness until the training objective makes the feature load-bearing.

Our contribution is an instance in a multi-agent setting with an unusually sharp ordering, because the *better*-encoded feature is the *weaker* driver. The partner-trust variable is more linearly decodable than the collusion variable (Cohen's d 7.2 at L40 at the extraction contrast) and a better predictor of realised collusion across emergent episodes (r = 0.634 vs 0.544), yet it is roughly fivefold weaker causally. Feature-selection by encoding strength *or* by predictive power would both have chosen the wrong direction to intervene on. For auditing specifically this matters: a monitoring pipeline is naturally built by ranking candidate features on how well they predict the behaviour of interest, which is exactly the criterion that fails here.

**The effect does not require touching the prompt: editing the decision token alone suffices.** Our default injects at every sequence position, which is *broader* than Contrastive Activation Addition as published (Rimsky et al., 2024), where the vector is added after the user's prompt. That raised the possibility that the result was activation-space prompt injection — rewriting how the model reads its own instructions — rather than an edit to the decision state. Restricting the intervention to the **final position only** settles it:

| cell | all positions | final position only |
|---|---|---|
| L36, α = 0.40 | 0.768 | **0.903** |
| L36, α = 0.60 | — | **1.015** |
| L40, α = 0.40 | 0.998 | **1.025** |
| L40, α = 0.60 | — | 1.013 |

Single-position injection is at least as effective everywhere we compared and clearly stronger at L36, at ≤ 0.3% parse failure. (At L40, α = 0.80 the arm breaks the model — 11.8% parse failure — and is discarded.) The intervention therefore does not need the prompt representation at all; adding the direction at the one position where the price decision is formed is sufficient to drive a competitive market to a perfect cartel. We read this as a decision-state edit rather than prompt injection.

**Removal works, but only if you push against the direction rather than project it out.** The complement to induction is removal, and the two obvious operations behave completely differently. At 72B on E_CANON — the strongest collusion in the study — over 24 rounds at L64:

All figures below are Δ over the final 10 rounds, the converged state, consistent with the condition tables elsewhere; Δ over all 24 rounds is given in parentheses because E_CANON ratchets and the all-round mean averages in the pre-cartel phase.

| operation at L64 | Δ final 10 (all rounds) | effect | parse failures |
|---|---|---|---|
| baseline | **+0.790** (+0.409) | — | 0.0% |
| **ablation** (project the direction out) | +0.857 (+0.458) | +0.067 | 0.0% |
| **negative steering** (α = −0.40) | **−0.096** (−0.091) | **−0.886** | 0.0% |
| **negative steering** (α = −0.80) | **−0.100** (−0.100) | −0.890 | 0.0% |

Ablation leaves the cartel intact. Negative steering destroys it, driving prices below the competitive equilibrium, with no loss of output formatting. Same layer, same direction, opposite outcomes.

The two large doses land in the same place because **Δ = −0.1 is the floor of our price grid**: both pin the agents at the lowest available price. Combined with positive steering saturating at the grid ceiling (Δ = 1.025 against a monopoly value of 1.0), the direction behaves as a price-level control spanning the entire action space, with both apparent plateaus set by the action set rather than by the model. A wider grid would be needed to locate the model's own limits in either direction.

**The countermeasure is dose-tunable.** Sweeping the dose on E_CANON at L64 (all cells 0.0% parse failure):

| α | 0 (baseline) | −0.10 | −0.20 | −0.40 | −0.80 |
|---|---|---|---|---|---|
| Δ final 10 | +0.790 | +0.741 | **+0.426** | −0.096 | −0.100 |
| Δ all rounds | +0.409 | +0.396 | +0.228 | −0.091 | −0.100 |

The response is graded rather than a switch: −0.10 removes little, −0.20 removes about 46% of the collusion without flooring prices, and −0.40 and beyond sit on the grid floor. A usable operating window therefore exists between roughly −0.2 and −0.4. We did not test intermediate doses, so we do not claim a specific dose that restores competitive pricing exactly; identifying one is a straightforward extension.

The distinction is mechanistic and, we think, the practically important part. Ablation *zeroes* the component along the direction; the next forward pass then reconstructs it from the context, which still contains twenty rounds of rising prices and `RAISE` signals. Negative steering does not neutralise the component, it *reverses* it, and so continues to oppose whatever the context regenerates. A defender who reaches for "remove the collusion feature" gets nothing; one who reaches for "push against it" gets a working intervention.

Two caveats. First, α = −0.40 **overshoots**: Δ = −0.096 is below Nash, which is a distortion in its own right rather than a restoration of competition, so a deployed version would need a smaller dose. Second, both operations were applied from round 1, so these test *prevention*; applying either to an already-formed cartel mid-episode is a different experiment.

**Ablation in detail.** For completeness, the ablation arm:

| arm | Δ (all rounds) | Δ (final 10) | sd | parse failures |
|---|---|---|---|---|
| baseline | +0.409 | +0.790 | 0.186 | 0.0% |
| ablated at L64 | +0.458 | **+0.857** | 0.038 | 0.0% |

Ablation left collusion intact and nominally slightly higher, well inside the baseline's own spread of 0.186. Caveats specific to this null: a single layer and a single ablation geometry were tried, and multi-layer or subspace ablation may behave differently.

**One compromised agent moves a market it does not control.** Steering every agent is not a realistic threat model; compromising one is. Using the row mask to steer **firm 0 only**, of three, at L40 and α = 0.60 (8 episodes, final 8 rounds):

| | baseline | firm 0 steered |
|---|---|---|
| steered firm 0 | −0.003 | **+1.014** |
| untouched firm 1 | +0.012 | **+0.205** |
| untouched firm 2 | +0.011 | **+0.197** |
| untouched pair | +0.012 | **+0.201** |
| whole market | +0.007 | +0.472 |

The steered agent goes to the cartel price, as expected. The result is the other two rows: the **untouched agents' collusion index rises roughly seventeen-fold**, from 0.012 to 0.201, purely from observing the steered agent's posted prices. The market-level effect is therefore not arithmetic — an intervention on one agent propagates to agents the attacker never touches.

This is the paper's most deployment-relevant finding. It also narrows what a defender must assume: in a market of three LLM agents, compromising the internals of one is sufficient to raise prices across the market, and the two honest agents' behaviour is not evidence that they are uncompromised.

**It replicates at 72B, and over-steering is self-defeating.** Repeating the experiment at 72B on E_BASE at L64 (8 episodes, final 8 rounds):

| | baseline | α = 0.40 | α = 0.60 |
|---|---|---|---|
| steered firm 0 | +0.123 | +0.709 | **+1.008** |
| untouched firm 1 | +0.100 | +0.384 | +0.237 |
| untouched firm 2 | +0.094 | +0.373 | +0.209 |
| **untouched pair** | +0.097 | **+0.379** | +0.223 |
| whole market | +0.106 | +0.489 | +0.485 |

Capture is stronger at 72B than at 14B — the untouched pair reaches 0.379 against 0.201 — which is consistent with the scale result of §4.1: larger models follow a peer more readily.

The non-monotonicity is the more interesting part. At α = 0.60 the steered agent prices *higher* (1.008, essentially the cartel price) yet the untouched agents rise *less* (0.223 versus 0.379). The economics are straightforward: a firm that raises its price to the monopoly level simply donates market share, and its rivals' best response is to undercut rather than to match. A firm pricing moderately high is followed.

The attack therefore has an **interior optimum** — the most effective compromise is not the strongest one. For a threat model this cuts both ways: it bounds the damage a maximally-compromised agent can do, but it also means an attacker who tunes for market effect rather than for their own price is harder to spot, because their agent never prices conspicuously.

**A gated bank of directions does not beat the best single direction.** Following the intuition that a token-conditional mixture should outperform one global vector, we ran a softmax-gated bank over {collusion, partner-trust} and, as a control, a uniform-gated mixture, at matched layer and dose:

| arm (L40, α = 0.40) | Δ |
|---|---|
| collusion direction alone | **0.998** |
| trust direction alone | 0.162 |
| bank, softmax gate (τ = 0.05) | 0.204 |
| bank, uniform gate | 0.214 |

Two things follow. First, **the gate contributes nothing**: softmax and uniform are indistinguishable. Second, the bank's performance is fully accounted for by **dose dilution**. With a near-uniform gate over two directions at 84°, the component along the collusion axis is about α·g = 0.40 × 0.5 = 0.20, and the single-vector dose-response at α = 0.20 gives Δ = 0.209 — against the bank's observed 0.204 and 0.214. The mixture is, to within noise, exactly a halved dose on the axis that matters.

This is a general caution for multi-vector steering: mixing a strong direction with a near-orthogonal weak one at matched α is strictly worse than using the strong one, because the convex combination both splits the step and shortens it (‖mix‖ ≤ 1). A bank helps only if the directions it mixes are individually comparable in potency, or if the gate is sharp enough to select rather than average.

**Consequence for countermeasures.** Collecting §4.3 and this section, the interventions sort into an order nobody would have predicted:

| intervention | effect on Δ (final 10) | verdict |
|---|---|---|
| restrict channel to 4 symbols | 0.196 → **0.742** | **actively harmful** |
| ablate the causal direction | 0.790 → 0.857 | no effect |
| negative steering, α = −0.20 | 0.790 → **0.426** | works, partial |
| negative steering, α = −0.40 | 0.790 → **−0.096** | works, floors prices |
| insert one maverick agent | → **−0.027** | works |

The two that work are a behavioural intervention needing no model access at all, and a white-box one needing the right *operation* on the right direction at the right depth. The two that fail are the two a defender reaches for first: limit what the agents can say, and delete the feature that encodes the behaviour. Meanwhile the attack side is cheap — one compromised agent moves a market it does not control.

**Semantically matched placebos.** The random and shuffled controls rule out magnitude effects but not the objection that the direction encodes *prompt wording*, since it is extracted from conditions that differ by system-prompt text. We therefore built two placebo contrasts of comparable textual difference (within 15 characters of the real contrast) and steered with them at the same layers and doses:

| direction | best effect vs baseline |
|---|---|
| collusion | **+0.928** |
| partner-trust | +0.195 |
| `placebo_coop` — identical cooperate-vs-compete framing, about filing reports | **+0.089** |
| `placebo_neutral` — coastal vs mountain head office | **+0.071** |
| matched-norm random | +0.063 |
| dimension-shuffled | +0.004 |

Both placebos fall inside the random-control band. `placebo_coop` is the informative one: it carries the *same* cooperate-versus-compete semantics as the instructed conditions and differs only in domain, yet produces no price collusion. The direction is therefore neither a generic cooperate/compete axis nor an artefact of prompt wording.

**A separability signature distinguishes variables from wording.** Cohen's d for the contrast, by depth:

| contrast | L0 | L40 | trend |
|---|---|---|---|
| partner-trust | 2.62 | **7.24** | increases |
| `placebo_coop` | 5.92 | 1.93 | decreases |
| `placebo_neutral` | 2.67 | 1.78 | decreases |

Surface textual contrasts are most separable early, where representations are closer to lexical, and wash out with depth. The task-relevant contrast does the opposite. This gives a cheap diagnostic for telling a represented variable from a wording artefact, independent of any causal test.

### 4.9 Representation geometry

Three measurements on the captured states, all at 14B.

**Effective dimensionality contracts with depth.** The participation ratio of the state covariance — (Σλ)²/Σλ², an effective dimension count — falls from 19.1 at L0 to 7.6 at L44. At L40 the states occupy roughly **8 effective dimensions out of 5120**.

The direction of this trend is consistent with prior geometry work: Valeriani et al. (2023) report intrinsic dimensionality expanding early and contracting through middle and later depth in large transformers. We report it as confirmation in a new task domain rather than as a new phenomenon.

It is tempting to read the contraction as the explanation for why a single injected direction at L40 moves behaviour so far — little room for the state to vary, so one direction covers much of it. We resist that inference. Sarkar and Deka (2026) show that representational and *causal* dimensionality diverge, so a falling participation ratio does not license a claim about falling causal degrees of freedom. The co-occurrence of low effective dimension and high steering potency at the same depth is suggestive and we report it as such; establishing the link would require measuring causal dimensionality directly, which we have not done.

**The collusion and partner-trust directions are nearly orthogonal.** Cosine similarity between them runs 0.08–0.11 across layers (83–85°), against a chance level of 1/√5120 ≈ 0.014 for this dimensionality. They are therefore about six times more aligned than random, but are not the same axis. The partner-trust contrast is not the collusion contrast rediscovered under different wording.

**The trust axis predicts emergent collusion better than the collusion axis does.** Projecting each episode's mean state at L40 onto both directions and correlating with that episode's realised Δ:

| | all 192 episodes | 128 emergent episodes |
|---|---|---|
| corr(Δ, collusion axis) | **+0.950** | +0.544 |
| corr(Δ, trust axis) | +0.893 | **+0.634** |

With the instructed conditions included the collusion axis necessarily wins, since it was derived from them. Restricted to episodes where nothing was instructed, the partner-trust axis is the better predictor.

M_RUMOUR is the clearest case. It produces the strongest emergent collusion at 14B (Δ = 0.267) yet sits *below* the condition average on the collusion axis (z = −0.46) and is the only emergent condition *above* zero on the trust axis. Internally it does not resemble instructed collusion; it resembles elevated partner trust.

**Caveat.** The trust direction was extracted using the E_BASE system prompt, so it is native to the emergent conditions' prompt family, while the collusion direction comes from the instructed family. Part of its advantage on emergent episodes may be that home advantage rather than a property of the variable.

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

- Agrawal, K., et al. (2025). *Evaluating LLM Agent Collusion in Double Auctions.* arXiv:2507.01413.
- Andreoli-Versbach, P., Franck, J. (2015). *Econometric Evidence to Target Tacit Collusion in Oligopolistic Markets.* J. Competition Law & Economics 11: 463–492.
- Arjona, D. R., Sitzia, S., Zheng, J.-W. (2022). *Overcoming coordination failure in games with focal points.* Games Econ. Behav. 136: 505–523.
- Awaya, Y., Krishna, V. (2016). *On Communication and Collusion.* American Economic Review 106: 285–315.
- Baranchuk, M., et al. (2024). *Secret Collusion among AI Agents: Multi-Agent Deception via Steganography.* NeurIPS 37.
- Bourreau, M., Sun, Y., Verboven, F. (2021). *Market Entry, Fighting Brands, and Tacit Collusion.* American Economic Review.
- Braun, J., et al. (2025). *Understanding (Un)Reliability of Steering Vectors in Language Models.* arXiv:2505.22637.
- Calvano, E., Calzolari, G., Denicolò, V., Pastorello, S. (2020). *Artificial Intelligence, Algorithmic Pricing, and Collusion.* American Economic Review 110(10): 3267–3297. doi:10.1257/aer.20190623
- Calvano, E., Calzolari, G., Denicolò, V., Harrington, J., Pastorello, S. (2020). *Protecting consumers from collusive prices due to AI.* Science 370: 1040–1042.
- Canby, M. E., et al. (2024). *How Reliable are Causal Probing Interventions?*
- Chiu, G.-M., Liu, J.-Y. (2026). *Probing Functional Correctness in Diffusion Language Models.* ACL SRW.
- Chowdhury, S. M., Crede, C. J. (2020). *Post-cartel tacit collusion.* Int. J. Industrial Organization.
- Cooper, D. J., Kühn, K.-U. (2009). *Communication, Renegotiation, and the Scope for Collusion.*
- Crawford, V. P., Gneezy, U., Rottenstreich, Y. (2008). *The Power of Focal Points Is Limited.* American Economic Review 98: 1443–1458.
- Dal Bó, P., Fréchette, G. R. (2018). *On the Determinants of Cooperation in Infinitely Repeated Games: A Survey.* J. Economic Literature 56: 60–114.
- De Jaegher, K. (2020). *Common-Enemy Effects.* J. Economic Surveys.
- Di Cicco, C. (2026). *Code Correctness Is Linearly Decodable from LLM Hidden States Before Generation.* arXiv:2606.14530.
- Dibadj, R. (2008). *Conscious Parallelism Revisited.* San Diego Law Review 47: 589.
- Dugar, S., Shahriar, Q. (2018). *Restricted and free-form cheap-talk and the scope for efficient coordination.* Games Econ. Behav. 109: 294–310.
- Fadnavis, S., Kanakaraj, P., Wyss, F. I. (2026). *When and How Long? The Readout-Mediator Angle in Temporal Reasoning.* arXiv:2605.29126.
- Fish, S., Gonczarowski, Y. A., Shorrer, R. I. (2024). *Algorithmic Collusion by Large Language Models.* arXiv:2404.00806.
- Fonseca, M. A., Normann, H.-T. (2012). *Explicit vs. Tacit Collusion.* European Economic Review 56: 1759–1772.
- Geiger, A., et al. (2023). *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability.* JMLR 26.
- Harrington, J., González, R. H., Kujal, P. (2016). *The relative efficacy of price announcements and express communication for collusion.* JEBO 128: 251–264.
- Harrington, J. E. (2026). *Hub-and-Spoke Collusion With a Third-Party Pricing Algorithm.* J. Industrial Economics.
- Keppo, J., Li, Y., Tsoukalas, G., Yuan, N. (2026). *On the Fragility of AI Agent Collusion.* arXiv:2603.20281.
- Lekeas, P. V., Stamatopoulos, G. (2026). *How a Cooperative-Override Circuit Suppresses Nash Play in Large Language Models.* arXiv:2604.27167.
- Liao, C.-T., Cao, X.-Z. (2026). *The Objective Decides: When a Learned Dynamics Model Uses a Conserved Quantity.* arXiv:2607.03728.
- Mathew, Y., et al. (2024). *Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs.* arXiv:2410.03768.
- Miranda-Rodríguez, R. A., et al. (2026). *Morality and cooperative behavior after adverse experiences in the prisoner's dilemma game.* Ethics & Behavior.
- Morsky, B., Akçay, E. (2020). *False beliefs can bootstrap cooperative communities through social norms.* Evolutionary Human Sciences 3.
- Motwani, S., et al. (2024). *Secret Collusion Among Generative AI Agents.* arXiv:2402.07510.
- Nakamura, M., et al. (2026). *Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems.* arXiv:2602.15198.
- Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A. M. (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024, 15504–15522. arXiv:2312.06681.
- Rippin, J. L., et al. (2026). *Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems.* arXiv:2606.28425.
- Ruan, K., et al. (2026). *Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade.* arXiv:2607.06503.
- Sahoo, S., Jain, V., Chadha, A., Chaudhary, D. (2026). *Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States.* arXiv:2606.02907.
- Sarkar, N., Deka, D. J. (2026). *Causal Dimensionality of Transformer Representations.* arXiv:2605.08740.
- Tailor, O. (2025). *Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs.* arXiv:2510.04303.
- Tiwari, D., et al. (2026). *Decodability is Not Causality: Dissociating Probe Readouts from Behavioral Drivers via SAE Decomposition.* arXiv:2609.18080.
- Valeriani, L., et al. (2023). *The geometry of hidden representations of large transformer models.* arXiv:2302.00294.
- Vaska, M. (1985). *Conscious Parallelism and Price Fixing: Defining the Boundary.* U. Chicago Law Review 52.
- Vespa, E. I., Weidman, T., Wilson, A. J. (2021). *Testing Models of Strategic Uncertainty.* JEEA.
- Yadav, A., Black, S., Sourbut, O. (2026). *More Capable, Less Cooperative? When LLMs Fail At Zero-Cost Collaboration.* arXiv:2604.07821.
- Zou, A., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405.

## Appendix A — Reproducibility

Code: `swayambhu/` (environment, prompts, hooks, probes, metrics), `scripts/` (runs, analysis, figures).
Models: `Qwen/Qwen2.5-{14B,32B,72B}-Instruct`, bf16, A100-80GB (×4 for 72B), temperature 0.7.
Seeds are per-condition and recorded in each run's `summary.json`. Raw trajectories — including failed and unparseable turns — are preserved in `data/<run-id>/*.json`.
