# Consensus queries for the lit review

Ordered by risk to the paper. The first block matters most: if any of those
return established results, a headline claim is not novel and the framing has
to change. The later blocks are for citations we should have rather than
claims we might lose.

For each: the query to paste, why we care, and what a hit would do to us.

---

## Block A — novelty checks on our three riskiest claims

### A1. Does restricting communication bandwidth increase collusion?
> `Does limiting communication bandwidth between agents increase or decrease tacit collusion?`
> `Do simpler communication channels make cartels more stable than rich communication?`
> `cheap talk vs restricted signalling in collusive agreements`

**Why:** our single most surprising result — a 4-symbol channel produced Δ = 0.742
against free text's 0.196, a 3.6× increase, replicated at fresh seeds.
**If it's known:** we lose the "counterintuitive" framing but gain a
mechanism explanation (our transcripts show free text anchoring on "fairness"
and capping the cartel). Still publishable, reframed as confirmation + mechanism.
**Related economics to check:** the industrial-organisation literature on
*focal points* and Schelling coordination predicts that simple, unambiguous
signals coordinate better than rich ones. If that's already established for
human cartels, our contribution is showing it transfers to LLM agents and
that it inverts the obvious AI-safety remedy.

### A2. Does a false belief about defection increase cooperation?
> `Does suspicion of defection increase or decrease cooperation in repeated games?`
> `Effect of false rumours of cheating on cartel stability`
> `belief about partner trustworthiness and tacit collusion`

**Why:** M_RUMOUR — telling agents a price-cutter may exist when none does —
produced our strongest 14B collusion (Δ = 0.267) with *zero* cartel language,
while an actual maverick produced none.
**If it's known:** likely some version is, in behavioural game theory. We'd
then cite it as predicted-and-confirmed and lean on the *mechanistic* half
(the trust direction) as the novel part.

### A3. Is emergent LLM collusion scale-dependent?
> `Does model size affect emergent coordination between language model agents?`
> `scaling and multi-agent cooperation in large language models`
> `Do smaller language models fail to collude in pricing games?`

**Why:** our ladder is 14B → 32B → 72B with cartel language rising 0.00 → 0.12 → 0.57.
**If it's known:** reduces §4.1's novelty, but our ladder is still a cleaner
controlled comparison than most, and the *message-content inversion* is a
distinct observation.

---

## Block B — our detection claims

### B1. Keyword and transcript auditing reliability
> `How reliable is keyword detection of collusion in communication transcripts?`
> `false positive rates of lexicon-based detection of collusive communication`

**Why:** our naive lexicon flagged 42–51% of messages against a true cartel
rate of 0–3%, an 18–25× over-count.

### B2. Price parallelism as evidence
> `Is price parallelism sufficient evidence of collusion?`
> `conscious parallelism versus agreement antitrust evidence standards`
> `Do collusive firms show more or less price responsiveness to rivals?`

**Why:** our reaction coefficient is **0.000** under a stable cartel and
+0.12 to +0.37 under competition — dynamic detectors rank the worst cartel as
least suspicious. The antitrust literature on conscious parallelism is the
right frame and we should cite it properly rather than deriving it ourselves.

### B3. Probing for intent before it is expressed
> `Can linear probes on hidden states predict model behaviour before generation?`
> `Do internal representations predict language model actions better than behavioural features?`

**Why:** our timing result — the probe loses to a price-history baseline at
the prompt boundary and wins after reasoning.

---

## Block C — interpretability method claims

### C1. Decodability versus causal relevance
> `Does probe accuracy predict causal importance of a feature in neural networks?`
> `Are linearly decodable features causally used by the model?`
> `probing versus causal intervention interpretability`

**Why:** our sharpest methodological point. The trust direction is *more*
decodable (d = 7.2) and a *better* predictor of realised collusion (r = 0.634
vs 0.544) yet ~5× weaker causally. This is a known concern in principle; we
should find who stated it and whether anyone measured both halves in one setting.

### C2. Steering vector depth and dose
> `At which layers are steering vectors most effective in transformer language models?`
> `dose-response of activation steering coefficient on model behaviour`
> `Do steering vectors degrade model capabilities?`

**Why:** our depth transition (symmetric below ~L24, antisymmetric above) and
the graded sigmoid dose-response. Also whether anyone reports the
competence-preservation check we ran.

### C3. Effective dimensionality of representations
> `effective dimensionality of transformer hidden states across layers`
> `participation ratio neural network representations depth`

**Why:** our participation ratio falls 19.1 → 7.6 from L0 to L44. Worth knowing
if this is a standard finding so we cite rather than present as new.

---

## Block D — grounding and countermeasures

### D1. Maverick firms
> `Do maverick firms prevent cartel formation?`
> `effect of a non-cooperative firm on tacit collusion stability`

**Why:** our strongest countermeasure (one maverick → Δ −0.027). This is
established antitrust doctrine and we should cite it, not claim it.

### D2. Algorithmic pricing and antitrust
> `Can tacit algorithmic collusion be addressed by existing antitrust law?`
> `regulation of algorithmic pricing collusion`

**Why:** the policy framing for §5, and Calvano et al. make this argument
explicitly.

### D3. Multi-agent LLM safety
> `What are the risks of collusion between large language model agents?`
> `steganographic communication between AI agents`

**Why:** general positioning, plus the covert-channel motivation for our
restricted-channel condition.

---

## How to read the results

Three outcomes worth distinguishing:

1. **Established for human/algorithmic agents, untested for LLMs** — best case.
   We cite the prior result and our contribution is the transfer plus mechanism.
2. **Established for LLMs** — we lose novelty on that claim. Check whether our
   version adds controls or scale that theirs lacked.
3. **Nothing found** — do not assume novelty. Consensus indexes papers, not
   working papers or very recent arXiv. Cross-check anything that looks novel
   against arXiv listings and Google Scholar before claiming it in the report.

The claims where a null search result would most tempt us into over-claiming
are A1 and C1. Be most sceptical there.
