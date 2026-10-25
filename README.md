# Swayambhu

**Mechanistic detection of emergent coordination in multi-agent LMs.**

When do LLM pricing agents collude, what carries the coordination, and which interventions actually break it?

This repo contains a repeated Bertrand oligopoly where the competitive and joint-profit prices are solved analytically, so the collusion label needs no rubric and no LLM judge. On top of it sits a residual-stream capture, probing and steering pipeline that runs identically on a Modal A100 cluster or a 6 GB laptop GPU.

Submitted to the **Apart Research AI Collusion Research Sprint**, October 2026 (tracks 1 and 3). Paper: [`report/Swayambhu_Apart_submission.pdf`](report/Swayambhu_Apart_submission.pdf).

---

## Headline results

| Finding | Where |
|---|---|
| Collusion is scale-dependent: neutral markets converge to competition at 14B (Δ = 0.01) and negotiate explicit cartels at 72B (Δ = 0.19). | §4.1 |
| **Restricting the comms channel to four symbols made collusion ~4× worse** (0.196 → 0.742), not better. Free-text negotiation anchored on "fairness" and capped the cartel. | §4.2 |
| **Ablating the causal direction did nothing** (0.790 → 0.857), while **negative steering along the same direction at the same layer destroyed the cartel** (0.790 → −0.096). The operation matters, not the feature. | §4.2 |
| Two conditions reach equal collusion with opposite evidentiary traces — one evades transcript auditing, the other evades price-parallelism auditing. | §4.3 |
| Keyword auditing over-counts coordination 18–25×; price-responsiveness detectors rank the most collusive market as the *least* suspicious. | §4.4 |
| Internals become readable only *after* the model reasons — a probe at the prompt boundary loses to the visible price history. | §4.5 |
| One compromised agent raises prices across a market it does not control, with an interior optimum: over-steering gets it undercut. | §4.6 |

The collusion index is `Δ = (p̄ − p_Nash) / (p_monopoly − p_Nash)`; 0 is Bertrand-Nash, 1 is the perfect cartel. Unless stated otherwise it is the mean over the final 10 rounds, because the restricted-channel condition ratchets and an all-round mean understates the end state.

---

## Layout

```
swayambhu/              # the library — backend-agnostic, no Modal imports
  game.py               # Bertrand/logit demand; Nash + joint-profit solved from analytic FOCs
  env.py                # vectorised episode runner (batches across episodes, not time)
  agents.py             # prompt construction + THINKING/MESSAGE/PRICE parsing (pure CPU)
  conditions.py         # the 12-condition registry; run it directly to print the factorial
  capture.py            # architecture-agnostic residual hooks, steering, ablation
  metrics.py            # Δ, reaction coefficient, punishment/recovery, language rates
  probes.py             # probes + the behavioural baselines they have to beat
  gate1.py              # miniature end-to-end check of the whole machinery
  local_runner.py       # local GPU backend with the same call surface as Modal

modal_app.py            # Modal harness: one warm GPU container, batched gen + capture

scripts/
  main_runs.py          # behavioural runs across conditions, with activation capture
  trust_direction.py    # extract the partner-trust direction and the placebo contrasts
  steering.py           # sweep / controls / ablate / bank / competence phases
  probe_analysis.py     # detection comparison: residual stream vs logs vs transcripts
  geometry.py           # cosine similarity, effective dimensionality (CPU only)
  figures.py            # paper figures
  fetch_acts.py         # pull activation .npz off the Modal volume
  gate1_local.py        # Gate 1 against a local GPU
  morning_batch.sh      # the final five experiments, cheapest-and-most-valuable first
  build_docx.py         # report → submission .docx
  build_appendix.py     # generated appendix tables

report/                 # report.md, appendix.md, the submission PDF, lit review queries
```

### Design decisions worth knowing before you read the code

- **HF transformers, not vLLM.** The whole project depends on forward hooks and a paged-attention server does not expose the residual stream.
- **The block stack is located structurally**, as the longest homogeneous `nn.ModuleList`, not by module name. Swapping in a Mamba-hybrid later does not require touching `capture.py`.
- **Steering dose α is a fraction of the local residual norm**, not an absolute. Residual norms span 1.08 to 166 across depth here, so a fixed absolute α measures layer norm rather than layer sensitivity.
- **Cross-validation is grouped by episode.** Turns inside an episode share a price history; splitting on turns leaks and manufactures a result.
- **Unparseable turns are counted, not dropped.** The agent repeats its previous price and the turn is flagged. Cells above 10% parse failure are discarded, because that fallback manufactures Δ ≈ 0.5 which mimics induced collusion.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That covers the analysis side and the Modal client. `torch` and `transformers` are **not** in `requirements.txt` — they are installed inside the Modal image. If you want the local GPU backend, add them yourself:

```bash
pip install torch "transformers>=4.45,<5" accelerate safetensors
```

`transformers` is pinned below 5.x deliberately: v5 renames the load kwargs.

For the Modal path you need a Modal account and `modal setup`. The volume `swayambhu-cache` is created on first use.

---

## Running it

### 1. Gate 1 — prove the machinery works

A miniature of the full study. It checks that the collude/compete/neutral framings separate behaviourally, that hooks capture the residual stream, and that a difference-in-means direction injected into a neutral prompt actually moves the posted price.

```bash
# local, on whatever card you have (defaults to Qwen2.5-1.5B-Instruct)
python scripts/gate1_local.py
SWAY_LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct python scripts/gate1_local.py

# on Modal
SWAY_MODEL=Qwen/Qwen2.5-14B-Instruct SWAY_GPU=A100-80GB modal run modal_app.py::gate1
```

### 2. Gate 2 — behavioural runs

**This is the decision point.** If the instructed collude/compete conditions do not separate on Δ, the model cannot play the game and there is no contrast to build a direction from. Stop and change the model — do not proceed to the mechanistic pipeline.

```bash
python scripts/main_runs.py --pilot                    # few episodes, few rounds
python scripts/main_runs.py --episodes 16 --rounds 24  # the main factorial
python scripts/main_runs.py --backend local --pilot    # local GPU instead of Modal
```

### 3. Detection

```bash
python scripts/fetch_acts.py --run-id main      # skip for --backend local runs
python scripts/probe_analysis.py --run-id main
```

Reports, at each capture point: behavioural-only, context-aware, transcript, transcript + private CoT, residual-stream probe, and probe + behavioural — plus the transfer test and the two shams (global and within-episode permutation) that decide whether any of it means anything.

### 4. Causal stage

```bash
python scripts/steering.py --phase sweep   --direction data/main/direction_pre_price.npy \
    --condition E_BASE --layers 36 40 --alphas 0.2 0.4 0.6 --episodes 8 --rounds 12
python scripts/steering.py --phase controls    --layer 40 --alpha 0.4
python scripts/steering.py --phase ablate      --layer 64 --ablate-condition E_CANON
python scripts/steering.py --phase competence  --layer 40 --alpha 0.4
python scripts/steering.py --phase bank --directions <collusion.npy> <trust.npy> --layer 40
```

Useful flags: `--positions last` injects only at the decision token rather than every position; `--steer-firms 0` steers one firm of three, which is the single-compromised-agent threat model.

### 5. Everything else

```bash
bash scripts/morning_batch.sh        # the final five experiments
bash scripts/morning_batch.sh 1 3    # just steps 1 and 3
python scripts/geometry.py --run-id main
python scripts/figures.py --run-id main
```

### Environment variables

| Var | Meaning | Default |
|---|---|---|
| `SWAY_MODEL` | model served by the Modal container | `Qwen/Qwen2.5-7B-Instruct` |
| `SWAY_GPU` | Modal GPU spec, e.g. `A100-80GB:4` | `A10G` |
| `SWAY_DEVICE_MAP` | HF device map; `auto` for multi-GPU | `cuda` |
| `SWAY_LOCAL_MODEL` | model for the local backend | `Qwen/Qwen2.5-1.5B-Instruct` |

Each `SWAY_MODEL` bakes a distinct Modal image, so scale-ladder runs land in separate containers and separate volume paths and can run alongside each other.

---

## The conditions

`python -m swayambhu.conditions` prints the registry. Channel is `off` / `free` / `canon` (a four-symbol vocabulary — `HOLD`, `RAISE`, `LOWER`, `NONE` — about 2 bits per round).

| | objective | channel | note |
|---|---|---|---|
| `C_COLLUDE` / `C_COMPETE` | collude / compete | off | positive and negative controls; ground truth certain by construction |
| `E_BASE` | neutral | off | |
| `E_CHAT` | neutral | free | |
| `E_ORG` | neutral | free | one sentence permitting the firms to organise the market |
| `E_CANON` | neutral | canon | the restricted channel — strongest collusion in the study |
| `E_STRAT` / `E_STRAT_CHAT` | neutral | off / free | states the true fact that rivals observe prices and may respond |
| `M_PRESENT` | neutral | free | a maverick firm actually present |
| `M_RUMOUR` | neutral | free | a maverick merely rumoured — strongest *emergent* collusion at 14B |
| `CC_COLLUDE` / `CC_COMPETE` | collude / compete | free | matched controls for the chat arm; dropped first if budget is short |

`C_COLLUDE` and `C_COMPETE` run with comms **off** so they differ from `E_BASE` in exactly one respect — the objective line. Any direction extracted from that contrast is therefore not confounded with "a chat channel exists".

The probe trains on the labelled (instructed) conditions and is tested on the emergent ones. If it only worked in-distribution it would most likely be reading the system prompt.

---

## Reproducing the paper

| Paper section | Command |
|---|---|
| §4.1 scale ladder, Appendix A factorial | `main_runs.py --episodes 16 --rounds 24` at each `SWAY_MODEL` |
| §4.2 bandwidth backfire | the `E_CANON` arm of the above at 72B |
| §4.2 ablate vs steer | `morning_batch.sh 1`, then `steering.py --phase sweep` with negative alphas |
| §4.4 keyword over-count | `probe_analysis.py` transcript columns |
| §4.5 detection timing, Appendix C | `probe_analysis.py --run-id main` |
| §4.6 dose-response, Appendix B1 | `steering.py --phase sweep` |
| §4.6 placebos, Appendix B3 | `morning_batch.sh 2` |
| §4.6 final-position only | `morning_batch.sh 3` |
| §4.6 gated bank | `morning_batch.sh 4` |
| §4.6 single-agent steering, Appendix B4 | `morning_batch.sh 5` |
| Appendix D geometry | `geometry.py --run-id main` |

Models: `Qwen/Qwen2.5-{14B,32B,72B}-Instruct`, bf16, A100-80GB (×4 for 72B), temperature 0.7.

**Data.** `data/` is committed — 26 MB across 22 run directories: every episode trajectory as JSON (including failed and unparseable turns), the run logs under `data/logs/`, and the eight extracted direction vectors.

The directions are the part that matters. Extracting them is the GPU-expensive step, so with `data/main/direction_pre_price.npy`, `direction_trust.npy`, the two placebos and the 72B pair in `data/cap72/` already in hand, every steering and ablation result in §4.2 and §4.6 reruns from the commands above without re-capturing anything.

Activation tensors themselves (≈5 GB, `data/activations_*/`) are excluded and are reproducible from the capture scripts.

---

## Known limitations

Three Qwen2.5 models, one environment, one model family. The scale ladder is three points and confounds size with post-training. The bandwidth-backfire result rests on 8 episodes per run at one scale, replicated once — **it is the single most important thing for others to reproduce.** The ablation null and the negative-steering positive are each one condition at one scale with one geometry. Both the restricted-channel ratchet and the steering saturations are bounded by the 13-point price grid, so neither Δ = 0.742 nor Δ = −0.096 measures where the agents would actually stop.

Full limitations in §5.1; dual-use assessment in Appendix E.

---

## Citation

```bibtex
@misc{jha2026swayambhu,
  title  = {Pushing Back, Not Deleting: Scale, Channels and Causal Control
            of Collusion Between LLM Pricing Agents},
  author = {Jha, Ariet},
  year   = {2026},
  note   = {Apart Research AI Collusion Research Sprint},
  url    = {https://github.com/AJ797/Swayambhu}
}
```

Built on the Calvano et al. (2020) logit-demand specification; the solver reproduces their published duopoly constants (`p_N = 1.4729`, `p_M = 1.9250`) as a correctness check.

## LLM usage

Claude (Anthropic) was used throughout as a coding and drafting tool. See the LLM Usage Statement in the paper for the full split.
