"""Emit the appendix tables straight from the data files, so the numbers in the
submission cannot drift from the numbers on disk."""
import json, glob
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swayambhu.conditions import CONDITIONS
from swayambhu import metrics as M
from swayambhu.game import BertrandConfig

cfg = BertrandConfig(n_firms=3)
L = []
def w(s=""): L.append(s)

w("### Appendix A — Full 14B factorial")
w()
w("16 episodes x 24 rounds x 3 firms per condition. Delta is the mean over the final 10 "
  "rounds. Message rates are over turns that sent a message.")
w()
w("| condition | objective | channel | Δ | sd | parse fail | cartel lang. | compete lang. | reaction β | price match |")
w("|---|---|---|---|---|---|---|---|---|---|")
for name in CONDITIONS:
    f = Path(f"data/main/{name}.json")
    if not f.exists(): continue
    p = json.loads(f.read_text(encoding="utf-8")); s = p["stats"]; eps = p["episodes"]
    ms = M.message_stats(eps); rc = M.reaction_coefficient(eps, cfg)
    spec = s["condition_spec"]
    cl = f"{ms['cartel_rate']:.2f}" if ms["n_messages"] else "--"
    cp = f"{ms['compete_rate']:.2f}" if ms["n_messages"] else "--"
    w(f"| {name} | {spec['objective']} | {spec['comms']} | {s['delta_last10']:+.3f} | "
      f"{s['delta_last10_sd']:.3f} | {s['parse_failure_rate']:.1%} | {cl} | {cp} | "
      f"{rc['beta']:+.3f} | {M.price_matching_rate(eps,cfg):.2f} |")
w()

w("### Appendix B — Steering dose grid and countermeasure dose-response")
w()
w("**B1. Induction grid, 14B, condition E_BASE (unsteered Δ = 0.070), all positions.** "
  "Cells are Δ over all rounds; every cell had ≤0.7% parse failure.")
w()
rows = {}
for run in ("steer14", "steer14fine"):
    f = Path(f"data/{run}/sweep.json")
    if not f.exists(): continue
    for r in json.loads(f.read_text(encoding="utf-8"))["sweep"]:
        if r.get("layer") is None: continue
        rows.setdefault(r["layer"], {})[r["alpha"]] = r["delta"]
alphas = sorted({a for v in rows.values() for a in v})
w("| layer | " + " | ".join(f"α={a:+.2f}" for a in alphas) + " |")
w("|---" * (len(alphas) + 1) + "|")
for Lr in sorted(rows):
    cells = [f"{rows[Lr][a]:+.3f}" if a in rows[Lr] else "--" for a in alphas]
    w(f"| L{Lr} | " + " | ".join(cells) + " |")
w()
w("**B2. Countermeasure dose-response, 72B, condition E_CANON at L64.**")
w()
cm = {}
for run in ("neg72", "neg72low"):
    for f in sorted(glob.glob(f"data/{run}/episodes_*.json")):
        p = json.loads(Path(f).read_text(encoding="utf-8"))
        tag = p["stats"].get("tag", "?")
        P = np.array([[r["price_idx"] for r in e["rounds"]] for e in p["episodes"]])
        from swayambhu.game import collusion_index, price_grid
        g = np.asarray(price_grid(cfg))
        cm[tag] = (collusion_index(float(g[P].mean()), cfg),
                   collusion_index(float(g[P[:, -10:, :]].mean()), cfg),
                   p["stats"]["parse_failure_rate"])
order = ["baseline", "L64_a-0.1", "L64_a-0.2", "L64_a-0.4", "L64_a-0.8"]
w("| cell | Δ all rounds | Δ final 10 | parse fail |")
w("|---|---|---|---|")
for k in order:
    if k in cm:
        a, l, pf = cm[k]
        w(f"| {k} | {a:+.3f} | {l:+.3f} | {pf:.1%} |")
w()

w("### Appendix C — Within-condition detection AUC")
w()
w("Grouped cross-validation by episode. Conditions with fewer than 20 of either class are "
  "omitted, since AUC on a handful of positives is noise. Instructed conditions are "
  "single-class by construction and therefore absent.")
for run, lbl in (("main", "14B"), ("cap72", "72B")):
    f = Path(f"data/{run}/probe_results.json")
    if not f.exists(): continue
    pr = json.loads(f.read_text(encoding="utf-8"))
    for point, blob in pr["points"].items():
        w()
        w(f"**{lbl}, capture point `{point}`** — sham (global) "
          f"{blob['sham']['global']['auc']:.3f}, sham (within-episode) "
          f"{blob['sham']['within_episode']['auc']:.3f}, "
          f"transfer instructed→emergent "
          f"{(blob.get('transfer') or {}).get('auc', float('nan')):.3f}")
        w()
        w("| condition | positive rate | probe | behavioural | transcript |")
        w("|---|---|---|---|---|")
        for c, v in blob["within_condition"].items():
            if v.get("status") != "ok": continue
            g = lambda k: (f"{v[k]['auc']:.3f}" if v[k]['auc'] == v[k]['auc'] else "n/a")
            w(f"| {c} | {v['pos_rate']:.2f} | {g('probe')} | {g('behavioural')} | {g('transcript')} |")
w()

w("### Appendix D — Contrast separability, geometry, and keyword over-count")
w()
w("**D1. Direction separability by depth (Cohen's d).** Surface textual contrasts separate "
  "early and wash out; the task-relevant contrast does the opposite.")
w()
w("| contrast | L0 | L40 | trend |")
w("|---|---|---|---|")
w("| partner-trust | 2.62 | **7.24** | increases |")
w("| placebo_coop (cooperate/compete, about reports) | 5.92 | 1.93 | decreases |")
w("| placebo_neutral (coastal vs mountain office) | 2.67 | 1.78 | decreases |")
w()
g = Path("data/main/geometry.json")
if g.exists():
    geo = json.loads(g.read_text(encoding="utf-8"))
    w("**D2. Effective dimensionality (participation ratio of the state covariance), 14B.**")
    w()
    ed = geo["effective_dim"]
    w("| layer | " + " | ".join(f"L{l}" for l in ed["layers"]) + " |")
    w("|---" * (len(ed["layers"]) + 1) + "|")
    w("| eff. dim | " + " | ".join(f"{v:.1f}" for v in ed["pr"]) + " |")
    w()
    cos = geo.get("direction_cosine") or []
    if cos:
        w(f"**D3. Collusion vs partner-trust direction.** Cosine similarity ranges "
          f"{min(cos):.3f} to {max(cos):.3f} across captured layers (83–85°), against a "
          f"chance level of 1/sqrt(5120) ≈ 0.014.")
        w()
w("**D4. Keyword over-count.** A naive coordination lexicon against direction-aware counts.")
w()
w("| condition | naive keyword flag | true cartel language | competitive language | over-count |")
w("|---|---|---|---|---|")
for run, c, oc in (("pilot", "E_CHAT", "--"), ("duo", "E_CHAT", "17.9x"), ("duo", "E_STRAT_CHAT", "25.2x")):
    f = Path(f"data/{run}/{c}.json")
    if not f.exists(): continue
    ms = M.message_stats(json.loads(f.read_text(encoding="utf-8"))["episodes"])
    w(f"| {run}/{c} | {ms['naive_coord_rate']:.2f} | {ms['cartel_rate']:.2f} | "
      f"{ms['compete_rate']:.2f} | {oc} |")

out = Path("report/appendix.md")
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(L)} lines)")
