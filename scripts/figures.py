"""Paper figures.

    python scripts/figures.py --run-id main

Every figure is skipped gracefully if its input file is absent, so this can be
run at any point to see what currently exists.

Colour: the four categorical slots are used in fixed order and were validated
with the dataviz validator against a white surface (all six checks PASS; the
aqua/yellow contrast WARN obligates visible labels, which is why every bar
carries a direct value label). These are static figures for a PDF, so there is
no hover layer and no dark variant. One y-axis per panel, never two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swayambhu import metrics as M  # noqa: E402
from swayambhu.game import BertrandConfig  # noqa: E402

# validated categorical slots, fixed order
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# which experimental arm each condition belongs to -> colour by arm, not by rank
ARM = {
    "C_COLLUDE": "instructed", "C_COMPETE": "instructed",
    "CC_COLLUDE": "instructed", "CC_COMPETE": "instructed",
    "E_BASE": "emergent", "E_CHAT": "emergent", "E_ORG": "emergent",
    "E_CANON": "emergent", "E_STRAT": "emergent", "E_STRAT_CHAT": "emergent",
    "M_PRESENT": "trust", "M_RUMOUR": "trust",
}
ARM_COLOUR = {"instructed": C1, "emergent": C2, "trust": C3}
ORDER = list(ARM.keys())


def load_run(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = BertrandConfig(**summary["config"])
    eps = {}
    for p in sorted(run_dir.glob("*.json")):
        if p.stem in ("summary", "probe_results"):
            continue
        eps[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return cfg, summary, eps


def _order(names):
    return [n for n in ORDER if n in names] + [n for n in names if n not in ARM]


def fig_delta_by_condition(cfg, eps, out: Path):
    names = _order(list(eps))
    vals, errs, cols = [], [], []
    for n in names:
        s = eps[n]["stats"]
        vals.append(s["delta_last10"])
        errs.append(s["delta_last10_sd"])
        cols.append(ARM_COLOUR.get(ARM.get(n, ""), C4))

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 1.5))
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=errs, height=0.62, color=cols,
            error_kw=dict(ecolor=INK2, lw=1.2, capsize=2.5))
    for yi, v in zip(y, vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=1.2)
    ax.axvline(1, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.text(1, len(names) - 0.3, " perfect cartel", fontsize=7.5, color=INK2, va="top")
    ax.text(0, len(names) - 0.3, " Bertrand-Nash", fontsize=7.5, color=INK2, va="top")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("collusion index  $\\Delta$   (final 10 rounds)")
    ax.set_title("Instructed collusion is executed perfectly; none emerges unprompted")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ARM_COLOUR.values()]
    ax.legend(handles, list(ARM_COLOUR), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_trajectory(cfg, eps, out: Path, keys=None):
    keys = [k for k in (keys or ["C_COLLUDE", "C_COMPETE", "E_BASE", "E_CHAT",
                                 "E_STRAT_CHAT", "M_PRESENT"]) if k in eps]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cols = [C1, C2, C3, C4, "#6250d6", "#e87ba4"]
    for k, c in zip(keys, cols):
        traj, se = M.delta_trajectory(eps[k]["episodes"], cfg)
        x = np.arange(1, len(traj) + 1)
        ax.plot(x, traj, lw=2.0, color=c, label=k, marker="o", ms=3.2)
        ax.fill_between(x, traj - se, traj + se, color=c, alpha=0.14, lw=0)
        ax.annotate(k, (x[-1], traj[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=c, va="center")
    ax.axhline(0, color=INK2, lw=1.2)
    ax.axhline(1, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("round")
    ax.set_ylabel("collusion index  $\\Delta$")
    ax.set_title("Unprompted markets converge toward the competitive equilibrium")
    ax.legend(fontsize=8, ncol=2, loc="center right")
    ax.set_xlim(1, len(traj) * 1.18)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_keyword_overcount(cfg, eps, out: Path):
    """What a keyword auditor sees vs what the messages actually say vs what the
    market actually does. All three series are rates in [0,1] -- ONE axis."""
    names = [n for n in _order(list(eps))
             if eps[n]["stats"]["condition_spec"]["comms"] != "off"]
    if not names:
        return
    naive, cartel, delta = [], [], []
    for n in names:
        ms = M.message_stats(eps[n]["episodes"])
        naive.append(ms["naive_coord_rate"])
        cartel.append(ms["cartel_rate"])
        delta.append(max(0.0, eps[n]["stats"]["delta_last10"]))

    fig, ax = plt.subplots(figsize=(max(3.4, min(7.0, 1.35 * len(names) + 2.0)), 3.5))
    x = np.arange(len(names))
    w = 0.26
    bars = [
        (x - w - 0.012, naive, C2, "flagged by naive\ncoordination keywords"),
        (x, cartel, C1, "actually cartel language\n(hold / keep high / agree)"),
        (x + w + 0.012, delta, C3, "realised collusion\nindex $\\Delta$"),
    ]
    for xs, vals, col, lab in bars:
        ax.bar(xs, vals, w, color=col, label=lab)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, names, fontsize=8, rotation=20, ha="right")
    ax.set_xlim(-0.62, len(names) - 0.38)
    ax.set_ylabel("rate  /  $\\Delta$")
    ax.set_ylim(0, 1.05)
    ax.set_title("Keyword transcript auditing over-counts coordination by ~20x")
    ax.legend(fontsize=7.5, loc="upper right", ncol=1)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_layer_sweep(probe: dict, out: Path):
    pts = probe.get("points", {})
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for (name, blob), c in zip(pts.items(), [C1, C2]):
        sweep = blob.get("sweep") or []
        if not sweep:
            continue
        xs = [r["layer"] for r in sweep]
        ys = [r["auc"] for r in sweep]
        ax.plot(xs, ys, lw=2.0, marker="o", ms=4.0, color=c, label=name)
    ax.axhline(0.5, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.01, 0.505, " chance", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=INK2)
    ax.set_xlabel("layer")
    ax.set_ylabel("probe AUC (grouped CV by episode)")
    ax.set_title("Where in the residual stream the signal lives")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_detection(probe: dict, out: Path):
    pts = probe.get("points", {})
    point = "pre_price" if "pre_price" in pts else next(iter(pts), None)
    if not point:
        return
    comp = pts[point].get("comparison") or {}
    if not comp:
        return
    labels = {
        "A_behavioural": "behavioural (price history)",
        "C_transcript_msg": "transcript: messages only",
        "C2_transcript_all": "transcript: + private reasoning",
        "D_probe": "residual-stream probe",
        "E_probe_plus_beh": "probe + behavioural",
    }
    keys = [k for k in labels if k in comp]
    vals = [comp[k]["auc"] for k in keys]
    cols = [C2 if "transcript" in labels[k] else (C1 if "probe" in labels[k] else C3)
            for k in keys]

    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(keys) + 1.4))
    y = np.arange(len(keys))
    ax.barh(y, vals, height=0.6, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.set_yticks(y, [labels[k] for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.45, max(vals) + 0.06)
    ax.set_xlabel("AUC")
    ax.set_title(f"Detection comparison ({point})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_steering(sweep: dict, out: Path):
    rows = [r for r in sweep.get("sweep", []) if r.get("layer") is not None]
    if not rows:
        return
    base = sweep.get("baseline", {}).get("delta_all")
    layers = sorted({r["layer"] for r in rows})[:4]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for L, c in zip(layers, [C1, C2, C3, C4]):
        sub = sorted((r for r in rows if r["layer"] == L), key=lambda r: r["alpha"])
        ax.plot([r["alpha"] for r in sub], [r["delta"] for r in sub],
                lw=2.0, marker="o", ms=4.5, color=c, label=f"layer {L}")
    if base is not None:
        ax.axhline(base, color=INK2, lw=1.2, ls=(0, (4, 3)))
        ax.text(0.01, base, " unsteered baseline", transform=ax.get_yaxis_transform(),
                fontsize=7.5, color=INK2, va="bottom")
    ax.set_xlabel(r"steering dose $\alpha$  (fraction of local residual norm)")
    ax.set_ylabel("collusion index  $\\Delta$")
    ax.set_title("Dose-response: injecting the collusion direction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_scale_ladder(runs: list[tuple[str, Path]], out: Path):
    """Delta vs model size for the key conditions.

    Either outcome is a result: flat lines make the non-emergence robust across
    scale; a rising emergent line locates an emergence threshold.
    """
    keys = ["C_COLLUDE", "C_COMPETE", "E_BASE", "E_CHAT"]
    labels, series = [], {k: [] for k in keys}
    for label, d in runs:
        if not (d / "summary.json").exists():
            continue
        labels.append(label)
        for k in keys:
            f = d / f"{k}.json"
            series[k].append(
                json.loads(f.read_text(encoding="utf-8"))["stats"]["delta_last10"]
                if f.exists() else np.nan
            )
    if len(labels) < 2:
        print("  (fewer than two scale points -- skipping ladder)")
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(labels))
    for k, c in zip(keys, [C1, C2, C3, C4]):
        ys = series[k]
        ax.plot(x, ys, lw=2.0, marker="o", ms=5.5, color=c, label=k)
        for xi, v in zip(x, ys):
            if np.isfinite(v):
                ax.annotate(f"{v:+.2f}", (xi, v), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=c)
    ax.axhline(0, color=INK2, lw=1.2)
    ax.axhline(1, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("collusion index  $\\Delta$   (final 10 rounds)")
    ax.set_xlabel("model")
    ax.set_ylim(-0.25, 1.2)
    ax.set_title("Does emergent collusion appear with scale?")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--steer-id", default="steer")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    run_dir = Path("data") / args.run_id
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "summary.json").exists():
        raise SystemExit(f"no run at {run_dir}")

    cfg, summary, eps = load_run(run_dir)
    print(f"conditions: {sorted(eps)}")
    tag = args.run_id

    fig_delta_by_condition(cfg, eps, out_dir / f"{tag}_f1_delta_by_condition.png")
    fig_trajectory(cfg, eps, out_dir / f"{tag}_f2_trajectory.png")
    fig_keyword_overcount(cfg, eps, out_dir / f"{tag}_f3_keyword_overcount.png")

    pr = run_dir / "probe_results.json"
    if pr.exists():
        probe = json.loads(pr.read_text(encoding="utf-8"))
        fig_layer_sweep(probe, out_dir / f"{tag}_f4_layer_sweep.png")
        fig_detection(probe, out_dir / f"{tag}_f5_detection.png")
    else:
        print("  (no probe_results.json yet -- skipping probe figures)")

    ladder = [
        ("14B", Path("data/main")),
        ("32B", Path("data/scale32")),
        ("72B", Path("data/scale72")),
    ]
    fig_scale_ladder([(a, b) for a, b in ladder if b.exists()],
                     out_dir / "f7_scale_ladder.png")

    sw = Path("data") / args.steer_id / "sweep.json"
    if sw.exists():
        fig_steering(json.loads(sw.read_text(encoding="utf-8")),
                     out_dir / f"{tag}_f6_steering.png")
    else:
        print("  (no steering sweep yet -- skipping dose-response figure)")


if __name__ == "__main__":
    main()
