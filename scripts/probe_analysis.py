"""Detection comparison: does reading the residual stream beat reading the logs?

    # pull activations off the Modal volume first (skip for --backend local runs)
    modal volume get swayambhu-cache /acts/main ./data/acts/main

    python scripts/probe_analysis.py --run-id main

Reports, at each capture point:
  A  behavioural-only        -- public price history, as an auditor sees it
  B  context-aware           -- A plus task state
  C  transcript (messages)   -- what an auditor can actually read
  C' transcript (+thinking)  -- generous upper bound, includes private CoT
  D  residual-stream probe
  E  probe + behavioural

Plus the two tests that decide whether D means anything:
  * TRANSFER: fit on instructed conditions, evaluate on emergent ones. If the
    probe only works in-distribution it is probably reading the instruction off
    nearby tokens rather than representing the strategy.
  * SHAM: labels shuffled within episode. Must land at chance, or the pipeline
    leaks and every other number here is void.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swayambhu.conditions import CONDITIONS  # noqa: E402
from swayambhu.game import BertrandConfig  # noqa: E402
from swayambhu.probes import (  # noqa: E402
    build_dataset,
    cv_score,
    diff_in_means_direction,
    layer_sweep,
    sham_control,
    within_condition_scores,
    talk_action_divergence,
    transfer_score,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--capture-points", nargs="*", default=["pre_reasoning", "pre_price"])
    ap.add_argument("--target", choices=["supra", "condition"], default="supra")
    ap.add_argument("--min-round", type=int, default=3)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--fetch", action="store_true",
                    help="run `modal volume get` for this run-id first")
    ap.add_argument("--act-dir", default=None,
                    help="where the .npz files live (default data/activations_<run-id>)")
    return ap.parse_args()


def fetch_activations(run_id: str) -> None:
    dest = Path("data/acts")
    dest.mkdir(parents=True, exist_ok=True)
    # `modal volume get` refuses to write into an existing directory, so never
    # pre-create the target -- only the parent.
    if (dest / run_id).exists() and any((dest / run_id).iterdir()):
        print(f"[fetch] data/acts/{run_id} already present, skipping")
        return
    # NB: no leading slash. `modal volume ls swayambhu-cache /acts` reports
    # "No such file or directory" even though the path exists.
    # --force is required even when the target is absent: the command appears to
    # create the directory before running its own existence check, so without it
    # the fetch fails on a clean filesystem.
    cmd = ["modal", "volume", "get", "--force", "swayambhu-cache",
           f"acts/{run_id}", str(dest / run_id)]
    print("[fetch] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def fmt(name: str, r: dict) -> str:
    return "  {:<28} AUC {:>6.3f}   AP {:>6.3f}   n={:<6} pos={:.2f}".format(
        name, r["auc"], r["ap"], r["n"], r["pos_rate"]
    )


def main() -> None:
    args = parse_args()
    if args.fetch:
        fetch_activations(args.run_id)

    run_dir = Path("data") / args.run_id
    act_dir = Path(args.act_dir) if args.act_dir else Path(f"data/activations_{args.run_id}")
    if not act_dir.exists() or not any(act_dir.glob("*.npz")):
        raise SystemExit(
            f"no .npz files at {act_dir}; run scripts/fetch_acts.py --run-id {args.run_id}"
        )

    # only real condition files -- the directory also holds summary.json and
    # this script's own probe_results.json output
    ep_files = {p.stem: p for p in sorted(run_dir.glob("*.json")) if p.stem in CONDITIONS}
    if not ep_files:
        raise SystemExit(f"no episode files in {run_dir}")
    print(f"conditions found: {sorted(ep_files)}")

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = BertrandConfig(**summary["config"])

    train_c = [k for k in ep_files if CONDITIONS[k].label is not None]
    test_c = [k for k in ep_files if CONDITIONS[k].label is None]
    print(f"instructed (train): {train_c}")
    print(f"emergent  (test) : {test_c}")

    results: dict = {"run_id": args.run_id, "target": args.target, "points": {}}

    for point in args.capture_points:
        print("\n" + "=" * 78)
        print(f"CAPTURE POINT: {point}")
        print("=" * 78)
        ds = build_dataset(
            ep_files, act_dir, cfg,
            capture_point=point, target=args.target, min_round=args.min_round,
        )
        print(f"  rows={len(ds)}  episodes={len(np.unique(ds.groups))}  "
              f"layers={list(ds.layers)}  positive rate={ds.y.mean():.3f}")

        # ---- layer sweep locates the signal
        sweep = layer_sweep(ds, n_splits=args.splits)
        best = max(sweep, key=lambda r: (r["auc"] if np.isfinite(r["auc"]) else -1))
        print("\n  layer sweep (probe AUC):")
        for r in sweep:
            bar = "#" * int(max(0.0, (r["auc"] - 0.5)) * 60) if np.isfinite(r["auc"]) else ""
            print(f"    L{r['layer']:>3}  {r['auc']:.3f}  {bar}")
        print(f"  best layer = L{best['layer']} (pos {best['layer_pos']}), "
              f"AUC {best['auc']:.3f}")
        lp = best["layer_pos"]

        # ---- baselines and the probe, all grouped by episode
        print("\n  detection comparison (grouped CV by episode):")
        rows = {}
        rows["A_behavioural"] = cv_score(ds.X_beh, ds.y, ds.groups, n_splits=args.splits)
        rows["C_transcript_msg"] = cv_score(
            ds.text_msg, ds.y, ds.groups, n_splits=args.splits, text=True
        )
        rows["C2_transcript_all"] = cv_score(
            ds.text_all, ds.y, ds.groups, n_splits=args.splits, text=True
        )
        rows["D_probe"] = cv_score(
            ds.X_act[:, lp, :].astype(np.float32), ds.y, ds.groups, n_splits=args.splits
        )
        combined = np.hstack([ds.X_act[:, lp, :].astype(np.float32), ds.X_beh])
        rows["E_probe_plus_beh"] = cv_score(combined, ds.y, ds.groups, n_splits=args.splits)
        for k, v in rows.items():
            print(fmt(k, v))

        # ---- shams: one checks the machinery, one checks the target
        sham = sham_control(ds, lp, n_splits=args.splits)
        print("\n  SHAM CONTROLS:")
        print(fmt("global shuffle (must be ~0.5)", sham["global"]))
        print(fmt("within-episode shuffle", sham["within_episode"]))
        if np.isfinite(sham["global"]["auc"]) and sham["global"]["auc"] > 0.60:
            print("    *** The machinery itself leaks. Discard everything above.")
        elif np.isfinite(sham["within_episode"]["auc"]) and sham["within_episode"]["auc"] > 0.60:
            print("    *** Target is CONFOUNDED WITH CONDITION: within-episode shuffling")
            print("    *** preserves each condition's base rate, so a probe that merely")
            print("    *** recognises the system prompt scores well. The POOLED numbers")
            print("    *** above are not evidence of collusion detection. Read the")
            print("    *** within-condition table below instead.")

        # ---- the confound-free comparison: inside one condition the prompt is fixed
        print("\n  WITHIN-CONDITION (prompt held constant -- the numbers that count):")
        wc = within_condition_scores(ds, lp, n_splits=args.splits)
        print("    {:<14} {:>7} {:>9} {:>11} {:>8}".format(
            "condition", "pos", "probe", "behavioural", "transcript"))
        for cond, r in wc.items():
            if r.get("status") != "ok":
                print(f"    {cond:<14} skipped ({r.get('n')} rows)")
                continue
            def g(k):
                v = r[k]["auc"]
                return f"{v:9.3f}" if np.isfinite(v) else f"{'n/a':>9}"
            print("    {:<14} {:>7.2f} {} {} {}".format(
                cond, r["pos_rate"], g("probe"), g("behavioural")[:11].rjust(11),
                g("transcript")[:8].rjust(8)))

        # ---- transfer: the test that decides whether the probe is real
        transfer = None
        if train_c and test_c:
            try:
                transfer = transfer_score(ds, train_c, test_c, lp)
                print("\n  TRANSFER (fit on instructed -> test on emergent):")
                print(fmt("transfer", transfer))
                per_cond = {}
                for c in test_c:
                    m = (ds.condition == c) & transfer["mask"]
                    sel = transfer["mask"]
                    sub_y = ds.y[sel][ds.condition[sel] == c]
                    sub_s = transfer["scores"][ds.condition[sel] == c]
                    if len(np.unique(sub_y)) > 1:
                        from sklearn.metrics import roc_auc_score

                        per_cond[c] = float(roc_auc_score(sub_y, sub_s))
                    else:
                        per_cond[c] = float("nan")
                print("    per emergent condition:")
                for c, a in per_cond.items():
                    print(f"      {c:<12} AUC {a:.3f}")
                transfer = {k: v for k, v in transfer.items() if k not in ("scores", "mask")}
                transfer["per_condition"] = per_cond
            except RuntimeError as e:
                print(f"\n  TRANSFER skipped: {e}")

        # ---- transcript detectability per condition: the channel-ban story
        print("\n  transcript (messages only) detectability by condition:")
        per_cond_txt = {}
        for c in sorted(ep_files):
            m = ds.condition == c
            if m.sum() < 20 or len(np.unique(ds.y[m])) < 2:
                per_cond_txt[c] = float("nan")
                print(f"      {c:<12} n/a (too few rows or single class)")
                continue
            r = cv_score(
                [ds.text_msg[i] for i in np.where(m)[0]],
                ds.y[m], ds.groups[m], n_splits=min(args.splits, len(np.unique(ds.groups[m]))),
                text=True,
            )
            per_cond_txt[c] = r["auc"]
            print(f"      {c:<12} AUC {r['auc']:.3f}  (n={r['n']})")

        # ---- talk vs action: can internals see through cheap talk?
        print("\n  TALK-ACTION DIVERGENCE (turns that sent coordination language):")
        tad = talk_action_divergence(ds, lp, n_splits=args.splits)
        if tad.get("status") != "ok":
            print(f"    skipped: {tad.get('status')} "
                  f"({tad['n_coord_turns']} coord turns of {tad['n_total']})")
        else:
            print(f"    {tad['n_coord_turns']} coordination-language turns; "
                  f"{tad['defection_rate']:.1%} of them undercut every rival anyway")
            print("    predicting WHICH of those turns defects:")
            for k in ("transcript", "behavioural", "probe"):
                print(fmt("  " + k, tad[k]))

        results["points"][point] = {
            "talk_action_divergence": tad,
            "n_rows": len(ds),
            "layers": [int(x) for x in ds.layers],
            "sweep": sweep,
            "best_layer": best,
            "comparison": {k: {m: v[m] for m in ("auc", "ap", "n", "pos_rate")}
                           for k, v in rows.items()},
            "sham": sham,
            "within_condition": wc,
            "transfer": transfer,
            "transcript_by_condition": per_cond_txt,
        }

        # ---- steering direction for the causal stage
        if train_c:
            pos = [c for c in train_c if CONDITIONS[c].label == 1]
            neg = [c for c in train_c if CONDITIONS[c].label == 0]
            if pos and neg:
                v = diff_in_means_direction(ds, pos, neg)
                out = Path("data") / args.run_id / f"direction_{point}.npy"
                np.save(out, v)
                print(f"\n  saved steering direction -> {out}  shape {v.shape}")

    out = Path("data") / args.run_id / "probe_results.json"
    out.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
