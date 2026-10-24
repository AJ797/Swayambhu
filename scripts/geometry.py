"""Representation geometry of the captured states. CPU only, no GPU needed.

Everything here runs on activations already on disk. It answers three
questions the probe and steering work leaves open:

1. Are the collusion and partner-trust directions the SAME direction?
   If cosine similarity is high, the "trust" axis is the collusion axis
   rediscovered under a different contrast, and the two steering results are
   one result. If it is low, they are distinct routes to the same behaviour,
   which is a stronger claim and a harder one to get by accident.

2. Where does each condition sit in the plane spanned by those two axes?
   A condition's mean state projected onto (collusion, trust) places it on a
   map. M_RUMOUR is the interesting case: it collides as hard as E_ORG but
   says nothing about it, so if belief-about-partners is what drives it, it
   should separate along the trust axis rather than the collusion axis.

3. How concentrated is the representation at each depth?
   Participation ratio of the activation covariance spectrum gives an
   effective dimensionality; a direction is more plausibly meaningful where
   the state is low-dimensional than where it is diffuse.

    python scripts/geometry.py --run-id main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swayambhu.conditions import CONDITIONS  # noqa: E402


def load_condition_states(act_dir: Path, cond: str, point: str):
    """-> [n_turns, n_layers, d_model] float32 for one condition."""
    files = sorted(act_dir.glob(f"{cond}__r*.npz"))
    if not files:
        return None, None
    blocks, layers = [], None
    for f in files:
        z = np.load(f)
        blocks.append(z[point].astype(np.float32))
        layers = z["layers"]
    return np.concatenate(blocks, axis=0), layers


def participation_ratio(X: np.ndarray) -> float:
    """Effective dimensionality: (sum l)^2 / sum l^2 over covariance eigenvalues.
    1 means a single direction carries everything; higher means diffuse."""
    Xc = X - X.mean(0, keepdims=True)
    # singular values of the centred matrix; eigenvalues of cov are s^2
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    if lam.sum() <= 0:
        return float("nan")
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--point", default="pre_price")
    ap.add_argument("--collusion", default="data/main/direction_pre_price.npy")
    ap.add_argument("--trust", default="data/main/direction_trust.npy")
    ap.add_argument("--act-dir", default=None)
    args = ap.parse_args()

    act_dir = Path(args.act_dir or f"data/activations_{args.run_id}")
    vc = np.load(args.collusion).astype(np.float32)
    has_trust = Path(args.trust).exists()
    vt = np.load(args.trust).astype(np.float32) if has_trust else None

    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    uc = unit(vc)
    ut = unit(vt) if has_trust else None

    out: dict = {"run_id": args.run_id, "point": args.point}

    # ---- 1. are the two directions the same direction?
    if has_trust:
        print("=" * 68)
        print("1. COLLUSION vs PARTNER-TRUST DIRECTION")
        print("=" * 68)
        print(f"  {'layer':>6} {'cosine':>9} {'angle_deg':>11}")
        cos_by_layer = []
        for i in range(uc.shape[0]):
            c = float(np.dot(uc[i], ut[i]))
            cos_by_layer.append(c)
            print(f"  {i*4:6d} {c:9.3f} {np.degrees(np.arccos(np.clip(c,-1,1))):11.1f}")
        out["direction_cosine"] = cos_by_layer
        # a random baseline for 5120 dims: E|cos| ~ 1/sqrt(d)
        d = uc.shape[1]
        print(f"\n  chance level for {d} dims: |cos| ~ {1/np.sqrt(d):.4f}")
        print(f"  mean |cos| across layers: {np.mean(np.abs(cos_by_layer)):.3f}")

    # ---- 2. condition map in the (collusion, trust) plane
    print("\n" + "=" * 68)
    print(f"2. CONDITION MAP at {args.point}  (projection of mean state)")
    print("=" * 68)
    conds = [c for c in CONDITIONS if (act_dir / f"{c}__r003.npz").exists()]
    deep = uc.shape[0] - 2          # a deep layer: L40 when stride is 4
    rows = {}
    print(f"  layer L{deep*4};  values are z-scored across conditions")
    raw = {}
    for c in conds:
        X, _ = load_condition_states(act_dir, c, args.point)
        if X is None:
            continue
        m = X[:, deep, :].mean(0)
        raw[c] = (float(np.dot(m, uc[deep])),
                  float(np.dot(m, ut[deep])) if has_trust else float("nan"))
    if raw:
        cx = np.array([v[0] for v in raw.values()])
        cy = np.array([v[1] for v in raw.values()])
        zx = (cx - cx.mean()) / (cx.std() + 1e-8)
        zy = (cy - cy.mean()) / (cy.std() + 1e-8) if has_trust else cy
        print(f"\n  {'condition':<14} {'collusion-axis':>15} {'trust-axis':>12}")
        for (c, _), a, b in zip(raw.items(), zx, zy):
            rows[c] = {"collusion_z": float(a), "trust_z": float(b)}
            print(f"  {c:<14} {a:>15.2f} {b:>12.2f}")
        out["condition_map"] = rows

    # ---- 3. effective dimensionality by depth
    print("\n" + "=" * 68)
    print("3. EFFECTIVE DIMENSIONALITY (participation ratio) by layer")
    print("=" * 68)
    X, layers = load_condition_states(act_dir, "E_BASE", args.point)
    if X is not None:
        pr = []
        print(f"  {'layer':>6} {'eff_dim':>9}  (of {X.shape[0]} states, {X.shape[2]} dims)")
        for i, L in enumerate(layers):
            v = participation_ratio(X[:, i, :])
            pr.append(v)
            print(f"  {int(L):6d} {v:9.1f}")
        out["effective_dim"] = {"layers": [int(x) for x in layers], "pr": pr}

    dest = Path("data") / args.run_id / "geometry.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
