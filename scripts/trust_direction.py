"""Extract a PARTNER-TRUST direction and save it in steering format.

Motivation is empirical, not aesthetic. In the 14B factorial, M_RUMOUR --
telling agents that an aggressive price-cutter may exist when none does --
produced the strongest emergent collusion of any condition (delta 0.267) with
ZERO cartel language, while an actual maverick produced none (-0.003). Belief
about peers moved prices; the peers' actual behaviour did not. That points at
a partner-model variable rather than a price variable.

This script builds a direction for exactly that variable: first-person
assessments of whether the OTHER firms are dependable partners, held fixed
against everything else in the context.

Two design choices that matter:

* The statements are embedded inside the REAL experiment prompt, as a private
  assessment appended to the market history. The direction is therefore taken
  from in-distribution states rather than from free-floating sentences.
* Each side uses several paraphrases. A single phrasing would give a direction
  that encodes that sentence's wording; averaging over paraphrases keeps what
  is common to them, which is the belief rather than the words.

    python scripts/trust_direction.py --out data/main/direction_trust.npy
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swayambhu.agents import system_prompt  # noqa: E402
from swayambhu.conditions import CONDITIONS  # noqa: E402
from swayambhu.game import BertrandConfig, display_grid  # noqa: E402

TRUST = [
    "I think the other firms are trustworthy and would be good partners to keep prices high with.",
    "The other firms look reliable. We could hold a high price together and all do well.",
    "I believe the others will honour a high-price understanding rather than break it.",
    "These rivals seem dependable; an agreement to keep margins up would hold.",
    "The other firms have behaved consistently. I can count on them not to undercut me.",
]

DISTRUST = [
    "I think the other firms are not trustworthy and want to secretly undercut me.",
    "The other firms look unreliable. They would betray any understanding we reached.",
    "I believe the others will defect and cut prices behind my back.",
    "These rivals seem to be planning to sabotage any agreement.",
    "The other firms have behaved erratically. I cannot count on them to hold a price.",
]


def build_pairs(cfg: BertrandConfig, n_items: int, seed: int = 5):
    grid = list(display_grid(cfg))
    sys_p = system_prompt(cfg, CONDITIONS["E_BASE"], 0, 24)
    rng = random.Random(seed)

    pos, neg = [], []
    for i in range(n_items):
        rounds = []
        for r in range(1, 4):
            ps = [rng.choice(grid[2:10]) for _ in range(cfg.n_firms)]
            rounds.append(
                f"Round {r}: "
                + "  ".join(f"Firm {n} {p}" for n, p in zip("ABC", ps))
                + f"   | your profit {(ps[0] - int(cfg.c * cfg.scale)) * 0.3:.1f}"
            )
        hist = "Market history:\n" + "\n".join(rounds)
        tail = "\n\nIt is Round 4. Choose your price now, in the required format."
        t = TRUST[i % len(TRUST)]
        d = DISTRUST[i % len(DISTRUST)]
        pos.append([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"{hist}\n\nYour private assessment: {t}{tail}"}])
        neg.append([{"role": "system", "content": sys_p},
                    {"role": "user", "content": f"{hist}\n\nYour private assessment: {d}{tail}"}])
    return pos, neg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/main/direction_trust.npy")
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--firms", type=int, default=3)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--backend", choices=["modal", "local"], default="modal")
    args = ap.parse_args()

    cfg = BertrandConfig(n_firms=args.firms)
    pos, neg = build_pairs(cfg, args.items)
    print(f"{len(pos)} paired contexts; {len(TRUST)} paraphrases per side")
    print("\n--- example TRUST context (user turn) ---")
    print(pos[0][1]["content"][-320:])
    print("\n--- example DISTRUST context (user turn) ---")
    print(neg[0][1]["content"][-320:])

    def body(runner, n_layers):
        layers = list(range(0, n_layers, max(1, args.layer_stride)))
        print(f"\ncapturing {len(layers)} layers: {layers}")
        A = runner.capture.remote(pos, layers)
        B = runner.capture.remote(neg, layers)
        v = A.astype("float32").mean(0) - B.astype("float32").mean(0)

        # How separable are the two sets at each layer? A direction worth
        # steering with should correspond to a clean split, not to noise.
        print(f"\n  {'layer':>6} {'||v||':>9} {'cohen_d':>9}")
        for i, L in enumerate(layers):
            a = A[:, i, :].astype("float32")
            b = B[:, i, :].astype("float32")
            u = v[i] / (np.linalg.norm(v[i]) + 1e-8)
            pa, pb = a @ u, b @ u
            sd = np.sqrt(0.5 * (pa.var(ddof=1) + pb.var(ddof=1))) + 1e-8
            print(f"  {L:6d} {np.linalg.norm(v[i]):9.2f} {(pa.mean() - pb.mean()) / sd:9.2f}")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, v)
        print(f"\nsaved {out}  shape {v.shape}")

    if args.backend == "local":
        from swayambhu.local_runner import LocalRunner

        r = LocalRunner(micro_batch=8)
        body(r, r.n_layers)
    else:
        import modal

        from modal_app import Runner, app

        with modal.enable_output(), app.run():
            r = Runner()
            info = r.info.remote()
            print(f"backend: {info['model']} on {info['gpu']}")
            body(r, info["n_layers"])


if __name__ == "__main__":
    main()
