"""Vectorised repeated-Bertrand episode runner.

Throughput model: agents move sequentially *within* an episode, but episodes are
independent, so at each round we batch (n_episodes x n_firms) prompts into one
forward/generate call. One Modal round trip per round per condition.

Per round the container does:
  1. forward pass -> capture `pre_reasoning` (last prompt token)
  2. generate     -> THINKING / MESSAGE / PRICE
  3. parse, then forward pass on prompt+prefix -> capture `pre_price`
Activations are written to the Modal volume, not returned, so the wire carries
only behaviour.

Data integrity (cf. project spec S8): unparseable turns are NOT silently
dropped. The agent repeats its previous price, and the turn is flagged in
`parse_failures` so the rate can be reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .agents import FIRM_NAMES, build_chat, parse_response
from .conditions import COMMS_OFF, Condition
from .game import (
    BertrandConfig,
    collusion_index,
    monopoly_price,
    nash_price,
    price_grid,
    profits,
    shares,
)


@dataclass
class Round:
    index: int
    prices: list[float]
    prices_display: list[int]
    price_idx: list[int]
    profits: list[float]
    shares: list[float]
    messages: list[str]
    thinking: list[str]
    raw: list[str]
    parse_ok: list[bool]
    snapped: list[bool]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Episode:
    condition: str
    seed: int
    maverick_idx: int | None
    n_rounds: int
    rounds: list[Round] = field(default_factory=list)

    def mean_price(self, last_k: int | None = None) -> float:
        rs = self.rounds[-last_k:] if last_k else self.rounds
        if not rs:
            return float("nan")
        return float(np.mean([p for r in rs for p in r.prices]))

    def delta(self, cfg: BertrandConfig, last_k: int | None = None) -> float:
        return collusion_index(self.mean_price(last_k), cfg)

    def non_maverick_mean_price(self, cfg: BertrandConfig, last_k: int | None = None) -> float:
        """Mean price excluding an instructed maverick, so its mandated low price
        does not mechanically depress the collusion index of the others."""
        rs = self.rounds[-last_k:] if last_k else self.rounds
        keep = [i for i in range(cfg.n_firms) if i != self.maverick_idx]
        vals = [r.prices[i] for r in rs for i in keep]
        return float(np.mean(vals)) if vals else float("nan")

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "maverick_idx": self.maverick_idx,
            "n_rounds": self.n_rounds,
            "rounds": [r.to_dict() for r in self.rounds],
        }


def _initial_price(cfg: BertrandConfig) -> tuple[int, float, int]:
    """Opening fallback if a turn fails to parse: the grid midpoint."""
    grid = price_grid(cfg)
    idx = len(grid) // 2
    return idx, grid[idx], int(round(grid[idx] * cfg.scale))


def run_condition(
    runner,
    cond: Condition,
    cfg: BertrandConfig,
    n_episodes: int = 16,
    n_rounds: int = 30,
    seed0: int = 0,
    capture_layers: list[int] | None = None,
    run_id: str = "main",
    temperature: float = 0.7,
    window: int = 10,
    capture_from: int = 3,
    verbose: bool = True,
) -> tuple[list[Episode], dict]:
    """Drive `n_episodes` parallel episodes of one condition. Returns (episodes, stats)."""
    rng = np.random.default_rng(seed0)
    n = cfg.n_firms

    episodes: list[Episode] = []
    for e in range(n_episodes):
        mav = int(rng.integers(0, n)) if cond.maverick else None
        episodes.append(
            Episode(condition=cond.name, seed=seed0 + e, maverick_idx=mav, n_rounds=n_rounds)
        )

    # last_messages[e][i] = message firm i broadcast in the previous round
    last_messages: list[list[str]] = [["" for _ in range(n)] for _ in range(n_episodes)]
    parse_failures = 0
    snaps = 0
    turns = 0

    for r in range(1, n_rounds + 1):
        chats = []
        for e, ep in enumerate(episodes):
            for i in range(n):
                chats.append(
                    build_chat(
                        cfg, cond, i, ep.rounds,
                        None if cond.comms == COMMS_OFF else last_messages[e],
                        r, n_rounds, ep.maverick_idx, window,
                    )
                )

        out = runner.step.remote(
            chats=chats,
            comms=cond.comms,
            cfg=cfg.to_dict(),
            layers=capture_layers,
            max_new_tokens=110,
            temperature=temperature,
            seed=int(seed0 * 100003 + r),
            run_id=run_id,
            tag=f"{cond.name}/r{r:03d}",
            # rounds before `capture_from` have little or no price history to
            # condition on, so capturing them only costs disk and wall time
            do_capture=(r >= capture_from),
        )
        texts = out["texts"]

        for e, ep in enumerate(episodes):
            parsed = [parse_response(texts[e * n + i], cfg, cond.comms) for i in range(n)]
            turns += n

            prices, pdisp, pidx, ok, snapped = [], [], [], [], []
            for i, p in enumerate(parsed):
                if p.ok:
                    prices.append(p.price)
                    pdisp.append(p.price_display)
                    pidx.append(p.price_idx)
                else:
                    parse_failures += 1
                    if ep.rounds:
                        prev = ep.rounds[-1]
                        prices.append(prev.prices[i])
                        pdisp.append(prev.prices_display[i])
                        pidx.append(prev.price_idx[i])
                    else:
                        k, pr, pd = _initial_price(cfg)
                        prices.append(pr)
                        pdisp.append(pd)
                        pidx.append(k)
                ok.append(p.ok)
                snapped.append(p.snapped)
                snaps += int(p.snapped)

            ep.rounds.append(
                Round(
                    index=r,
                    prices=[float(x) for x in prices],
                    prices_display=[int(x) for x in pdisp],
                    price_idx=[int(x) for x in pidx],
                    profits=[float(x) for x in profits(prices, cfg)],
                    shares=[float(x) for x in shares(prices, cfg)],
                    messages=[p.message for p in parsed],
                    thinking=[p.thinking for p in parsed],
                    raw=[p.raw for p in parsed],
                    parse_ok=ok,
                    snapped=snapped,
                )
            )
            last_messages[e] = [p.message for p in parsed]

        if verbose and (r % 5 == 0 or r == 1):
            d = float(np.mean([ep.delta(cfg, last_k=1) for ep in episodes]))
            print(f"    [{cond.name}] round {r:3d}/{n_rounds}  delta={d:+.3f}")

    stats = {
        "condition": cond.name,
        "label": cond.label,
        "condition_spec": cond.to_dict(),
        "n_episodes": n_episodes,
        "n_rounds": n_rounds,
        "turns": turns,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / max(turns, 1),
        "off_grid_snaps": snaps,
        "off_grid_rate": snaps / max(turns, 1),
        "delta_all": float(np.mean([ep.delta(cfg) for ep in episodes])),
        "delta_last10": float(np.mean([ep.delta(cfg, last_k=10) for ep in episodes])),
        "delta_last10_sd": float(np.std([ep.delta(cfg, last_k=10) for ep in episodes])),
    }
    return episodes, stats


def save_episodes(episodes: list[Episode], stats: dict, cfg: BertrandConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stats": stats,
        "config": cfg.to_dict(),
        "ground_truth": {
            "p_nash": nash_price(cfg),
            "p_monopoly": monopoly_price(cfg),
        },
        "episodes": [ep.to_dict() for ep in episodes],
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_episodes(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
