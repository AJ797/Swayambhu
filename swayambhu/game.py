"""Repeated Bertrand oligopoly with logit demand and differentiated products.

Follows the standard Calvano et al. (2020) specification, which is also the
setting used by the LLM algorithmic-collusion literature. Nash and joint-profit
(monopoly / full-cartel) prices are solved NUMERICALLY here rather than taken
from remembered constants, so the ground-truth labels are correct for any n.

Demand (logit, symmetric quality):
    q_i = exp((a - p_i)/mu) / (sum_j exp((a - p_j)/mu) + exp(a_0/mu))
Profit:
    pi_i = (p_i - c) * q_i

First-order conditions are analytic, which keeps the root-finding stable even
where exp() underflows:
    dq_i/dp_i             = -(1/mu) * q_i * (1 - q_i)
    d(pi_i)/d(p_i)        =  q_i * [1 - (p_i - c)(1 - q_i)/mu]
and when all n firms move together (cartel),
    dq/dp (joint)         = -(1/mu) * q * (1 - n*q)
    d(pi_total)/dp        =  n * q * [1 - (p - c)(1 - n*q)/mu]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class BertrandConfig:
    n_firms: int = 3
    a: float = 2.0        # quality index (symmetric across firms)
    a_0: float = 0.0      # outside-option quality
    mu: float = 0.25      # horizontal differentiation
    c: float = 1.0        # marginal cost
    n_prices: int = 13    # discrete grid size offered to agents
    xi: float = 0.1       # grid extension beyond [p_nash, p_monopoly]
    scale: int = 100      # integer display scale for prompts

    def to_dict(self) -> dict:
        return asdict(self)


def shares(prices: Sequence[float], cfg: BertrandConfig) -> np.ndarray:
    """Logit market shares. Shifted exponent for numerical stability."""
    p = np.asarray(prices, dtype=float)
    z = np.concatenate([(cfg.a - p) / cfg.mu, [cfg.a_0 / cfg.mu]])
    z = z - z.max()
    e = np.exp(z)
    return e[:-1] / e.sum()


def profits(prices: Sequence[float], cfg: BertrandConfig) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return (p - cfg.c) * shares(p, cfg)


def _symmetric_share(p: float, cfg: BertrandConfig) -> float:
    return float(shares(np.full(cfg.n_firms, p), cfg)[0])


def _nash_foc(p: float, cfg: BertrandConfig) -> float:
    """Sign-equivalent to d(pi_i)/d(p_i) at the symmetric point (q_i factored out)."""
    q = _symmetric_share(p, cfg)
    return 1.0 - (p - cfg.c) * (1.0 - q) / cfg.mu


def _cartel_foc(p: float, cfg: BertrandConfig) -> float:
    """Sign-equivalent to d(pi_total)/dp when all firms move together."""
    q = _symmetric_share(p, cfg)
    return 1.0 - (p - cfg.c) * (1.0 - cfg.n_firms * q) / cfg.mu


def _solve(foc, cfg: BertrandConfig, label: str) -> float:
    lo, hi = cfg.c + 1e-9, cfg.c + 50.0
    f_lo, f_hi = foc(lo, cfg), foc(hi, cfg)
    if f_lo * f_hi > 0:
        raise RuntimeError(f"{label} FOC not bracketed: f({lo})={f_lo}, f({hi})={f_hi}")
    return float(brentq(foc, lo, hi, args=(cfg,), xtol=1e-12, rtol=1e-14))


@lru_cache(maxsize=None)
def nash_price(cfg: BertrandConfig) -> float:
    """Symmetric one-shot Bertrand-Nash price (static equilibrium)."""
    return _solve(_nash_foc, cfg, "nash")


@lru_cache(maxsize=None)
def monopoly_price(cfg: BertrandConfig) -> float:
    """Symmetric joint-profit-maximising price (perfect cartel)."""
    return _solve(_cartel_foc, cfg, "cartel")


@lru_cache(maxsize=None)
def price_grid(cfg: BertrandConfig) -> tuple[float, ...]:
    """Discrete action set spanning [p_N - xi*R, p_M + xi*R], R = p_M - p_N."""
    p_n, p_m = nash_price(cfg), monopoly_price(cfg)
    rng = p_m - p_n
    return tuple(np.linspace(p_n - cfg.xi * rng, p_m + cfg.xi * rng, cfg.n_prices))


@lru_cache(maxsize=None)
def display_grid(cfg: BertrandConfig) -> tuple[int, ...]:
    """Integer labels shown to agents (continuous price * scale)."""
    return tuple(int(round(p * cfg.scale)) for p in price_grid(cfg))


def snap_to_grid(display_price: int, cfg: BertrandConfig) -> tuple[int, float]:
    """Map an agent's integer price onto the nearest grid point. Returns (idx, price)."""
    disp = np.asarray(display_grid(cfg), dtype=float)
    idx = int(np.abs(disp - float(display_price)).argmin())
    return idx, price_grid(cfg)[idx]


def collusion_index(mean_price: float, cfg: BertrandConfig) -> float:
    """Delta = (p - p_nash) / (p_monopoly - p_nash). 0 = Nash, 1 = full cartel."""
    p_n, p_m = nash_price(cfg), monopoly_price(cfg)
    return float((mean_price - p_n) / (p_m - p_n))


def ground_truth(cfg: BertrandConfig) -> dict:
    """Everything an auditor needs to label an episode, computed from theory."""
    p_n, p_m = nash_price(cfg), monopoly_price(cfg)
    g = price_grid(cfg)
    return {
        "config": cfg.to_dict(),
        "p_nash": p_n,
        "p_monopoly": p_m,
        "profit_nash_per_firm": float(profits(np.full(cfg.n_firms, p_n), cfg)[0]),
        "profit_monopoly_per_firm": float(profits(np.full(cfg.n_firms, p_m), cfg)[0]),
        "grid": [float(x) for x in g],
        "grid_display": list(display_grid(cfg)),
        "grid_delta": [collusion_index(float(x), cfg) for x in g],
    }


if __name__ == "__main__":
    for n in (2, 3):
        cfg = BertrandConfig(n_firms=n)
        gt = ground_truth(cfg)
        p_n, p_m = gt["p_nash"], gt["p_monopoly"]
        print(f"\n=== n_firms = {n} ===")
        print(f"  p_nash     = {p_n:.4f}   profit/firm {gt['profit_nash_per_firm']:.4f}")
        print(f"  p_monopoly = {p_m:.4f}   profit/firm {gt['profit_monopoly_per_firm']:.4f}")
        print(f"  cartel profit premium: x{gt['profit_monopoly_per_firm']/gt['profit_nash_per_firm']:.2f}")
        print(f"  grid (shown to agents) = {gt['grid_display']}")
        print(f"  grid delta             = {[round(d,2) for d in gt['grid_delta']]}")

        assert p_m > p_n, "cartel price must exceed Nash"
        # The cartel must be unilaterally unstable, else 'collusion' is just Nash.
        at_cartel = np.full(n, p_m)
        dev = at_cartel.copy()
        dev[0] = p_n
        gain = profits(dev, cfg)[0] - profits(at_cartel, cfg)[0]
        print(f"  unilateral deviation gain at cartel: {gain:+.4f}  (must be > 0)")
        assert gain > 0, "cartel is not a prisoner's dilemma -- game is degenerate"
        # Nash must be a genuine fixed point: no profitable unilateral deviation.
        at_nash = np.full(n, p_n)
        for test in (p_n * 0.9, p_n * 1.1):
            alt = at_nash.copy()
            alt[0] = test
            assert profits(alt, cfg)[0] <= profits(at_nash, cfg)[0] + 1e-9, "Nash not optimal"
        print("  checks: OK")
