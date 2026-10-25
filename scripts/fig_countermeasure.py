"""The paper's key figure: what each intervention does, all at 72B, all on the
converged statistic (mean Delta over the final 10 rounds)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"figure.dpi": 170, "font.size": 9, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": .7, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

rows = [
    ("free-text channel",                 0.196, C3,  "reference"),
    ("4-symbol channel",                  0.790, C2,  "intervention"),
    ("  + ablate direction (L64)",        0.857, C2,  "intervention"),
    ("  + negative steer  a=-0.20",       0.426, C1,  "intervention"),
    ("  + negative steer  a=-0.40",      -0.096, C1,  "intervention"),
    ("one maverick agent",               -0.027, C1,  "intervention"),
]
fig, ax = plt.subplots(figsize=(6.6, 3.1))
y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], height=.62, color=[r[2] for r in rows])
for yi, r in zip(y, rows):
    pos = r[1] >= 0
    ax.text(r[1] + (.02 if pos else -.02), yi, f"{r[1]:+.3f}", va="center",
            ha="left" if pos else "right", fontsize=8.5, color=INK)
ax.axvline(0, color=INK2, lw=1.2)
ax.axvline(0.790, color=INK2, lw=1.1, ls=(0, (4, 3)))
ax.text(0.785, -0.62, "cartel baseline ", fontsize=7.5, color=INK2,
        va="center", ha="right")
ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("collusion index  $\Delta$   (mean over final 10 rounds)")
ax.set_xlim(-0.22, 1.02)
ax.grid(axis="y", visible=False)
ax.set_title("Restricting the channel made collusion worse; only the right operation removes it",
             fontsize=9.5, fontweight="bold")
fig.tight_layout()
out = Path("figures/f11_countermeasures.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
