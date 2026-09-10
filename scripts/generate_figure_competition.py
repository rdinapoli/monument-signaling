"""Generate the supplementary competitive-investment figure (M11 + M2).

Panel (a): the parity hump (M11). Total dyadic monument labor X* against capacity
asymmetry delta_q, holding the mean capacity fixed. X* = V q_i q_j/(q_i+q_j) =
(V qbar/2)(1 - delta_q^2) is an inverted-U, maximized at parity (delta_q = 0) and
falling to zero as one group dwarfs the other. Aggregated over adjacent polities,
this is the regional prediction: total monument investment is highest where
neighboring capacities are most equal (low regional inequality), lower under a
single hegemon and under dispersal.

Panel (b): the directed-scale peak (M2). Group i's labor directed at neighbor j,
x_i* = V q_i^2 q_j/(q_i+q_j)^2, against the rival's capacity q_j at fixed own
capacity q_i. The directed effort peaks when the rival is matched (q_j = q_i),
not when the rival is strongest, ruling out monotone power/wealth display.

Both panels are computed from the closed forms in signaling.competition (the
dyadic Tullock contest-success-function equilibrium), which are themselves
verified symbolically and numerically in tests/test_competition.py.

Standalone, runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_figure_competition.py

Output: output/figures/figS29_competition_investment.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the signaling package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signaling.calibration import COMPETITION
from signaling.competition import dyadic_total, dyadic_effort
from signaling.plotting import PAPER_COLORS, setup_paper_style

FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

setup_paper_style()


def figure_competition() -> Path:
    """Two-panel competition figure (parity hump + directed peak).

    Returns
    -------
    Path
        Absolute path to the saved PDF.
    """
    V = COMPETITION["V"]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.0, 3.4), constrained_layout=True
    )

    # --- Panel (a): parity hump, total dyadic labor vs asymmetry. ---
    qbar = 5.0
    deltas = np.linspace(-0.95, 0.95, 241)
    X = np.array([dyadic_total(qbar * (1 + d), qbar * (1 - d), V) for d in deltas])
    axa.plot(deltas, X, color=PAPER_COLORS[0], linewidth=2.4)
    axa.axvline(0.0, color="gray", linestyle=":", linewidth=1.1)
    Xpeak = dyadic_total(qbar, qbar, V)
    axa.plot([0.0], [Xpeak], marker="o", color=PAPER_COLORS[0], markersize=6,
             markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    axa.annotate("peak at parity", xy=(0.0, Xpeak), xytext=(0.30, Xpeak * 0.72),
                 fontsize=8.5, ha="left",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
    axa.set_xlabel(r"capacity asymmetry $\delta_q$")
    axa.set_ylabel(r"total dyadic investment $X^*$")
    axa.set_xlim(-1.0, 1.0)
    axa.set_ylim(bottom=0.0)
    axa.text(0.03, 0.97, "(a)", transform=axa.transAxes, va="top",
             fontsize=11, fontweight="bold")

    # --- Panel (b): directed-scale peak vs rival capacity. ---
    q_i = 5.0
    q_js = np.linspace(0.2, 20.0, 300)
    xi = np.array([dyadic_effort(q_i, qj, V) for qj in q_js])
    axb.plot(q_js, xi, color=PAPER_COLORS[1], linewidth=2.4)
    axb.axvline(q_i, color="gray", linestyle=":", linewidth=1.1)
    xpeak = dyadic_effort(q_i, q_i, V)
    axb.plot([q_i], [xpeak], marker="o", color=PAPER_COLORS[1], markersize=6,
             markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    axb.annotate("peak at matched rival", xy=(q_i, xpeak),
                 xytext=(q_i + 1.5, xpeak * 0.62), fontsize=8.5, ha="left",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
    axb.set_xlabel(r"rival capacity $q_j$")
    axb.set_ylabel(r"directed investment $x_i^*$")
    axb.set_xlim(0.0, 20.0)
    axb.set_ylim(bottom=0.0)
    axb.text(0.03, 0.97, "(b)", transform=axb.transAxes, va="top",
             fontsize=11, fontweight="bold")

    out = FIGURE_DIR / "figS29_competition_investment.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = figure_competition()
    print(f"  Saved {out}")
