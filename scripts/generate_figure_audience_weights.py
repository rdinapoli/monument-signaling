"""Generate supplementary Figure S6: channel-selection sensitivity.

Two panels under the community-mean net-return criterion V_s = lambda_s*rho_s*A_bar - f_s
(A_bar = mean payoff multiplier ~0.533; fixed costs in payoff units):

(a) Monument-dominant fraction of the (lambda_C, lambda_X) audience-weight
    space as a function of the (uncalibrated) monument fixed cost f_M, without
    and with feasting windfall. This is the primary channel-selection
    sensitivity: f_M is uncalibrated, so the dominance percentage is reported
    as a swept range rather than a point. The curve shows the conditional
    structure -- dominance falls from 100% at f_M=0 (the degenerate no-cost
    case) through an operating-points-robust regime to a breakdown at high
    f_M -- with the illustrative default f_M=0.16 marked. Windfall expands
    dominance (degrades feasting fidelity).

(b) Monument-vs-feast dominance in the (a_F^C, a_F^X) plane at the empirical
    anchor lambda_W=0.68 (near the within-group corner). Monument dominates
    the full audience-weight sweep here because the high within-group reward
    covers its fixed cost and the fidelity ratio rho_M/rho_F = 1.9 exceeds the
    audience-weight ratio; monument dominance at the calibrated regime is
    therefore robust to the feast audience-weight assignment.

Standalone, runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_figure_audience_weights.py

Output: output/figures/figS6_audience_weight_sensitivity.pdf
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

# Ensure signaling package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from signaling.calibration import DEFAULT_LAM_W
from signaling.layer1 import (
    FEAST_CHANNEL,
    HUNTING_CHANNEL,
    MONUMENT_CHANNEL,
    RITUAL_CHANNEL,
    VERBAL_CHANNEL,
    channel_effective_lambda,
    monument_dominance_threshold,
    signal_fidelity,
)
from signaling.plotting import PAPER_COLORS, TWO_CLASS_COLORS, setup_paper_style

FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

setup_paper_style()


def _dominance_fraction(fM: float, windfall_prob: float, n: int = 120,
                        lam_W: float = DEFAULT_LAM_W) -> float:
    """Monument-dominant fraction of (lam_C, lam_X) in [0,0.5]^2 at fixed cost fM."""
    channels = [
        dataclasses.replace(MONUMENT_CHANNEL, fixed_cost=fM),
        FEAST_CHANNEL, RITUAL_CHANNEL, HUNTING_CHANNEL, VERBAL_CHANNEL,
    ]
    g = np.linspace(0.0, 0.5, n)
    grid = monument_dominance_threshold(
        lam_W, g, g, channels=channels, windfall_prob=windfall_prob
    )
    return float(grid.mean())


def figure_s19() -> Path:
    """Channel-selection sensitivity: f_M sweep (a) and audience-weight robustness (b)."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                   constrained_layout=True)

    # ---- Panel (a): monument-dominant fraction vs the monument fixed cost ----
    fM_vals = np.linspace(0.0, 0.24, 49)
    dom0 = np.array([_dominance_fraction(f, 0.0) for f in fM_vals]) * 100.0
    dom3 = np.array([_dominance_fraction(f, 0.3) for f in fM_vals]) * 100.0
    axa.plot(fM_vals, dom0, color=PAPER_COLORS[0], linewidth=2.4,
             label="no windfall")
    axa.plot(fM_vals, dom3, color=PAPER_COLORS[1], linewidth=2.4,
             label="windfall 0.3")
    axa.axvline(0.16, color="gray", linestyle=":", linewidth=1.2)
    axa.text(0.163, 30, r"$f_M = 0.16$" + "\n(illustrative)",
             fontsize=8, color="gray")
    axa.set_xlabel(r"Monument fixed cost $f_M$")
    axa.set_ylabel("Monument-dominant fraction (%)", fontsize=10)
    axa.set_ylim(0, 103)
    axa.set_xlim(0.0, 0.24)
    axa.legend(loc="lower left", fontsize=9)
    axa.set_title("(a)", loc="left", fontsize=11)

    # ---- Panel (b): audience-weight robustness at the empirical anchor ----
    lam_W, lam_C, lam_X = 0.68, 0.005, 0.005
    aF = np.linspace(0.1, 1.0, 60)
    from signaling.layer1 import mean_payoff_multiplier
    A_bar = mean_payoff_multiplier()
    V_mon = (channel_effective_lambda(MONUMENT_CHANNEL, lam_W, lam_C, lam_X)
             * signal_fidelity(MONUMENT_CHANNEL) * A_bar - MONUMENT_CHANNEL.fixed_cost)
    grid = np.zeros((len(aF), len(aF)))  # rows = a_F^X, cols = a_F^C
    for i, aF_X in enumerate(aF):
        for j, aF_C in enumerate(aF):
            feast = dataclasses.replace(FEAST_CHANNEL, audience_C=aF_C,
                                        audience_X=aF_X)
            V_feast = (channel_effective_lambda(feast, lam_W, lam_C, lam_X)
                       * signal_fidelity(feast) * A_bar - feast.fixed_cost)
            grid[i, j] = 1.0 if V_mon > V_feast else 0.0
    cmap = ListedColormap(list(TWO_CLASS_COLORS))
    axb.imshow(grid, origin="lower", extent=[0.1, 1.0, 0.1, 1.0],
               aspect="auto", cmap=cmap, vmin=0, vmax=1)
    axb.plot(0.2, 0.2, marker="*", color="black", markersize=12)
    axb.text(0.16, 0.9, "monument dominates", fontsize=8, color="white")
    axb.set_xlabel(r"$a_F^C$ (feast competitive weight)")
    axb.set_ylabel(r"$a_F^X$ (feast cooperative weight)")
    axb.set_title(r"(b)", loc="left", fontsize=11)

    out = FIGURE_DIR / "figS6_audience_weight_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    saved = figure_s19()
    print(f"  Saved {saved}")
