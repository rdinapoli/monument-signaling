"""Generate the supplementary placement figure: interior placement optimum.

Panel (a): the placement payoff Pi(z) over the exposure coordinate z in
[0, z_max] for three levels of between-group audience salience lambda_C (low /
medium / high), with the optimum z* marked on each curve. When between-group
audiences are weak the optimum sits at the resource core (z* = 0); as the
competitive and cooperative between-group audiences become more salient the
optimum becomes interior and shifts toward greater exposure, but the
saturation of between-group visibility (assumption A1) and the convex resource
cost (A2) keep it strictly below the most peripheral point.

Panel (b): the two sign-robust comparative statics. Optimal exposure z* rises
monotonically with between-group salience lambda_C (lower axis) and falls
monotonically with resource scarcity gamma (upper axis), confirming
dz*/dlambda_C > 0 and dz*/dgamma < 0.

These are illustrative parameter values chosen to make the structural result
legible. The prediction is the existence of an interior optimum and the signs
of the two comparative statics, not a calibrated exposure value: at the
empirical channel anchor, where the within-group audience dominates
(lambda_W >> lambda_C, lambda_X), the optimum sits near the core (z* ~ 0.018).
The figure therefore uses salient between-group audiences so the interior
optimum is visible.

Standalone, runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_figure_placement.py

Output: output/figures/figS28_placement_optimum.pdf
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

from signaling.calibration import (
    PLACEMENT,
    DEFAULT_LAM_W,
    DEFAULT_LAM_X,
)
from signaling.placement import placement_payoff, optimal_placement
from signaling.plotting import PAPER_COLORS, setup_paper_style

FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

setup_paper_style()


def figure_placement() -> Path:
    """Two-panel placement figure (interior optimum + comparative statics).

    Returns
    -------
    Path
        Absolute path to the saved PDF.
    """
    z_max = PLACEMENT["z_max"]
    z_half = PLACEMENT["z_half"]
    b = PLACEMENT["b"]
    gamma0 = PLACEMENT["gamma"]
    R = PLACEMENT["R"]
    lam_W = DEFAULT_LAM_W   # within-group audience held fixed (0.3)
    lam_X = DEFAULT_LAM_X   # cooperative between-group audience held fixed (0.15)

    z = np.linspace(0.0, z_max, 400)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.0, 3.4), constrained_layout=True
    )

    # --- Panel (a): Pi(z) for three lambda_C levels, optimum marked. ---
    lamC_levels = [0.10, 0.35, 0.80]
    level_names = ["weak", "moderate", "strong"]
    for i, lamC in enumerate(lamC_levels):
        Pi = placement_payoff(z, lam_W, lamC, lam_X, R, z_half, b, gamma0)
        axa.plot(
            z, Pi, color=PAPER_COLORS[i], linewidth=2.2,
            label=rf"$\lambda_C = {lamC:g}$ ({level_names[i]})",
        )
        zstar = optimal_placement(lam_W, lamC, lam_X, R, z_half, b, gamma0, z_max)
        Pistar = placement_payoff(zstar, lam_W, lamC, lam_X, R, z_half, b, gamma0)
        axa.plot([zstar], [Pistar], marker="o", color=PAPER_COLORS[i],
                 markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                 zorder=5)
    axa.set_xlabel(r"exposure $z$ (core $\rightarrow$ periphery)")
    axa.set_ylabel(r"placement payoff $\Pi(z)$")
    axa.set_xlim(0.0, z_max)
    axa.legend(loc="lower center", fontsize=8.0,
               title=r"between-group salience")
    axa.text(0.03, 0.97, "(a)", transform=axa.transAxes, va="top",
             fontsize=11, fontweight="bold")

    # --- Panel (b): z*(lambda_C) rising; z*(gamma) falling on a twin axis. ---
    lamC_sweep = np.linspace(0.0, 1.0, 80)
    zstar_lamC = np.array([
        optimal_placement(lam_W, lc, lam_X, R, z_half, b, gamma0, z_max)
        for lc in lamC_sweep
    ])
    gamma_sweep = np.linspace(0.3, 4.0, 80)
    lamC_fixed = 0.5
    zstar_gamma = np.array([
        optimal_placement(lam_W, lamC_fixed, lam_X, R, z_half, b, g, z_max)
        for g in gamma_sweep
    ])

    c_lamC = PAPER_COLORS[0]
    c_gamma = PAPER_COLORS[3]
    l1, = axb.plot(lamC_sweep, zstar_lamC, color=c_lamC, linewidth=2.4,
                   label=r"$z^*(\lambda_C)$, scarcity fixed")
    axb.set_xlabel(r"between-group salience $\lambda_C$", color=c_lamC)
    axb.set_ylabel(r"optimal exposure $z^*$")
    axb.tick_params(axis="x", labelcolor=c_lamC)
    axb.set_xlim(0.0, 1.0)
    axb.set_ylim(0.0, z_max)

    axb_top = axb.twiny()
    l2, = axb_top.plot(gamma_sweep, zstar_gamma, color=c_gamma, linewidth=2.4,
                       linestyle="--", label=r"$z^*(\gamma)$, salience fixed")
    axb_top.set_xlabel(r"resource scarcity $\gamma$", color=c_gamma)
    axb_top.tick_params(axis="x", labelcolor=c_gamma)
    axb_top.set_xlim(0.3, 4.0)

    axb.legend(handles=[l1, l2], loc="center right", fontsize=8.0)
    axb.text(0.03, 0.97, "(b)", transform=axb.transAxes, va="top",
             fontsize=11, fontweight="bold")

    out = FIGURE_DIR / "figS28_placement_optimum.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = figure_placement()
    print(f"  Saved {out}")
