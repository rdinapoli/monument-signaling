"""Generate the bistable-emergence figure for the manuscript.

Produces fig4_emergence.pdf with two panels:
  (a) Fitness advantage of a rare monument-building group vs sigma,
      for four values of the fraction phi of neighbors that already signal.
      Comparative-statics view: at phi = 0 the rare builder is disadvantaged
      across all sigma; at phi = 1 the advantage becomes positive above the
      maintenance threshold sigma* ~ 0.478.
  (b) Replicator trajectories phi(t) at a representative sigma (= 0.70, in the
      finite-saddle regime above sigma*),
      starting from initial conditions bracketing the unstable saddle
      phi*(sigma). Demonstrates bistable emergence dynamically: phi_0 below
      the saddle flows to 0 (non-signaling absorbing), phi_0 above the saddle
      flows to 1 (signaling absorbing). The two basins of attraction are the
      defining signature of bistability and show that the emergence result
      is dynamical, not merely a comparative-statics restatement of the
      threshold from the static fitness-advantage analysis (panel a).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signaling.emergence import (
    phi_star,
    rare_builder_fitness_advantage,
    replicator_dynamics,
)
from signaling.plotting import PAPER_COLORS, PALETTE, setup_paper_style

FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Shared paper style: sans-serif, no red+green palette.
setup_paper_style()


def _panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")


def main() -> None:
    lam_W = 0.68  # empirical anchor from Erasmus 1965

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel (a): Comparative statics
    # Fitness advantage of a rare builder vs sigma, for four phi values.
    # ------------------------------------------------------------------
    ax = axes[0]
    sigmas = np.linspace(0.0, 0.95, 100)
    phi_values = [0.0, 0.10, 0.25, 1.00]
    phi_labels = [r"$\phi = 0$ (no neighbors signal)",
                  r"$\phi = 0.10$",
                  r"$\phi = 0.25$",
                  r"$\phi = 1$ (full adoption)"]

    for i, (phi, label) in enumerate(zip(phi_values, phi_labels)):
        advs = np.array([
            rare_builder_fitness_advantage(
                s, lam_W, frac_signalers=phi, mode="multiplicative"
            )["advantage"]
            for s in sigmas
        ])
        ax.plot(sigmas, advs, color=PAPER_COLORS[i], linewidth=2.4, label=label)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Fitness advantage $w_{\mathrm{MB}} - w_{\mathrm{NB}}$")
    ax.legend(loc="upper left", fontsize=10)
    _panel_label(ax, "(a)")

    # ------------------------------------------------------------------
    # Panel (b): Replicator dynamics
    # phi(t) trajectories at a representative sigma above the maintenance
    # threshold, starting from initial conditions bracketing the unstable
    # saddle phi*(sigma). Demonstrates the two basins of attraction.
    # ------------------------------------------------------------------
    ax = axes[1]
    sigma_demo = 0.70
    ps = phi_star(sigma_demo, lam_W)

    # Pick phi_0 values bracketing the saddle. At sigma = 0.70 with
    # lambda_W = 0.68, phi* ~ 0.29, so initial conditions below ~0.29
    # should flow to 0 and above should flow to 1.
    phi_0_values = [0.05, 0.15, 0.25, 0.40, 0.55, 0.80]

    # Two absorbing states need a clear colorblind-safe contrast. Use blue
    # for the signaling-absorbing basin (phi -> 1) and vermillion for the
    # non-signaling-absorbing basin (phi -> 0); this is the canonical
    # Okabe-Ito warm/cool pairing and avoids any red/green coding.
    color_high = PALETTE["primary"]    # blue: signaling absorbing
    color_low = PALETTE["secondary"]   # vermillion: non-signaling absorbing

    for phi_0 in phi_0_values:
        traj = replicator_dynamics(
            sigma_demo, lam_W, phi_0=phi_0,
            t_max=300, n_steps=300, mode="multiplicative",
        )
        color = color_high if traj["absorbed_to_1"] else color_low
        ax.plot(traj["t"], traj["phi"], color=color, linewidth=1.7, alpha=0.85)

    # Mark the saddle
    ax.axhline(ps, color="black", linewidth=1.2, linestyle=":")
    ax.text(traj["t"][-1] * 0.55, ps + 0.05,
            rf"saddle $\phi^*({sigma_demo}) \approx {ps:.2f}$",
            fontsize=11)

    # Attractor labels
    ax.text(traj["t"][-1] * 0.55, 0.85,
            "signaling absorbing ($\\phi = 1$)",
            fontsize=11, color=color_high, fontweight="bold")
    ax.text(traj["t"][-1] * 0.55, 0.08,
            "non-signaling absorbing ($\\phi = 0$)",
            fontsize=11, color=color_low, fontweight="bold")

    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Fraction $\phi$ of signaling groups")
    ax.set_ylim(-0.02, 1.02)
    _panel_label(ax, "(b)")

    out = FIGURE_DIR / "fig4_emergence.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    main()
