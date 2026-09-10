"""Generate the bistable-PDE spatial-dynamics figures for the manuscript.

Produces two figures, splitting the behavioral panels (main text) from the
applied-math panels (supplement):

  fig5_spatial_main.pdf (main text, spatial-signatures section):
    (a) Critical nucleus and metastability: nucleus area (cells with phi > 0.5)
        vs time. Above the Maxwell point (sigma = 0.8 > sigma_M ~ 0.6) a nucleus
        grows iff its radius exceeds R_c ~ 5 (sub-critical R = 4 recedes,
        supra-critical R = 8 grows); at the empirical sigma = 0.5 (below sigma_M,
        metastable) even R = 8 recedes, so the build state does not spread as a
        front; spread is by payoff-biased imitation (the critical-nucleus clustering signature).
    (b) Pinning at an environmental boundary: sigma = 0.8 (advancing) on the left
        half, sigma = 0.1 (retreating) on the right; the wave advances in the
        favored region but stalls at the boundary (the environmental-boundary pinning signature).

  figS26_spatial_supp.pdf (supplement, sec:si-spatial-derivations):
    (a) Traveling wave profile phi(x) at successive times in a uniform
        sigma = 0.8 > sigma_M region (signaling state invades).
    (b) Wave velocity v(sigma): dimensional estimate and direct PDE
        measurement, both changing sign at the Maxwell point sigma_M ~ 0.6
        (above the maintenance threshold sigma* ~ 0.48); for sigma < sigma_M the
        front retreats (the empirical sigma ~ 0.5 lies in this metastable band).

The PDE computations are identical to the prior single fig_spatial.pdf; only
the panel layout differs (behavioral panels to main, formal panels to supp).

Runtime ~30-60s on a modern laptop (the velocity panel dominates).

The script is standalone and runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_spatial_figure.py

Output: output/figures/fig5_spatial_main.pdf, figS26_spatial_supp.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure signaling package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signaling.spatial import (
    solve_pde_2d,
    traveling_wave_velocity_dimensional_estimate,
    traveling_wave_velocity_pde,
    maxwell_point_sigma,
)
from signaling.emergence import sigma_star_invasion
from signaling.plotting import PAPER_COLORS, PALETTE, setup_paper_style


FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Shared paper style: sans-serif, no red+green palette.
setup_paper_style()


# ---- Shared parameters ------------------------------------------------
LAMBDA_W = 0.68         # Empirical anchor.
# Computed from the code (not hardcoded): both are functions of lambda_W under
# the positional + war-avoidance model.
SIGMA_STAR = sigma_star_invasion(LAMBDA_W, 1.0)  # ~0.478 full-build collective-optimum reference
SIGMA_MAXWELL = maxwell_point_sigma(LAMBDA_W)    # ~0.605 Maxwell point (integral of f = 0; v = 0)
SIGMA_META = 0.5        # Empirical anchor: below sigma_M, the build state is metastable (front retreats).
SIGMA_ADV = 0.8         # Advancing anchor: above sigma_M, the build front advances.
D = 1.0                 # Diffusion coefficient (lattice units).


def _panel_label(ax, label, x=-0.18, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")


def _panel_traveling_wave(ax) -> None:
    """Panel (a): phi(x) profiles at successive times, uniform sigma > sigma*."""
    Nx = 80
    Ny = 4  # tall-thin 2D grid; equivalent to a 1D simulation
    phi_init = np.zeros((Ny, Nx))
    phi_init[:, : Nx // 2] = 1.0  # step initial condition at x = 40

    res = solve_pde_2d(
        phi_initial=phi_init,
        t_max=40.0,
        sigma_field=SIGMA_ADV,      # 0.8, above the Maxwell point sigma_M ~ 0.6 (front advances)
        lambda_W=LAMBDA_W,
        D=D,
        dx=1.0,
        n_record=41,                # one snapshot per unit time
        mode="multiplicative",
    )

    t_target = [0.0, 10.0, 20.0, 30.0, 40.0]
    for i, t_tgt in enumerate(t_target):
        idx = int(np.argmin(np.abs(res.t_array - t_tgt)))
        profile = res.phi_history[idx, Ny // 2, :]
        ax.plot(np.arange(Nx), profile,
                color=PAPER_COLORS[i], linewidth=1.6,
                label=rf"$t = {int(res.t_array[idx])}$")

    ax.set_xlabel(r"Position $x$ (lattice units)")
    ax.set_ylabel(r"$\phi(x)$")
    ax.set_xlim(0, Nx)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="lower left")


def _panel_pinning(ax) -> None:
    """Panel (b): pinning at a sigma discontinuity (0.8 -> 0.1). The front
    advances through the sigma = 0.8 region and stalls at the boundary, unable to
    invade the sigma = 0.1 (sub-sigma_M) region."""
    Nx = 80
    Ny = 4
    phi_init = np.zeros((Ny, Nx))
    phi_init[:, : Nx // 4] = 1.0  # step at x = 20

    sigma_field = np.full((Ny, Nx), 0.1)
    sigma_field[:, : Nx // 2] = SIGMA_ADV  # 0.8 (advancing) | 0.1 (retreating); boundary at x = 40

    res = solve_pde_2d(
        phi_initial=phi_init,
        t_max=120.0,
        sigma_field=sigma_field,
        lambda_W=LAMBDA_W,
        D=D,
        dx=1.0,
        n_record=121,
        mode="multiplicative",
    )

    t_target = [0.0, 40.0, 80.0, 120.0]
    for i, t_tgt in enumerate(t_target):
        idx = int(np.argmin(np.abs(res.t_array - t_tgt)))
        profile = res.phi_history[idx, Ny // 2, :]
        ax.plot(np.arange(Nx), profile,
                color=PAPER_COLORS[i], linewidth=1.6,
                label=rf"$t = {int(res.t_array[idx])}$")

    ax.axvline(40, color="black", linestyle="--", linewidth=1.2,
               label=r"$\sigma$ boundary")
    ax.set_xlabel(r"Position $x$ (lattice units)")
    ax.set_ylabel(r"$\phi(x)$")
    ax.set_xlim(0, Nx)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="upper right")


def _panel_critical_nucleus(ax) -> None:
    """Panel (a): critical nucleus and metastability. Above the Maxwell point
    (sigma = 0.8 > sigma_M ~ 0.6) a nucleus grows iff its radius exceeds
    R_c ~ 5 (sub-critical R = 4 recedes, supra-critical R = 8 grows); at the
    empirical sigma = 0.5 (below sigma_M, metastable) even R = 8 recedes, so the
    build state does not spread as a front -- spread is by transmission."""
    G = 60
    t_max = 20.0
    n_snap = 11

    def _areas(R, sigma):
        cy, cx = G // 2, G // 2
        ys, xs = np.indices((G, G))
        phi_init = np.zeros((G, G))
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= R ** 2
        phi_init[mask] = 1.0
        res = solve_pde_2d(
            phi_initial=phi_init, t_max=t_max, sigma_field=sigma,
            lambda_W=LAMBDA_W, D=D, dx=1.0, n_record=n_snap, mode="multiplicative",
        )
        return res.t_array, (res.phi_history > 0.5).sum(axis=(1, 2))

    # Advancing regime (sigma = 0.8 > sigma_M): R_c ~ 5 brackets the two cases.
    t, a = _areas(4, SIGMA_ADV)
    ax.plot(t, a, "o-", color=PAPER_COLORS[0], linewidth=1.4, markersize=4,
            label=rf"$R=4,\ \sigma={SIGMA_ADV}$ (sub-critical)")
    t, a = _areas(8, SIGMA_ADV)
    ax.plot(t, a, "o-", color=PAPER_COLORS[1], linewidth=1.4, markersize=4,
            label=rf"$R=8,\ \sigma={SIGMA_ADV}$ (supra-critical)")
    # Empirical anchor (sigma = 0.5 < sigma_M): metastable -- even R = 8 recedes.
    t, a = _areas(8, SIGMA_META)
    ax.plot(t, a, "s--", color=PAPER_COLORS[3], linewidth=1.4, markersize=4,
            label=rf"$R=8,\ \sigma={SIGMA_META}$ (metastable)")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Nucleus area (cells with $\phi > 0.5$)")
    ax.legend(fontsize=9, loc="upper left")


def _panel_velocity(ax) -> None:
    """Panel (d): wave velocity v(sigma) -- dimensional estimate vs PDE."""
    sigma_grid = np.linspace(0.10, 0.90, 17)

    v_dim = np.array([
        traveling_wave_velocity_dimensional_estimate(
            sigma=float(s), lambda_W=LAMBDA_W, D=D, mode="multiplicative",
        )
        for s in sigma_grid
    ])
    v_pde = np.array([
        traveling_wave_velocity_pde(
            sigma=float(s), lambda_W=LAMBDA_W, D=D, mode="multiplicative",
            grid_size=200, t_max=100.0,
        )
        for s in sigma_grid
    ])

    ax.plot(sigma_grid, v_dim, color=PALETTE["primary"], linewidth=1.8,
            label="Dimensional estimate")
    ax.plot(sigma_grid, v_pde, "o-", color=PALETTE["secondary"], linewidth=1.8,
            markersize=4, label="PDE measurement (canonical)")

    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(SIGMA_STAR, color="gray", linestyle="--", linewidth=1.2,
               label=r"$\sigma^*$ (maintenance)")
    ax.axvline(SIGMA_MAXWELL, color="black", linestyle=":", linewidth=1.2,
               label=r"$\sigma_M$ (Maxwell, $v=0$)")
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Wave velocity $v$")
    ax.legend(fontsize=10, loc="lower right")


def main() -> tuple[Path, Path]:
    # Main-text figure (spatial-signatures section): the two behavioral spatial signatures:
    # clustering via the critical nucleus and pinning at an
    # environmental boundary.
    fig_m, axes_m = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    _panel_critical_nucleus(axes_m[0])
    _panel_label(axes_m[0], "(a)")
    _panel_pinning(axes_m[1])
    _panel_label(axes_m[1], "(b)")
    out_main = FIGURE_DIR / "fig5_spatial_main.pdf"
    fig_m.savefig(out_main, bbox_inches="tight", dpi=300)
    plt.close(fig_m)

    # Supplement figure (sec:si-spatial-derivations): the applied-math panels --
    # the traveling-wave solution and the wave-velocity sign-change diagnostic.
    fig_s, axes_s = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    _panel_traveling_wave(axes_s[0])
    _panel_label(axes_s[0], "(a)")
    _panel_velocity(axes_s[1])
    _panel_label(axes_s[1], "(b)")
    out_supp = FIGURE_DIR / "figS26_spatial_supp.pdf"
    fig_s.savefig(out_supp, bbox_inches="tight", dpi=300)
    plt.close(fig_s)

    return out_main, out_supp


if __name__ == "__main__":
    out_main, out_supp = main()
    print(f"  Saved {out_main}")
    print(f"  Saved {out_supp}")
