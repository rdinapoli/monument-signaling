"""Generate supplementary Figure S19: joint sensitivity of sigma* in (C, n).

Computes the self-consistent critical threshold sigma*(C, n) on a 2D grid
spanning the in-scope per-builder cost bracket (C in [0.20, 0.50]) and the
face-to-face audience range (n in [10, 250]). Plots a filled contour with a
diverging colormap centered on the MLS baseline sigma*_base ~ 0.497, with the
in-scope rectangle outlined and the central-tendency point (C = 0.35, n = 60)
marked. Overlays the sigma*_base contour where it falls within the rectangle.

The framework's headline threshold is presented as a range over
this in-scope (C, n) rectangle rather than as a point: sigma* in approximately
[0.24, 0.65] across the rectangle, with central tendency ~0.43 at the central
grid point (C at the linear midpoint of its range, n at the geometric midpoint
of the log-sampled audience range). The C axis sweeps the per-builder peak labor
fraction during active
construction (re-anchored from per-household-year; see
calibration.py RAPA_NUI['C']); the n axis sweeps the face-to-face audience size
(the lower bound is set by where the framework's signals are operative, the
upper bound by the face-to-face CRED audience constraint).

The C-lambda_W coupling follows the manuscript's analytic interpolation
(the main-text C_model calibration relation): under q_min = 0.1 and q_max = 2.0,
lambda_W ~ 1.93 * C. For each (C, n) grid point the script sets
lambda_W = 1.93 * C and calls sigma_star_self_consistent(lambda_W=lambda_W,
n=n), reproducing the self-consistent parameterization used elsewhere in the
manuscript.

The script complements the existing supplementary sensitivity figures:
- Fig. S21 (sec:si-bivariate): joint (C, lambda_W) at fixed n = 12.
- Fig. S17 (sec:si-omega-k0-joint): joint (omega, k_0) at fixed C, n.
- THIS Fig. S19 (sec:si-Cn-joint): joint (C, n) at the self-consistent locus,
  the SI section "Joint sensitivity in (C, n) space".
  The (C, n) sweep is the framework's headline sensitivity: expanding from the
  single-source Roscoe 2009 n = 12 anchor makes both the cost calibration and
  the audience-size calibration uncertain in tandem.

The script is standalone and runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_figure_Cn_joint.py

Output: output/figures/figS19_Cn_joint_sensitivity.pdf

Runtime ~30-60 s on a modern laptop (the self-consistent fixed-point iteration
is the dominant cost; the grid is 13 C-values x 8 n-values = 104 points).
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
from matplotlib.colors import TwoSlopeNorm

from signaling.calibration import DEFAULT_LAMBDA_W_ANCHOR
from signaling.plotting import setup_paper_style
from signaling.price_equation import (
    initial_model_sigma_star,
    sigma_star_self_consistent,
)


FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Shared paper style: sans-serif, no red+green palette, RdBu_r diverging.
setup_paper_style()


# Analytic interpolation under q_min = 0.1, q_max = 2.0 (the main-text C_model(lambda_W) self-consistent parameterization).
LAMBDA_W_PER_C: float = 1.93

# Standard MLS baseline as the diverging-colormap center. Derived from the
# analytic MLS threshold at C = 0.35 (initial_model_sigma_star), so the figure
# tracks the code; numerically ~0.497.
SIGMA_STAR_BASE: float = float(initial_model_sigma_star(0.35))

# In-scope per-builder cost bracket (per-builder re-anchor of Erasmus 1965 +
# Abrams 1994 architectural-energetics range). 13 grid points at 0.025 step.
C_VALS: np.ndarray = np.linspace(0.20, 0.50, 13)

# Face-to-face audience range. 8 points spaced quasi-logarithmically to give
# resolution at the lower end where Layer-3 saturation drives the variation.
N_VALS: np.ndarray = np.array([10, 15, 25, 40, 60, 100, 150, 250], dtype=int)

# Central-tendency reference point: C at the linear midpoint of its range, n at
# the geometric midpoint of the log-sampled audience range (NOT the arithmetic
# midpoint of [10, 250], which is 130; n = 60 is the central grid point).
C_CENTRAL: float = 0.35
N_CENTRAL: int = 60


def compute_grid(
    c_vals: np.ndarray = C_VALS,
    n_vals: np.ndarray = N_VALS,
) -> np.ndarray:
    """Compute sigma*(C, n) on the joint grid.

    For each (C, n) grid point sets lambda_W = LAMBDA_W_PER_C * C (the
    self-consistent C_model inversion under q_min = 0.1, q_max = 2.0) and
    calls :func:`sigma_star_self_consistent` with the default multiplicative
    specification. Entries may be ``inf`` where signaling is never favored
    (no finite threshold), or ``nan`` on a genuine root-find failure; both
    are preserved so the contour plot can mask them.

    Parameters
    ----------
    c_vals, n_vals : array
        Grid values to sweep.

    Returns
    -------
    np.ndarray of shape (len(c_vals), len(n_vals)) with sigma*(C, n).
    """
    nC, nN = len(c_vals), len(n_vals)
    ss = np.full((nC, nN), np.nan)

    for i, c in enumerate(c_vals):
        lambda_W = LAMBDA_W_PER_C * float(c)
        for j, n_val in enumerate(n_vals):
            try:
                result = sigma_star_self_consistent(
                    lambda_W=lambda_W,
                    n=int(n_val),
                )
                ss[i, j] = result["sigma_star"]
            except (ValueError, RuntimeError):
                pass  # remains nan

    return ss


def plot_heatmap(
    ss: np.ndarray,
    out_path: Path,
    c_vals: np.ndarray = C_VALS,
    n_vals: np.ndarray = N_VALS,
) -> None:
    """Render the sigma*(C, n) heatmap to PDF.

    Uses RdBu_r diverging colormap centered on SIGMA_STAR_BASE. Overlays:
    - Rectangle outline at C in [0.20, 0.50], n in [10, 250] (the in-scope
      bracket; redundant with axis limits but emphasized for clarity).
    - Central marker at (C_CENTRAL, N_CENTRAL) = (0.35, 60).
    - Black contour at sigma* = SIGMA_STAR_BASE (~0.497) where it crosses
      the rectangle.
    - Black contours at sigma* = 0.24 and sigma* = 0.65 to mark the
      headline range (0.24 is the grid floor; a 0.20 contour would not
      render since min sigma* over the rectangle is ~0.236).
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Color span: derive from data, but use a diverging norm centered at MLS
    # baseline so the visual asymmetry around the baseline is preserved.
    finite_mask = np.isfinite(ss)
    if not finite_mask.any():
        raise RuntimeError("No finite sigma* values in the grid; check inputs.")
    vmin = float(np.nanmin(ss))
    vmax = float(np.nanmax(ss))
    # Ensure the baseline sits strictly inside the color range.
    eps = 1e-3
    vmin = min(vmin, SIGMA_STAR_BASE - eps)
    vmax = max(vmax, SIGMA_STAR_BASE + eps)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=SIGMA_STAR_BASE, vmax=vmax)

    # pcolormesh expects edges; build them from cell-centered axes.
    c_edges = _cell_edges(c_vals)
    n_edges = _cell_edges(n_vals.astype(float))

    # Mask non-finite entries (inf = no finite threshold; nan = solver fail).
    ss_plot = np.ma.masked_invalid(ss)
    mesh = ax.pcolormesh(
        c_edges,
        n_edges,
        ss_plot.T,  # transpose: rows -> y (n), cols -> x (C)
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"Critical threshold $\sigma^*$", fontsize=11)
    cbar.ax.axhline(SIGMA_STAR_BASE, color="black", linewidth=0.8)

    # Headline-range contours (sigma* = 0.20 and 0.65) where they cross the
    # grid. Use a quadrangle-aware contour: contourf-like contour on the
    # cell-centered data.
    C_grid, N_grid = np.meshgrid(c_vals, n_vals.astype(float), indexing="ij")
    cs = ax.contour(
        C_grid,
        N_grid,
        ss,
        levels=[0.24, SIGMA_STAR_BASE, 0.65],
        colors=["black", "black", "black"],
        linewidths=[1.0, 1.8, 1.0],
        linestyles=["dashed", "solid", "dashed"],
    )
    ax.clabel(cs, fmt={0.24: r"$\sigma^*=0.24$",
                       SIGMA_STAR_BASE: r"$\sigma^*_{\mathrm{base}}\approx 0.50$",
                       0.65: r"$\sigma^*=0.65$"},
              fontsize=8, inline=True)

    # Central-tendency marker.
    ax.plot(C_CENTRAL, N_CENTRAL, marker="o", markerfacecolor="white",
            markeredgecolor="black", markersize=8, zorder=5)
    ax.annotate(
        rf"central $(C, n) = (0.35, 60)$" + "\n" +
        rf"$\sigma^* \approx {ss[c_vals.tolist().index(0.35), n_vals.tolist().index(60)]:.2f}$",
        xy=(C_CENTRAL, N_CENTRAL),
        xytext=(C_CENTRAL + 0.04, N_CENTRAL + 30),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
    )

    ax.set_xlabel(r"Per-builder peak cost fraction $C$ (in-scope bracket)",
                  fontsize=11)
    ax.set_ylabel(r"Group size $n$ (face-to-face audience range)", fontsize=11)
    ax.set_xlim(C_VALS.min(), C_VALS.max())
    ax.set_ylim(N_VALS.min(), N_VALS.max())

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Convert cell centers to pcolormesh edges (midpoints between centers,
    extended to boundaries by mirroring the half-cell offsets)."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        raise ValueError("Need at least 2 centers to build edges.")
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def main() -> None:
    print("Computing sigma*(C, n) on the in-scope rectangle...")
    ss = compute_grid()

    # Report central values for the SI caption / test pins.
    i_C035 = list(C_VALS).index(0.35)
    j_n60 = list(N_VALS).index(60)
    j_n10 = list(N_VALS).index(10)
    j_n250 = list(N_VALS).index(250)
    i_C020 = list(C_VALS).index(0.20)
    i_C050 = list(C_VALS).index(0.50)

    finite = ss[np.isfinite(ss)]
    print(f"  Finite grid points: {finite.size} / {ss.size}")
    print(f"  sigma* range over rectangle: "
          f"[{finite.min():.4f}, {finite.max():.4f}]")
    print(f"  Central tendency sigma*(C=0.35, n=60) = {ss[i_C035, j_n60]:.4f}")
    print(f"  Rectangle corners:")
    print(f"    sigma*(C=0.20, n=10)  = {ss[i_C020, j_n10]:.4f}")
    print(f"    sigma*(C=0.20, n=250) = {ss[i_C020, j_n250]:.4f}")
    print(f"    sigma*(C=0.50, n=10)  = {ss[i_C050, j_n10]:.4f}")
    print(f"    sigma*(C=0.50, n=250) = {ss[i_C050, j_n250]:.4f}")
    print(f"  MLS baseline sigma*_base = {SIGMA_STAR_BASE:.4f} (RdBu_r center)")

    out_path = FIGURE_DIR / "figS19_Cn_joint_sensitivity.pdf"
    plot_heatmap(ss, out_path)
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
