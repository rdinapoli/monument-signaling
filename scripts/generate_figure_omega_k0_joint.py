"""Generate supplementary Figure S17: joint sensitivity of sigma* in (omega, k_0).

Computes the self-consistent critical threshold sigma*(omega, k_0) on a 2D grid
at the empirical anchor lambda_W = 0.68, then plots a filled contour with a
diverging colormap centered on the MLS baseline sigma*_base ~ 0.497. Marks the
default parameter point (omega = 0, k_0 = 0.5) and overlays the
sigma* = sigma*_base contour where it falls within the swept region.

Under the framework parameterization, sigma* is corrected upward:
across the joint (omega, k_0) sweep the great majority of grid points sit at
or above the MLS baseline, and a substantial high-k_0 region has no finite
threshold at all (signaling is NEVER favored there -- a high non-signaler
baseline network leaves no survival advantage to building -- plotted as an
unfilled "no finite sigma*" region). This inverts the earlier reading in
which most of the grid lay below the baseline. The two swept parameters are
the two whose individual variation (the single-parameter omega and k_0
sensitivity analyses in the SI) moves sigma* the most; this joint sweep is
the most acute version of the
"corner-of-parameter-space" robustness check reported in the main text
sensitivity analysis.

The script is standalone and runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_figure_omega_k0_joint.py

Output: output/figures/figS17_omega_k0_joint_sensitivity.pdf
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

from signaling.price_equation import (
    sigma_star_self_consistent,
    initial_model_sigma_star,
)
from signaling.plotting import setup_paper_style


FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Shared paper style: sans-serif, no red+green palette, RdBu_r diverging.
setup_paper_style()


LAMBDA_W_ANCHOR = 0.68     # Empirical anchor for the within-group status weight
                           # (main text C-lambda_W calibration).
# Standard MLS baseline used as the diverging-colormap center. Derived from
# the analytic MLS threshold at C = 0.35 (initial_model_sigma_star) rather than
# hardcoded, so the figure tracks the code; numerically ~0.497.
SIGMA_STAR_BASE = float(initial_model_sigma_star(0.35))
DEFAULT_OMEGA = 0.0        # Primary multiplicative specification (the assembled multilevel fitness comparison).
DEFAULT_K_0 = 0.5          # Calibrated non-signaler baseline degree (calibration.py).


def compute_grid(
    omega_vals: np.ndarray,
    k0_vals: np.ndarray,
    lambda_W: float = LAMBDA_W_ANCHOR,
) -> np.ndarray:
    """Compute sigma*(omega, k_0) on the joint grid.

    Calls :func:`sigma_star_self_consistent` with ``mode='convex'`` for each
    (omega, k_0) pair. At omega = 0 the convex form reduces to the
    multiplicative specification; at omega = 1 it is fully additive. Entries
    may be ``inf`` where signaling is never favored (no finite threshold;
    typical at high k_0, where the non-signaler baseline network leaves no
    survival advantage to building), or ``nan`` on a genuine root-find
    failure; both are preserved so the contour plot can mask them.

    Parameters
    ----------
    omega_vals : ndarray
        1D array of omega values in [0, 1].
    k0_vals : ndarray
        1D array of k_0 values in [0.1, 5.0].
    lambda_W : float
        Within-group reward (default 0.68, empirical anchor).

    Returns
    -------
    ndarray
        2D array of shape (len(k0_vals), len(omega_vals)) giving sigma* at
        each grid point. Rows index k_0; columns index omega; this is the
        natural orientation for :func:`matplotlib.pyplot.pcolormesh` /
        ``contourf`` with omega on the x-axis and k_0 on the y-axis.
    """
    sigma_star = np.full((len(k0_vals), len(omega_vals)), np.nan, dtype=float)
    for i, k0 in enumerate(k0_vals):
        for j, om in enumerate(omega_vals):
            res = sigma_star_self_consistent(
                lambda_W=lambda_W,
                mode="convex",
                omega=float(om),
                k_0=float(k0),
            )
            sigma_star[i, j] = float(res["sigma_star"])
    return sigma_star


def figure_s21() -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Joint sensitivity figure: sigma*(omega, k_0) at the empirical anchor.

    Returns
    -------
    Path
        Absolute path to the saved PDF.
    ndarray, ndarray, ndarray
        The omega grid, k_0 grid, and sigma* array; returned so the caller
        can report representative grid values for the supplementary section.
    """
    omega_vals = np.linspace(0.0, 1.0, 21)        # 21 points: 0, 0.05, ..., 1.0
    k0_vals = np.linspace(0.1, 5.0, 25)           # 25 points
    sigma_star = compute_grid(omega_vals, k0_vals)

    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)

    # Mask non-finite entries (inf = no finite threshold, signaling never
    # favored; nan = solver failure) so contourf leaves them blank. They are
    # marked separately below as the "no finite sigma*" region.
    sigma_plot = np.where(np.isfinite(sigma_star), sigma_star, np.nan)
    sigma_plot_masked = np.ma.masked_invalid(sigma_plot)

    # Diverging norm centered on the MLS baseline so that values below the
    # baseline read blue and values above read red. Under the framework most of
    # the grid sits at or above the baseline (red). RdBu_r is colorblind-
    # friendly and standard for this "above / below threshold" contrast.
    finite = sigma_star[np.isfinite(sigma_star)]
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    # TwoSlopeNorm requires vmin < vcenter < vmax; pad if the grid never
    # crosses the baseline so the colormap is still well-defined.
    if vmax <= SIGMA_STAR_BASE:
        vmax = SIGMA_STAR_BASE + 0.01
    if vmin >= SIGMA_STAR_BASE:
        vmin = SIGMA_STAR_BASE - 0.01
    norm = TwoSlopeNorm(vmin=vmin, vcenter=SIGMA_STAR_BASE, vmax=vmax)

    # Shade the masked (no-finite-threshold) cells in light gray underneath.
    no_finite = np.isinf(sigma_star)
    if np.any(no_finite):
        ax.contourf(
            omega_vals, k0_vals, no_finite.astype(float),
            levels=[0.5, 1.5], colors=["0.82"], zorder=0,
        )

    levels = np.linspace(vmin, vmax, 25)
    cf = ax.contourf(
        omega_vals, k0_vals, sigma_plot_masked,
        levels=levels, norm=norm, cmap="RdBu_r", extend="both",
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r"$\sigma^*$")
    cbar.ax.axhline(
        SIGMA_STAR_BASE, color="black", linewidth=1.2,
    )

    # Label the no-finite-threshold region if present.
    if np.any(no_finite):
        # Place the label near the high-k_0 edge where the region sits.
        ax.text(
            0.5, 0.96, "no finite $\\sigma^*$\n(signaling never favored)",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            color="0.25",
            bbox=dict(boxstyle="round,pad=0.25", fc="0.92", ec="0.6", lw=0.6),
        )

    # MLS baseline contour if it lies within the swept region.
    if vmin < SIGMA_STAR_BASE < vmax:
        cs = ax.contour(
            omega_vals, k0_vals, sigma_plot_masked,
            levels=[SIGMA_STAR_BASE], colors="black", linewidths=1.5,
        )
        ax.clabel(cs, fmt=rf"$\sigma^*_{{\rm base}} = {SIGMA_STAR_BASE:.2f}$",
                  inline=True, fontsize=11)

    # Mark the default operating point (omega = 0, k_0 = 0.5).
    ax.plot(
        DEFAULT_OMEGA, DEFAULT_K_0,
        marker="o", color="black", markersize=8, markerfacecolor="white",
        markeredgewidth=1.5, zorder=5,
    )
    ax.annotate(
        r"default $(\omega = 0,\ k_0 = 0.5)$",
        xy=(DEFAULT_OMEGA, DEFAULT_K_0),
        xytext=(0.10, 1.2),
        fontsize=11,
        arrowprops=dict(arrowstyle="-", color="black", linewidth=1.0),
    )

    ax.set_xlabel(r"$\omega$ (additive--multiplicative mixing weight)")
    ax.set_ylabel(r"$k_0$ (non-signaler baseline network degree)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.1, 5.0)

    out = FIGURE_DIR / "figS17_omega_k0_joint_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out, omega_vals, k0_vals, sigma_star


def _interp(omega_vals: np.ndarray, k0_vals: np.ndarray,
            sigma_star: np.ndarray, omega: float, k0: float) -> float:
    """Bilinear lookup; if the requested point coincides with a grid node
    return that node exactly, otherwise interpolate. Used for the
    "representative grid values" reported in the supplementary section."""
    # Use the closest-grid-node value to avoid implying sub-grid accuracy.
    j = int(np.argmin(np.abs(omega_vals - omega)))
    i = int(np.argmin(np.abs(k0_vals - k0)))
    return float(sigma_star[i, j])


if __name__ == "__main__":
    out, omega_vals, k0_vals, sigma_star = figure_s21()
    print(f"  Saved {out}")
    # Report representative grid values for use in the supplementary text.
    pts = [(0.0, 0.5), (0.5, 2.0), (1.0, 5.0), (0.0, 5.0), (1.0, 0.5),
           (0.25, 1.0), (0.5, 0.5)]
    print("  Representative sigma* values:")
    for om, k0 in pts:
        s = _interp(omega_vals, k0_vals, sigma_star, om, k0)
        print(f"    sigma*(omega={om:.2f}, k_0={k0:.2f}) = {s:.4f}")
    finite = sigma_star[np.isfinite(sigma_star)]
    n_inf = int(np.sum(np.isinf(sigma_star)))
    n_total = sigma_star.size
    print(f"  MLS baseline sigma*_base = {SIGMA_STAR_BASE:.4f}")
    print(f"  finite min sigma* on grid: {finite.min():.4f}")
    print(f"  finite max sigma* on grid: {finite.max():.4f}")
    print(f"  no-finite-threshold cells: {n_inf}/{n_total} "
          f"({100.0 * n_inf / n_total:.1f}%)")
    # Fraction of grid points strictly below the MLS baseline (inf counts as
    # not-below). Under the framework this fraction is small: most of the grid sits at or
    # above the baseline.
    below = np.sum(sigma_star < SIGMA_STAR_BASE) / n_total
    at_or_above = np.sum(sigma_star >= SIGMA_STAR_BASE) / n_total
    print(f"  fraction of grid with sigma* < baseline: {below:.3f}")
    print(f"  fraction of grid with sigma* >= baseline: {at_or_above:.3f}")
