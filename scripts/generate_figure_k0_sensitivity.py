"""Generate supplementary Figure S16: sensitivity of sigma* to the non-signaler
baseline network degree k_0, at the empirical anchor lambda_W = 0.68.

Sweeps k_0 over [0.1, 5.0], computing the self-consistent threshold
sigma*(k_0) = sigma_star_self_consistent(lambda_W=0.68, k_0=k_0). Under the
Under the framework parameterization, sigma* RISES with k_0: at the default
k_0 = 0.5 it is ~0.48 (just below the MLS baseline ~0.50); as k_0 approaches
parity with builders the survival differential beta_eff - alpha_eff shrinks and
sigma* rises above the baseline, with no finite threshold (signaling never
favored) by k_0 ~ 5.

This regenerates a figure that previously had no standalone generator (the legacy
the former figS17 (now figS16) showed the old 0.21 -> 0.59 curve and could not be reproduced). It is
the 1-D k_0 slice of the joint (omega, k_0) sensitivity in figS17.

Runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_figure_k0_sensitivity.py

Output: output/figures/figS16_k0_sensitivity.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signaling.price_equation import (
    sigma_star_self_consistent,
    initial_model_sigma_star,
)
from signaling.plotting import setup_paper_style, PAPER_COLORS


FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

setup_paper_style()

LAMBDA_W = 0.68                                   # empirical anchor (Erasmus 1965).
SIGMA_BASE = float(initial_model_sigma_star(0.35))  # MLS baseline ~0.497.
DEFAULT_K0 = 0.5                                  # calibrated non-signaler degree.


def figure_s17() -> Path:
    """sigma*(k_0) at the empirical anchor; marks the default and the baseline."""
    k0_vals = np.linspace(0.1, 5.0, 60)
    sigma_star = np.array([
        float(sigma_star_self_consistent(lambda_W=LAMBDA_W, k_0=float(k0))["sigma_star"])
        for k0 in k0_vals
    ])
    finite = np.isfinite(sigma_star)

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    ax.plot(k0_vals[finite], sigma_star[finite], linewidth=2.4, color=PAPER_COLORS[0],
            label=r"$\sigma^*(k_0)$ (self-consistent, $\lambda_W = 0.68$)")

    # No-finite-threshold region (signaling never favored), shaded.
    if np.any(~finite):
        k_inf = float(k0_vals[~finite].min())
        ax.axvspan(k_inf, 5.0, color="0.88", zorder=0)
        ax.text((k_inf + 5.0) / 2.0, 0.5,
                "no finite $\\sigma^*$\n(signaling never favored)",
                ha="center", va="center", fontsize=10, color="0.3")

    ax.axhline(SIGMA_BASE, color=PAPER_COLORS[1], linestyle="--", linewidth=1.6,
               label=rf"MLS baseline $\sigma^*_{{\mathrm{{base}}}} \approx {SIGMA_BASE:.2f}$")

    s_def = float(sigma_star[int(np.argmin(np.abs(k0_vals - DEFAULT_K0)))])
    ax.axvline(DEFAULT_K0, color="gray", linestyle=":", linewidth=1.2)
    ax.annotate(rf"default $k_0 = 0.5$ ($\sigma^* \approx {s_def:.2f}$)",
                xy=(DEFAULT_K0, s_def), xytext=(DEFAULT_K0 + 0.5, s_def - 0.12),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel(r"Non-signaler baseline network degree $k_0$")
    ax.set_ylabel(r"Critical threshold $\sigma^*$")
    ax.set_xlim(0.1, 5.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=10, loc="upper left")

    out = FIGURE_DIR / "figS16_k0_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = figure_s17()
    print(f"  Saved {out}")
