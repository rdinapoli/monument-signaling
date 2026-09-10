"""Generate supplementary Figure S11: self-assessment and Bourgeois alternatives.

The two alternative assessment regimes derived alongside the mutual model
(Supp. Fig. S9 shows mutual assessment alone). Panel (a) is the
self-assessment conflict surface (Supp. Eq. pconflict_self), panel (b) the
Bourgeois convention (ownership threshold M_own = 2). Both apply the shared
absolute-deterrence factor 1/(1 + beta_d (M_g + M_h)). Supp. Fig. S9
shows mutual assessment alone; these panels are the supplementary
alternatives.

Standalone, runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_figure_assessment_alternatives.py

Output: output/figures/figS11_assessment_alternatives.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signaling.layer2 import compare_assessment_models
from signaling.plotting import setup_paper_style, SEQUENTIAL_CMAP

FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
setup_paper_style()


def _panel_label(ax, label):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def figure_s25() -> Path:
    M_range = np.linspace(0.1, 15, 100)
    comparison = compare_assessment_models(M_range)
    extent = [float(M_range[0]), float(M_range[-1]),
              float(M_range[0]), float(M_range[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    # Honest per-panel maxima (as in the former main-text figure): the
    # normalized self model peaks near 0.32; Bourgeois is bounded by P_base.
    for ax, key, vmax, lab in zip(axes, ["self", "bourgeois"], [0.33, 0.01], ["(a)", "(b)"]):
        im = ax.imshow(comparison[key], origin="lower", extent=extent,
                       aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax)
        ax.set_xlabel(r"$M_g$")
        ax.set_ylabel(r"$M_h$")
        fig.colorbar(im, ax=ax, label=r"$P_{\mathrm{conflict}}$", fraction=0.046, pad=0.04)
        _panel_label(ax, lab)
    out = FIGURE_DIR / "figS11_assessment_alternatives.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")
    return out


if __name__ == "__main__":
    figure_s25()
