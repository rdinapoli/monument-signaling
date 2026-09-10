"""Generate supplementary Figure S12: self-assessment M_thresh sensitivity.

Plots the absolute symmetric-dyad conflict probability under self-assessment,
P_self(M, M) (normalized self model, Eq. pconflict_self), against the
escalation threshold M_thresh, for three fixed symmetric monument stocks
M in {3, 5, 10}, with the mutual-assessment probability P_mutual(M, M) shown
as a low reference. Both models are anchored to the common no-signaling
baseline P_base at M = 0.

The figure demonstrates the robustness of the mutual-vs-self discriminator:
P_self stays far above P_mutual across the entire
M_thresh range, with the contrast if anything widening at high M_thresh,
where the self conflict probability approaches its ceiling of 1 (near-certain
conflict between two confident high-investors -- the self model behaving as
intended). Reporting absolute probabilities (bounded in [0, 1]) rather than
the reduction ratio r_self avoids the 1 - 1/P_base floor that the ratio
hits when P_self saturates. Because both conflict probabilities share the
common baseline at M = 0, the discriminator holds for all M_thresh, not
only the M_thresh < M_g region.

The script is standalone and runnable from the project root:

    PYTHONPATH=src python3 scripts/generate_figure_self_assessment.py

Output: output/figures/figS12_self_assessment_M_thresh_sensitivity.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure signaling package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signaling.layer2 import (
    self_assessment_conflict_prob,
    mutual_assessment_conflict_prob,
)
from signaling.plotting import PAPER_COLORS, setup_paper_style


FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Shared paper style: sans-serif, no red+green palette.
setup_paper_style()


def figure_s20() -> Path:
    """Self-assessment vs mutual conflict probability sensitivity to M_thresh.

    Sweeps M_thresh over [0.5, 8.0] (61 points) and plots the absolute
    symmetric-dyad conflict probability P_self(M, M) for three fixed
    monument stocks M in {3.0, 5.0, 10.0}, with P_mutual(M, M) (independent
    of M_thresh) as dashed reference lines. Both use the default beta = 0.1
    and the common baseline P_base; the self model is normalized so
    P_self(0, 0) = P_base, matching mutual assessment. Absolute probabilities
    are bounded in [0, 1], so the saturation toward 1 at high M_thresh is a
    real ceiling (near-certain conflict), not a statistic artifact.

    Returns
    -------
    Path
        Absolute path to the saved PDF.
    """
    M_thresh_vals = np.linspace(0.5, 8.0, 61)
    M_fixed = [3.0, 5.0, 10.0]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)

    for i, M in enumerate(M_fixed):
        P_self = np.array([
            float(self_assessment_conflict_prob(M, M, M_thresh=float(mt)))
            for mt in M_thresh_vals
        ])
        ax.plot(M_thresh_vals, P_self, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$M = {M:g}$")
        # Mutual reference (does not depend on M_thresh): flat dashed line.
        P_mut = float(mutual_assessment_conflict_prob(M, M))
        ax.plot([M_thresh_vals[0], M_thresh_vals[-1]], [P_mut, P_mut],
                color=PAPER_COLORS[i], linewidth=1.3, linestyle="--")

    # Default M_thresh reference.
    ax.axvline(2.0, color="gray", linestyle=":", linewidth=1.2)
    ax.text(2.1, 0.93, r"$M_{\rm thresh} = 2$ (default)", fontsize=9, color="gray")
    ax.text(0.6, 0.06,
            r"dashed: mutual assessment ($P^{\rm mutual}_{\rm conflict} < 0.01$)",
            fontsize=9)

    ax.set_xlabel(r"$M_{\rm thresh}$")
    ax.set_ylabel(r"$P_{\rm conflict}$ (symmetric dyad $M_g = M_h = M$)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="center right", fontsize=10,
              title=r"$P^{\rm self}_{\rm conflict}$ (solid)")

    out = FIGURE_DIR / "figS12_self_assessment_M_thresh_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = figure_s20()
    print(f"  Saved {out}")
