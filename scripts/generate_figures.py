"""Generate the main-text figures and several supplementary figures.

Function -> output file -> rendered number (EHS v7, 2026-09-08):
  figure1  -> fig2_signaling_equilibrium.pdf   (main Fig. 2)
  figure2  -> figS9_conflict_assessment.pdf    (Supp. Fig. S9)
  figure3  -> figS13_network_buffering.pdf     (Supp. Fig. S13)
  figure4  -> figS25_feedback_loop.pdf         (Supp. Fig. S25)
  figure5  -> fig3_sigma_star.pdf              (main Fig. 3)
  figure6  -> figS23_phase_space.pdf           (Supp. Fig. S23)
  figure8  -> fig6_depreciation.pdf            (main Fig. 6)
  figure9  -> figS22_sensitivity.pdf           (Supp. Fig. S22)
  figure10 -> figS21_bivariate_sensitivity.pdf (Supp. Fig. S21)
  figure11 -> figS1_intensity.pdf              (Supp. Fig. S1)
  figure12 -> figS10_war_avoidance.pdf         (Supp. Fig. S10)
Function names are historical (numbered at the JTB submission) and are kept
so that tests importing them keep working.

All figures saved as PDF in output/figures/.
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
from scipy.optimize import brentq

from signaling.layer1 import (
    equilibrium_investment,
    equilibrium_fitness,
    fitness_gain,
    free_rider_deterrent,
)
from signaling.layer1 import effective_monument_stock, monument_stock_trajectory, signal_half_life
from signaling.layer2 import mutual_assessment_conflict_prob, compare_assessment_models, compute_lambda_C
from signaling.layer3 import (
    network_degree,
    survival_probability,
    vulnerability_coefficient,
    derive_vulnerability_differential,
    compute_lambda_X,
    lambda_sigma_sweep,
    lambda_total_at_sigma,
)
from signaling.price_equation import (
    initial_model_sigma_star,
    fitness_advantage,
    sigma_star_comparison,
    phase_space,
    critical_threshold_sigma_star,
    sigma_star_self_consistent,
    sigma_star_vs_param,
    sigma_star_bivariate,
    sensitivity_tornado,
)
from signaling.emergence import rare_builder_fitness_advantage
from signaling.layer1 import average_equilibrium_cost
from signaling.calibration import (
    DEFAULT_Q_MIN,
    DEFAULT_Q_MAX,
    DEFAULT_GAMMA,
    DEFAULT_K_0,
    DEFAULT_K_MAX,
    DEFAULT_M_HALF,
)
from signaling.plotting import (
    PAPER_COLORS,
    PALETTE,
    SEQUENTIAL_CMAP,
    DIVERGING_CMAP,
    TWO_CLASS_COLORS,
    setup_paper_style,
)

FIGURE_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Apply the shared paper style (sans-serif, Okabe-Ito-derived palette,
# no red+green, viridis sequential, RdBu_r diverging).
setup_paper_style()


def _panel_label(ax, label, x=-0.12, y=1.05):
    """Add a bold panel label at the upper-left corner."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")


# =======================================================================
# Figure 1: Signaling equilibrium (Layer 1)
# =======================================================================

def figure1():
    q = np.linspace(DEFAULT_Q_MIN + 1e-6, DEFAULT_Q_MAX, 300)
    # Emergence-stage reward, empirical anchor, and top of the calibrated range.
    lam_values = [0.3, 0.68, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    # (a) Equilibrium investment
    ax = axes[0]
    for i, lam in enumerate(lam_values):
        x_star = equilibrium_investment(q, DEFAULT_Q_MIN, lam)
        ax.plot(q, x_star, color=PAPER_COLORS[i], label=rf"$\lambda_W = {lam}$")
    ax.set_xlabel(r"Capacity $q$")
    ax.set_ylabel(r"Investment $x^*(q)$")
    ax.legend()
    _panel_label(ax, "(a)")

    # (b) Equilibrium payoff vs unilateral free-riding at the empirical anchor lambda_W = 0.68.
    # The deviation comparison, not the no-signaling counterfactual: a
    # free-rider (x = 0) is read at the floor q_min (on-path schedule
    # inversion) and earns
    # q + lambda_W * q_min, so the two curves meet at q_min and the gap is
    # the free-rider deterrent Delta_fr(q) >= 0 (positional-reward-consistent).
    ax = axes[1]
    lam_b = 0.68
    w_star = equilibrium_fitness(q, DEFAULT_Q_MIN, lam_b)
    w_freeride = q + lam_b * DEFAULT_Q_MIN
    ax.plot(q, w_star, color=PAPER_COLORS[0], linewidth=2.4,
            label=r"$w^*(q)$ (build)")
    ax.plot(q, w_freeride, "--", color="gray", linewidth=1.4,
            label=r"$q + \lambda_W q_{\min}$ (free-ride)")
    ax.fill_between(q, w_freeride, w_star, alpha=0.2, color=PAPER_COLORS[0],
                    label=r"Deterrent $\Delta_{\mathrm{fr}}(q)$")
    ax.set_xlabel(r"Capacity $q$")
    ax.set_ylabel("Fitness")
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Free-rider deterrent Delta_fr(q) = lambda_W (q - q_min)^2 / (2q)
    ax = axes[2]
    for i, lam in enumerate(lam_values):
        d_fr = free_rider_deterrent(q, lam, DEFAULT_Q_MIN)
        ax.plot(q, d_fr, color=PAPER_COLORS[i], label=rf"$\lambda_W = {lam}$")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"Capacity $q$")
    ax.set_ylabel(r"$\Delta_{\mathrm{fr}}(q)$")
    ax.legend()
    _panel_label(ax, "(c)")

    out = FIGURE_DIR / "fig2_signaling_equilibrium.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 2: Conflict assessment models (Layer 2)
# =======================================================================

def figure2():
    # Mutual assessment only; the self-assessment and Bourgeois alternatives
    # are Supp. Fig. S25 (scripts/generate_figure_assessment_alternatives.py).
    M_range = np.linspace(0.1, 15, 100)
    comparison = compare_assessment_models(M_range)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), constrained_layout=True)
    extent = [float(M_range[0]), float(M_range[-1]),
              float(M_range[0]), float(M_range[-1])]
    # Mutual assessment is bounded by P_base = 0.01.
    im = ax.imshow(
        comparison["mutual"], origin="lower", extent=extent,
        aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0, vmax=0.01,
    )
    ax.set_xlabel(r"$M_g$")
    ax.set_ylabel(r"$M_h$")
    fig.colorbar(im, ax=ax, label=r"$P_{\mathrm{conflict}}$",
                 fraction=0.046, pad=0.04)

    out = FIGURE_DIR / "figS9_conflict_assessment.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 3: Network buffering (Layer 3)
# =======================================================================

def figure3():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # (a) Network degree k(M_g)
    ax = axes[0, 0]
    M_vals = np.linspace(0, 20, 300)
    k_vals = network_degree(M_vals)
    ax.plot(M_vals, k_vals, linewidth=2.4, color=PAPER_COLORS[0])
    ax.axhline(DEFAULT_K_0, color="gray", linestyle="--", linewidth=1.0,
               label=rf"$k_0 = {DEFAULT_K_0}$")
    ax.axhline(DEFAULT_K_0 + DEFAULT_K_MAX, color="gray", linestyle=":",
               linewidth=1.0,
               label=rf"$k_0 + k_{{\max}} = {DEFAULT_K_0 + DEFAULT_K_MAX}$")
    ax.axvline(DEFAULT_M_HALF, color=PAPER_COLORS[0], linestyle=":", linewidth=1.0,
               alpha=0.5, label=rf"$M_{{1/2}} = {DEFAULT_M_HALF}$")
    ax.set_xlabel(r"Group monument stock $M_g$")
    ax.set_ylabel(r"Network degree $k$")
    ax.legend(fontsize=10)
    _panel_label(ax, "(a)")

    # (b) Survival surface S(sigma, k)
    ax = axes[0, 1]
    sigma_arr = np.linspace(0.01, 0.8, 100)
    k_arr = np.linspace(0, 10, 100)
    Sg, Kg = np.meshgrid(sigma_arr, k_arr)
    S = survival_probability(Sg, Kg, DEFAULT_GAMMA)
    extent_b = [0.01, 0.8, 0, 10]
    im = ax.imshow(S, origin="lower", extent=extent_b, aspect="auto",
                   cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Network degree $k$")
    fig.colorbar(im, ax=ax, label=r"$S(\sigma, k)$",
                 fraction=0.046, pad=0.04)
    _panel_label(ax, "(b)")

    # (c) Vulnerability coefficient alpha(k)
    ax = axes[1, 0]
    k_plot = np.linspace(0, 12, 300)
    alpha_vals = vulnerability_coefficient(k_plot, DEFAULT_GAMMA)
    ax.plot(k_plot, alpha_vals, linewidth=2.4, color=PAPER_COLORS[0],
            label=r"$\alpha(k) = 1/(1+\gamma k)$")
    # Standard MLS baseline reference lines (exogenous assumptions)
    ax.axhline(0.30, color=PAPER_COLORS[1], linestyle="--", linewidth=1.4,
               label=r"Baseline $\alpha = 0.30$")
    ax.axhline(0.90, color=PAPER_COLORS[2], linestyle="--", linewidth=1.4,
               label=r"Baseline $\beta = 0.90$")
    # Effective beta for non-signalers at k_0
    beta_eff_val = float(vulnerability_coefficient(DEFAULT_K_0, DEFAULT_GAMMA))
    ax.axhline(beta_eff_val, color=PAPER_COLORS[3], linestyle=":", linewidth=1.4,
               label=rf"$\beta_{{\mathrm{{eff}}}} = \alpha(k_0) = {beta_eff_val:.2f}$")
    ax.set_xlabel(r"Network degree $k$")
    ax.set_ylabel(r"Vulnerability coefficient $\alpha(k)$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    _panel_label(ax, "(c)")

    # (d) lambda_X as function of sigma for several M_g values
    ax = axes[1, 1]
    sigma_plot = np.linspace(0.01, 0.8, 200)
    M_g_values = [2, 5, 10, 20]
    for i, Mg in enumerate(M_g_values):
        lam_X_arr = [compute_lambda_X(float(Mg), float(s)) for s in sigma_plot]
        ax.plot(sigma_plot, lam_X_arr, color=PAPER_COLORS[i],
                label=rf"$M_g = {Mg}$")
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"$\lambda_X$")
    ax.legend(fontsize=10)
    _panel_label(ax, "(d)")

    out = FIGURE_DIR / "figS13_network_buffering.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 4: Lambda-sigma feedback loop
# =======================================================================

def figure4():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sigma_range = np.linspace(0.01, 0.8, 80)

    # (a) Feedback percentage of total lambda across lambda_W values
    # This shows the REGIME STRUCTURE: when does the feedback matter?
    ax = axes[0]
    lw_values = [0.01, 0.05, 0.10, 0.30, 0.68]
    colors_a = [PAPER_COLORS[0], PAPER_COLORS[1], PAPER_COLORS[2], PAPER_COLORS[3], PAPER_COLORS[4]]
    for lw, color in zip(lw_values, colors_a):
        pcts = []
        for s in sigma_range:
            loop = lambda_total_at_sigma(float(s), lambda_W=lw)
            fb = (loop["lambda_C"] + loop["lambda_X"]) / loop["lambda_total"] * 100
            pcts.append(fb)
        ax.plot(sigma_range, pcts, linewidth=2.4, color=color,
                label=rf"$\lambda_W = {lw}$")
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel("Feedback contribution (%)")
    ax.axhline(10, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.annotate("10% threshold", xy=(0.02, 11), fontsize=10, color="gray")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 85)
    _panel_label(ax, "(a)")

    # (b) Network saturation: k/k_max as a function of lambda_W
    # Shows WHY the feedback is negligible at high lambda_W: the network saturates
    ax = axes[1]
    lw_range = np.linspace(0.01, 1.0, 60)
    sat_pcts = []
    M_g_vals = []
    for lw in lw_range:
        loop = lambda_total_at_sigma(0.3, lambda_W=lw)
        k_val = loop["k"]
        sat = (k_val - 0.5) / 8.0 * 100  # (k - k_0) / k_max
        sat_pcts.append(sat)
        M_g_vals.append(loop["M_g"])
    ax.plot(lw_range, sat_pcts, linewidth=2.4, color=PAPER_COLORS[0])
    ax.axvline(0.68, color="gray", linestyle=":", linewidth=1.2,
               label=r"Empirical anchor $\lambda_W = 0.68$")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.annotate("50% saturated", xy=(0.02, 52), fontsize=10, color="gray")
    ax.set_xlabel(r"Within-group social reward $\lambda_W$")
    ax.set_ylabel("Network saturation (%)")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    _panel_label(ax, "(b)")

    out = FIGURE_DIR / "figS25_feedback_loop.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 5: sigma* comparison
# =======================================================================

def figure5():
    """Self-consistent critical threshold under the positional + war-avoidance model.

    (a) sigma*(lambda_W) along the self-consistent locus C = C_model(lambda_W),
        with the empirical anchor point (lambda_W ~ 0.68, sigma* ~ 0.477) marked.
        The threshold rises with lambda_W and sits near the standard MLS baseline
        sigma*_base ~ 0.50: the framework corrects the threshold upward, it does
        not lower it.
    (b) Rare-builder fitness advantage vs sigma at the empirical anchor; it
        crosses zero at sigma* ~ 0.477, close to the MLS baseline.

    Uses the canonical sigma_star_self_consistent function (derived r,
    closed lambda_C feedback, internally consistent C = C_model(lambda_W))
    to ensure the figure matches the manuscript text's described specification.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # (a) Self-consistent threshold sigma*(lambda_W)
    ax = axes[0]
    lam_W_range = np.linspace(0.05, 1.0, 30)
    sigma_star_sc = []
    for lam_W in lam_W_range:
        try:
            res = sigma_star_self_consistent(
                lambda_W=float(lam_W), mode="multiplicative",
            )
            sigma_star_sc.append(res["sigma_star"])
        except (ValueError, RuntimeError):
            sigma_star_sc.append(np.nan)
    sigma_star_sc = np.array(sigma_star_sc)
    valid = np.isfinite(sigma_star_sc) & (sigma_star_sc > 0)

    ax.plot(lam_W_range[valid], sigma_star_sc[valid],
            linewidth=2.5, color=PAPER_COLORS[0],
            label=r"$\sigma^*(\lambda_W)$, multiplicative, self-consistent")
    ax.axhline(0.50, color=PAPER_COLORS[1], linestyle="--", linewidth=1.7,
               label=r"Standard MLS baseline $\sigma^*_{\mathrm{base}} = 0.50$")

    # Empirical anchor point
    lam_emp = 0.68
    res_emp = sigma_star_self_consistent(lambda_W=lam_emp, mode="multiplicative")
    sigma_emp = res_emp["sigma_star"]
    ax.scatter([lam_emp], [sigma_emp], color=PALETTE["secondary"], s=80, zorder=5,
               label=rf"Empirical anchor $(\lambda_W \approx {lam_emp},\ \sigma^* \approx {sigma_emp:.2f})$")
    ax.annotate(rf"$C_{{\mathrm{{model}}}} \approx 0.35$",
                xy=(lam_emp, sigma_emp), xytext=(lam_emp + 0.03, sigma_emp - 0.08),
                fontsize=11, color=PALETTE["secondary"])

    ax.set_xlabel(r"Within-group social reward $\lambda_W$")
    ax.set_ylabel(r"Critical threshold $\sigma^*$")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 0.65)
    ax.set_xlim(0, 1.0)
    _panel_label(ax, "(a)")

    # (b) Rare-builder fitness advantage vs sigma at the empirical anchor.
    # The advantage crosses zero at the self-consistent threshold sigma* ~ 0.477,
    # close to the naive MLS baseline ~ 0.50: the positional within-group reward
    # and the derived (less favorable) coefficients correct the threshold upward
    # toward the baseline, rather than lowering it.
    ax = axes[1]
    sigma_range = np.linspace(0.01, 0.8, 100)
    adv = np.array([
        rare_builder_fitness_advantage(
            float(s), lam_emp, frac_signalers=1.0, mode="multiplicative",
        )["advantage"]
        for s in sigma_range
    ])
    ax.plot(sigma_range, adv, linewidth=2.4, color=PAPER_COLORS[0],
            label=r"Rare-builder advantage ($\phi = 1$)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.fill_between(sigma_range, 0, adv, where=adv > 0, alpha=0.1,
                    color=PAPER_COLORS[0])

    # Mark the self-consistent threshold (advantage zero-crossing = panel-a anchor)
    # and the MLS baseline; they nearly coincide (the threshold is corrected upward).
    sigma_base = initial_model_sigma_star(C=res_emp["C_model"])
    ax.axvline(sigma_emp, color=PAPER_COLORS[0], linestyle=":", linewidth=1.3)
    ax.annotate(rf"$\sigma^* \approx {sigma_emp:.2f}$",
                xy=(sigma_emp, 0), xytext=(sigma_emp - 0.22, 0.015),
                fontsize=11, color=PAPER_COLORS[0])
    ax.axvline(sigma_base, color=PAPER_COLORS[1], linestyle="--", linewidth=1.6,
               label=rf"MLS baseline $\sigma^*_{{\mathrm{{base}}}} \approx {sigma_base:.2f}$")

    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Fitness advantage $\Delta w$")
    ax.legend(fontsize=10, loc="upper left")
    _panel_label(ax, "(b)")

    out = FIGURE_DIR / "fig3_sigma_star.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 6: Phase space
# =======================================================================

def figure6():
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    sigma_range = np.linspace(0.01, 0.8, 50)
    C_range = np.linspace(0.05, 0.7, 50)

    result = phase_space(sigma_range, C_range, lambda_W=0.3)

    # Compute standard MLS baseline boundary sigma*_base for each C
    sigma_star_init = np.array([
        initial_model_sigma_star(C=float(c)) for c in C_range
    ])

    # Build three-zone classification:
    #   0 = not favored (gray)
    #   1 = newly accessible: favored by extended model but NOT by baseline
    #   2 = favored by both models (full green)
    favored_ext = result["fitness_advantage"] > 0  # shape (len(C_range), len(sigma_range))
    Sg, _ = np.meshgrid(sigma_range, C_range)

    # Baseline favored where sigma > sigma*_base for that C
    # sigma_star_init is per-C, broadcast across sigma dimension
    sigma_init_2d = sigma_star_init[:, np.newaxis] * np.ones_like(Sg)
    favored_init = Sg > sigma_init_2d

    zones = np.zeros_like(Sg)
    zones[favored_ext & ~favored_init] = 1  # newly accessible region
    zones[favored_ext & favored_init] = 2   # favored by both

    from matplotlib.colors import ListedColormap
    # Three-class sequential blue ramp (light gray, light blue, mid blue),
    # colorblind-safe, giving an ordered "off, newly accessible, favored"
    # reading of the three zones.
    cmap = ListedColormap(["#D9D9D9", "#C6DBEF", "#6BAED6"])
    extent = [0.01, 0.8, 0.05, 0.7]
    ax.imshow(zones, origin="lower", extent=extent,
              aspect="auto", cmap=cmap, vmin=0, vmax=2)

    # Plot the baseline boundary as a dashed line
    # Clip to visible range
    valid_init = (sigma_star_init > 0.01) & (sigma_star_init < 0.8)
    ax.plot(sigma_star_init[valid_init], C_range[valid_init],
            "k--", linewidth=2.4, label=r"$\sigma^*_{\mathrm{base}}$", zorder=4)

    # Case study points removed: Rapa Nui and Rapa Iti no longer in paper

    # The derived (war-avoidance) between-group boundary nearly coincides with the
    # MLS baseline: the framework corrects sigma* upward toward the baseline rather
    # than dramatically expanding the favored region.
    ax.annotate(
        "Derived boundary\n$\\approx$ MLS baseline\n($\\sigma^*$ corrected upward)",
        xy=(0.46, 0.28), xytext=(0.55, 0.10),
        fontsize=10, ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        zorder=6,
    )

    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Reproductive cost $C$")

    import matplotlib.patches as mpatches
    favored_patch = mpatches.Patch(color="#6BAED6", label="Favored (both)")
    newly_patch = mpatches.Patch(color="#C6DBEF",
                                 label="Extended-favored only (thin band)")
    gray_patch = mpatches.Patch(color="#D9D9D9", label="Not favored")
    legend_handles = [favored_patch, newly_patch, gray_patch]
    if ax.lines:
        legend_handles.append(ax.lines[0])  # baseline boundary
    ax.legend(handles=legend_handles, fontsize=11, loc="upper right")

    out = FIGURE_DIR / "figS23_phase_space.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 8: Signal depreciation
# =======================================================================

def figure8():
    from signaling.calibration import (
        DEFAULT_DELTA, DEFAULT_N_COMPUTE, DEFAULT_Q_MIN, DEFAULT_Q_MAX,
    )
    from signaling.layer1 import expected_monument_stock

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # (a) M_g(t) trajectories for different delta values
    ax = axes[0]
    # Anchor investment flow I_g ~ 10.31 (lambda_W = 0.68, n = 12), the same
    # flow behind the depreciation-results disclosure M_g* = I_g/delta ~ 103
    # at delta = 0.10. A hard-coded illustrative I_g = 5.0 previously made
    # panel (b) read M* ~ 50 at the default delta, contradicting the text.
    I_g = expected_monument_stock(
        DEFAULT_N_COMPUTE, DEFAULT_Q_MIN, DEFAULT_Q_MAX, 0.68
    )
    T = 80
    for i, delta in enumerate([0.05, 0.10, 0.20]):
        traj = monument_stock_trajectory(I_g, delta, T)
        ax.plot(range(T + 1), traj, linewidth=2.4, color=PAPER_COLORS[i],
                label=rf"$\delta = {delta}$")
        ax.axhline(I_g / delta, color=PAPER_COLORS[i], linestyle=":", linewidth=1.0,
                   alpha=0.5)
    ax.set_xlabel("Time period $t$")
    ax.set_ylabel(r"Monument stock $M_g(t)$")
    ax.legend(fontsize=11)
    _panel_label(ax, "(a)")

    # (b) Steady-state M_g* vs delta
    ax = axes[1]
    delta_vals = np.linspace(0.02, 0.5, 100)
    M_star = I_g / delta_vals
    ax.plot(delta_vals, M_star, linewidth=2.4, color=PAPER_COLORS[0])
    ax.axvline(DEFAULT_DELTA, color="gray", linestyle="--", linewidth=1.2,
               label=rf"Default $\delta = {DEFAULT_DELTA}$")
    ax.set_xlabel(r"Depreciation rate $\delta$")
    ax.set_ylabel(r"Steady-state $M_g^* = I_g / \delta$")
    ax.legend(fontsize=11)
    _panel_label(ax, "(b)")

    out = FIGURE_DIR / "fig6_depreciation.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 9: Parameter sensitivity analysis
# =======================================================================

def figure9():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    # Readable parameter labels for the tornado chart
    PARAM_LABELS = {
        "C": r"$C$ (reproductive cost)",
        "C_exogenous": r"$C$ (off-locus cost)",
        "lambda_W": r"$\lambda_W$ (within-group reward)",
        "gamma": r"$\gamma$ (buffering efficiency)",
        "n": r"$n$ (group size)",
        "r": r"$r$ (conflict reduction)",
        "delta": r"$\delta$ (depreciation rate)",
        "k_max": r"$k_{\max}$ (max network gain)",
        "P_base": r"$P_{\mathrm{base}}$ (conflict probability)",
    }

    # ------------------------------------------------------------------
    # (a) sigma* vs lambda_W
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    lam_W_range = np.linspace(0.05, 1.0, 25)
    # Self-consistent family (C = C_model(lambda_W), derived r = r_bb): the
    # canonical sigma*(lambda_W), rising and correcting upward toward the baseline.
    ss_lw = np.array([
        sigma_star_self_consistent(lambda_W=float(lw))["sigma_star"]
        for lw in lam_W_range
    ])
    valid = np.isfinite(ss_lw) & (ss_lw > 0)
    ax.plot(lam_W_range[valid], ss_lw[valid], linewidth=2.4, color=PAPER_COLORS[0])
    ss_init = float(sigma_star_self_consistent(lambda_W=0.68)["sigma_star_base"])
    ax.axhline(ss_init, color=PAPER_COLORS[1], linestyle="--", linewidth=1.7,
               label=rf"$\sigma^*_{{\mathrm{{base}}}} \approx {ss_init:.2f}$")
    ax.axvline(0.68, color="gray", linestyle=":", linewidth=1.2,
               label=r"Empirical anchor $\lambda_W = 0.68$")
    ax.set_xlabel(r"Within-group reward $\lambda_W$")
    ax.set_ylabel(r"Critical threshold $\sigma^*$")
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=10)
    _panel_label(ax, "(a)")

    # ------------------------------------------------------------------
    # (b) sigma* vs group size n
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    n_range = np.arange(4, 42, 2, dtype=float)
    sweep_n = sigma_star_vs_param("n", n_range, lambda_W=0.68)
    valid_n = np.isfinite(sweep_n["sigma_star"])
    ax.plot(n_range[valid_n], sweep_n["sigma_star"][valid_n],
            "o-", linewidth=2.4, color=PAPER_COLORS[0], markersize=3)
    ax.axhline(ss_init, color=PAPER_COLORS[1], linestyle="--", linewidth=1.7,
               label=rf"$\sigma^*_{{\mathrm{{base}}}}$")
    ax.axvline(12, color="gray", linestyle=":", linewidth=1.2,
               label=r"Default $n = 12$")
    ax.set_xlabel(r"Group size $n$")
    ax.set_ylabel(r"Critical threshold $\sigma^*$")
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # ------------------------------------------------------------------
    # (c) Tornado chart: one-at-a-time sensitivity
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    # Conflict reduction r is now DERIVED (r = r_bb at equilibrium M_g), not a
    # free parameter, so it is not swept; the war-avoidance reduction enters via
    # the equilibrium monument stock rather than as a stipulated constant.
    # The cost row sweeps C_exogenous (off-locus mode): on the canonical
    # self-consistent path C = C_model(lambda_W) is endogenous, so a bare-C
    # sweep would leave the framework's sigma* flat by construction. The
    # off-locus sweep decouples the reproductive cost from lambda_W (holding
    # the derived coefficients at the lambda_W equilibrium) over the empirical
    # bracket C_emp in [0.20, 0.50].
    tornado = sensitivity_tornado({
        "lambda_W": (0.1, 0.5),
        "gamma": (0.1, 0.5),
        "C_exogenous": (0.2, 0.5),
        "n": (6, 24),
        "delta": (0.0, 0.2),
        "k_max": (4, 12),
        "P_base": (0.005, 0.02),
    })

    params = list(tornado.keys())  # sorted by swing (largest first)
    n_params = len(params)
    y_pos = np.arange(n_params)
    baseline_ss = list(tornado.values())[0]["sigma_star_baseline"]

    # Single-colour bars: parameters are distinguished by their y-axis
    # label, so cycling through eight different hues adds visual noise
    # without information.
    for i, pname in enumerate(params):
        info = tornado[pname]
        lo = info["sigma_star_low"]
        hi = info["sigma_star_high"]
        left = min(lo, hi)
        width = abs(hi - lo)
        ax.barh(n_params - 1 - i, width, left=left, height=0.6,
                color=PALETTE["primary"], alpha=0.7,
                edgecolor="black", linewidth=0.7)

    ax.axvline(baseline_ss, color="black", linewidth=1.7, linestyle="-",
               label=rf"Baseline $\sigma^* = {baseline_ss:.2f}$")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in reversed(params)],
                       fontsize=11)
    ax.set_xlabel(r"Critical threshold $\sigma^*$")
    ax.legend(fontsize=10, loc="upper right")
    _panel_label(ax, "(c)")

    # ------------------------------------------------------------------
    # (d) Fitness mode comparison: sigma*(C) under mixed vs multiplicative
    # ------------------------------------------------------------------
    # Off-locus C sweep (C_exogenous): the framework's cost is decoupled from
    # lambda_W so the mode curves genuinely vary with C, comparable to the
    # C-parameterized initial-model baseline below.
    ax = axes[1, 1]
    C_range_mode = np.linspace(0.1, 0.6, 20)
    for mode_name, ls, color, label in [
        ("mixed", "-", PAPER_COLORS[0], "Mixed (additive-reward)"),
        ("multiplicative", "--", PAPER_COLORS[2], "Multiplicative"),
        ("additive", ":", PAPER_COLORS[3], "Additive"),
    ]:
        ss_vals = []
        for c_val in C_range_mode:
            try:
                res = critical_threshold_sigma_star(
                    C_exogenous=float(c_val), lambda_W=0.68, mode=mode_name
                )
                ss_vals.append(res["sigma_star"])
            except (ValueError, RuntimeError):
                ss_vals.append(np.nan)
        ss_arr = np.array(ss_vals)
        v = np.isfinite(ss_arr)
        ax.plot(C_range_mode[v], ss_arr[v], linestyle=ls, linewidth=2.4,
                color=color, label=label)

    # Initial model baseline
    ss_base_arr = np.array([initial_model_sigma_star(C=float(c)) for c in C_range_mode])
    ax.plot(C_range_mode, ss_base_arr, "k--", linewidth=1.7, alpha=0.5,
            label=r"$\sigma^*_{\mathrm{base}}$")
    ax.set_xlabel(r"Reproductive cost $C$")
    ax.set_ylabel(r"Critical threshold $\sigma^*$")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    _panel_label(ax, "(d)")

    out = FIGURE_DIR / "figS22_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 10: Bivariate sensitivity sigma*(C, lambda_W)
# =======================================================================

def figure10():
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), constrained_layout=True)

    C_range = np.linspace(0.10, 0.60, 20)
    lam_range = np.linspace(0.05, 1.00, 25)

    # Off-locus sweep: C_exogenous decouples the framework's reproductive cost
    # from lambda_W (the derived coefficients stay at the lambda_W equilibrium),
    # so the contours genuinely display independent (C, lambda_W) dependence.
    # A bare-C sweep would leave sigma* flat: on the canonical path the cost is
    # the endogenous C_model(lambda_W).
    result = sigma_star_bivariate("C_exogenous", C_range, "lambda_W", lam_range)
    ss = np.ma.masked_invalid(result["sigma_star"])

    # Contour plot: C on x-axis, lambda_W on y-axis
    cf = ax.contourf(
        result["param1_grid"], result["param2_grid"], ss,
        levels=np.arange(0.10, 0.90, 0.05),
        cmap="viridis_r",
    )
    cs = ax.contour(
        result["param1_grid"], result["param2_grid"], ss,
        levels=np.arange(0.2, 0.9, 0.1),
        colors="white", linewidths=0.8,
    )
    ax.clabel(cs, inline=True, fontsize=11, fmt="%.1f")

    # Self-consistent locus C = C_model(lambda_W): the diagonal along which the
    # main text reports sigma* (cost and reward moving together).
    from signaling.layer1 import average_equilibrium_cost
    lam_locus = np.linspace(lam_range[0], lam_range[-1], 200)
    C_locus = np.array([average_equilibrium_cost(float(l), 0.1, 2.0) for l in lam_locus])
    on_plot = (C_locus >= C_range[0]) & (C_locus <= C_range[-1])
    ax.plot(C_locus[on_plot], lam_locus[on_plot], "-", color="black",
            linewidth=2.0, zorder=4,
            label=r"Self-consistent locus $C = C_{\mathrm{model}}(\lambda_W)$")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    # Mark the empirical anchor point (C ~ 0.35, lambda_W ~ 0.68)
    import matplotlib.patheffects as pe
    ax.plot(0.35, 0.68, "o", color=PALETTE["secondary"], markersize=8,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.annotate(
        r"Empirical anchor ($C=0.35, \lambda_W=0.68$)",
        xy=(0.35, 0.68), xytext=(0.42, 0.85),
        fontsize=11, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

    cbar = fig.colorbar(cf, ax=ax, label=r"Critical threshold $\sigma^*$")
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(r"Reproductive cost $C$")
    ax.set_ylabel(r"Within-group reward $\lambda_W$")

    out = FIGURE_DIR / "figS21_bivariate_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 11: Equilibrium signaling intensity p*(sigma)
# =======================================================================

def figure11():
    """Equilibrium signaling intensity p*(sigma): sigma governs intensity, not
    binary emergence. At the empirical anchor lambda_W = 0.68 the interior
    equilibrium within-group builder frequency p* rises with environmental
    uncertainty and then saturates; the saturation level depends on the
    (archaeologically unobservable) assortment F, shown for two values."""
    from signaling.price_equation import equilibrium_intensity

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sigma_range = np.linspace(0.01, 0.95, 80)
    lam_W = 0.68
    for i, F in enumerate([0.5, 0.65]):
        ps = np.array([equilibrium_intensity(float(s), lam_W, F) for s in sigma_range])
        ax.plot(sigma_range, ps, linewidth=2.4, color=PAPER_COLORS[i], label=rf"$F = {F}$")
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Equilibrium signaling intensity $p^*(\sigma)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11, title=r"assortment $F$")

    out = FIGURE_DIR / "figS1_intensity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure 12: War-avoidance decomposition
# =======================================================================

def figure12():
    """War-avoidance: scarcity-scaled mutual deterrence. The conflict (war) cost
    W(sigma) = w0*sigma is reduced on build-build dyads, which settle by mutual
    assessment (reduction r_bb), so a builder among a fraction phi of building
    neighbors retains more survival than a non-builder. The resulting advantage
    W(sigma)*phi*r_bb rises with scarcity (sigma) and coordination (phi)."""
    from signaling.calibration import DEFAULT_WAR_COST
    from signaling.layer2 import derived_conflict_reduction

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    sigma_range = np.linspace(0.0, 0.95, 80)
    w0 = DEFAULT_WAR_COST
    r_bb = float(derived_conflict_reduction(10.0, 10.0))  # ~0.43 at equilibrium M_g

    # (a) Conflict-survival factor K: builder (phi=1, build-build) vs non-builder.
    ax = axes[0]
    K_non = 1.0 - w0 * sigma_range
    K_build = 1.0 - w0 * sigma_range * (1.0 - 1.0 * r_bb)
    ax.plot(sigma_range, K_build, linewidth=2.4, color=PAPER_COLORS[0],
            label=r"builder ($\phi=1$): $1 - W(\sigma)(1-\phi r_{bb})$")
    ax.plot(sigma_range, K_non, linewidth=2.4, color=PAPER_COLORS[1], linestyle="--",
            label=r"non-builder: $1 - W(\sigma)$")
    ax.set_xlabel(r"Environmental uncertainty $\sigma$ (scarcity)")
    ax.set_ylabel(r"Conflict-survival factor $K$")
    ax.legend(fontsize=10, loc="lower left")
    _panel_label(ax, "(a)")

    # (b) War-avoidance advantage W(sigma)*phi*r_bb vs sigma for several phi.
    ax = axes[1]
    for i, phi in enumerate([0.25, 0.5, 1.0]):
        adv = w0 * sigma_range * phi * r_bb
        ax.plot(sigma_range, adv, linewidth=2.4, color=PAPER_COLORS[i], label=rf"$\phi = {phi}$")
    ax.set_xlabel(r"Environmental uncertainty $\sigma$ (scarcity)")
    ax.set_ylabel(r"War-avoidance advantage $W(\sigma)\,\phi\,r_{bb}$")
    ax.legend(fontsize=10, title=r"coordination $\phi$")
    _panel_label(ax, "(b)")

    out = FIGURE_DIR / "figS10_war_avoidance.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Main
# =======================================================================

def main():
    print("Generating publication figures...")
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure8()
    figure9()
    figure10()
    figure11()
    figure12()
    print("Done. All figures saved to", FIGURE_DIR)


if __name__ == "__main__":
    main()
