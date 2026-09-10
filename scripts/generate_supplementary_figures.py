"""Generate supplementary figures for the Supplementary Information document.

Produces supplementary figures covering robustness analyses, sensitivity tests,
and supporting results from the multilevel costly signaling framework:

  Fig S2: Functional form robustness (cost and benefit functions)
  Fig S3: Incentive compatibility and equilibrium selection
  Fig S4: Quality distribution sensitivity
  Fig S5: Channel selection analysis
  Fig S7: Assessment noise decomposition
  Fig S8: Assessment model parameter sensitivity
  (not in the SI) War of attrition dynamics: figure_s7() kept for reference, not called by main()
  Fig S14: Network calibration and formation
  Fig S20: Price-equation selection partition (beta_0, beta_1, criterion)
  Fig S24: Feedback loop dynamics
  Fig S18: Convex-combination fitness sensitivity
  (former Fig S12 removed; see the header comment before the saddle-locus figure for the rationale)
  Fig S27: Bistable emergence saddle locus (replaces former main Fig (b))
  Fig S15: Network degree functional-form robustness

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

from signaling.layer1 import (
    ALL_CHANNELS,
    equilibrium_investment,
    fitness_gain,
    fitness_at_deviation,
    expected_monument_stock,
    expected_fitness_gain,
    cost_function_sensitivity,
    benefit_function_sensitivity,
    channel_effective_lambda,
    channel_dominance_condition,
    monument_dominance_threshold,
)
from signaling.layer2 import (
    constrained_sigma_0,
    derived_conflict_reduction,
    war_of_attrition_cost,
    compute_lambda_C,
)
from signaling.layer3 import (
    network_degree,
    vulnerability_coefficient,
    lambda_total_at_sigma,
    lambda_sigma_sweep,
    form_network_signal_based,
    form_network_baseline,
    calibrate_to_initial_vulnerability,
)
from signaling.price_equation import (
    average_signaling_benefit,
    within_group_selection,
    between_group_selection,
    fitness_advantage,
)
from signaling.calibration import (
    DEFAULT_Q_MIN,
    DEFAULT_Q_MAX,
    DEFAULT_N,
    DEFAULT_GAMMA,
    DEFAULT_K_0,
    DEFAULT_LAM_W,
    DEFAULT_LAM_C,
    DEFAULT_LAM_X,
    DEFAULT_DELTA,
    DEFAULT_RHO_PROD_FIGHT,
    RAPA_NUI,
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
# Figure S2: Functional form robustness
# =======================================================================

def figure_s1():
    """Cost and benefit function sensitivity analysis.

    Shows that the separating equilibrium's qualitative properties
    (monotonicity, positive fitness gain) are invariant to the choice
    of cost and benefit functional forms.
    """
    q = np.linspace(DEFAULT_Q_MIN + 1e-6, DEFAULT_Q_MAX, 300)
    lam = 0.5

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # (a) Cost function comparison: x*(q)
    ax = axes[0, 0]
    cost_types = [
        ("quadratic", {}, PAPER_COLORS[0], "-"),
        ("power", {"a": 2.5, "b": 1.5}, PAPER_COLORS[1], "--"),
        ("exponential", {}, PAPER_COLORS[2], ":"),
    ]
    for ctype, params, color, ls in cost_types:
        q_out, x_out = cost_function_sensitivity(q, DEFAULT_Q_MIN, lam,
                                                  cost_type=ctype, **params)
        label = ctype.capitalize()
        if ctype == "power":
            label += f" (a={params['a']}, b={params['b']})"
        ax.plot(q_out, x_out, color=color, linestyle=ls, linewidth=2.4,
                label=label)
    ax.set_xlabel(r"Capacity $q$")
    ax.set_ylabel(r"Investment $x^*(q)$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(a)")

    # (b) Benefit function comparison: x*(q)
    ax = axes[0, 1]
    benefit_types = [
        ("linear", {}, PAPER_COLORS[0], "-"),
        ("concave", {"gamma": 0.8}, PAPER_COLORS[1], "--"),
        ("concave", {"gamma": 0.5}, PAPER_COLORS[2], ":"),
    ]
    for btype, params, color, ls in benefit_types:
        q_out, x_out = benefit_function_sensitivity(q, DEFAULT_Q_MIN, lam,
                                                     benefit_type=btype,
                                                     **params)
        label = btype.capitalize()
        if btype == "concave":
            label += rf" ($\gamma = {params['gamma']}$)"
        ax.plot(q_out, x_out, color=color, linestyle=ls, linewidth=2.4,
                label=label)
    ax.set_xlabel(r"Capacity $q$")
    ax.set_ylabel(r"Investment $x^*(q)$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Fitness gain under cost function variants
    ax = axes[1, 0]
    # Only quadratic has closed-form fitness gain; show fitness gain
    # by computing w*(q) - q for the quadratic case at different lambdas
    for i, l in enumerate([0.2, 0.5, 1.0]):
        dw = fitness_gain(q, DEFAULT_Q_MIN, l)
        ax.plot(q, dw, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$\lambda_W = {l}$")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"Capacity $q$")
    ax.set_ylabel(r"Fitness gain $\Delta w(q)$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(c)")

    # (d) Summary: monotonicity check across all forms
    ax = axes[1, 1]
    # Show that x*(q) is monotonically increasing for all forms
    # by plotting the numerical derivative dx*/dq
    for ctype, params, color, ls in cost_types:
        q_out, x_out = cost_function_sensitivity(q, DEFAULT_Q_MIN, lam,
                                                  cost_type=ctype, **params)
        dx_dq = np.gradient(x_out, q_out)
        ax.plot(q_out, dx_dq, color=color, linestyle=ls, linewidth=2.4,
                label=ctype.capitalize())
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"Capacity $q$")
    ax.set_ylabel(r"$dx^*/dq$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(d)")

    out = FIGURE_DIR / "figS2_functional_form_robustness.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S3: Incentive compatibility and equilibrium selection
# =======================================================================

def figure_s2():
    """Incentive compatibility verification and pooling comparison.

    Demonstrates that the separating equilibrium x*(q) is a global
    fitness maximum for each quality type, and that pooling equilibria
    are unstable (high types profitably deviate).
    """
    lam = 0.5
    q_types = [0.3, 0.8, 1.3, 1.8]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # (a-d) Fitness landscape for 4 quality types
    for idx, q_val in enumerate(q_types):
        ax = axes[idx // 2, idx % 2]
        x_star = equilibrium_investment(q_val, DEFAULT_Q_MIN, lam)
        x_range = np.linspace(0, float(x_star) * 2.5, 300)
        w_vals = fitness_at_deviation(x_range, q_val, DEFAULT_Q_MIN, lam,
                                       q_max=DEFAULT_Q_MAX)
        w_star = fitness_at_deviation(x_star, q_val, DEFAULT_Q_MIN, lam,
                                       q_max=DEFAULT_Q_MAX)

        ax.plot(x_range, w_vals, linewidth=2.4, color=PAPER_COLORS[0])
        ax.axvline(float(x_star), color=PAPER_COLORS[1], linestyle="--",
                   linewidth=1.7, label=rf"$x^*(q) = {float(x_star):.2f}$")
        ax.plot(float(x_star), float(w_star), "o", color=PAPER_COLORS[1],
                markersize=8, zorder=5)

        # Pooling investment (mean quality)
        q_mean = (DEFAULT_Q_MIN + DEFAULT_Q_MAX) / 2
        x_pool = equilibrium_investment(q_mean, DEFAULT_Q_MIN, lam)
        ax.axvline(float(x_pool), color=PAPER_COLORS[2], linestyle=":",
                   linewidth=1.4, label=rf"$x_{{\mathrm{{pool}}}} = {float(x_pool):.2f}$")

        ax.set_xlabel(r"Investment $x$")
        ax.set_ylabel(r"Fitness $w(x, q)$")
        # Panel identity via annotation instead of title
        ax.annotate(rf"$q = {q_val}$", xy=(0.95, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=12)
        ax.legend(fontsize=9, loc="lower left")
        _panel_label(ax, f"({'abcd'[idx]})")

    # Suptitle removed: descriptive info in caption

    out = FIGURE_DIR / "figS3_incentive_compatibility.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S4: Quality distribution sensitivity
# =======================================================================

def figure_s3():
    """Sensitivity of expected monument stock and fitness gain to quality
    distribution assumptions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    lam_range = np.linspace(0.1, 1.5, 30)
    distributions = [
        ("uniform", {}, "Uniform"),
        ("truncated_normal", {"mu": 1.0, "sigma": 0.4}, "Trunc. Normal"),
        ("beta", {"a": 2.0, "b": 5.0}, "Beta(2, 5)"),
    ]

    # (a) Expected monument stock E[M_g] vs lambda
    ax = axes[0]
    for i, (dist, params, label) in enumerate(distributions):
        E_M = [expected_monument_stock(DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX,
                                        float(l), distribution=dist, **params)
               for l in lam_range]
        ax.plot(lam_range, E_M, color=PAPER_COLORS[i], linewidth=2.4, label=label)
    ax.set_xlabel(r"$\lambda_W$")
    ax.set_ylabel(r"$E[M_g]$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(a)")

    # (b) Positional within-group status differential s_W(lambda) vs the
    #     equilibrium signaling cost C_model(lambda). Under the positional
    #     (zero-sum) specification, the within-group reward is status, not a
    #     multiplicative fitness benefit; the quantity that matters for the
    #     Price within-group regression is s_W = lambda*(E[q] - q_min), which
    #     exceeds the average equilibrium cost C_model at every lambda. The
    #     positive gap s_W - C_model is what makes beta_0 > 0 (free-riding
    #     resolved at all costs). s_W is distribution-invariant in functional
    #     form; we plot it once against the three C_model curves implied by the
    #     same quality distributions used in panel (a).
    from signaling.price_equation import within_group_status_differential
    from signaling.layer1 import average_equilibrium_cost
    s_W_vals = np.array([
        within_group_status_differential(float(l), DEFAULT_Q_MIN, DEFAULT_Q_MAX)
        for l in lam_range
    ])
    # C_model here is the UNIFORM-quality equilibrium cost, paired with each
    # distribution's s_W in panel (c). For the skewed distributions the true
    # equilibrium cost is lower, so the plotted margin s_W - C_model is
    # conservative and the beta_0 > 0 conclusion holds a fortiori.
    C_model_vals = np.array([
        average_equilibrium_cost(float(l), DEFAULT_Q_MIN, DEFAULT_Q_MAX)
        for l in lam_range
    ])
    ax = axes[1]
    ax.plot(lam_range, s_W_vals, color=PAPER_COLORS[0], linewidth=2.4,
            label=r"$s_W(\lambda_W) = \lambda_W\,(E[q] - q_{\min})$")
    ax.plot(lam_range, C_model_vals, color=PAPER_COLORS[1], linewidth=2.4,
            linestyle="--", label=r"$C_{\mathrm{model}}(\lambda_W)$")
    ax.fill_between(lam_range, C_model_vals, s_W_vals, alpha=0.18,
                    color=PAPER_COLORS[0],
                    label=r"$s_W - C_{\mathrm{model}} > 0$")
    ax.set_xlabel(r"$\lambda_W$")
    ax.set_ylabel(r"$s_W(\lambda_W),\ C_{\mathrm{model}}(\lambda_W)$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Positional differential margin s_W - C_model across quality
    #     distributions. s_W depends on E[q], which shifts with the
    #     distribution; the margin stays strictly positive throughout,
    #     confirming beta_0 > 0 is robust to the quality distribution.
    ax = axes[2]
    # Distribution-specific E[q]: recompute s_W = lambda*(E[q]-q_min) per dist.
    def _mean_quality(dist, params):
        if dist == "uniform":
            return 0.5 * (DEFAULT_Q_MIN + DEFAULT_Q_MAX)
        if dist == "truncated_normal":
            # Sample-based mean of the truncated normal on [q_min, q_max].
            from scipy.stats import truncnorm
            mu = params["mu"]; sd = params["sigma"]
            a = (DEFAULT_Q_MIN - mu) / sd
            b = (DEFAULT_Q_MAX - mu) / sd
            return float(truncnorm.mean(a, b, loc=mu, scale=sd))
        if dist == "beta":
            from scipy.stats import beta as beta_dist
            a = params["a"]; b = params["b"]
            return float(DEFAULT_Q_MIN
                         + (DEFAULT_Q_MAX - DEFAULT_Q_MIN) * beta_dist.mean(a, b))
        raise ValueError(dist)

    for i, (dist, params, label) in enumerate(distributions):
        Eq = _mean_quality(dist, params)
        s_W_dist = lam_range * (Eq - DEFAULT_Q_MIN)
        margin = s_W_dist - C_model_vals
        ax.plot(lam_range, margin, color=PAPER_COLORS[i], linewidth=2.4,
                label=label)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\lambda_W$")
    ax.set_ylabel(r"$s_W(\lambda_W) - C_{\mathrm{model}}(\lambda_W)$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(c)")

    out = FIGURE_DIR / "figS4_quality_distribution.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S5: Channel selection analysis
# =======================================================================

def figure_s4():
    """Channel comparison, dominance regions, and fidelity sensitivity."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # (a) Channel investment comparison
    ax = axes[0, 0]
    q = np.linspace(DEFAULT_Q_MIN + 1e-6, DEFAULT_Q_MAX, 300)
    channel_names = []
    for i, ch in enumerate(ALL_CHANNELS):
        lam_s = channel_effective_lambda(ch, DEFAULT_LAM_W, DEFAULT_LAM_C,
                                          DEFAULT_LAM_X)
        x_star = equilibrium_investment(q, DEFAULT_Q_MIN, lam_s)
        ax.plot(q, x_star, color=PAPER_COLORS[i], linewidth=2.4,
                label=f"{ch.name.capitalize()} ($\\lambda_s = {lam_s:.2f}$)")
        channel_names.append(ch.name)
    ax.set_xlabel(r"Capacity $q$")
    ax.set_ylabel(r"Investment $x^*(q)$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=9)
    _panel_label(ax, "(a)")

    # (b) Net signaling return R_s bar chart
    ax = axes[0, 1]
    ranking = channel_dominance_condition(ALL_CHANNELS, DEFAULT_LAM_W,
                                          DEFAULT_LAM_C, DEFAULT_LAM_X,
                                          windfall_prob=0.3)
    names = [r[0] for r in ranking]
    R_vals = [r[3] for r in ranking]
    lam_vals = [r[1] for r in ranking]
    rho_vals = [r[2] for r in ranking]
    colors = [PAPER_COLORS[channel_names.index(n)] for n in names]
    bars = ax.barh(range(len(names)), R_vals, color=colors, alpha=0.8,
                   edgecolor="black", linewidth=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.capitalize() for n in names], fontsize=11)
    ax.set_xlabel(r"Net return $\bar{V}_s = \lambda_s \rho_s \bar{A} - f_s$")
    # Title removed: descriptive info in caption

    # Annotate with lambda and rho
    for i, (bar, l, r) in enumerate(zip(bars, lam_vals, rho_vals)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                rf"$\lambda$={l:.2f}, $\rho$={r:.2f}",
                va="center", fontsize=9)
    _panel_label(ax, "(b)")

    # (c) Monument dominance region (lam_C vs lam_X)
    ax = axes[1, 0]
    lam_C_range = np.linspace(0, 0.5, 50)
    lam_X_range = np.linspace(0, 0.5, 50)
    dom_grid = monument_dominance_threshold(DEFAULT_LAM_W, lam_C_range,
                                             lam_X_range, windfall_prob=0.0)
    extent = [0, 0.5, 0, 0.5]
    from matplotlib.colors import ListedColormap
    cmap_dom = ListedColormap(["#f4a582", "#92c5de"])
    ax.imshow(dom_grid.astype(float), origin="lower", extent=extent,
              aspect="auto", cmap=cmap_dom, vmin=0, vmax=1)
    ax.plot(DEFAULT_LAM_C, DEFAULT_LAM_X, "k*", markersize=12, zorder=5,
            label="Default")
    ax.set_xlabel(r"$\tilde{\lambda}_C$ (competitive)")
    ax.set_ylabel(r"$\tilde{\lambda}_X$ (cooperative)")
    # Title removed: descriptive info in caption
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color="#92c5de", label="Monument dominates")
    red_patch = mpatches.Patch(color="#f4a582", label="Alternative dominates")
    ax.legend(handles=[blue_patch, red_patch], fontsize=10, loc="upper right")
    _panel_label(ax, "(c)")

    # (d) Same with windfall = 0.3
    ax = axes[1, 1]
    dom_grid_wf = monument_dominance_threshold(DEFAULT_LAM_W, lam_C_range,
                                                lam_X_range, windfall_prob=0.3)
    ax.imshow(dom_grid_wf.astype(float), origin="lower", extent=extent,
              aspect="auto", cmap=cmap_dom, vmin=0, vmax=1)
    ax.plot(DEFAULT_LAM_C, DEFAULT_LAM_X, "k*", markersize=12, zorder=5,
            label="Default")
    ax.set_xlabel(r"$\tilde{\lambda}_C$ (competitive)")
    ax.set_ylabel(r"$\tilde{\lambda}_X$ (cooperative)")
    # Title removed: descriptive info in caption
    ax.legend(handles=[blue_patch, red_patch], fontsize=10, loc="upper right")
    _panel_label(ax, "(d)")

    out = FIGURE_DIR / "figS5_channel_selection.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S7: Assessment noise decomposition
# =======================================================================

def figure_s5():
    """Constrained sigma_0 decomposition into deception, lag, and mismatch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # (a) Noise components vs depreciation rate delta
    ax = axes[0]
    delta_range = np.linspace(0.02, 0.5, 50)
    dec_arr, lag_arr, mis_arr, total_arr = [], [], [], []
    for d in delta_range:
        result = constrained_sigma_0(delta=float(d))
        dec_arr.append(result["sigma_deception_sq"])
        lag_arr.append(result["sigma_lag_sq"])
        mis_arr.append(result["sigma_mismatch_sq"])
        total_arr.append(result["sigma_0_sq"])

    dec_arr = np.array(dec_arr)
    lag_arr = np.array(lag_arr)
    mis_arr = np.array(mis_arr)
    total_arr = np.array(total_arr)

    ax.fill_between(delta_range, 0, dec_arr, alpha=0.6, color=PAPER_COLORS[0],
                    label="Deception")
    ax.fill_between(delta_range, dec_arr, dec_arr + lag_arr, alpha=0.6,
                    color=PAPER_COLORS[1], label="Lag")
    ax.fill_between(delta_range, dec_arr + lag_arr,
                    dec_arr + lag_arr + mis_arr, alpha=0.6,
                    color=PAPER_COLORS[2], label="Mismatch")
    ax.plot(delta_range, total_arr, "k-", linewidth=2.4, label=r"Total $\sigma_0^2$")
    ax.axvline(DEFAULT_DELTA, color="gray", linestyle="--", linewidth=1.2,
               label=rf"Default $\delta = {DEFAULT_DELTA}$")
    ax.set_xlabel(r"Depreciation rate $\delta$")
    ax.set_ylabel(r"Variance $\sigma^2$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=9, loc="upper right")
    _panel_label(ax, "(a)")

    # (b) sigma_0 vs productive-fighting correlation
    ax = axes[1]
    rho_pf_range = np.linspace(0.3, 1.0, 50)
    sigma_0_vals = []
    for rpf in rho_pf_range:
        result = constrained_sigma_0(rho_prod_fight=float(rpf))
        sigma_0_vals.append(result["sigma_0"])
    ax.plot(rho_pf_range, sigma_0_vals, linewidth=2.4, color=PAPER_COLORS[0])
    ax.axvline(DEFAULT_RHO_PROD_FIGHT, color="gray", linestyle="--",
               linewidth=1.2,
               label=rf"Default $\rho_{{\mathrm{{pf}}}} = {DEFAULT_RHO_PROD_FIGHT}$")
    ax.set_xlabel(r"$\rho_{\mathrm{prod-fight}}$")
    ax.set_ylabel(r"$\sigma_0$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Component bar chart at default parameters
    ax = axes[2]
    default_result = constrained_sigma_0()
    components = ["Deception", "Lag", "Mismatch"]
    variances = [
        default_result["sigma_deception_sq"],
        default_result["sigma_lag_sq"],
        default_result["sigma_mismatch_sq"],
    ]
    bars = ax.bar(components, variances, color=[PAPER_COLORS[0], PAPER_COLORS[1], PAPER_COLORS[2]],
                  alpha=0.8, edgecolor="black", linewidth=0.7)
    ax.set_ylabel(r"Variance $\sigma^2$")
    ax.annotate(rf"$\sigma_0 = {default_result['sigma_0']:.2f}$",
                xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top",
                fontsize=11)

    # Annotate with percentage
    total_var = sum(variances)
    for bar, v in zip(bars, variances):
        pct = 100 * v / total_var if total_var > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{pct:.0f}%", ha="center", fontsize=10)
    _panel_label(ax, "(c)")

    out = FIGURE_DIR / "figS7_noise_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S8: Assessment model parameter sensitivity
# =======================================================================

def figure_s6():
    """Sensitivity of conflict reduction r to Layer 2 parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    M_range = np.linspace(0.5, 20, 100)

    # (a) Sensitivity to sigma_0 (baseline noise)
    ax = axes[0, 0]
    for i, s0 in enumerate([0.5, 1.0, 2.0]):
        r_vals = [derived_conflict_reduction(float(m), float(m), sigma_0=s0)
                  for m in M_range]
        ax.plot(M_range, r_vals, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$\sigma_0 = {s0}$")
    ax.set_xlabel(r"Monument stock $M$ (symmetric)")
    ax.set_ylabel(r"Conflict reduction $r$")
    # Title removed: descriptive info in caption
    ax.axhline(0.0, color='gray', linewidth=0.7, linestyle=':')
    ax.set_ylim(-1.3, 1.05)
    ax.legend(fontsize=10)
    _panel_label(ax, "(a)")

    # (b) Sensitivity to kappa (information gain)
    ax = axes[0, 1]
    for i, kap in enumerate([0.0, 0.1, 0.3]):
        r_vals = [derived_conflict_reduction(float(m), float(m), kappa=kap)
                  for m in M_range]
        ax.plot(M_range, r_vals, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$\kappa = {kap}$")
    ax.set_xlabel(r"Monument stock $M$ (symmetric)")
    ax.set_ylabel(r"Conflict reduction $r$")
    # Title removed: descriptive info in caption
    ax.axhline(0.0, color='gray', linewidth=0.7, linestyle=':')
    ax.set_ylim(-1.3, 1.05)
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Sensitivity to beta_conflict (absolute deterrence)
    ax = axes[1, 0]
    for i, beta_c in enumerate([0.0, 0.1, 0.3]):
        r_vals = [derived_conflict_reduction(float(m), float(m), beta=beta_c)
                  for m in M_range]
        ax.plot(M_range, r_vals, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$\beta_d = {beta_c}$")
    ax.set_xlabel(r"Monument stock $M$ (symmetric)")
    ax.set_ylabel(r"Conflict reduction $r$")
    # Title removed: descriptive info in caption
    ax.axhline(0.0, color='gray', linewidth=0.7, linestyle=':')
    ax.set_ylim(-1.3, 1.05)
    ax.legend(fontsize=10)
    _panel_label(ax, "(c)")

    # (d) War-avoidance settlement reduction r_bb(M_g), symmetric and
    #     asymmetric. r_bb is the sigma-scaled mutual war-avoidance reduction
    #     derived in Layer 2; there is no exogenous constant-r assumption to
    #     compare against. At the symmetric anchor M_g = M_h = 10 the derived
    #     reduction is r_bb ~ 0.43.
    ax = axes[1, 1]
    M_asym = np.linspace(0.5, 20, 100)
    # Symmetric case
    r_symmetric = [derived_conflict_reduction(float(m), float(m))
                   for m in M_asym]
    # Asymmetric: M_h = 5
    r_asymmetric = [derived_conflict_reduction(float(m), 5.0)
                    for m in M_asym]
    ax.plot(M_asym, r_symmetric, linewidth=2.4, color=PAPER_COLORS[0],
            label=r"Symmetric ($M_g = M_h$)")
    ax.plot(M_asym, r_asymmetric, linewidth=2.4, color=PAPER_COLORS[1],
            linestyle="--", label=r"Asymmetric ($M_h = 5$)")
    r_bb_anchor = float(derived_conflict_reduction(10.0, 10.0))
    ax.scatter([10.0], [r_bb_anchor], color=PAPER_COLORS[0], s=45, zorder=5)
    ax.annotate(rf"$r_{{bb}}(10) \approx {r_bb_anchor:.2f}$",
                xy=(10.0, r_bb_anchor),
                xytext=(11.0, r_bb_anchor - 0.12), fontsize=10,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="gray"))
    ax.set_xlabel(r"Monument stock $M_g$")
    ax.set_ylabel(r"War-avoidance settlement reduction $r_{bb}$")
    # Title removed: descriptive info in caption
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    _panel_label(ax, "(d)")

    out = FIGURE_DIR / "figS8_assessment_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# War of attrition dynamics (not in the SI; figure_s7() kept for reference, not called by main())
# =======================================================================

def figure_s7():
    """War of attrition: duration, win probability, and payoffs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    M_g_range = np.linspace(0.5, 20, 100)
    M_h_values = [2.0, 5.0, 10.0]

    # (a) Expected contest duration
    ax = axes[0]
    for i, Mh in enumerate(M_h_values):
        durations = [war_of_attrition_cost(float(mg), Mh)["duration"]
                     for mg in M_g_range]
        ax.plot(M_g_range, durations, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$M_h = {Mh}$")
    ax.set_xlabel(r"Monument stock $M_g$")
    ax.set_ylabel("Expected duration")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(a)")

    # (b) Win probability for group g
    ax = axes[1]
    for i, Mh in enumerate(M_h_values):
        p_wins = [war_of_attrition_cost(float(mg), Mh)["p_g_wins"]
                  for mg in M_g_range]
        ax.plot(M_g_range, p_wins, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$M_h = {Mh}$")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Monument stock $M_g$")
    ax.set_ylabel(r"$P(g \mathrm{\ wins})$")
    # Title removed: descriptive info in caption
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Net payoff to group g
    ax = axes[2]
    for i, Mh in enumerate(M_h_values):
        payoffs = [war_of_attrition_cost(float(mg), Mh)["payoff_g"]
                   for mg in M_g_range]
        ax.plot(M_g_range, payoffs, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$M_h = {Mh}$")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel(r"Monument stock $M_g$")
    ax.set_ylabel("Net payoff to group $g$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(c)")

    out = FIGURE_DIR / "reference_war_of_attrition.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S14: Network calibration and formation
# =======================================================================

def figure_s8():
    """Network calibration to initial model vulnerability parameters and
    comparison of signal-based vs. baseline network structure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # (a) Vulnerability curves: derived vs initial model assumptions
    ax = axes[0, 0]
    k_plot = np.linspace(0, 12, 300)
    alpha_vals = vulnerability_coefficient(k_plot, DEFAULT_GAMMA)
    ax.plot(k_plot, alpha_vals, linewidth=2.4, color=PAPER_COLORS[0],
            label=r"$\alpha(k) = 1/(1+\gamma k)$")
    ax.axhline(0.30, color=PAPER_COLORS[1], linestyle="--", linewidth=1.4,
               label=r"Initial $\alpha = 0.30$")
    ax.axhline(0.90, color=PAPER_COLORS[2], linestyle="--", linewidth=1.4,
               label=r"Initial $\beta = 0.90$")

    # Derived values ON the plotted gamma = 0.30 curve, at this section's
    # illustrative lambda_W = 0.5 (alpha at the signal-derived degree, beta
    # at the baseline degree k_0). The exact initial-model recovery re-solves
    # gamma ~ 0.22 and therefore sits OFF this curve; it is shown as open
    # markers so the two conventions stay distinct (filled markers on the
    # curve would misread as the prose values 0.34/0.87).
    M_e_s8 = float(expected_monument_stock(
        DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX, 0.5))
    k_sig_s8 = float(network_degree(M_e_s8))
    a_eff_s8 = float(vulnerability_coefficient(k_sig_s8, DEFAULT_GAMMA))
    b_eff_s8 = float(vulnerability_coefficient(DEFAULT_K_0, DEFAULT_GAMMA))
    ax.plot(k_sig_s8, a_eff_s8, "o", color=PAPER_COLORS[0], markersize=8,
            zorder=5,
            label=rf"Derived $\alpha_{{\mathrm{{eff}}}} = {a_eff_s8:.2f}$ ($\gamma = 0.30$)")
    ax.plot(DEFAULT_K_0, b_eff_s8, "s", color=PAPER_COLORS[0], markersize=8,
            zorder=5,
            label=rf"Derived $\beta_{{\mathrm{{eff}}}} = {b_eff_s8:.2f}$ ($\gamma = 0.30$)")
    try:
        cal = calibrate_to_initial_vulnerability()
        ax.plot(cal["k_signal"], cal["alpha_achieved"], "o", mfc="none",
                color=PAPER_COLORS[3], markersize=8, zorder=5,
                label=rf"Recovery $\gamma \approx {cal['gamma']:.2f}$: $\alpha = {cal['alpha_achieved']:.2f}$, $\beta = {cal['beta_achieved']:.2f}$")
        ax.plot(cal["k_nonsignal"], cal["beta_achieved"], "s", mfc="none",
                color=PAPER_COLORS[3], markersize=8, zorder=5)
    except Exception:
        pass

    ax.set_xlabel(r"Network degree $k$")
    ax.set_ylabel(r"Vulnerability $\alpha(k)$")
    # Title removed: descriptive info in caption
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    _panel_label(ax, "(a)")

    # (b) Network degree sensitivity to gamma
    ax = axes[0, 1]
    M_vals = np.linspace(0, 20, 200)
    for i, gam in enumerate([0.15, 0.30, 0.50]):
        k_vals = network_degree(M_vals)
        alpha_at_k = vulnerability_coefficient(k_vals, gam)
        ax.plot(M_vals, alpha_at_k, color=PAPER_COLORS[i], linewidth=2.4,
                label=rf"$\gamma = {gam}$")
    ax.axhline(0.30, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Monument stock $M_g$")
    ax.set_ylabel(r"Vulnerability $\alpha(k(M_g))$")
    # Title removed: descriptive info in caption
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Signal-based network: degree distribution
    ax = axes[1, 0]
    np.random.seed(42)
    n_agents = 60
    qualities = np.random.uniform(DEFAULT_Q_MIN, DEFAULT_Q_MAX, n_agents)
    investments = np.asarray(equilibrium_investment(qualities, DEFAULT_Q_MIN, 0.5))
    G_signal = form_network_signal_based(qualities, investments, rho=0.5,
                                          threshold_quantile=0.5, seed=42)
    G_baseline = form_network_baseline(n_agents, p_connect=0.15, seed=42)

    deg_signal = [d for _, d in G_signal.degree()]
    deg_baseline = [d for _, d in G_baseline.degree()]

    bins = np.arange(0, max(max(deg_signal), max(deg_baseline)) + 2) - 0.5
    ax.hist(deg_signal, bins=bins, alpha=0.6, color=PAPER_COLORS[0],
            label=f"Signal-based (mean={np.mean(deg_signal):.1f})",
            density=True)
    ax.hist(deg_baseline, bins=bins, alpha=0.6, color=PAPER_COLORS[1],
            label=f"Baseline (mean={np.mean(deg_baseline):.1f})",
            density=True)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    _panel_label(ax, "(c)")

    # (d) Degree heterogeneity: investment predicts degree in signal-based network
    ax = axes[1, 1]
    node_inv = [G_signal.nodes[n].get("investment", 0) for n in G_signal.nodes()]
    node_deg = [G_signal.degree(n) for n in G_signal.nodes()]
    ax.scatter(node_inv, node_deg, alpha=0.6, color=PAPER_COLORS[0],
               edgecolors="black", linewidth=0.7, s=40)
    if len(node_inv) > 2:
        z = np.polyfit(node_inv, node_deg, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(node_inv), max(node_inv), 50)
        corr = np.corrcoef(node_inv, node_deg)[0, 1]
        ax.plot(x_fit, p(x_fit), "--", color="gray", linewidth=1.7,
                label=rf"$r = {corr:.2f}$")
        ax.legend(fontsize=10)
    ax.set_xlabel("Node investment $x^*_i$")
    ax.set_ylabel("Network degree $k_i$")
    ax.set_ylim(-0.5, max(node_deg) + 2)
    _panel_label(ax, "(d)")

    out = FIGURE_DIR / "figS14_network_calibration.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S20: Selection term decomposition
# =======================================================================

def figure_s9():
    """Price-equation selection partition across sigma at the empirical anchor.

    Decomposes selection on the signaling trait into the within-group regression
    coefficient beta_0 = w_S - w_N and the between-group regression coefficient
    beta_1 = dW_g/dp, evaluated at lambda_W = 0.68 and the emergence (rare-builder)
    composition p -> 0. Under the positional status specification beta_0 is strictly
    positive: honest signalers out-reproduce free-riders within the group, so the
    free-rider problem is resolved at all costs (the sign reversal of Hamilton
    1975, where a costly trait always loses within-group selection). beta_1 is a
    reinforcement term whose sign at small p is sensitive to the network
    functional form (positive here under the Michaelis-Menten reference). The
    emergence criterion (Hamilton 1975 Eq. 3, large-n limit) is positive across
    the full relatedness range F, so favorability does not hinge on the
    archaeologically unobservable F.
    """
    from signaling.price_equation import (
        within_group_regression,
        between_group_regression,
        emergence_criterion,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    sigma_range = np.linspace(0.01, 0.8, 100)
    lambda_W = 0.68      # empirical anchor (Erasmus 1965)
    p_rep = 1e-4         # emergence (rare-builder) composition

    beta_0 = np.array([within_group_regression(p_rep, float(s), lambda_W)
                       for s in sigma_range])
    beta_1 = np.array([between_group_regression(p_rep, float(s), lambda_W)
                       for s in sigma_range])

    # (a) Within-group regression beta_0 (strictly positive)
    ax = axes[0]
    ax.plot(sigma_range, beta_0, linewidth=2.4, color=PAPER_COLORS[0],
            label=r"$\beta_0 = w_S - w_N$")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.fill_between(sigma_range, 0, beta_0, where=beta_0 > 0, alpha=0.15,
                    color=PAPER_COLORS[0])
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"Within-group regression $\beta_0$")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    _panel_label(ax, "(a)")

    # (b) Between-group regression beta_1 (reinforcement term)
    ax = axes[1]
    ax.plot(sigma_range, beta_1, linewidth=2.4, color=PAPER_COLORS[1],
            label=r"$\beta_1 = dW_g/dp$")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"Between-group regression $\beta_1$")
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Emergence criterion (Hamilton 1975 Eq. 3, large-n) at three assortment
    #     weights F, at the emergence composition. It is positive across all F and
    #     sigma: favorability does not hinge on the unobservable relatedness F.
    ax = axes[2]
    for i, F in enumerate([0.2, 0.5, 0.8]):
        crit = np.array([emergence_criterion(float(s), lambda_W, F, p=p_rep)
                         for s in sigma_range])
        ax.plot(sigma_range, crit, linewidth=2.4, color=PAPER_COLORS[i],
                label=rf"$F = {F}$")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"Emergence criterion (selection on signaling)")
    ax.legend(fontsize=10, title=r"$F$ (assortment)")
    _panel_label(ax, "(c)")

    out = FIGURE_DIR / "figS20_selection_decomposition.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S24: Feedback loop dynamics
# =======================================================================

def figure_s10():
    """Full lambda-sigma feedback sweep with component breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    sigma_range = np.linspace(0.01, 0.8, 80)
    lambda_W = 0.3

    def _lambda_C_wrapper(M_g):
        return compute_lambda_C(M_g)

    sweep = lambda_sigma_sweep(sigma_range, lambda_W,
                                compute_lambda_C_func=_lambda_C_wrapper)

    # (a) Total lambda and components
    ax = axes[0, 0]
    ax.plot(sweep["sigma"], sweep["lambda_total"], linewidth=2.4,
            color="black", label=r"$\lambda_{\mathrm{total}}$")
    ax.plot(sweep["sigma"], [lambda_W]*len(sweep["sigma"]), linewidth=1.7,
            color=PAPER_COLORS[0], linestyle="--", label=r"$\lambda_W$")
    ax.plot(sweep["sigma"], sweep["lambda_C"], linewidth=1.7,
            color=PAPER_COLORS[1], linestyle="--", label=r"$\lambda_C$")
    ax.plot(sweep["sigma"], sweep["lambda_X"], linewidth=1.7,
            color=PAPER_COLORS[2], linestyle="--", label=r"$\lambda_X$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\lambda$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(a)")

    # (b) Positional within-group status differential s_W(lambda(sigma)) vs
    #     the equilibrium signaling cost C_model(lambda(sigma)). Under the
    #     positional specification, the within-group reward is zero-sum status, so the
    #     relevant quantity is the status differential s_W, not a multiplicative
    #     fitness benefit B. s_W tracks lambda_total(sigma) through the feedback
    #     loop and stays above C_model throughout, keeping beta_0 > 0.
    from signaling.price_equation import within_group_status_differential
    from signaling.layer1 import average_equilibrium_cost
    ax = axes[0, 1]
    s_W_vals = [within_group_status_differential(lt, DEFAULT_Q_MIN, DEFAULT_Q_MAX)
                for lt in sweep["lambda_total"]]
    C_model_vals = [average_equilibrium_cost(lt, DEFAULT_Q_MIN, DEFAULT_Q_MAX)
                    for lt in sweep["lambda_total"]]
    ax.plot(sweep["sigma"], s_W_vals, linewidth=2.4, color=PAPER_COLORS[0],
            label=r"$s_W(\lambda(\sigma))$")
    ax.plot(sweep["sigma"], C_model_vals, linewidth=2.4, color=PAPER_COLORS[1],
            linestyle="--", label=r"$C_{\mathrm{model}}(\lambda(\sigma))$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Positional status differential")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(b)")

    # (c) Effective alpha(sigma) and beta_eff
    ax = axes[1, 0]
    # Compute alpha_eff at each sigma
    alpha_eff_vals = []
    for s in sigma_range:
        eq = lambda_total_at_sigma(float(s), lambda_W)
        alpha_eff_vals.append(eq["alpha_eff"])
    beta_eff = float(vulnerability_coefficient(DEFAULT_K_0, DEFAULT_GAMMA))

    ax.plot(sigma_range, alpha_eff_vals, linewidth=2.4, color=PAPER_COLORS[0],
            label=r"$\alpha_{\mathrm{eff}}(\sigma)$")
    ax.axhline(beta_eff, color=PAPER_COLORS[1], linestyle="--", linewidth=1.7,
               label=rf"$\beta_{{\mathrm{{eff}}}} = {beta_eff:.2f}$")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Vulnerability")
    # Title removed: descriptive info in caption
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    _panel_label(ax, "(c)")

    # (d) Fitness advantage with fully feedback-derived parameters. Both the
    #     equilibrium reward lambda_total(sigma) and the between-group conflict
    #     reduction are derived endogenously: r is the sigma-scaled mutual
    #     war-avoidance settlement reduction r_bb evaluated at the equilibrium
    #     monument stock M_g(sigma) (symmetric, M_g = M_h). There is no
    #     exogenous constant-r assumption. The within-group reward enters the
    #     group fitness positionally through lambda_total and the quality range
    #     (canonical fitness_advantage signature); the within-group
    #     contribution to threshold placement is carried by beta_0 (Fig. S9),
    #     not by a separate multiplicative benefit term.
    from signaling.layer2 import derived_conflict_reduction
    ax = axes[1, 1]
    adv_ext = []
    for idx, s in enumerate(sigma_range):
        eq = lambda_total_at_sigma(float(s), lambda_W)
        ae = eq["alpha_eff"]
        lam_total = eq["lambda_total"]
        M_g = eq["M_g"]
        r_bb = float(derived_conflict_reduction(M_g, M_g))
        adv_ext.append(fitness_advantage(float(s), lam_total, ae, beta_eff,
                                          r_bb))

    adv_ext = np.array(adv_ext)
    ax.plot(sigma_range, adv_ext, linewidth=2.4, color=PAPER_COLORS[0],
            label="Feedback-derived ($r_{bb}$, positional)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.fill_between(sigma_range, 0, adv_ext,
                    where=adv_ext > 0, alpha=0.1, color=PAPER_COLORS[0])
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"Fitness advantage $\Delta w$")
    # Title removed: descriptive info in caption
    ax.legend(fontsize=10)
    _panel_label(ax, "(d)")

    out = FIGURE_DIR / "figS24_feedback_dynamics.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S18: Convex-combination fitness sensitivity
# =======================================================================

def figure_s11():
    """Parameterized sensitivity to the multiplicative-vs-additive choice.

    sigma*(omega) for omega in [0, 1], where omega = 0 is pure multiplicative
    and omega = 1 is pure additive. Under the framework parameterization, the
    framework corrects sigma* upward: at the empirical anchor lambda_W = 0.68
    the multiplicative threshold is sigma* ~ 0.48, just below the MLS baseline
    sigma*_base ~ 0.50, and it rises with omega to sit at or above the baseline
    in the additive limit. The qualitative robustness claim is therefore that
    sigma* stays near the baseline across the full omega range, not that it
    falls far below it.
    """
    from signaling.price_equation import (
        sigma_star_self_consistent,
        sigma_star_vs_omega,
        critical_threshold_sigma_star,
        fitness_advantage,
        average_signaling_benefit as B_of_lam,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    omegas = np.linspace(0.0, 1.0, 41)

    # (a) sigma*(omega) at the empirical anchor lambda_W = 0.68
    ax = axes[0, 0]
    sweep_anchor = sigma_star_vs_omega(omegas, lambda_W=0.68)
    sigma_base = sweep_anchor["sigma_star_base"]
    s0 = float(sweep_anchor["sigma_star"][0])
    s1 = float(sweep_anchor["sigma_star"][-1])
    ax.plot(sweep_anchor["omega"], sweep_anchor["sigma_star"],
            linewidth=2.2, color=PAPER_COLORS[0], label=r"$\sigma^*(\omega)$")
    ax.axhline(sigma_base, color="gray", linestyle="--", linewidth=1.7,
               label=rf"Baseline $\sigma^*_{{\mathrm{{base}}}} = {sigma_base:.2f}$")
    ax.scatter([0.0], [s0], color=PAPER_COLORS[0], zorder=5, s=40)
    ax.annotate(rf"$\omega=0$: multiplicative ($\sigma^* = {s0:.2f}$)",
                xy=(0.0, s0),
                xytext=(0.06, s0 - 0.10),
                fontsize=11, arrowprops=dict(arrowstyle="-", lw=0.6, color="gray"))
    ax.scatter([1.0], [s1], color=PAPER_COLORS[0], zorder=5, s=40)
    ax.annotate(rf"$\omega=1$: additive ($\sigma^* = {s1:.2f}$)",
                xy=(1.0, s1),
                xytext=(0.40, s1 + 0.06),
                fontsize=11, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="gray"))
    ax.set_xlabel(r"Convex weight $\omega$")
    ax.set_ylabel(r"Critical threshold $\sigma^*$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=11, loc="lower right")
    _panel_label(ax, "(a)")

    # (b) sigma*(omega) at three lambda_W values
    ax = axes[0, 1]
    for i, lam_W in enumerate([0.30, 0.50, 0.68]):
        sweep = sigma_star_vs_omega(omegas, lambda_W=lam_W)
        ax.plot(sweep["omega"], sweep["sigma_star"],
                linewidth=2.4, color=PAPER_COLORS[i],
                label=rf"$\lambda_W = {lam_W:.2f}$")
    ax.axhline(sigma_base, color="gray", linestyle="--", linewidth=1.4,
               label=r"Baseline $\sigma^*_{\mathrm{base}}$")
    ax.set_xlabel(r"Convex weight $\omega$")
    ax.set_ylabel(r"$\sigma^*(\omega)$")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=11)
    _panel_label(ax, "(b)")

    # (c) Fitness-advantage curves Delta_w(sigma; omega) at lambda_W = 0.68.
    ax = axes[1, 0]
    # Pre-compute the equilibrium state at the multiplicative anchor; use the
    # same lam_total, alpha_eff, beta_eff, r for all curves so the only thing
    # changing is omega. The canonical fitness_advantage takes the equilibrium
    # reward lam_total as its second argument (the within-group reward enters
    # positionally through lam_total and the quality range, not through a
    # separate multiplicative benefit term), and derives the war-avoidance
    # reduction r_bb internally to the self-consistent solve.
    res_anchor = sigma_star_self_consistent(lambda_W=0.68, mode="multiplicative")
    lam_total_anchor = res_anchor["lambda_total"]
    alpha_eff_anchor = res_anchor["alpha_eff"]
    beta_eff_anchor = res_anchor["beta_eff"]
    r_anchor = res_anchor["r"]   # derived war-avoidance reduction r_bb

    sigma_range = np.linspace(0.001, 0.62, 200)
    for i, w in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        dw = [
            fitness_advantage(
                float(s), lam_total_anchor, alpha_eff_anchor, beta_eff_anchor,
                r_anchor, mode="convex", omega=w,
            )
            for s in sigma_range
        ]
        ax.plot(sigma_range, dw, linewidth=2.4, color=PAPER_COLORS[i],
                label=rf"$\omega = {w}$")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Fitness advantage $\Delta w$")
    ax.set_xlim(0, 0.62)
    ax.legend(fontsize=11)
    _panel_label(ax, "(c)")

    # (d) sigma_star(omega) family at the anchor, shown against the MLS
    #     baseline. sigma* rises monotonically with omega from the
    #     multiplicative value, crossing the baseline in the additive limit;
    #     the framework corrects the threshold upward across the full range.
    ax = axes[1, 1]
    sweep_anchor_dense = sigma_star_vs_omega(np.linspace(0.0, 1.0, 81), lambda_W=0.68)
    s0_dense = float(sweep_anchor_dense["sigma_star"][0])
    ax.plot(sweep_anchor_dense["omega"], sweep_anchor_dense["sigma_star"],
            linewidth=2.2, color=PAPER_COLORS[0])
    ax.fill_between(sweep_anchor_dense["omega"], 0,
                    sweep_anchor_dense["sigma_star"], alpha=0.18, color=PAPER_COLORS[0])
    ax.axhline(sigma_base, color="gray", linestyle="--", linewidth=1.4,
               label=rf"Baseline $\sigma^*_{{\mathrm{{base}}}} = {sigma_base:.2f}$")
    ax.axhline(s0_dense, color=PAPER_COLORS[1],
               linestyle=":", linewidth=1.4,
               label=rf"Multiplicative $\sigma^* = {s0_dense:.2f}$")
    ax.set_xlabel(r"Convex weight $\omega$")
    ax.set_ylabel(r"$\sigma^*(\omega)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=11, loc="lower right")
    _panel_label(ax, "(d)")

    out = FIGURE_DIR / "figS18_convex_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Former Figure S12: REMOVED (rationale in header comments below).
#
# The former Fig. S12 plotted the within-group multiplicative benefit
# B(lambda) against the equilibrium cost C_model(lambda) and shaded the
# "informational rent" B - C_model. Under the positional specification, the
# within-group reward is zero-sum status, not a multiplicative fitness
# benefit, so B(lambda) is no longer a model quantity and this figure has
# no referent. The within-group differential is now carried by the
# positional status margin s_W - C_model (Fig. S3b) and the Price
# within-group regression beta_0 > 0 (Fig. S9a). The figure is no longer
# generated; the stale PDF (figS12_B_vs_Cmodel.pdf) has been removed.
# NOTE: the .tex no longer includes this figure (reference removed).
# =======================================================================


# =======================================================================
# Figure S27: Bistable emergence saddle locus (sigma_star_inv(phi))
# =======================================================================

def figure_s15():
    """Saddle locus of bistable emergence: sigma*_inv(phi) at lambda_W = 0.68.

    Displays the curve previously shown in panel (b) of fig4_emergence.pdf,
    which was redesigned around dynamical replicator trajectories. The
    sigma*_inv(phi) curve is the saddle locus in (sigma, phi) phase space:
    above the curve the rare-builder advantage is positive and trajectories
    flow to phi = 1; below it the advantage is negative and trajectories
    flow to phi = 0. The dual representation phi*(sigma) plotted in the
    main text figure is the same locus parameterized by sigma instead of phi.
    """
    from signaling.emergence import sigma_star_invasion

    lam_W = 0.68

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    phis = np.linspace(0.005, 1.0, 80)
    sigma_inv = np.array([
        sigma_star_invasion(lam_W, float(phi), mode="multiplicative")
        for phi in phis
    ])
    finite_mask = np.isfinite(sigma_inv)

    ax.plot(phis[finite_mask], sigma_inv[finite_mask],
            color=PAPER_COLORS[0], linewidth=2.5,
            label=r"$\sigma^*_{\mathrm{inv}}(\phi)$ (saddle locus)")

    # Annotate values read directly from the recomputed curve (no hardcoded
    # numbers). For each target phi we take the nearest finite grid sample.
    def _nearest(phi_target):
        idx_fin = np.where(finite_mask)[0]
        j = idx_fin[int(np.argmin(np.abs(phis[idx_fin] - phi_target)))]
        return float(phis[j]), float(sigma_inv[j])

    # Mark the finite values quoted in the main text (phi = 0.25, 0.5, 1);
    # the main text's phi <= 0.05 entry is the left-edge divergence, which
    # has no finite marker. Evaluate sigma*_inv AT the exact target phi
    # (nearest-grid sampling would annotate phi = 0.24 where the caption
    # promises 0.25).
    from signaling.emergence import sigma_star_invasion as _ssi
    for phi_target in (0.25, 0.50, 1.0):
        _exact = float(_ssi(lambda_W=0.68, frac_signalers=phi_target))
        if np.isfinite(_exact):
            phi_q, sig_q = phi_target, _exact
        else:
            phi_q, sig_q = _nearest(phi_target)
        ax.scatter([phi_q], [sig_q], color=PAPER_COLORS[1], s=55, zorder=6,
                   edgecolors="black", linewidths=0.7)
        ax.annotate(
            rf"$\sigma^*_{{\mathrm{{inv}}}}({phi_q:.2f}) \approx {sig_q:.2f}$",
            xy=(phi_q, sig_q), xytext=(phi_q + 0.04, sig_q + 0.05),
            fontsize=11, color=PAPER_COLORS[1],
            arrowprops=dict(arrowstyle="-", lw=0.5, color=PAPER_COLORS[1]),
        )

    # Maintenance threshold at phi -> 1 (full participation), read from the
    # curve. Under the framework, this asymptote is ~0.48 (the empirical-anchor
    # maintenance threshold), not the old ~0.20.
    maintenance_sigma = float(sigma_inv[finite_mask][-1])
    ax.axhline(maintenance_sigma, color="gray", linewidth=1.2, linestyle=":",
               label=rf"Maintenance $\sigma^* \approx {maintenance_sigma:.2f}$")

    # Shade the two basins
    ax.fill_between(phis[finite_mask], sigma_inv[finite_mask], 0.95,
                    alpha=0.13, color=PAPER_COLORS[2],
                    label="Signaling absorbing")
    ax.fill_between(phis[finite_mask], 0.0, sigma_inv[finite_mask],
                    alpha=0.13, color=PAPER_COLORS[3],
                    label="Non-signaling absorbing")
    # Left of the finite-locus region the invasion threshold is infinite
    # (no finite sigma admits invasion), so the whole visible column belongs
    # to the non-signaling basin; extend the shading to the axis.
    ax.fill_between([0.0, float(phis[finite_mask][0])], 0.0, 0.95,
                    alpha=0.13, color=PAPER_COLORS[3])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel(r"Fraction $\phi$ of signaling neighbors")
    ax.set_ylabel(r"Invasion threshold $\sigma^*_{\mathrm{inv}}(\phi)$")
    # Title removed: descriptive info belongs in the caption (figure style).
    ax.legend(loc="upper right", fontsize=11)

    out = FIGURE_DIR / "figS27_bistable_emergence_saddle.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Figure S15: Network degree functional-form robustness
# =======================================================================

def figure_s16():
    """Robustness of sigma*(lambda_W) to network-degree functional form.

    Tests the network functional-form robustness claim that the Michaelis-Menten
    assumption for k(M_g) is not load-bearing. The four alternative forms
    (Michaelis-Menten reference, exponential, Hill n=2, piecewise-linear)
    are each calibrated to match exactly at the empirical M_g anchor; their
    shapes elsewhere differ. The downstream sigma*(lambda_W) curves
    nevertheless overlap closely, demonstrating that the threshold
    prediction does not depend on the functional form. Under the
    framework parameterization, sigma*(lambda_W) rises with lambda_W and
    sits near or above the MLS baseline at the empirical anchor (sigma* ~
    0.48 vs sigma*_base ~ 0.50); the framework corrects the threshold
    upward, and that correction is robust to the network functional form.
    """
    from signaling.layer1 import expected_monument_stock
    from signaling.layer3 import calibrate_alternative_network_forms
    from signaling.price_equation import (
        sigma_star_self_consistent,
        initial_model_sigma_star,
    )
    from signaling.calibration import (
        DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF,
        DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX,
    )

    sigma_base = float(initial_model_sigma_star(0.35))   # MLS baseline ~0.497

    # Anchor: equilibrium M_g at the empirical lambda_W
    lam_W_anchor = 0.68
    M_anchor = float(expected_monument_stock(
        DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX, lam_W_anchor,
    ))
    specs = calibrate_alternative_network_forms(
        M_anchor, DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    # Display names and colors
    display = {
        "michaelis_menten": ("Michaelis-Menten (reference)", PAPER_COLORS[0], "-"),
        "exponential": ("Exponential", PAPER_COLORS[1], "--"),
        "hill_n2": ("Hill (n=2)", PAPER_COLORS[2], "-."),
        "piecewise_linear": ("Piecewise linear", PAPER_COLORS[3], ":"),
    }

    # ------------------------------------------------------------------
    # Panel (a): network-degree functions k(M)
    # ------------------------------------------------------------------
    ax = axes[0]
    M_range = np.linspace(0, 30, 200)
    for name, spec in specs.items():
        label, color, ls = display[name]
        k_vals = np.array([spec.k(float(M)) for M in M_range])
        ax.plot(M_range, k_vals, color=color, linestyle=ls, linewidth=2.4,
                label=label)
    ax.axvline(M_anchor, color="gray", linestyle=":", linewidth=1.2)
    ax.text(M_anchor - 0.35, 8.7,
            rf"calibration anchor $M_e \approx {M_anchor:.1f}$",
            rotation=90, va="top", ha="right",
            fontsize=11, color="gray")
    ax.set_xlabel(r"Group monument stock $M_g$")
    ax.set_ylabel(r"Network degree $k(M_g)$")
    # Title removed: descriptive info in caption (figure style).
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, DEFAULT_K_0 + DEFAULT_K_MAX + 0.5)
    _panel_label(ax, "(a)")

    # ------------------------------------------------------------------
    # Panel (b): downstream sigma*(lambda_W) under each form
    # ------------------------------------------------------------------
    ax = axes[1]
    lam_W_range = np.linspace(0.05, 1.0, 25)
    for name, spec in specs.items():
        label, color, ls = display[name]
        sigma_stars = []
        for lam_W in lam_W_range:
            try:
                res = sigma_star_self_consistent(
                    lambda_W=float(lam_W),
                    mode="multiplicative",
                    network_degree_spec=spec,
                )
                sigma_stars.append(res["sigma_star"])
            except (ValueError, RuntimeError):
                sigma_stars.append(np.nan)
        sigma_stars = np.array(sigma_stars)
        ax.plot(lam_W_range, sigma_stars, color=color, linestyle=ls,
                linewidth=2.4, label=label)

    ax.axhline(sigma_base, color="black", linestyle="-", linewidth=1.0, alpha=0.4,
               label=rf"Standard MLS baseline $\sigma^*_{{\mathrm{{base}}}} = {sigma_base:.2f}$")
    ax.axvline(lam_W_anchor, color="gray", linestyle=":", linewidth=1.2)
    ax.text(lam_W_anchor + 0.02, 0.62,
            rf"empirical anchor $\lambda_W = {lam_W_anchor}$",
            fontsize=11, color="gray", rotation=90, va="top")

    ax.set_xlabel(r"Within-group social reward $\lambda_W$")
    ax.set_ylabel(r"Critical threshold $\sigma^*(\lambda_W)$")
    # Title removed: descriptive info in caption (figure style).
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.72)
    _panel_label(ax, "(b)")

    out = FIGURE_DIR / "figS15_network_robustness.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


# =======================================================================
# Main
# =======================================================================

def main():
    print("Generating supplementary figures...")
    figure_s1()
    figure_s2()
    figure_s3()
    figure_s4()
    figure_s5()
    figure_s6()
    # S7 and S12-S14 are not figures in the SI; figure_s7() is kept for reference only.
    figure_s8()
    figure_s9()
    figure_s10()
    figure_s11()
    figure_s15()
    figure_s16()
    print("Done. All supplementary figures saved to", FIGURE_DIR)


if __name__ == "__main__":
    main()
