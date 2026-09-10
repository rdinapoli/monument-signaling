"""Shared plotting utilities for publication figures.

All plotting functions accept an optional ax parameter for composability
into multi-panel figures. When ax is None, a new figure and axes are
created. Designed for publication-quality output suitable for journals
such as Journal of Theoretical Biology, Evolution and Human Behavior,
and Proceedings B.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from signaling.layer1 import (
    ALL_CHANNELS,
    SignalChannel,
    channel_dominance_condition,
    channel_effective_lambda,
    equilibrium_fitness,
    equilibrium_investment,
    fitness_gain,
    incremental_lambda,
    monument_dominance_threshold,
    signal_fidelity,
)


# =====================================================================
# Shared paper style: sans-serif, colorblind-safe palette, no red+green
# =====================================================================
# Drawn from the Okabe-Ito 8-colour palette (Okabe & Ito 2008), omitting
# the bluish-green entry so no figure pairs red and green. Vermillion
# (#D55E00) is retained as the warm pole because it is distinguishable
# from pure red and is the Okabe-Ito recommended warm contrast.

PAPER_COLORS: tuple[str, ...] = (
    "#0072B2",  # 0: blue              (primary)
    "#D55E00",  # 1: vermillion        (secondary, warm pole)
    "#000000",  # 2: black             (tertiary, neutral)
    "#CC79A7",  # 3: reddish purple
    "#56B4E9",  # 4: sky blue
    "#E69F00",  # 5: orange
    "#525252",  # 6: dark gray
    "#F0E442",  # 7: yellow
)

PALETTE = {
    "primary":   PAPER_COLORS[0],
    "secondary": PAPER_COLORS[1],
    "tertiary":  PAPER_COLORS[2],
    "accent":    PAPER_COLORS[3],
    "neutral":   "#525252",
    "grid":      "#E5E5E5",
}

# Sequential colormap for unidirectional quantities (survival, conflict
# probability, etc.). viridis is perceptually uniform and colorblind safe.
SEQUENTIAL_CMAP = "viridis"

# Diverging colormap for above/below-threshold quantities. RdBu_r is
# blue (low) to red (high) through near-white; safe because red and
# blue are distinguishable under all common forms of color vision.
DIVERGING_CMAP = "RdBu_r"

# Discrete two-class colormap: light gray (off) and light blue (on).
# Used in dominance / phase-space plots in place of the earlier
# gray + light-green pair.
TWO_CLASS_COLORS: tuple[str, str] = ("#D9D9D9", "#9ECAE1")


def setup_paper_style() -> None:
    """Apply the project-wide matplotlib style.

    Centralizes the rcParams previously duplicated across the figure
    generation scripts. Sans-serif throughout (DejaVu Sans, bundled
    with matplotlib, so output is machine-independent); sans-serif math via the
    matplotlib ``dejavusans`` math fontset; top and right spines off;
    moderately larger fonts and line weights so that figures remain
    legible after the journal scales them down to column width.

    Idempotent: safe to call once per script at import time.
    """
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "sans-serif",
        # DejaVu Sans is bundled with matplotlib, so every machine renders
        # the same glyphs; an Arial-first list silently substitutes whatever
        # fontconfig offers and makes figure output machine-dependent.
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=list(PAPER_COLORS)),
        "lines.linewidth": 2.2,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _get_ax(ax: Axes | None) -> tuple[plt.Figure | None, Axes]:
    """Return (fig, ax), creating a new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return None, ax


def plot_investment_function(
    q_range: NDArray[np.float64],
    q_min: float,
    lam_values: Sequence[float],
    ax: Axes | None = None,
) -> Axes:
    r"""Plot equilibrium investment x*(q) for several lambda values.

    Parameters
    ----------
    q_range : array
        Quality values at which to evaluate x*(q).
    q_min : float
        Minimum quality.
    lam_values : sequence of float
        Lambda values to plot.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    _, ax = _get_ax(ax)
    for lam in lam_values:
        x_star = equilibrium_investment(q_range, q_min, lam)
        ax.plot(q_range, x_star, label=rf"$\lambda = {lam}$")
    ax.set_xlabel(r"Quality $q$")
    ax.set_ylabel(r"Investment $x^*(q)$")
    ax.legend()
    return ax


def plot_fitness_landscape(
    q_range: NDArray[np.float64],
    q_min: float,
    lam: float,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot equilibrium fitness w*(q) and baseline q, shading the gain.

    Parameters
    ----------
    q_range : array
        Quality values.
    q_min : float
        Minimum quality.
    lam : float
        Lambda parameter.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    _, ax = _get_ax(ax)
    w_star = equilibrium_fitness(q_range, q_min, lam)
    ax.plot(q_range, w_star, label=r"$w^*(q)$")
    ax.plot(q_range, q_range, "--", color="gray", label=r"$w = q$ (no signaling)")
    ax.fill_between(q_range, q_range, w_star, alpha=0.2, label=r"$\Delta w(q)$")
    ax.set_xlabel(r"Quality $q$")
    ax.set_ylabel("Fitness")
    ax.legend()
    return ax


def plot_fitness_gain(
    q_range: NDArray[np.float64],
    q_min: float,
    lam_values: Sequence[float],
    ax: Axes | None = None,
) -> Axes:
    r"""Plot fitness gain Delta_w(q) for several lambda values.

    Parameters
    ----------
    q_range : array
        Quality values.
    q_min : float
        Minimum quality.
    lam_values : sequence of float
        Lambda values to plot.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    _, ax = _get_ax(ax)
    for lam in lam_values:
        dw = fitness_gain(q_range, q_min, lam)
        ax.plot(q_range, dw, label=rf"$\lambda = {lam}$")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"Quality $q$")
    ax.set_ylabel(r"$\Delta w(q)$")
    ax.legend()
    return ax


# =====================================================================
# Channel selection plots
# =====================================================================


def plot_channel_comparison(
    q_range: NDArray[np.float64],
    q_min: float,
    channels: Sequence[SignalChannel],
    lam_W: float,
    lam_C: float,
    lam_X: float,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot equilibrium investment x*(q) for each channel at its effective lambda.

    Each channel produces a different investment curve because its effective
    lambda (audience-weighted sum) differs. Channels with higher effective
    lambda produce steeper investment schedules.

    Parameters
    ----------
    q_range : array
        Quality values at which to evaluate x*(q).
    q_min : float
        Minimum quality.
    channels : sequence of SignalChannel
        Channels to compare.
    lam_W, lam_C, lam_X : float
        Audience-specific lambda components.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    _, ax = _get_ax(ax)
    for ch in channels:
        lam_eff = channel_effective_lambda(ch, lam_W, lam_C, lam_X)
        if lam_eff > 0:
            x_star = equilibrium_investment(q_range, q_min, lam_eff)
            ax.plot(q_range, x_star, label=rf"{ch.name} ($\lambda_s = {lam_eff:.3f}$)")
    ax.set_xlabel(r"Quality $q$")
    ax.set_ylabel(r"Investment $x^*(q)$")
    ax.legend()
    return ax


def plot_dominance_region(
    lam_C_range: NDArray[np.float64],
    lam_X_range: NDArray[np.float64],
    lam_W: float,
    channels: Sequence[SignalChannel] | None = None,
    windfall_prob: float = 0.0,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot the region in ($\lambda_C$, $\lambda_X$) space where monuments dominate.

    Produces a heatmap where green indicates monument dominance and gray
    indicates an alternative channel dominates.

    Parameters
    ----------
    lam_C_range : array
        Values of competitive between-group lambda.
    lam_X_range : array
        Values of cooperative between-group lambda.
    lam_W : float
        Within-group lambda (fixed).
    channels : sequence of SignalChannel or None
        Channels to compare. Defaults to ALL_CHANNELS.
    windfall_prob : float
        Windfall probability for feasting.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    _, ax = _get_ax(ax)
    ch_list = list(channels) if channels is not None else ALL_CHANNELS
    grid = monument_dominance_threshold(
        lam_W, lam_C_range, lam_X_range,
        channels=ch_list, windfall_prob=windfall_prob,
    )
    # Plot with imshow; origin='lower' so lam_X increases upward
    extent = [
        float(lam_C_range[0]), float(lam_C_range[-1]),
        float(lam_X_range[0]), float(lam_X_range[-1]),
    ]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(list(TWO_CLASS_COLORS))
    ax.imshow(
        grid.astype(float), origin="lower", extent=extent,
        aspect="auto", cmap=cmap, vmin=0, vmax=1,
    )
    ax.set_xlabel(r"$\tilde{\lambda}_C$ (competitive audience return)")
    ax.set_ylabel(r"$\tilde{\lambda}_X$ (cooperative audience return)")
    # Add legend patches
    import matplotlib.patches as mpatches
    on_patch = mpatches.Patch(color=TWO_CLASS_COLORS[1], label="Monument dominates")
    off_patch = mpatches.Patch(color=TWO_CLASS_COLORS[0], label="Alternative dominates")
    ax.legend(handles=[on_patch, off_patch], loc="upper left")
    return ax


def plot_fidelity_sensitivity(
    windfall_range: NDArray[np.float64],
    lam_W: float,
    lam_C: float,
    lam_X: float,
    channels: Sequence[SignalChannel] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot channel net return V_s = lambda_s*rho_s*A_bar - f_s vs feasting windfall probability.

    Shows how increasing the windfall probability reduces feasting's net
    return while leaving other channels unchanged, potentially shifting
    the dominant channel.

    Parameters
    ----------
    windfall_range : array
        Windfall probability values to sweep.
    lam_W, lam_C, lam_X : float
        Audience-specific lambda components.
    channels : sequence of SignalChannel or None
        Channels to plot. Defaults to ALL_CHANNELS.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    from signaling.layer1 import mean_payoff_multiplier

    _, ax = _get_ax(ax)
    A_bar = mean_payoff_multiplier()
    ch_list = list(channels) if channels is not None else ALL_CHANNELS
    for ch in ch_list:
        lam_s = channel_effective_lambda(ch, lam_W, lam_C, lam_X)
        V_values = []
        for wp in windfall_range:
            # Only feast is affected by windfall
            wp_ch = float(wp) if ch.name == "feast" else 0.0
            rho = signal_fidelity(ch, windfall_prob=wp_ch)
            V_values.append(lam_s * rho * A_bar - ch.fixed_cost)
        ax.plot(windfall_range, V_values, label=ch.name)
    ax.set_xlabel("Windfall probability (feasting)")
    ax.set_ylabel(r"Net return $V_s = \lambda_s \rho_s \bar{A} - f_s$")
    ax.legend()
    return ax


# =====================================================================
# Layer 2: Intergroup assessment plots
# =====================================================================


def plot_conflict_probability_surface(
    M_range: NDArray[np.float64],
    sigma_0: float = 1.0,
    V: float = 1.0,
    D: float = 2.0,
    T_0: float = 0.5,
    beta: float = 0.1,
    kappa: float = 0.1,
    P_base: float = 0.01,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot the mutual assessment conflict probability as a heatmap.

    Displays $P_{\text{conflict}}(M_g, M_h)$ over a 2D grid. The surface
    should show high probability near the origin (no assessment basis),
    low probability along the diagonal at high investment (deterrence),
    and low probability far from the diagonal (clear asymmetry).

    Parameters
    ----------
    M_range : array
        1D array of monument stock values for both axes.
    sigma_0, V, D, T_0, beta, kappa, P_base : float
        Mutual assessment model parameters.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : Axes
    """
    from signaling.layer2 import mutual_assessment_conflict_prob

    _, ax = _get_ax(ax)
    Mg, Mh = np.meshgrid(M_range, M_range)
    P = mutual_assessment_conflict_prob(Mg, Mh, sigma_0, V, D, T_0, beta, kappa, P_base)

    extent = [float(M_range[0]), float(M_range[-1]),
              float(M_range[0]), float(M_range[-1])]
    im = ax.imshow(
        P, origin="lower", extent=extent, aspect="auto",
        cmap=SEQUENTIAL_CMAP, vmin=0, vmax=float(P_base),
    )
    ax.set_xlabel(r"$M_g$ (group $g$ monument stock)")
    ax.set_ylabel(r"$M_h$ (group $h$ monument stock)")
    plt.colorbar(im, ax=ax, label=r"$P_{\mathrm{conflict}}$")
    return ax


def plot_assessment_model_comparison(
    M_range: NDArray[np.float64],
    sigma_0: float = 1.0,
    V: float = 1.0,
    D: float = 2.0,
    T_0: float = 0.5,
    beta: float = 0.1,
    kappa: float = 0.1,
    P_base: float = 0.01,
    M_thresh: float = 2.0,
    M_scale: float = 1.0,
    M_own: float = 2.0,
    steepness: float = 2.0,
    axes: Sequence[Axes] | None = None,
) -> list[Axes]:
    r"""Plot all three assessment models side by side for comparison.

    Produces a 1x3 panel of heatmaps: mutual assessment, self-assessment,
    and Bourgeois convention. The key visual takeaway is that mutual and
    self-assessment produce opposite gradient directions for high-M dyads.

    Parameters
    ----------
    M_range : array
        1D array of monument stock values for both axes.
    sigma_0, V, D, T_0, beta, kappa, P_base : float
        Mutual assessment parameters.
    M_thresh, M_scale : float
        Self-assessment parameters.
    M_own, steepness : float
        Bourgeois parameters.
    axes : sequence of 3 Axes, optional
        Axes to plot on. If None, a new 1x3 figure is created.

    Returns
    -------
    list of 3 Axes
    """
    from signaling.layer2 import compare_assessment_models

    if axes is None:
        fig, axes_arr = plt.subplots(1, 3, figsize=(15, 4.5))
        ax_list = list(axes_arr)
    else:
        ax_list = list(axes)
        if len(ax_list) != 3:
            raise ValueError("axes must have exactly 3 elements")

    comparison = compare_assessment_models(
        M_range, sigma_0, V, D, T_0, beta, kappa, P_base,
        M_thresh, M_scale, M_own, steepness,
    )

    extent = [float(M_range[0]), float(M_range[-1]),
              float(M_range[0]), float(M_range[-1])]

    # Mutual assessment
    im0 = ax_list[0].imshow(
        comparison["mutual"], origin="lower", extent=extent,
        aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0, vmax=float(P_base),
    )
    ax_list[0].set_xlabel(r"$M_g$")
    ax_list[0].set_ylabel(r"$M_h$")
    plt.colorbar(im0, ax=ax_list[0], label=r"$P_{\mathrm{conflict}}$")

    # Self-assessment (different scale: P can reach ~1)
    im1 = ax_list[1].imshow(
        comparison["self"], origin="lower", extent=extent,
        aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1.0,
    )
    ax_list[1].set_xlabel(r"$M_g$")
    ax_list[1].set_ylabel(r"$M_h$")
    plt.colorbar(im1, ax=ax_list[1], label=r"$P_{\mathrm{conflict}}$")

    # Bourgeois
    im2 = ax_list[2].imshow(
        comparison["bourgeois"], origin="lower", extent=extent,
        aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0, vmax=float(P_base),
    )
    ax_list[2].set_xlabel(r"$M_g$")
    ax_list[2].set_ylabel(r"$M_h$")
    plt.colorbar(im2, ax=ax_list[2], label=r"$P_{\mathrm{conflict}}$")

    return ax_list


# =====================================================================
# Layer 3: Network and buffering plots
# =====================================================================


def plot_network_degree(
    M_range: NDArray[np.float64],
    k_0: float = 0.5,
    k_max: float = 8.0,
    M_half: float = 3.0,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot network degree k(M_g) as a function of monument stock.

    Shows the saturating Michaelis-Menten relationship between group
    monument investment and intergroup exchange network size.

    Parameters
    ----------
    M_range : array
        Monument stock values.
    k_0, k_max, M_half : float
        Network formation parameters.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : Axes
    """
    from signaling.layer3 import network_degree

    _, ax = _get_ax(ax)
    k_vals = network_degree(M_range, k_0, k_max, M_half)
    ax.plot(M_range, k_vals, linewidth=2.4, color="C0")
    ax.axhline(k_0, color="gray", linestyle="--", linewidth=1.0, label=rf"$k_0 = {k_0}$")
    ax.axhline(k_0 + k_max, color="gray", linestyle=":", linewidth=1.0,
               label=rf"$k_0 + k_{{\max}} = {k_0 + k_max}$")
    ax.axvline(M_half, color="C0", linestyle=":", linewidth=1.0, alpha=0.5,
               label=rf"$M_{{1/2}} = {M_half}$")
    ax.set_xlabel(r"Group monument stock $M_g$")
    ax.set_ylabel(r"Network degree $k$")
    ax.legend(fontsize=11)
    return ax


def plot_survival_surface(
    sigma_range: NDArray[np.float64],
    k_range: NDArray[np.float64],
    gamma: float = 0.3,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot survival probability S(sigma, k) as a heatmap.

    Shows how survival depends jointly on environmental uncertainty and
    network degree. The key visual: dense networks (high k) maintain
    high survival even under severe uncertainty.

    Parameters
    ----------
    sigma_range : array
        Environmental uncertainty values.
    k_range : array
        Network degree values.
    gamma : float
        Buffering efficiency.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : Axes
    """
    from signaling.layer3 import survival_probability

    _, ax = _get_ax(ax)
    Sg, Kg = np.meshgrid(sigma_range, k_range)
    S = survival_probability(Sg, Kg, gamma)

    extent = [float(sigma_range[0]), float(sigma_range[-1]),
              float(k_range[0]), float(k_range[-1])]
    im = ax.imshow(
        S, origin="lower", extent=extent, aspect="auto",
        cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1,
    )
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Network degree $k$")
    plt.colorbar(im, ax=ax, label=r"$S(\sigma, k)$")
    return ax


def plot_vulnerability_differential(
    k_signal_range: NDArray[np.float64],
    k_nonsignal: float = 0.5,
    gamma: float = 0.3,
    alpha_initial: float = 0.30,
    beta_initial: float = 0.90,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot derived vulnerability vs. initial model assumed values.

    Shows how the vulnerability coefficient alpha_eff decreases with
    signaler network degree, compared to the fixed beta_eff for
    non-signalers and the initial model's assumed values.

    Parameters
    ----------
    k_signal_range : array
        Range of signaler network degrees.
    k_nonsignal : float
        Non-signaler baseline degree.
    gamma : float
        Buffering efficiency.
    alpha_initial, beta_initial : float
        Initial model assumed values for comparison.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : Axes
    """
    from signaling.layer3 import vulnerability_coefficient

    _, ax = _get_ax(ax)
    alpha_vals = vulnerability_coefficient(k_signal_range, gamma)
    beta_val = float(vulnerability_coefficient(k_nonsignal, gamma))

    ax.plot(k_signal_range, alpha_vals, linewidth=2.4, color="C0",
            label=r"$\alpha_{\mathrm{eff}}(k_{\mathrm{signal}})$")
    ax.axhline(beta_val, color="C1", linewidth=2.4,
               label=rf"$\beta_{{\mathrm{{eff}}}} = {beta_val:.2f}$ (non-signaler)")
    ax.axhline(alpha_initial, color="C0", linestyle="--", linewidth=1.2,
               label=rf"$\alpha_{{\mathrm{{initial}}}} = {alpha_initial}$")
    ax.axhline(beta_initial, color="C1", linestyle="--", linewidth=1.2,
               label=rf"$\beta_{{\mathrm{{initial}}}} = {beta_initial}$")
    ax.set_xlabel(r"Signaler network degree $k_{\mathrm{signal}}$")
    ax.set_ylabel("Vulnerability coefficient")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    return ax


def plot_lambda_sigma_feedback(
    sigma_range: NDArray[np.float64],
    lambda_W: float = 0.3,
    gamma: float = 0.3,
    k_0: float = 0.5,
    k_max: float = 8.0,
    M_half: float = 3.0,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot equilibrium lambda(sigma) showing the feedback loop.

    The lambda(sigma) curve is the central output of the feedback
    analysis: it shows how environmental uncertainty drives signaling
    intensity through the cooperation network mechanism.

    Parameters
    ----------
    sigma_range : array
        Environmental uncertainty values.
    lambda_W : float
        Within-group lambda component (shown as baseline).
    gamma, k_0, k_max, M_half : float
        Network parameters.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : Axes
    """
    from signaling.layer3 import lambda_sigma_sweep

    _, ax = _get_ax(ax)
    sweep = lambda_sigma_sweep(sigma_range, lambda_W, gamma, k_0, k_max, M_half)

    ax.plot(sweep["sigma"], sweep["lambda_total"], linewidth=2.4, color="C0",
            label=r"$\lambda(\sigma) = \lambda_W + \lambda_C + \lambda_X$")
    ax.axhline(lambda_W, color="gray", linestyle="--", linewidth=1.2,
               label=rf"$\lambda_W = {lambda_W}$ (baseline)")
    ax.fill_between(sweep["sigma"], lambda_W, sweep["lambda_total"],
                    alpha=0.15, color="C0", label=r"$\lambda_C + \lambda_X$ (feedback)")
    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Total $\lambda$")
    ax.legend(fontsize=11)
    return ax


# =====================================================================
# Price equation: assembled plots
# =====================================================================


def plot_sigma_star_comparison(
    C_range: NDArray[np.float64],
    lambda_W: float = 0.3,
    gamma: float = 0.3,
    k_0: float = 0.5,
    k_max: float = 8.0,
    M_half: float = 3.0,
    sigma_rapa_nui: float = 0.20,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot sigma* comparison: extended model vs. initial model.

    Shows the critical threshold as a function of reproductive cost C,
    with the extended model's lower threshold due to signaling benefits.
    Rapa Nui's sigma is marked for reference.

    Parameters
    ----------
    C_range : array
        Reproductive cost values.
    lambda_W : float
        Within-group lambda.
    gamma, k_0, k_max, M_half : float
        Network parameters.
    sigma_rapa_nui : float
        Rapa Nui sigma for reference line.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : Axes
    """
    from signaling.price_equation import sigma_star_comparison

    _, ax = _get_ax(ax)
    result = sigma_star_comparison(
        C_range, lambda_W=lambda_W, gamma=gamma,
        k_0=k_0, k_max=k_max, M_half=M_half,
    )

    ax.plot(result["C"], result["sigma_star_initial"], "--", linewidth=2.4,
            color="C1", label=r"$\sigma^*_{\mathrm{initial}}$")
    valid = np.isfinite(result["sigma_star_extended"])
    ax.plot(result["C"][valid], result["sigma_star_extended"][valid],
            linewidth=2.4, color="C0", label=r"$\sigma^*_{\mathrm{extended}}$")
    ax.axhline(sigma_rapa_nui, color="black", linestyle=":", linewidth=1.2,
               label=rf"Rapa Nui $\sigma = {sigma_rapa_nui}$")

    # Shade the region between the two thresholds
    if np.any(valid):
        ax.fill_between(
            result["C"][valid],
            result["sigma_star_extended"][valid],
            result["sigma_star_initial"][valid],
            alpha=0.1, color="C0",
            label="Newly accessible region",
        )

    ax.set_xlabel(r"Reproductive cost $C$")
    ax.set_ylabel(r"Critical threshold $\sigma^*$")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    return ax


def plot_phase_space(
    sigma_range: NDArray[np.float64],
    C_range: NDArray[np.float64],
    lambda_W: float = 0.3,
    gamma: float = 0.3,
    k_0: float = 0.5,
    k_max: float = 8.0,
    M_half: float = 3.0,
    sigma_rapa_nui: float = 0.20,
    C_rapa_nui: float = 0.35,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot the (sigma, C) phase space of monument building favorability.

    Green region: monument building favored (fitness advantage > 0).
    Gray region: monument building not favored. The boundary is
    the sigma*(C) curve. Case study points can be overlaid.

    Parameters
    ----------
    sigma_range, C_range : array
        Values for the grid axes.
    lambda_W : float
        Within-group lambda.
    gamma, k_0, k_max, M_half : float
        Network parameters.
    sigma_rapa_nui, C_rapa_nui : float
        Rapa Nui case study coordinates.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : Axes
    """
    from signaling.price_equation import phase_space

    _, ax = _get_ax(ax)
    result = phase_space(
        sigma_range, C_range, lambda_W=lambda_W,
        gamma=gamma, k_0=k_0, k_max=k_max, M_half=M_half,
    )

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(list(TWO_CLASS_COLORS))
    favored = (result["fitness_advantage"] > 0).astype(float)

    extent = [float(sigma_range[0]), float(sigma_range[-1]),
              float(C_range[0]), float(C_range[-1])]
    ax.imshow(
        favored, origin="lower", extent=extent,
        aspect="auto", cmap=cmap, vmin=0, vmax=1,
    )

    # Case study points
    ax.plot(sigma_rapa_nui, C_rapa_nui, "k^", markersize=10,
            label="Rapa Nui", zorder=5)

    ax.set_xlabel(r"Environmental uncertainty $\sigma$")
    ax.set_ylabel(r"Reproductive cost $C$")

    import matplotlib.patches as mpatches
    on_patch = mpatches.Patch(color=TWO_CLASS_COLORS[1], label="Signaling favored")
    off_patch = mpatches.Patch(color=TWO_CLASS_COLORS[0], label="Signaling not favored")
    ax.legend(handles=[on_patch, off_patch, ax.lines[-1]], fontsize=10, loc="upper right")
    return ax
