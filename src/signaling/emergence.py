"""Bistable emergence analysis.

The maintenance threshold :math:`\\sigma^*` (computed in price_equation.py)
answers whether an established signaling population outperforms a non-signaling
one. The *emergence* question is whether a rare monument-building group can
invade a population in which no one else signals. This module implements that
analysis.

Key result: under the multiplicative fitness specification with
self-consistent reproductive cost :math:`C = C_{\\mathrm{model}}(\\lambda_W)`,
a lone builder facing entirely non-signaling neighbors cannot invade: the
within-group positional reward nets out of the group mean (it never enters
group fitness), so the lone builder's group carries the cost factor
:math:`1 - C` while, under the partner-availability rule (main-text
Eq. "krare"), it gains no extra exchange partners (:math:`k = k_0` at
:math:`\\phi = 0`) and no war-avoidance either (mixed dyads are unassessed
and pay the full war cost; the :math:`\\phi`-weighting zeroes the avoided
fraction at :math:`\\phi = 0`). The measured deficit at the anchor is about
0.11 fitness units at :math:`\\sigma = 0.68`. Invasion becomes possible once
a critical fraction :math:`\\phi^*` of neighboring groups has already
adopted. Below this fraction, non-signaling is locally stable; above it, the
signaling equilibrium is reachable. Emergence is therefore bistable and
predicts clustered adoption.

See the main text ("Where monuments appear"; "Bistable spread and the
pattern of independent traditions") and the supplementary section "Bistable
emergence saddle locus".
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from signaling.calibration import (
    DEFAULT_CONFLICT_MORTALITY,
    DEFAULT_GAMMA,
    DEFAULT_K_0,
    DEFAULT_K_MAX,
    DEFAULT_M_HALF,
    DEFAULT_N,
    DEFAULT_P_BASE,
    DEFAULT_Q_MAX,
    DEFAULT_Q_MIN,
    DEFAULT_WAR_COST,
)
from signaling.layer1 import average_equilibrium_cost, expected_monument_stock
from signaling.layer2 import (
    DEFAULT_BETA_CONFLICT,
    DEFAULT_D,
    DEFAULT_KAPPA,
    DEFAULT_SIGMA_0,
    DEFAULT_T_0,
    DEFAULT_V,
    mutual_assessment_conflict_prob,
)
from signaling.layer3 import (
    NetworkDegreeSpec,
    network_degree,
    survival_probability,
    vulnerability_coefficient,
)
from signaling.price_equation import average_signaling_benefit


def rare_builder_fitness_advantage(
    sigma: float,
    lambda_W: float,
    frac_signalers: float = 0.0,
    mode: str = "multiplicative",
    use_self_consistent_cost: bool = True,
    C_exogenous: float = 0.35,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    network_degree_spec: NetworkDegreeSpec | None = None,
) -> dict[str, float]:
    r"""Fitness advantage of a rare monument-builder group.

    Computes :math:`w_{\mathrm{MB}} - w_{\mathrm{NB}}` for a rare builder
    group facing a population in which a fraction :math:`\phi` of neighbors
    already signal. The three channels through which neighboring adoption
    helps the rare builder are:

    - **Network partners**: :math:`k_{\mathrm{rare}} = k_0 + \phi (k(M_g) - k_0)`,
      a linear interpolation reflecting that assortative partner-choice
      connects the rare builder to other signalers. This is the
      partner-availability rule displayed in the main text (Eq. "krare",
      Section "Where monuments appear"), of which the network-degree
      equation k(M_g) is the full-adoption case phi = 1. It is a modeling
      assumption and load-bearing: with the full k(M_g) available at
      phi = 0, the lone builder would be favored at the anchor
      (advantage ~ +0.07 at sigma = 0.68) and there would be no
      critical mass.
    - **War-avoidance**: the conflict reduction :math:`r_{\mathrm{eff}}` is
      the symmetric build--build reduction :math:`r_{bb} = 1 -
      P_{\mathrm{conflict}}(M_g, M_g)/P_{\mathrm{base}}` (carried by the
      absolute-deterrence factor net of parity amplification; relative
      assessment alone gives no reduction at parity), and the rare builder
      captures only the fraction :math:`\phi` of it through the
      :math:`\phi`-weighting of the conflict term: at :math:`\phi = 0` the
      builder's war-avoidance advantage is exactly zero (mixed dyads are
      unassessed and pay the full war cost). Neighbor asymmetry enters
      through this :math:`\phi` weighting, not through :math:`r_{\mathrm{eff}}`
      itself.
    - **Within-group recognition**: :math:`B(\lambda_W)` is unilateral and
      does not depend on :math:`\phi`.

    Parameters
    ----------
    sigma : float
        Environmental uncertainty.
    lambda_W : float
        Within-group social reward component.
    frac_signalers : float in [0, 1]
        Fraction of the broader population that signals.
    mode : "multiplicative" or "mixed"
        Fitness combination form.
    use_self_consistent_cost : bool
        If True, :math:`C = C_{\mathrm{model}}(\lambda_W)` (default).
        If False, :math:`C = C_{\mathrm{exogenous}}`.
    C_exogenous : float
        Exogenous cost used when ``use_self_consistent_cost`` is False.
    n, q_min, q_max : float
        Group-size and quality range parameters.

    Returns
    -------
    dict with keys: sigma, frac_signalers, M_g, M_h, C, B, r_eff, k_rare,
    alpha_eff, beta_eff, w_mb, w_nb, advantage.
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    if not 0.0 <= frac_signalers <= 1.0:
        raise ValueError(f"frac_signalers must be in [0, 1], got {frac_signalers}")
    if lambda_W <= 0.0:
        raise ValueError(
            "lambda_W must be > 0: at lambda_W = 0 there is no signaling "
            "equilibrium (zero schedule, zero stock) and the advantage is "
            "sign-noise around 0."
        )
    C = average_equilibrium_cost(lambda_W, q_min, q_max) if use_self_consistent_cost else C_exogenous
    B = average_signaling_benefit(lambda_W, q_min, q_max)
    M_g = expected_monument_stock(n, q_min, q_max, lambda_W)
    M_h = frac_signalers * M_g

    P_c = float(mutual_assessment_conflict_prob(
        M_g, M_g,  # build-build: r_eff is the reduction r_bb that settles a build-build dyad
        sigma_0=DEFAULT_SIGMA_0, V=DEFAULT_V, D=DEFAULT_D, T_0=DEFAULT_T_0,
        beta=DEFAULT_BETA_CONFLICT, kappa=DEFAULT_KAPPA, P_base=DEFAULT_P_BASE,
    ))
    r_eff = 1.0 - P_c / DEFAULT_P_BASE

    if network_degree_spec is None:
        k_full = float(network_degree(M_g, DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF))
    else:
        k_full = float(network_degree_spec.k(M_g))
    k_rare = DEFAULT_K_0 + frac_signalers * (k_full - DEFAULT_K_0)

    alpha_eff = float(vulnerability_coefficient(k_rare, DEFAULT_GAMMA))
    beta_eff = float(vulnerability_coefficient(DEFAULT_K_0, DEFAULT_GAMMA))

    survival_mb = float(survival_probability(sigma, k_rare, DEFAULT_GAMMA))
    # Mutual war-avoidance (sigma-scaled, scarcity-driven): the war cost
    # W(sigma) = DEFAULT_WAR_COST * sigma is avoided on build-build dyads (settled by
    # mutual assessment, reduction r_eff = r_bb). A focal builder among a fraction phi
    # of building neighbors avoids it on its phi build-build dyads, paying it on the rest.
    W_war = DEFAULT_WAR_COST * sigma
    conflict_mb = 1.0 - W_war * (1.0 - frac_signalers * r_eff)

    survival_nb = float(survival_probability(sigma, DEFAULT_K_0, DEFAULT_GAMMA))
    conflict_nb = 1.0 - W_war  # non-builder cannot be assessed: full war cost on every dyad

    # Positional within-group reward: within-group status nets out at the group
    # level, so the rare builder's group fitness is the cost factor (1 - C) times
    # the public goods (survival via networks, conflict via deterrence). With no
    # separate reward term, 'mixed' coincides with 'multiplicative'.
    if mode in ("multiplicative", "mixed"):
        w_mb = (1.0 - C) * survival_mb * conflict_mb
        w_nb = survival_nb * conflict_nb
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'multiplicative' or 'mixed'.")

    return {
        "sigma": sigma,
        "frac_signalers": frac_signalers,
        "M_g": M_g,
        "M_h": M_h,
        "C": C,
        "B": B,
        "r_eff": r_eff,
        "k_rare": k_rare,
        "alpha_eff": alpha_eff,
        "beta_eff": beta_eff,
        "w_mb": w_mb,
        "w_nb": w_nb,
        "advantage": w_mb - w_nb,
    }


def sigma_star_invasion(
    lambda_W: float,
    frac_signalers: float,
    mode: str = "multiplicative",
    use_self_consistent_cost: bool = True,
    C_exogenous: float = 0.35,
    n_grid: int = 400,
) -> float:
    r"""Smallest :math:`\sigma` at which a rare builder gains advantage over non-builders.

    Scans :math:`\sigma \in [0, 0.95]` and locates the crossing of
    :math:`w_{\mathrm{MB}} - w_{\mathrm{NB}}` from negative to positive via
    linear interpolation between grid points.

    Returns
    -------
    float
        - Finite value in [0, 0.95] if a crossing exists.
        - ``0.0`` if the rare builder is favored at all :math:`\sigma`.
        - ``float('inf')`` if the rare builder cannot invade at any
          :math:`\sigma` (the non-signaling state is absorbing).
    """
    sigmas = np.linspace(0.0, 0.95, n_grid)
    prev_adv: float | None = None
    for i, s in enumerate(sigmas):
        adv = rare_builder_fitness_advantage(
            float(s), lambda_W, frac_signalers, mode,
            use_self_consistent_cost=use_self_consistent_cost,
            C_exogenous=C_exogenous,
        )["advantage"]
        if prev_adv is not None and prev_adv <= 0 and adv > 0:
            s_prev = sigmas[i - 1]
            return float(s_prev + (0.0 - prev_adv) / (adv - prev_adv) * (s - s_prev))
        prev_adv = adv
    if prev_adv is not None and prev_adv > 0:
        return 0.0
    return float("inf")


def phi_star(
    sigma: float,
    lambda_W: float,
    mode: str = "multiplicative",
    use_self_consistent_cost: bool = True,
    C_exogenous: float = 0.35,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    phi_bounds: tuple[float, float] = (1e-4, 1.0 - 1e-4),
    network_degree_spec: NetworkDegreeSpec | None = None,
) -> float:
    r"""Unstable interior fixed point :math:`\phi^*(\sigma)` of the replicator dynamics.

    At fixed :math:`\sigma`, locates the fraction :math:`\phi^*` of signaling
    neighbors at which the rare-builder fitness advantage vanishes:
    :math:`w_{\mathrm{MB}}(\phi^*) - w_{\mathrm{NB}}(\phi^*) = 0`. Below
    :math:`\phi^*` the advantage is negative and the replicator flow carries
    :math:`\phi \to 0` (non-signaling absorbing). Above :math:`\phi^*` the
    advantage is positive and :math:`\phi \to 1` (signaling absorbing). The
    interior fixed point is therefore the saddle separating the two basins
    of attraction.

    The saddle is dual to :func:`sigma_star_invasion`: locating
    :math:`\phi^*(\sigma)` and :math:`\sigma^*_{\mathrm{inv}}(\phi)` gives
    the same curve, parameterized differently.

    Returns
    -------
    float
        :math:`\phi^*` in :math:`(0, 1)`, or NaN if no interior saddle
        exists at this :math:`\sigma` (the rare builder is either
        disadvantaged at all :math:`\phi` or advantaged at all :math:`\phi`,
        depending on which side of the maintenance threshold one sits).
    """
    def adv(phi: float) -> float:
        return rare_builder_fitness_advantage(
            sigma, lambda_W, float(phi), mode=mode,
            use_self_consistent_cost=use_self_consistent_cost,
            C_exogenous=C_exogenous, n=n, q_min=q_min, q_max=q_max,
            network_degree_spec=network_degree_spec,
        )["advantage"]

    adv_lo = adv(phi_bounds[0])
    adv_hi = adv(phi_bounds[1])
    if adv_lo * adv_hi > 0:
        return float("nan")
    return float(brentq(adv, phi_bounds[0], phi_bounds[1], xtol=1e-6))


def replicator_dynamics(
    sigma: float,
    lambda_W: float,
    phi_0: float,
    t_max: float = 200.0,
    n_steps: int = 400,
    mode: str = "multiplicative",
    use_self_consistent_cost: bool = True,
    C_exogenous: float = 0.35,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    network_degree_spec: NetworkDegreeSpec | None = None,
) -> dict[str, object]:
    r"""Replicator trajectory for the fraction :math:`\phi` of signaling groups.

    Integrates the replicator equation

    .. math::
        \dot{\phi} = \phi (1 - \phi) \, [w_{\mathrm{MB}}(\phi) - w_{\mathrm{NB}}(\phi)]

    where the rare-builder fitness advantage at fraction :math:`\phi` of
    signaling neighbors is computed by
    :func:`rare_builder_fitness_advantage`. The replicator form is the
    standard mean-field dynamics for type-frequency change in a large
    well-mixed multi-group population, equivalent here to payoff-biased
    imitation: groups copy monument-building in proportion to its realized
    payoff advantage, so :math:`\phi` rises by payoff-biased cultural
    transmission rather than demographic turnover. In this framework
    :math:`\phi` is the fraction of groups that adopt monument signaling,
    not the within-group strategy frequency. The non-signaling and full-adoption
    states :math:`\phi = 0` and :math:`\phi = 1` are absorbing boundaries
    of the dynamics; the interior fixed point :math:`\phi^*(\sigma)`
    (see :func:`phi_star`) is the unstable saddle.

    This function provides the dynamical demonstration of bistable
    emergence: trajectories starting below :math:`\phi^*(\sigma)`
    converge to :math:`\phi = 0`; trajectories starting above converge to
    :math:`\phi = 1`. The two basins of attraction are the defining
    signature of bistability, distinct from comparative-statics
    statements about sign changes in the fitness advantage at the
    threshold.

    Parameters
    ----------
    sigma : float
        Environmental uncertainty (held fixed along the trajectory).
    lambda_W : float
        Within-group social reward.
    phi_0 : float in [0, 1]
        Initial fraction of signaling neighbors.
    t_max : float
        Integration horizon. Default 200 is sufficient to see convergence
        to absorbing states at calibrated parameters where the advantage
        magnitude is roughly 0.01 to 0.10.
    n_steps : int
        Number of output time points.
    mode : "multiplicative" or "mixed"
        Fitness combination form (see :func:`rare_builder_fitness_advantage`).
    use_self_consistent_cost : bool
        If True, :math:`C = C_{\mathrm{model}}(\lambda_W)`.
    C_exogenous : float
        Cost used when ``use_self_consistent_cost`` is False.
    n, q_min, q_max : float
        Group-size and quality range parameters.
    rtol, atol : float
        Tolerances passed to ``solve_ivp``.

    Returns
    -------
    dict with keys:
        t : ndarray
            Output time grid.
        phi : ndarray
            :math:`\phi(t)` along the trajectory.
        phi_0 : float
            Initial condition.
        sigma : float
            Environmental uncertainty used.
        lambda_W : float
            Within-group social reward used.
        terminal_phi : float
            :math:`\phi(t_{\max})`.
        absorbed_to_1 : bool
            True if ``terminal_phi`` > 0.5.
        absorbed_to_0 : bool
            True if ``terminal_phi`` < 0.5.
        success : bool
            ``solve_ivp`` success flag.
    """
    if not (0.0 <= phi_0 <= 1.0):
        raise ValueError(f"phi_0 must lie in [0, 1]; got {phi_0}.")

    def rhs(t: float, y: np.ndarray) -> list[float]:
        phi = float(y[0])
        # Clip to avoid drifting outside [0, 1] from numerical noise; the
        # boundaries are absorbing for the replicator equation so this is
        # the dynamically correct behavior.
        if phi <= 0.0 or phi >= 1.0:
            return [0.0]
        adv = rare_builder_fitness_advantage(
            sigma, lambda_W, phi, mode=mode,
            use_self_consistent_cost=use_self_consistent_cost,
            C_exogenous=C_exogenous, n=n, q_min=q_min, q_max=q_max,
            network_degree_spec=network_degree_spec,
        )["advantage"]
        return [phi * (1.0 - phi) * adv]

    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(
        rhs, (0.0, t_max), [phi_0], t_eval=t_eval,
        method="RK45", rtol=rtol, atol=atol,
    )

    phi_t = sol.y[0]
    terminal_phi = float(phi_t[-1])

    return {
        "t": sol.t,
        "phi": phi_t,
        "phi_0": float(phi_0),
        "sigma": float(sigma),
        "lambda_W": float(lambda_W),
        "terminal_phi": terminal_phi,
        "absorbed_to_1": terminal_phi > 0.5,
        "absorbed_to_0": terminal_phi < 0.5,
        "success": bool(sol.success),
    }
