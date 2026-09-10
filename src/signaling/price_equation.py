"""Assembled multilevel Price equation.

Combines Layers 1-3 into the full multilevel Price equation, deriving
the critical environmental uncertainty threshold sigma* above which
monument building is favored by selection.

Key results:
- Individual fitness w(x_i, q_i, sigma) combines reproduction, survival,
  conflict deterrence, and signaling rewards
- Within-group selection is positional: the status reward nets out of
  group-mean fitness, so it does not lower the threshold
- sigma* is corrected upward toward the standard multilevel-selection
  baseline by the derived between-group coefficients, not below it
- Three equilibrium types: monument, alternative-channel, no-signaling

**Canonical assembly (lambda_W-only)**: the individual schedule and hence
the group stock, network degree, signaler vulnerability, and derived
war-avoidance reduction all derive from the within-group reward lambda_W
alone; the between-group marginal returns lambda_C (Layer 2) and lambda_X
(Layer 3) are diagnostics evaluated at M_g(lambda_W), realized in group
fitness through the war-avoidance factor K and the survival factor S. The
superseded composite-reward variant (schedule priced by
lambda_W + lambda_C + lambda_X, solved by fixed-point iteration) is retained
only as a robustness check (layer3.lambda_total_at_sigma; SI Banach
section). Channel selection is a community-level, emergence-stage-scoped
comparison handled in layer1 (channel_dominance_condition); its incremental
net-return gap (layer1.incremental_lambda) is a descriptive diagnostic of
channel competition, NOT a schedule input.

The assembled group fitness (w_MB, w_NB), the modified critical threshold
sigma*, and the within/between-group Price partition are set out in the
manuscript's "Assembled multilevel fitness comparison" section; the SI
sections "Selection term decomposition" and "Explicit multilevel
decomposition: where selection lives" give the full multilevel decomposition.
The channel-selection cascade that determines whether the active channel is
monument construction is covered in the main-text subsection "Channel
selection: why monuments specifically" and the SI section "Channel selection
analysis".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
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
    INITIAL_MODEL_PARAMS,
    RAPA_NUI,
)


# =====================================================================
# Individual fitness
# =====================================================================


def individual_fitness(
    x_i: float | NDArray[np.float64],
    q_i: float | NDArray[np.float64],
    sigma: float,
    k: float,
    P_conflict: float,
    lam: float,
    gamma: float = DEFAULT_GAMMA,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    mode: str = "multiplicative",
    omega: float | None = None,
    q_min: float = DEFAULT_Q_MIN,
) -> float | NDArray[np.float64]:
    r"""Composed per-member fitness across the three layers (illustrative diagnostic).

    SCOPE (load-bearing): this composition is a bookkeeping diagnostic that
    shows how the layer components multiply through a member's fitness. It is
    NOT the objective that generates the Layer-1 equilibrium, and the
    manuscript's Eq. 2 schedule is not stationary under it. Eq. 2 is derived
    from the additive Layer-1 payoff :math:`q + \lambda_W \hat q(x) -
    c(x, q)`; under the multiplicative composition below,
    :math:`(1 - c)(1 + \lambda \hat q)`, the cost enters as a proportional
    factor and the first-order condition changes. Measured at
    :math:`q = 1`, :math:`\lambda_W = 0.68`: the composition's own argmax is
    :math:`x \approx 0.50` versus the schedule value
    :math:`x^*(1) \approx 0.82`, with
    :math:`\partial w/\partial x|_{x^*} \approx -0.83` before the
    survival/conflict multiplier. The
    canonical threshold and group-level comparisons (manuscript Eqs. 10--13)
    do not use this function, and a population mean of this quantity must
    not be read as the assembled-model fitness advantage.

    Four combination modes are supported.

    **Multiplicative** (default):

    .. math::
        w = (1 - c(x_i, q_i)) \cdot S(\sigma, k) \cdot (1 - m \cdot P_{\text{conflict}})
            \cdot (1 + \lambda \cdot \hat{q}(x_i))

    All components are proportional modifiers; builders must survive and
    avoid conflict before any signaling reward is realized.

    **Mixed** (additive-reward bounding sensitivity):

    .. math::
        w = (1 - c(x_i, q_i)) \cdot S(\sigma, k) \cdot (1 - m \cdot P_{\text{conflict}})
            + \lambda \cdot \hat{q}(x_i)

    Reproduction, survival, and conflict combine multiplicatively; signaling
    reward is additive. Reflects partial decoupling of social rewards from
    immediate survival (Bliege Bird and Smith 2005). At self-consistent
    parameters this predicts sigma* approaching 0 for any lambda_W > 0;
    retained as a bounding sensitivity.

    **Additive**:

    .. math::
        w = (1 - c(x_i, q_i)) + S(\sigma, k) - 1
            - m \cdot P_{\text{conflict}} + \lambda \cdot \hat{q}(x_i)

    Independent contributions summed.

    **Convex** (parameterized blend between multiplicative and additive):

    .. math::
        w = (1 - \omega) \, w_{\text{mult}} + \omega \, w_{\text{add}}

    With ``omega in [0, 1]``: ``omega = 0`` recovers multiplicative,
    ``omega = 1`` recovers additive. Used to characterize sensitivity to
    the multiplicative-vs-additive specification when social capital is
    only partially decoupled from survival (Bliege Bird and Smith 2005).

    Parameters
    ----------
    x_i : float or array
        Individual monument investment. Must be >= 0.
    q_i : float or array
        Individual quality. Must be > 0.
    sigma : float
        Environmental uncertainty.
    k : float
        Network degree for this individual's group.
    P_conflict : float
        Conflict probability for this group.
    lam : float
        Lambda (fitness value of social rewards).
    gamma : float
        Buffering efficiency per partner.
    conflict_mortality : float
        Group mortality fraction per conflict event.
    mode : str
        Fitness combination mode: 'mixed', 'multiplicative', 'additive',
        or 'convex'.
    omega : float, optional
        Convex-combination weight in [0, 1]; required when ``mode='convex'``.

    Returns
    -------
    float or array
        Individual fitness.

    See the manuscript's "Individual fitness" equations (w_S, w_N) under the
    assembled multilevel fitness comparison.
    """
    x_i = np.asarray(x_i, dtype=np.float64)
    q_i = np.asarray(q_i, dtype=np.float64)

    # Layer 1: reproductive cost (quadratic cost function)
    c = x_i ** 2 / (2.0 * q_i)
    reproduction = 1.0 - c

    # Layer 3: survival
    from signaling.layer3 import survival_probability
    S = float(survival_probability(sigma, k, gamma))

    # Layer 2: conflict effect
    conflict_effect = 1.0 - conflict_mortality * P_conflict

    # Layer 1: signaling reward b(x_i) = lambda * q_hat(x_i), where q_hat
    # inverts the separating schedule x*(q) = sqrt(lambda (q^2 - q_min^2)):
    #     q_hat(x) = sqrt(q_min^2 + x^2 / lambda),
    # so q_hat(x*(q)) = q on the equilibrium path and q_hat(0) = q_min (the
    # floor reading: x = 0 is the floor type's on-path action). The reward
    # depends on the OBSERVED investment, not on true quality; using
    # lambda * q_i at every x_i would misprice off-equilibrium play.
    x_arr = np.asarray(x_i, dtype=np.float64)
    if lam > 0.0:
        q_hat = np.sqrt(q_min**2 + x_arr**2 / lam)
        signaling_reward = lam * q_hat
    else:
        signaling_reward = np.zeros_like(x_arr)
    if np.ndim(x_i) == 0 and np.ndim(signaling_reward) > 0:
        signaling_reward = float(signaling_reward)

    if mode == "mixed":
        return reproduction * S * conflict_effect + signaling_reward
    elif mode == "multiplicative":
        return reproduction * S * conflict_effect * (1.0 + signaling_reward)
    elif mode == "additive":
        return reproduction + (S - 1.0) - conflict_mortality * P_conflict + signaling_reward
    elif mode == "convex":
        if omega is None or not (0.0 <= omega <= 1.0):
            raise ValueError("mode='convex' requires omega in [0, 1].")
        w_mult = reproduction * S * conflict_effect * (1.0 + signaling_reward)
        w_add = reproduction + (S - 1.0) - conflict_mortality * P_conflict + signaling_reward
        return (1.0 - omega) * w_mult + omega * w_add
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'mixed', 'multiplicative', 'additive', or 'convex'."
        )


# =====================================================================
# Average signaling benefit B(lambda)
# =====================================================================


def average_signaling_benefit(
    lam: float,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
) -> float:
    r"""Average of the equilibrium-vs-baseline diagnostic B(lambda_W) = E[Delta_w].

    From Layer 1, the equilibrium-vs-no-signaling difference for an individual
    of quality q is:

    .. math::
        \Delta w(q) = \frac{\lambda_W q}{2} + \frac{\lambda_W q_{\min}^2}{2q}

    Averaging over a uniform distribution on [q_min, q_max]:

    .. math::
        B(\lambda_W) = E[\Delta w(q)]
            = \frac{\lambda_W}{2} \cdot \frac{q_{\min} + q_{\max}}{2}
              + \frac{\lambda_W q_{\min}^2}{2(q_{\max} - q_{\min})}
                \ln\!\left(\frac{q_{\max}}{q_{\min}}\right)

    B is a DIAGNOSTIC, not the within-group result and not a threshold input:
    the within-group reward is positional (zero-sum), so the group cannot gain
    B while paying the aggregate cost, and the assembled threshold uses
    C_model(lambda_W) with the group-level factors S and K, never B. The
    load-bearing within-group quantities are the free-rider deterrent
    Delta_fr (signaling.layer1.free_rider_deterrent) and the status
    differential s_W (within_group_status_differential). B is retained for
    the SI diagnostic derivation and reporting only.

    Parameters
    ----------
    lam : float
        Within-group informational reward lambda_W. Must be >= 0.
    q_min : float
        Minimum quality. Must be > 0.
    q_max : float
        Maximum quality. Must be > q_min.

    Returns
    -------
    float
        B(lambda) >= 0.

    See the SI sections "Layer 1 equilibrium-vs-baseline diagnostic" and
    "Self-consistency: B(lambda_W) and C_model(lambda_W) derivations".
    """
    if lam <= 0:
        return 0.0

    q_range = q_max - q_min

    # E[lambda*q/2] over uniform(q_min, q_max)
    term1 = lam / 2.0 * (q_min + q_max) / 2.0

    # E[lambda*q_min^2/(2*q)] over uniform(q_min, q_max)
    # integral of 1/q from q_min to q_max = ln(q_max/q_min)
    term2 = lam * q_min ** 2 / (2.0 * q_range) * np.log(q_max / q_min)

    return term1 + term2


def average_signaling_benefit_numerical(
    lam: float,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
) -> float:
    """Numerical verification of average_signaling_benefit.

    Computes B(lambda) by numerical integration over the uniform
    quality distribution, for cross-checking the analytical formula.

    Parameters
    ----------
    lam, q_min, q_max : float
        Same as average_signaling_benefit.

    Returns
    -------
    float
        B(lambda) computed numerically.
    """
    from signaling.layer1 import fitness_gain

    def integrand(q: float) -> float:
        return float(fitness_gain(q, q_min, lam))

    result, _ = quad(integrand, q_min, q_max)
    return result / (q_max - q_min)


def within_group_fitness_factor(
    lam: float,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
) -> float:
    r"""Exact within-group reproduction-and-reward factor for a builder group.

    .. math::
        W_{\mathrm{in}}(\lambda)
            = \mathbb{E}_q\!\left[(1 - c^*(q))(1 + \lambda\,\hat q(x^*(q)))\right]

    This is the quality-distribution expectation of the within-group part of
    :func:`individual_fitness` (the reproduction factor :math:`1 - c^*(q)` times
    the signaling-reward factor :math:`1 + \lambda \hat q`, with the gross reward
    :math:`\lambda \hat q(x^*(q)) = \lambda q` and equilibrium cost
    :math:`c^*(q) = \lambda (q^2 - q_{\min}^2) / (2q)`).

    It **replaces** the mean-field product :math:`(1 - C)(1 + B)` used previously,
    which double-counted the construction cost: the cost sat in both the
    :math:`(1 - C)` factor and inside the *net* benefit
    :math:`B = \mathbb{E}[\Delta w] = \mathbb{E}[\lambda \hat q] - C`, so the cost
    was subtracted twice (the algebraic fingerprint was the spurious
    :math:`C(1 - C)` gap between the individual and group fitness). The exact
    expectation also retains the :math:`\mathrm{Cov}(c, \hat q)` term that the
    mean-field product dropped.

    Closed form for uniform quality on :math:`[q_{\min}, q_{\max}]`:

    .. math::
        W_{\mathrm{in}}(\lambda) = 1 + \lambda \mathbb{E}[q] - C_{\mathrm{model}}(\lambda)
            - \tfrac{\lambda^2}{2}\left(\mathbb{E}[q^2] - q_{\min}^2\right)

    with :math:`\mathbb{E}[q] = (q_{\min}+q_{\max})/2`,
    :math:`\mathbb{E}[q^2] = (q_{\max}^3 - q_{\min}^3)/(3(q_{\max}-q_{\min}))`, and
    :math:`C_{\mathrm{model}}(\lambda) = \mathbb{E}[c^*]`.

    Parameters
    ----------
    lam : float
        Total signaling reward weight :math:`\lambda` (>= 0). ``W_in(0) = 1``.
    q_min, q_max : float
        Quality range.

    Returns
    -------
    float
        :math:`W_{\mathrm{in}}(\lambda) \ge 0`. Exceeds 1 when the within-group
        signaling benefit outweighs the cost.

    See the SI section "Self-consistency: B(lambda_W) and C_model(lambda_W)
    derivations".
    """
    if lam <= 0:
        return 1.0
    Eq = (q_min + q_max) / 2.0
    Eq2 = (q_max ** 3 - q_min ** 3) / (3.0 * (q_max - q_min))
    C_model = (lam / 2.0) * (
        Eq - q_min ** 2 * np.log(q_max / q_min) / (q_max - q_min)
    )
    return 1.0 + lam * Eq - C_model - (lam ** 2 / 2.0) * (Eq2 - q_min ** 2)


def within_group_fitness_factor_numerical(
    lam: float,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
) -> float:
    """Numerical verification of :func:`within_group_fitness_factor` via quadrature."""
    if lam <= 0:
        return 1.0

    def integrand(q: float) -> float:
        c_star = lam * (q ** 2 - q_min ** 2) / (2.0 * q)  # equilibrium Spence cost
        return (1.0 - c_star) * (1.0 + lam * q)           # qhat(x*(q)) = q (gross reward)

    result, _ = quad(integrand, q_min, q_max)
    return result / (q_max - q_min)


# =====================================================================
# Group fitness comparison
# =====================================================================


def group_fitness_signaler(
    sigma: float,
    lam: float,
    alpha_eff: float,
    r: float,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    mode: str = "multiplicative",
    omega: float | None = None,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    C_exogenous: float | None = None,
) -> float:
    r"""Average fitness of a monument-building group member.

    The within-group reward is positional: :math:`\lambda_W` (status / mating)
    is a zero-sum good that reshuffles fitness by revealed quality within an
    all-signaling group but adds nothing to the group average, so it does not
    enter the group-level fitness. The between-group rewards :math:`\lambda_C`
    (deterrence) and :math:`\lambda_X` (networks) are realized *materially*
    through the war-avoidance factor :math:`K` and the survival factor
    :math:`S` via the equilibrium monument stock :math:`M_g(\lambda_W)`, not
    as a separate reward term and not through the schedule. The group fitness
    therefore reduces to the multilevel-selection form
    :math:`(1 - C_{\mathrm{model}}(\lambda_W))\, S\, K` with all coefficients
    derived. This replaces the former mean-field product :math:`(1 - C)(1 + B)`,
    which double-counted the construction cost.

    **Multiplicative** (default; the assembled multilevel fitness comparison; 'mixed' is identical,
    there being no separate reward term to leave undiscounted):

    .. math::
        w_{\text{MB}} = (1 - C_{\mathrm{model}})\,(1 - \alpha_{\text{eff}} \sigma)
                        \,(1 - W(\sigma)(1 - r)), \qquad W(\sigma) = w_0\,\sigma

    **Additive**:

    .. math::
        w_{\text{MB}} = (1 - C_{\mathrm{model}}) - \alpha_{\text{eff}} \sigma
                        - W(\sigma)(1 - r)

    **Convex** (parameterized blend; the supplementary convex-combination sensitivity analysis):

    .. math::
        w_{\text{MB}}(\omega) = (1-\omega) \, w_{\text{MB,mult}}
                                + \omega \, w_{\text{MB,add}}

    Parameters
    ----------
    sigma : float
        Environmental uncertainty.
    lam : float
        Within-group informational reward :math:`\lambda_W`. The equilibrium
        cost :math:`C_{\mathrm{model}}(\lambda_W)` derives from it. (Under the
        lambda_W-only first-order condition this is the only reward component
        that prices the schedule; passing a composite lambda here reproduces
        the superseded variant.)
    alpha_eff : float
        Signaler vulnerability coefficient (derived from network degree).
    r : float
        Conflict reduction from mutual assessment (derived from Layer 2).
    P_base : float
        INERT under the war-avoidance formulation: retained for API
        stability only. The war cost enters as the expected per-period
        W(sigma) = w0*sigma and the derived reduction r = 1 -
        P_conflict/P_base cancels P_base from the ratio, so sweeping this
        argument cannot move the output (the tornado's zero-width P_base
        bar is genuine; SI sensitivity caption).
    conflict_mortality : float
        INERT under the war-avoidance formulation (the legacy mortality
        multiplier 1 - m*P_conflict was replaced by W(sigma); SI
        war-avoidance section); retained for API stability only.
    mode : str
        Fitness combination mode: 'mixed', 'multiplicative', 'additive',
        or 'convex'.
    omega : float, optional
        Convex-combination weight in [0, 1]; required when ``mode='convex'``.
    q_min, q_max : float
        Quality range (for the within-group expectation).
    C_exogenous : float or None
        Reproductive cost mode. ``None`` (default, the canonical
        self-consistent locus): the cost is the model-implied equilibrium
        cost :math:`C_{\mathrm{model}}(\lambda_W)`, so cost and reward move
        together. DOMAIN NOTE: past the C = 1 line (endogenously at
        lambda_W ~ 1.934 for the default quality range, or an off-locus
        C_exogenous >= 1) the builder factor 1 - C is negative and the
        returned "fitness" is negative; the function returns it honestly
        rather than guarding, and the threshold solver classifies such
        profiles as never-favored (sigma* = inf). Callers treating fitness
        as positive must stay inside C < 1. A float: the *off-locus* mode used by the SI bivariate
        sensitivity, in which the reproductive cost in the
        :math:`(1 - C)` factor is set independently of ``lam`` while the
        signaling-derived quantities (stock, network degree, vulnerability,
        conflict reduction) remain those of the :math:`\lambda_W`
        equilibrium. This is the counterfactual in which measured labor
        diversion differs from the model-implied signaling cost.

    Returns
    -------
    float
        Average fitness of a monument builder.

    See the manuscript's "Modified critical threshold" (w_MB) and the SI
    sections "Explicit multilevel decomposition: where selection lives" and
    "Bivariate sensitivity in (C, lambda_W) space".
    """
    if omega is not None and mode != "convex":
        raise ValueError(
            "omega is consumed only under mode='convex'; passing omega with "
            f"mode='{mode}' would be silently ignored (the same "
            "silent-override guard as for r)."
        )
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(
            f"sigma must be in [0, 1], got {sigma}: outside it the linear "
            "survival factor is not a probability (S < 0 for sigma > 1 + "
            "gamma*k) and the returned 'fitness' is silent garbage "
            "rather than a probability."
        )
    survival = 1.0 - alpha_eff * sigma
    conflict = 1.0 - DEFAULT_WAR_COST * sigma * (1.0 - r)
    # Fully-positional within-group reward. lambda_W (within-group status/mating) is
    # a positional good: in an all-signaling group it reshuffles status by quality
    # but adds nothing to the group average, so it contributes no direct group-level
    # reward. The between-group rewards lambda_C (deterrence) and lambda_X (networks)
    # are realized *materially* as the conflict and survival factors via the
    # equilibrium monument stock M_g(lambda_W) -- not as a separate reward term and
    # not through the schedule (lambda_W-only FOC). Hence the within-group x
    # reproduction factor is the cost factor 1 - C_model(lambda_W) alone, and w_MB
    # reduces to the multilevel-selection form (1 - C_model) * S * K with all
    # parameters derived rather than assumed.
    if C_exogenous is not None:
        # Off-locus mode: reproductive cost decoupled from the lambda_W
        # equilibrium (SI "Bivariate sensitivity in (C, lambda_W) space").
        C_model = float(C_exogenous)
    elif lam > 0:
        Eq = (q_min + q_max) / 2.0
        C_model = (lam / 2.0) * (
            Eq - q_min ** 2 * np.log(q_max / q_min) / (q_max - q_min)
        )
    else:
        C_model = 0.0
    within_group = 1.0 - C_model
    if mode in ("multiplicative", "mixed"):
        # With no separate reward term, 'mixed' coincides with 'multiplicative'.
        return within_group * survival * conflict
    elif mode == "additive":
        return within_group - alpha_eff * sigma - DEFAULT_WAR_COST * sigma * (1.0 - r)
    elif mode == "convex":
        if omega is None or not (0.0 <= omega <= 1.0):
            raise ValueError("mode='convex' requires omega in [0, 1].")
        w_mult = within_group * survival * conflict
        w_add = within_group - alpha_eff * sigma - DEFAULT_WAR_COST * sigma * (1.0 - r)
        return (1.0 - omega) * w_mult + omega * w_add
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            "Use 'mixed', 'multiplicative', 'additive', or 'convex'."
        )


def group_fitness_nonsignaler(
    sigma: float,
    beta_eff: float,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    mode: str = "multiplicative",
    omega: float | None = None,
) -> float:
    r"""Average fitness of a non-monument-building group member.

    Non-signalers pay no reproductive cost (C = 0), gain no war-avoidance
    (they cannot be assessed, so they pay the full scarcity-driven war cost
    :math:`W(\sigma)`), and have higher vulnerability
    (:math:`\beta_{\text{eff}} > \alpha_{\text{eff}}`).

    **Mixed / Multiplicative** (identical):

    .. math::
        w_{\text{NB}} = (1 - \beta_{\text{eff}} \sigma)\,(1 - W(\sigma)),
        \qquad W(\sigma) = w_0\,\sigma

    **Additive**:

    .. math::
        w_{\text{NB}} = 1 - \beta_{\text{eff}} \sigma - W(\sigma)

    **Convex**:

    .. math::
        w_{\text{NB}}(\omega) = (1-\omega) \, w_{\text{NB,mult}} + \omega \, w_{\text{NB,add}}

    Parameters
    ----------
    sigma : float
        Environmental uncertainty.
    beta_eff : float
        Non-signaler vulnerability coefficient.
    P_base : float
        INERT under the war-avoidance formulation: retained for API
        stability only. The war cost enters as the expected per-period
        W(sigma) = w0*sigma and the derived reduction r = 1 -
        P_conflict/P_base cancels P_base from the ratio, so sweeping this
        argument cannot move the output (the tornado's zero-width P_base
        bar is genuine; SI sensitivity caption).
    conflict_mortality : float
        INERT under the war-avoidance formulation (the legacy mortality
        multiplier 1 - m*P_conflict was replaced by W(sigma); SI
        war-avoidance section); retained for API stability only.
    mode : str
        Fitness combination mode: 'mixed', 'multiplicative', 'additive',
        or 'convex'.
    omega : float, optional
        Convex-combination weight in [0, 1]; required when ``mode='convex'``.

    Returns
    -------
    float
        Average fitness of a non-signaler.
    """
    if omega is not None and mode != "convex":
        raise ValueError(
            "omega is consumed only under mode='convex'; passing omega with "
            f"mode='{mode}' would be silently ignored (the same "
            "silent-override guard as for r)."
        )
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(
            f"sigma must be in [0, 1], got {sigma}: outside it the linear "
            "survival factor is not a probability (S < 0 for sigma > 1 + "
            "gamma*k) and the returned 'fitness' is silent garbage "
            "rather than a probability."
        )
    survival = 1.0 - beta_eff * sigma
    conflict = 1.0 - DEFAULT_WAR_COST * sigma
    if mode == "additive":
        return 1.0 - beta_eff * sigma - DEFAULT_WAR_COST * sigma
    if mode == "convex":
        if omega is None or not (0.0 <= omega <= 1.0):
            raise ValueError("mode='convex' requires omega in [0, 1].")
        w_mult = survival * conflict
        w_add = 1.0 - beta_eff * sigma - DEFAULT_WAR_COST * sigma
        return (1.0 - omega) * w_mult + omega * w_add
    # mixed and multiplicative are identical for non-signalers (B = 0)
    return survival * conflict


def fitness_advantage(
    sigma: float,
    lam: float,
    alpha_eff: float,
    beta_eff: float,
    r: float,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    mode: str = "multiplicative",
    omega: float | None = None,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    C_exogenous: float | None = None,
) -> float:
    r"""Fitness advantage of monument builders over non-builders.

    .. math::
        \Delta w = w_{\text{MB}} - w_{\text{NB}}

    Signaling is favored when Delta_w > 0. The threshold sigma* is
    where Delta_w = 0.

    Parameters
    ----------
    [same as group_fitness_signaler and group_fitness_nonsignaler;
    ``C_exogenous=None`` is the self-consistent locus
    :math:`C = C_{\mathrm{model}}(\lambda_W)`, a float is the off-locus
    reproductive cost of the SI bivariate sensitivity]

    Returns
    -------
    float
        Fitness difference. Positive means monument building is favored.
    """
    if omega is not None and mode != "convex":
        raise ValueError(
            "omega is consumed only under mode='convex'; passing omega with "
            f"mode='{mode}' would be silently ignored (the same "
            "silent-override guard as for r)."
        )
    w_MB = group_fitness_signaler(
        sigma, lam, alpha_eff, r, P_base, conflict_mortality, mode, omega, q_min, q_max,
        C_exogenous=C_exogenous,
    )
    w_NB = group_fitness_nonsignaler(
        sigma, beta_eff, P_base, conflict_mortality, mode, omega,
    )
    return w_MB - w_NB


# =====================================================================
# Critical threshold sigma*
# =====================================================================


def initial_model_sigma_star(
    C: float = INITIAL_MODEL_PARAMS["C"],
    alpha: float = INITIAL_MODEL_PARAMS["alpha"],
    beta: float = INITIAL_MODEL_PARAMS["beta"],
) -> float:
    r"""Compute sigma* from the initial model (simplified formula).

    .. math::
        \sigma^*_{\text{initial}} = \frac{C}{\beta - (1 - C)\alpha}

    This is the simplified threshold from the initial Price equation model,
    without conflict reduction terms. The full formula including conflict
    gives sigma* ~ 0.39; the simplified version gives ~0.496.

    Parameters
    ----------
    C : float
        Reproductive cost. Default 0.35.
    alpha : float
        Signaler vulnerability. Default 0.30.
    beta : float
        Non-signaler vulnerability. Default 0.90.

    Returns
    -------
    float
        sigma* for the initial model.
    """
    denominator = beta - (1.0 - C) * alpha
    if denominator <= 0:
        return float("inf")
    return C / denominator


def critical_threshold_sigma_star(
    C: float = RAPA_NUI["C"],
    lambda_W: float = 0.3,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    r: float | None = None,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    compute_lambda_C_func: Any = None,
    sigma_bounds: tuple[float, float] = (0.001, 0.99),
    use_derived_r: bool = True,
    max_outer_iter: int = 10,
    r_tol: float = 1e-4,
    delta: float = 0.0,
    mode: str = "multiplicative",
    omega: float | None = None,
    network_degree_spec: Any = None,
    C_exogenous: float | None = None,
) -> dict[str, float]:
    r"""Solve for the critical threshold sigma* numerically.

    This is a lower-level building block. For the manuscript's canonical
    self-consistent threshold (internally consistent $C = C_{\mathrm{model}}(\lambda_W)$,
    derived $r(M_g)$, closed $\lambda_C$ feedback), use
    :func:`sigma_star_self_consistent` instead.

    Finds sigma* where the fitness advantage of monument builders
    over non-builders equals zero.

    Under the lambda_W-only first-order condition (the canonical
    specification), the schedule, the group stock M_g, the network degree,
    and the signaler vulnerability alpha_eff all derive from lambda_W alone
    and are sigma-independent, so the threshold is a single Brent root-find
    on fitness_advantage(sigma, lambda_W, ...). The between-group returns
    lambda_C and lambda_X are evaluated once at M_g(lambda_W) as diagnostics
    (:func:`signaling.layer3.between_group_lambda_diagnostics`); they enter
    the fitness comparison through the survival factor S and the
    war-avoidance factor K, not through the schedule. (The superseded
    composite-lambda variant iterated a fixed point on lambda per candidate
    sigma; it is retained as :func:`signaling.layer3.lambda_total_at_sigma`
    for the SI Banach robustness comparison and moves sigma* by ~0.001.)

    Note: this function treats ``r`` as a scalar and ``compute_lambda_C_func``
    as an optional closure (diagnostic only). When ``use_derived_r`` is True,
    r is computed directly as r_bb at the depreciation-adjusted equilibrium
    stock M_g(lambda_W); no outer iteration is needed because M_g no longer
    depends on r or sigma*. :func:`sigma_star_self_consistent` wraps this
    function with the canonical defaults and a default $\lambda_C$ factory.

    Parameters
    ----------
    C : float
        Reproductive cost used by the *assumed-coefficient baseline*
        (``sigma_star_initial``) only. The framework's own threshold does
        not consume this argument: on the canonical path its cost is the
        endogenous :math:`C_{\mathrm{model}}(\lambda_W)`; to move the
        framework's cost independently, use ``C_exogenous``.
    lambda_W : float
        Within-group lambda component.
    gamma, k_0, k_max, M_half : float
        Network and survival parameters.
    n : int
        Group size.
    q_min, q_max : float
        Quality range.
    r : float or None
        Exogenous conflict reduction, honored ONLY with
        ``use_derived_r=False`` (the sensitivity mode behind the
        manuscript's "varying r over [0, 0.95] moves sigma* only within
        [0.42, 0.54]" claim; measured 0.5376 at r = 0 down to 0.4206 at
        r = 0.95). On the canonical path (``use_derived_r=True``, default)
        r is derived as r_bb at the equilibrium stock and an explicitly
        passed r would be ignored, so passing one raises a ValueError
        rather than producing a silently flat sweep (the same hazard class
        as an exogenous-C sweep that ignores the passed C). ``None`` with
        ``use_derived_r=False`` falls back to the legacy stipulated 0.75.
    P_base : float
        Baseline conflict probability.
    conflict_mortality : float
        Mortality per conflict event.
    compute_lambda_C_func : callable(M_g) -> float, or None
        Layer 2 feedback function.
    sigma_bounds : tuple
        Bounds for root search.
    delta : float
        Signal depreciation rate in [0, 1]. Passed to the feedback loop.
        delta = 1 is the flow-assessment limit, M_g = I_g, and reproduces
        the delta = 0 reference path exactly (the manuscript's reference
        threshold is reported as this limit); smaller delta accumulates a
        larger steady-state stock and lowers sigma* monotonically.
        When delta > 0, effective monument stock is I_g / delta.
    C_exogenous : float or None
        ``None`` (default): the framework's reproductive cost is the
        self-consistent :math:`C_{\mathrm{model}}(\lambda_W)`, so a sweep
        over ``C`` leaves the framework's sigma* flat by construction (only
        the baseline moves). A float: off-locus mode, in which the
        framework's cost in Eqs. 12--13 is set to this value while the
        derived coefficients (alpha_eff, beta_eff, r, M_g) still come from
        the ``lambda_W`` equilibrium. This is the mode behind the SI
        "Bivariate sensitivity in (C, lambda_W) space" section: it answers
        how sigma* would respond if measured labor diversion and the
        within-group reward were treated as independent parameters.

    Returns
    -------
    dict with keys:
        'sigma_star' : critical threshold. The LOWEST zero of Delta w(sigma)
            in sigma_bounds (the uncertainty at which building first becomes
            favored). Found by a dense sign-scan that brackets every
            crossing, so a profile favored only on an interior band (two
            roots, both endpoint signs negative) is detected rather than
            misclassified as "never favored". Under the multiplicative
            specification Delta w is an exact quadratic in sigma, so at most
            two roots exist; at the calibrated anchor and across the
            reported (C, n) rectangle exactly one lies in [0, 1] (pinned in
            the test suite). Existence and uniqueness are otherwise an
            empirical property of the coefficients, not a theorem; consult
            'n_roots'. (For derived coefficients alpha_eff <= beta_eff holds
            structurally, since k_signal >= k_0.)
        'sigma_roots' : list of all zeros found in sigma_bounds (ascending)
        'n_roots' : number of zeros found
        'sigma_star_initial' : initial model threshold for comparison
        'B_lambda' : signaling benefit at threshold
        'lambda_total' : total lambda at threshold
        'alpha_eff' : signaler vulnerability at threshold
        'beta_eff' : non-signaler vulnerability at threshold
        'k_signal' : signaler network degree at threshold
        'converged' : bool; True when at least one crossing was found
    """
    if omega is not None and mode != "convex":
        raise ValueError(
            "omega is consumed only under mode='convex'; passing omega with "
            f"mode='{mode}' would be silently ignored (the same "
            "silent-override guard as for r)."
        )
    from signaling.layer3 import between_group_lambda_diagnostics, vulnerability_coefficient
    from signaling.layer2 import derived_conflict_reduction as _derived_r

    if lambda_W <= 0.0:
        raise ValueError(
            "lambda_W must be > 0: at lambda_W = 0 the signaling equilibrium "
            "collapses (zero schedule, zero stock, r from vanishing-stock "
            "noise) and the threshold classification is sign-noise, not a "
            "result."
        )

    beta_eff = float(vulnerability_coefficient(k_0, gamma))

    # lambda_W-only schedule: the group stock, network degree, and signaler
    # vulnerability derive from lambda_W alone and are sigma-independent
    # (lambda_X enters group fitness through S, not through the schedule).
    # sigma below affects only the lambda_X diagnostic, so evaluate at 0 here
    # and re-evaluate at sigma* for the returned diagnostics.
    diag = between_group_lambda_diagnostics(
        0.0, lambda_W, gamma, k_0, k_max, M_half,
        n, q_min, q_max, compute_lambda_C_func, delta=delta,
        network_degree_spec=network_degree_spec,
    )
    alpha_eff_star = diag["alpha_eff"]
    k_signal = diag["k"]
    M_eq = diag["M_g"]

    # r = r_bb at the (depreciation-adjusted) equilibrium monument stock.
    # Direct computation: M_eq depends only on lambda_W, so no outer
    # iteration is needed (max_outer_iter and r_tol are retained in the
    # signature for API stability but are no longer consumed).
    if use_derived_r:
        if r is not None:
            raise ValueError(
                "An exogenous r was supplied but use_derived_r=True would "
                "silently ignore it (r is derived as r_bb at the equilibrium "
                "stock on the canonical path). Pass use_derived_r=False to "
                "sweep r exogenously."
            )
        r = float(_derived_r(M_eq, M_eq))
    elif r is None:
        r = 0.75  # legacy stipulated value for the non-derived mode

    def _fitness_diff(sigma: float) -> float:
        """Fitness advantage as function of sigma (for root finding)."""
        return fitness_advantage(
            sigma, lambda_W, alpha_eff_star, beta_eff, r,
            P_base, conflict_mortality, mode, omega, q_min, q_max,
            C_exogenous=C_exogenous,
        )

    # Initial-model baseline (exogenous alpha/beta; r-independent).
    sigma_star_initial = initial_model_sigma_star(
        C, INITIAL_MODEL_PARAMS["alpha"], INITIAL_MODEL_PARAMS["beta"]
    )

    # Dense sign-scan over sigma_bounds, bracketing EVERY crossing of
    # Delta w(sigma). An endpoint-only check would misclassify a profile with
    # two interior roots (favored only on an interior band, both endpoint
    # signs negative) as "never favored"; under the multiplicative
    # specification Delta w is an exact quadratic in sigma (and linear in the
    # additive mode), so a 201-point scan brackets all roots. The reported
    # sigma_star is the LOWEST root in the bounds: the uncertainty at which
    # building first becomes favored (the maintenance threshold). All roots
    # are returned in 'sigma_roots' and their count in 'n_roots'; at the
    # calibrated anchor and across the reported (C, n) rectangle the quadratic
    # has exactly one root in [0, 1] (pinned in the test suite).
    scan = np.linspace(sigma_bounds[0], sigma_bounds[1], 201)
    f_scan = np.array([_fitness_diff(float(s)) for s in scan])
    sigma_roots: list[float] = []
    for _i in range(len(scan) - 1):
        fa, fb = f_scan[_i], f_scan[_i + 1]
        if fa == 0.0:
            if not sigma_roots or abs(scan[_i] - sigma_roots[-1]) > 1e-10:
                sigma_roots.append(float(scan[_i]))
            continue
        if fa * fb < 0:
            sigma_roots.append(float(brentq(
                _fitness_diff, float(scan[_i]), float(scan[_i + 1]), xtol=1e-8
            )))
    if f_scan[-1] == 0.0:
        sigma_roots.append(float(scan[-1]))

    if sigma_roots:
        sigma_star = sigma_roots[0]
        converged = True
    else:
        # No crossing anywhere on the scan: always or never favored.
        sigma_star = 0.0 if f_scan[0] > 0 else float("inf")
        converged = False

    # Diagnostic composite lambda at sigma* (lambda_C, lambda_X evaluated at
    # M_g(lambda_W); they do not feed back into the schedule).
    sigma_diag = sigma_star if (converged and np.isfinite(sigma_star)) else 0.0
    diag_star = between_group_lambda_diagnostics(
        sigma_diag, lambda_W, gamma, k_0, k_max, M_half,
        n, q_min, q_max, compute_lambda_C_func, delta=delta,
        network_degree_spec=network_degree_spec,
    )
    lam_composite = diag_star["lambda_composite"]

    if C_exogenous is not None:
        _C_model_eq = float(C_exogenous)
    elif lambda_W > 0:
        _Eq = (q_min + q_max) / 2.0
        _C_model_eq = (lambda_W / 2.0) * (_Eq - q_min ** 2 * np.log(q_max / q_min) / (q_max - q_min))
    else:
        _C_model_eq = 0.0
    within_group = 1.0 - _C_model_eq  # positional within-group x reproduction factor (status nets out)
    # B_lambda: average *net* individual gain E[Delta w], a within-group diagnostic
    # only (the group fitness is (1 - C_model)*S*K under the positional model).
    B_lambda = average_signaling_benefit(lambda_W, q_min, q_max)

    return {
        "sigma_star": sigma_star,
        "sigma_roots": sigma_roots,
        "n_roots": len(sigma_roots),
        "sigma_star_initial": sigma_star_initial,
        "within_group_factor": within_group,
        "B_lambda": B_lambda,
        "lambda_total": lam_composite,
        "alpha_eff": alpha_eff_star,
        "beta_eff": beta_eff,
        "k_signal": k_signal,
        "r": r,
        "converged": converged,
    }


def _default_lambda_C_factory() -> Any:
    """Default lambda_C(M_g) closure using calibration.py assessment parameters.

    Used by sigma_star_self_consistent to close the competitive-feedback loop
    with reasonable defaults. Kept private; callers should use the keyword
    argument of sigma_star_self_consistent rather than invoking directly.
    """
    from signaling.layer2 import (
        DEFAULT_BETA_CONFLICT,
        DEFAULT_D,
        DEFAULT_KAPPA,
        DEFAULT_SIGMA_0,
        DEFAULT_T_0,
        DEFAULT_V,
        compute_lambda_C,
    )

    def _f(M_g: float) -> float:
        return compute_lambda_C(
            M_g=M_g, N=DEFAULT_N,
            sigma_0=DEFAULT_SIGMA_0, V=DEFAULT_V, D=DEFAULT_D, T_0=DEFAULT_T_0,
            beta=DEFAULT_BETA_CONFLICT, kappa=DEFAULT_KAPPA, P_base=DEFAULT_P_BASE,
        )
    return _f


def sigma_star_self_consistent(
    lambda_W: float,
    mode: str = "multiplicative",
    use_derived_r: bool = True,
    use_lambda_C: bool = True,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    n: int = DEFAULT_N,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    max_outer_iter: int = 10,
    r_tol: float = 1e-4,
    omega: float | None = None,
    network_degree_spec: Any = None,
    delta: float = 0.0,
) -> dict[str, float]:
    r"""Self-consistent critical threshold $\sigma^*(\lambda_W)$.

    This is the canonical threshold calculation for the revised framework
    (the main-text critical-threshold results). It enforces the framework's
    consistency conditions on top of the lower-level
    :func:`critical_threshold_sigma_star`:

    1. **Internally consistent cost.** $C = C_{\mathrm{model}}(\lambda_W)$ from
       :func:`signaling.layer1.average_equilibrium_cost` (Eq.~21). Under the
       lambda_W-only first-order condition the schedule, the cost, and the
       free-rider deterrent all carry the same $\lambda_W$.
    2. **Derived conflict reduction.** $r = r_{bb}(M_g(\lambda_W))$ via
       :func:`signaling.layer2.derived_conflict_reduction` at the equilibrium
       monument stock, rather than held at a stipulated constant. Because
       $M_g$ derives from $\lambda_W$ alone, this is a direct computation,
       not an iteration.
    3. **Between-group returns as diagnostics.** $\lambda_C$ and $\lambda_X$
       are evaluated at $M_g(\lambda_W)$ (via
       :func:`signaling.layer3.between_group_lambda_diagnostics`) and
       reported in ``lambda_total``; they enter fitness through S and K,
       not through the schedule.

    Parameters
    ----------
    lambda_W : float
        Within-group social reward. The single free parameter of the
        self-consistent threshold calculation.
    mode : {"multiplicative", "mixed", "convex"}
        Fitness combination specification. Multiplicative is the
        manuscript's primary specification. Mixed is an alias of
        multiplicative in the current positional specification (no separate
        additive reward term), returning the same $\sigma^* = 0.4771$; it is
        retained only as an accepted mode name. Convex
        interpolates between multiplicative and additive forms,
        parameterized by ``omega`` (the supplementary convex-combination sensitivity analysis).
    use_derived_r : bool
        If True, compute $r = r_{bb}(M_g(\lambda_W))$ via
        :func:`derived_conflict_reduction` at the equilibrium stock. If
        False, uses the legacy scalar $r = 0.75$.
    use_lambda_C : bool
        If True, evaluate the $\lambda_C$ diagnostic via
        :func:`compute_lambda_C` at $M_g(\lambda_W)$. If False,
        $\lambda_C = 0$. Either way $\lambda_C$ does not enter the schedule.
    q_min, q_max, n, gamma, k_0, k_max, M_half : float
        Standard parameters; defaults from ``calibration.py``.
    P_base, conflict_mortality : float
        Conflict parameters.
    max_outer_iter : int
        Retained for API stability; no longer consumed (the former $r$
        outer loop collapsed to a direct computation under the
        lambda_W-only schedule).
    r_tol : float
        Retained for API stability; no longer consumed.
    omega : float, optional
        Convex-combination weight in [0, 1]; required when ``mode='convex'``.

    Returns
    -------
    dict with keys:
        sigma_star : float
            Self-consistent threshold $\sigma^*(\lambda_W)$.
        sigma_star_base : float
            Standard MLS baseline $\sigma^*_{\mathrm{base}}$ for comparison,
            using exogenous $\alpha=0.30$, $\beta=0.90$ and the same $C$.
        C_model : float
            Internally consistent reproductive cost
            $C = C_{\mathrm{model}}(\lambda_W)$.
        B_lambda : float
            Average within-group signaling benefit at equilibrium.
        r : float
            Derived conflict reduction $r_{bb}(M_g(\lambda_W))$ (or 0.75 if
            ``use_derived_r=False``).
        lambda_total : float
            Diagnostic composite $\lambda_W + \lambda_C + \lambda_X$ at
            $M_g(\lambda_W)$; the schedule itself carries $\lambda_W$ alone.
        alpha_eff, beta_eff : float
            Vulnerability coefficients at equilibrium.
        converged : bool
            Whether the inner threshold root-find succeeded.

    Examples
    --------
    >>> result = sigma_star_self_consistent(lambda_W=0.68)
    >>> result["sigma_star"]  # doctest: +ELLIPSIS
    0.4770...
    >>> result["C_model"]  # doctest: +ELLIPSIS
    0.3516...
    """
    if omega is not None and mode != "convex":
        raise ValueError(
            "omega is consumed only under mode='convex'; passing omega with "
            f"mode='{mode}' would be silently ignored (the same "
            "silent-override guard as for r)."
        )
    from signaling.layer1 import average_equilibrium_cost

    C_model = average_equilibrium_cost(lambda_W, q_min, q_max)
    lambda_C_func = _default_lambda_C_factory() if use_lambda_C else None

    # Under the lambda_W-only schedule, M_g depends only on lambda_W, so the
    # derived r = r_bb(M_g(lambda_W)) is computed directly inside
    # critical_threshold_sigma_star; no outer r iteration is needed
    # (max_outer_iter and r_tol are retained in the signature for API
    # stability but are no longer consumed).
    result = critical_threshold_sigma_star(
        C=C_model,
        lambda_W=lambda_W,
        gamma=gamma, k_0=k_0, k_max=k_max, M_half=M_half,
        n=n, q_min=q_min, q_max=q_max,
        r=None,  # derived on the canonical path; legacy 0.75 when use_derived_r=False
        use_derived_r=use_derived_r,
        P_base=P_base, conflict_mortality=conflict_mortality,
        compute_lambda_C_func=lambda_C_func,
        mode=mode,
        omega=omega,
        network_degree_spec=network_degree_spec,
        delta=delta,
    )
    r = result["r"]

    # Baseline comparison using the same C but exogenous alpha/beta
    sigma_star_base = initial_model_sigma_star(
        C_model, INITIAL_MODEL_PARAMS["alpha"], INITIAL_MODEL_PARAMS["beta"]
    )

    return {
        "sigma_star": result["sigma_star"],
        "sigma_roots": result["sigma_roots"],
        "n_roots": result["n_roots"],
        "sigma_star_base": sigma_star_base,
        "C_model": C_model,
        "within_group_factor": result["within_group_factor"],
        "B_lambda": result["B_lambda"],
        "r": r,
        "lambda_total": result["lambda_total"],
        "alpha_eff": result["alpha_eff"],
        "beta_eff": result["beta_eff"],
        "converged": bool(result["converged"]),
    }


# =====================================================================
# Analytical Lipschitz bound for the lambda-feedback map T(lambda)
# =====================================================================


def _J_integral_uniform(q_min: float, q_max: float) -> float:
    r"""Closed form of $J(q_{\min}, q_{\max}) = \int_{q_{\min}}^{q_{\max}} \sqrt{q^2 - q_{\min}^2}\, dq$.

    Antiderivative: $F(q) = \tfrac{q \sqrt{q^2 - a^2}}{2} - \tfrac{a^2}{2} \ln(q + \sqrt{q^2 - a^2})$
    with $a = q_{\min}$. Used by the closed-form expression for
    $dM/d\lambda$ under a uniform quality distribution.

    See the Banach contraction analysis.
    """
    a = q_min

    def F(q: float) -> float:
        s = np.sqrt(max(q * q - a * a, 0.0))
        # log(q + s) is well-defined for q >= a > 0.
        return q * s / 2.0 - a * a / 2.0 * np.log(q + s)

    return float(F(q_max) - F(q_min))


def lipschitz_derivative_analytical(
    lam: float,
    sigma: float,
    lambda_W: float = 0.68,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    include_lambda_C: bool = False,
    lambda_C_dx: float = 0.01,
    **lambda_C_kwargs: Any,
) -> float:
    r"""Analytical Lipschitz bound on the lambda-feedback map T(lambda).

    Supports the composite-lambda ROBUSTNESS VARIANT (superseded as the
    canonical path; the canonical schedule carries lambda_W alone,
    see :func:`signaling.layer3.between_group_lambda_diagnostics`).
    Implements the closed-form derivation in the Banach contraction analysis
    (``sec:si-banach``), which now quantifies that the variant's fixed point
    sits within 1% of lambda_W. The feedback map is

    .. math::
        T(\lambda) = \lambda_W + \lambda_C(M(\lambda)) + \lambda_X(M(\lambda); \sigma),

    so by the chain rule
    $T'(\lambda) = (\partial \lambda_C / \partial M + \partial \lambda_X / \partial M) \cdot dM/d\lambda$.

    Closed-form components:

    1. ``dM/dlambda = M(lambda) / (2 lambda)`` under uniform quality with
       ``x*(q) = sqrt(lambda * (q^2 - q_min^2))``. Equivalently
       $dM/d\lambda = n J / (2 \sqrt{\lambda} (q_{\max} - q_{\min}))$
       where $J$ is the closed-form integral in :func:`_J_integral_uniform`.

    2. ``d(lambda_X)/dM`` analytic. Writing
       ``f(M) = dk/dM = k_max M_half / (M_half + M)^2``
       and ``k(M) = k_0 + k_max M / (M_half + M)``,
       ``lambda_X = sigma * gamma * f / (1 + gamma k)^2``, so the product rule gives
       $\partial \lambda_X / \partial M = \sigma \gamma (1 + \gamma k)^{-3}\bigl[f'(M)(1 + \gamma k) - 2 \gamma f(M)^2\bigr]$
       with $f'(M) = -2 k_{\max} M_{1/2} / (M_{1/2} + M)^3$.

    3. ``d(lambda_C)/dM`` numerical fallback. The mutual-assessment
       conflict probability (Layer 2, the war-avoidance section) embeds a sum over
       neighbors with logistic-cost ratios and an absolute-deterrence
       factor; closed form is unwieldy and contributes negligibly at the
       empirical anchor (lambda_C + lambda_X < 1% of lambda; see
       Supp. S25). Off by default; enable via ``include_lambda_C=True``.

    The lambda_W term is constant in lambda and contributes 0 to T'(lambda).

    Parameters
    ----------
    lam : float
        Current feedback iterate lambda. Must be > 0.
    sigma : float
        Environmental uncertainty. Must be >= 0.
    lambda_W : float
        Within-group component (constant under T; carried only for API
        symmetry with the existing numerical test).
    gamma, k_0, k_max, M_half : float
        Layer 3 network and survival parameters.
    n, q_min, q_max : float
        Group size and quality bounds for Layer 1 ``M(lambda)``.
    include_lambda_C : bool
        If True, add the numerically computed ``|d(lambda_C)/dM|``.
        Default False because the term is < 0.5% of the bound at all
        empirical anchors and adds a numerical step.
    lambda_C_dx : float
        Finite-difference step for ``d(lambda_C)/dM``. Used only when
        ``include_lambda_C=True``.
    **lambda_C_kwargs :
        Extra keyword arguments forwarded to ``compute_lambda_C`` (e.g.
        ``n_neighbors``, ``dispute_freq``, ``conflict_cost``).

    Returns
    -------
    float
        $|T'(\lambda)|$ at ``(lam, sigma)``. Compare against the numerical
        bound enforced by ``test_lipschitz_bound_below_one``.

    Notes
    -----
    The maximum of $|T'(\lambda)|$ over the manuscript operating range
    $\sigma \in [0.1, 0.9]$, $\lambda \in [0.3, 1.5]$ occurs at the
    corner $(\sigma, \lambda) = (0.9, 0.3)$ and equals 0.0246, matching
    the numerical bound $\leq 0.025$ reported in Supp. S25.

    See Also
    --------
    signaling.layer3.compute_lambda_X
    signaling.layer1.expected_monument_stock
    """
    # Local imports avoid circular import at module load.
    from signaling.layer1 import expected_monument_stock

    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if q_min <= 0 or q_max <= q_min:
        raise ValueError(
            f"require 0 < q_min < q_max; got q_min={q_min}, q_max={q_max}"
        )

    # Step 1: dM/dlambda in closed form.
    # M(lambda) = n * sqrt(lambda) * J / (q_max - q_min)  under uniform quality.
    # => dM/dlambda = M / (2 lambda).
    M = expected_monument_stock(n=n, q_min=q_min, q_max=q_max, lam=lam)
    dM_dlambda = M / (2.0 * lam)

    # Step 2: d(lambda_X)/dM in closed form.
    k = k_0 + k_max * M / (M_half + M)
    f = k_max * M_half / (M_half + M) ** 2
    f_prime = -2.0 * k_max * M_half / (M_half + M) ** 3
    one_plus_gk = 1.0 + gamma * k
    dLambdaX_dM = (
        sigma * gamma / one_plus_gk**3
        * (f_prime * one_plus_gk - 2.0 * gamma * f * f)
    )

    # Step 3: d(lambda_C)/dM numerical (optional; negligible at anchor).
    if include_lambda_C:
        from signaling.layer2 import compute_lambda_C

        lam_C_plus = compute_lambda_C(M + lambda_C_dx, **lambda_C_kwargs)
        lam_C_minus = compute_lambda_C(
            max(M - lambda_C_dx, 0.0), **lambda_C_kwargs
        )
        dLambdaC_dM = (lam_C_plus - lam_C_minus) / (2.0 * lambda_C_dx)
    else:
        dLambdaC_dM = 0.0

    # Step 4: assemble |T'(lambda)|.
    # T = lambda_W + lambda_C + lambda_X, lambda_W constant => dT/dlam = 0 from it.
    return float(abs(dLambdaX_dM + dLambdaC_dM) * dM_dlambda)


# =====================================================================
# Stability and invasion analysis
# =====================================================================


def stability_analysis(
    C: float = RAPA_NUI["C"],
    lambda_W: float = 0.3,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    r: float | None = None,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    compute_lambda_C_func: Any = None,
    delta: float = 0.0,
    mode: str = "multiplicative",
    dsigma: float = 1e-5,
    n_initial_conditions: int = 20,
    sigma_check_values: list[float] | None = None,
) -> dict[str, Any]:
    """Minimal stability analysis for the critical threshold sigma*.

    Checks three properties:
    1. **Threshold gradient**: d(fitness_advantage)/d_sigma at sigma* is
       positive, confirming that crossing the threshold tips selection
       toward monument building.
    2. **Fixed-point uniqueness**: The lambda feedback loop converges to
       the same fixed point from multiple initial conditions, providing
       numerical evidence of uniqueness. When ``sigma_check_values`` is
       supplied, uniqueness is checked at each value rather than at the
       default sigma_star + 0.05.
    3. **Advantage positive above threshold**: At sigma just above sigma*,
       a rare monument builder has positive fitness advantage over
       non-builders. (This is structural -- sigma_star is defined as the
       root of the advantage function -- and serves as a consistency check
       on the numerical root-finding, not an independent dynamical claim.)

    Parameters
    ----------
    C, lambda_W, gamma, etc. : float
        Standard model parameters (see critical_threshold_sigma_star).
    dsigma : float
        Step size for numerical gradient computation.
    n_initial_conditions : int
        Number of initial lambda values to test for uniqueness.
    sigma_check_values : list[float] or None
        If supplied, perform the uniqueness check at each of these sigma
        values. The returned ``fixed_point_unique_per_sigma`` is a list of
        booleans and ``fixed_point_spread_per_sigma`` is a list of spreads.

    Returns
    -------
    dict with keys:
        sigma_star : float
        gradient_at_threshold : float
            d(Delta_w)/d_sigma at sigma*; positive means crossing the
            threshold favors monument building.
        gradient_positive : bool
        fixed_point_unique : bool
            True if all initial conditions converge to the same lambda
            at the default test sigma (sigma_star + 0.05).
        fixed_point_values : list[float]
            Converged lambda values from each initial condition.
        fixed_point_spread : float
            Max - min of converged values (should be ~0 if unique).
        advantage_positive_above_threshold : bool
            True if the fitness advantage is positive at sigma* + 0.01.
            Note: this is structural since sigma_star is defined as the
            advantage's root; the flag confirms the root-finding worked
            rather than an independent dynamical property.
        invasion_advantage : float
            Fitness advantage at sigma* + 0.01.
        fixed_point_unique_per_sigma : list[bool], optional
            Uniqueness flag at each value in ``sigma_check_values``.
        fixed_point_spread_per_sigma : list[float], optional
            Spread of converged values at each ``sigma_check_values``.
    """
    from signaling.layer3 import lambda_total_at_sigma

    # Step 1: Compute sigma*. r=None (default) resolves to the derived
    # r_bb at the equilibrium stock; an explicit r switches the whole
    # analysis to the exogenous-r mode. The SAME resolved r is then used
    # for the gradient below; mixing the two (derived r for sigma*, the
    # passed 0.75 for the gradient) would be an inconsistency hidden by a
    # silent override.
    threshold_result = critical_threshold_sigma_star(
        C=C, lambda_W=lambda_W, gamma=gamma, k_0=k_0, k_max=k_max,
        M_half=M_half, n=n, q_min=q_min, q_max=q_max, r=r,
        P_base=P_base, conflict_mortality=conflict_mortality,
        compute_lambda_C_func=compute_lambda_C_func, delta=delta,
        mode=mode, use_derived_r=(r is None),
    )
    sigma_star = threshold_result["sigma_star"]
    r_resolved = float(threshold_result["r"])

    # Step 2: Numerical gradient of fitness advantage at sigma*
    # lambda_W-only canonical path: the schedule/cost carry lambda_W and
    # alpha_eff is evaluated at M_g(lambda_W) (sigma-independent).
    def _advantage_at(sig: float) -> float:
        from signaling.layer3 import between_group_lambda_diagnostics
        diag = between_group_lambda_diagnostics(
            sig, lambda_W, gamma, k_0, k_max, M_half,
            n, q_min, q_max, compute_lambda_C_func, delta=delta,
        )
        return fitness_advantage(
            sig, lam=lambda_W, alpha_eff=diag["alpha_eff"],
            beta_eff=1.0 / (1.0 + gamma * k_0),
            r=r_resolved, P_base=P_base,
            conflict_mortality=conflict_mortality,
            mode=mode, q_min=q_min, q_max=q_max,
        )

    adv_plus = _advantage_at(sigma_star + dsigma)
    adv_minus = _advantage_at(sigma_star - dsigma)
    gradient = (adv_plus - adv_minus) / (2 * dsigma)

    # Step 3: Fixed-point uniqueness check
    # Sweep initial lambda values and check convergence to the same point
    sigma_test = sigma_star + 0.05  # test above threshold
    if sigma_test > 0.99:
        sigma_test = sigma_star - 0.05
    init_lambdas = np.linspace(0.05, 2.0, n_initial_conditions)
    converged_lambdas = []
    for init_lam in init_lambdas:
        loop = lambda_total_at_sigma(
            sigma_test, lambda_W=lambda_W, gamma=gamma, k_0=k_0,
            k_max=k_max, M_half=M_half, n=n, q_min=q_min,
            q_max=q_max, compute_lambda_C_func=compute_lambda_C_func,
            delta=delta, lambda_init=float(init_lam),
        )
        converged_lambdas.append(loop["lambda_total"])

    converged_arr = np.array(converged_lambdas)
    fp_spread = float(converged_arr.max() - converged_arr.min())

    # Step 4: Advantage above threshold (consistency check, not an
    # independent dynamical claim).
    epsilon = 0.01
    sigma_invade = min(sigma_star + epsilon, 0.99)
    invasion_adv = _advantage_at(sigma_invade)

    result: dict[str, Any] = {
        "sigma_star": sigma_star,
        "gradient_at_threshold": gradient,
        "gradient_positive": gradient > 0,
        "fixed_point_unique": fp_spread < 1e-4,
        "fixed_point_values": converged_lambdas,
        "fixed_point_spread": fp_spread,
        "advantage_positive_above_threshold": invasion_adv > 0,
        "invasion_advantage": invasion_adv,
        "mode": mode,
    }

    # Optional Step 5: multi-sigma uniqueness sweep
    if sigma_check_values is not None:
        unique_flags: list[bool] = []
        spreads: list[float] = []
        for sig in sigma_check_values:
            convs = []
            for init_lam in init_lambdas:
                loop = lambda_total_at_sigma(
                    sig, lambda_W=lambda_W, gamma=gamma, k_0=k_0,
                    k_max=k_max, M_half=M_half, n=n, q_min=q_min,
                    q_max=q_max, compute_lambda_C_func=compute_lambda_C_func,
                    delta=delta, lambda_init=float(init_lam),
                )
                convs.append(loop["lambda_total"])
            arr = np.array(convs)
            spread = float(arr.max() - arr.min())
            spreads.append(spread)
            unique_flags.append(spread < 1e-4)
        result["fixed_point_unique_per_sigma"] = unique_flags
        result["fixed_point_spread_per_sigma"] = spreads
        result["sigma_check_values"] = list(sigma_check_values)

    return result


def stability_analysis_self_consistent(
    lambda_W: float = 0.68,
    mode: str = "multiplicative",
    use_derived_r: bool = True,
    use_lambda_C: bool = True,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    n: int = DEFAULT_N,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    delta: float = 0.0,
    dsigma: float = 1e-5,
    n_initial_conditions: int = 20,
    sigma_check_values: list[float] | None = None,
) -> dict[str, Any]:
    """Stability diagnostics at the canonical self-consistent threshold.

    Calls :func:`sigma_star_self_consistent` to obtain the canonical
    threshold with C = C_model(lambda_W), derived r, and the lambda
    feedback closed, then runs the three stability checks of
    :func:`stability_analysis` at that threshold using the self-consistent
    C, r, and alpha_eff. This is the canonical wrapper used to back the
    manuscript's three-property stability statement (main-text bistable-emergence results).

    Parameters
    ----------
    lambda_W : float
        Within-group social reward (default 0.68, the empirical anchor).
    mode : str
        Fitness composition mode ('multiplicative' is the primary
        specification; 'mixed' is an alias of it, and the additive-reward
        bound is the omega=1 convex limit).
    use_derived_r : bool
        If True, derive r from mutual assessment at the equilibrium M_g.
    use_lambda_C : bool
        If True, include the lambda_C feedback in the loop.
    sigma_check_values : list[float] or None
        Sigma values at which to verify fixed-point uniqueness. Default
        ``None`` reuses the single sigma_star + 0.05 check.

    Returns
    -------
    dict
        Same keys as :func:`stability_analysis` plus ``C_model``,
        ``r_used``, and ``alpha_eff``.
    """
    sc = sigma_star_self_consistent(
        lambda_W=lambda_W, mode=mode,
        use_derived_r=use_derived_r, use_lambda_C=use_lambda_C,
        q_min=q_min, q_max=q_max, n=n,
        gamma=gamma, k_0=k_0, k_max=k_max, M_half=M_half,
        P_base=P_base, conflict_mortality=conflict_mortality,
        delta=delta,
    )

    # Now run stability_analysis at the self-consistent parameters.
    stab = stability_analysis(
        C=sc["C_model"], lambda_W=lambda_W,
        gamma=gamma, k_0=k_0, k_max=k_max, M_half=M_half,
        n=n, q_min=q_min, q_max=q_max,
        r=sc["r"], P_base=P_base,
        conflict_mortality=conflict_mortality,
        delta=delta, mode=mode,
        dsigma=dsigma, n_initial_conditions=n_initial_conditions,
        sigma_check_values=sigma_check_values,
    )

    stab["C_model"] = sc["C_model"]
    stab["r_used"] = sc["r"]
    stab["alpha_eff"] = sc["alpha_eff"]
    return stab


# =====================================================================
# Within-group and between-group selection terms
# =====================================================================


def within_group_selection(
    lam: float,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    C: float = RAPA_NUI["C"],
) -> float:
    r"""Within-group selection term per unit S*K: the positional net s_W - C.

    In the standard Price treatment the within-group term is negative for
    purely costly group-beneficial traits (free-rider advantage). Here the
    POSITIONAL status differential s_W = lambda_W (E[q] - q_min) offsets the
    cost C, so the within-group term is positive exactly when s_W > C
    (beta_0 = SK (s_W - C) > 0; manuscript within-group differential and
    SI Eq. beta0-si).

    NOTE (superseded rule): an earlier docstring stated the sign rule in
    terms of the Layer-1 diagnostic B(lambda) ("B > C favors"). That is the
    earlier claim the manuscript superseded: B is a partial-equilibrium
    diagnostic a positional reward cannot deliver groupwide, and it does not
    govern this sign. The rules genuinely differ on an interval: at
    lambda = 0.45 (defaults), B = 0.240 < C = 0.35 yet
    s_W - C = +0.078 > 0 (pinned in the test suite).

    The sign of this term is the key difference from the initial model,
    which assumes within-group selection is always negative at rate C.

    Parameters
    ----------
    lam : float
        Total lambda at the environmental conditions considered.
    q_min, q_max : float
        Quality range.
    C : float
        Reproductive cost.

    Returns
    -------
    float
        Within-group selection term (s_W - C), the positional within-group net
        (status differential minus cost). Positive favors signaling.
    """
    s_W = within_group_status_differential(lam, q_min, q_max)
    return s_W - C


def between_group_selection(
    sigma: float,
    C: float,
    alpha_eff: float,
    beta_eff: float,
    r: float,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
) -> float:
    r"""Two-homogeneous-group fitness difference Delta w(sigma) = w_MB - w_NB.

    The direction of the between-group comparison between an all-building
    and an all-non-building group (main Eqs. 12-13 at full adoption), with
    the reproductive cost passed EXOGENOUSLY through ``C``.

    OBJECT DISTINCTION (load-bearing near the threshold): this is NOT the
    Price between-group covariance Cov_G(W_g, p_g), and the two have
    opposite signs on an interval at the anchor: Delta w(sigma) > 0
    immediately above sigma* = 0.477, while the ensemble covariance stays
    slightly negative until sigma ~ 0.73 (main-text within/between section;
    measured Delta w(0.55) = +0.045 against Cov_between = -0.006). The
    covariance lives in :func:`price_partition` and is tested there.

    Positive when the survival (Layer 3) and war-avoidance (Layer 2)
    advantages outweigh the reproductive cost C.

    Parameters
    ----------
    [same as fitness_advantage]

    Returns
    -------
    float
        Between-group selection indicator. Positive favors signaling.
    """
    # Positional between-group advantage with given (C, alpha_eff, beta_eff, r):
    # w_MB = (1-C)*S*K, w_NB = S_NB*K_NB; no signaling reward (status nets out).
    w_MB = (1.0 - C) * (1.0 - alpha_eff * sigma) * (1.0 - DEFAULT_WAR_COST * sigma * (1.0 - r))
    w_NB = (1.0 - beta_eff * sigma) * (1.0 - DEFAULT_WAR_COST * sigma)
    return w_MB - w_NB


# =====================================================================
# Equilibrium classification
# =====================================================================


def classify_equilibrium(
    sigma: float,
    sigma_star: float,
    monument_dominates: bool = True,
) -> str:
    r"""Classify the equilibrium type for given environmental conditions.

    Three equilibrium types (see the main-text subsection "Channel selection:
    why monuments specifically" and the SI section "Channel selection analysis"):

    1. **Monument equilibrium**: sigma > sigma*, monument channel dominates.
       High investment, dense networks, all three audience channels active.
    2. **Alternative-channel equilibrium**: sigma > some threshold but monument
       channel does not dominate. Cooperation via cheaper signal (feasting,
       ritual). Lambda_C and lambda_X too low to justify monument cost.
    3. **No-signaling equilibrium**: sigma < all thresholds. Zero investment,
       sparse networks, no incentive to signal.

    Parameters
    ----------
    sigma : float
        Environmental uncertainty.
    sigma_star : float
        Critical threshold for monument building.
    monument_dominates : bool
        Whether the monument channel dominates alternatives (from
        channel selection analysis).

    Returns
    -------
    str
        'monument', 'alternative', or 'no_signaling'.
    """
    if sigma >= sigma_star and monument_dominates:
        return "monument"
    elif sigma >= sigma_star and not monument_dominates:
        return "alternative"
    else:
        return "no_signaling"


# =====================================================================
# Phase space
# =====================================================================


def phase_space(
    sigma_range: NDArray[np.float64],
    C_range: NDArray[np.float64],
    lambda_W: float = 0.3,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    r: float | None = None,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    compute_lambda_C_func: Any = None,
    delta: float = 0.0,
    mode: str = "multiplicative",
) -> dict[str, NDArray[np.float64]]:
    r"""Compute the phase space of fitness advantage over (sigma, C).

    For each (sigma, C) pair, computes the equilibrium lambda, signaling
    benefit, and fitness advantage. The sigma* boundary is where the
    advantage equals zero. C is exogenous here by design (the off-locus
    reading of the phase plane); r is derived at each cell's equilibrium
    stock when ``r=None`` (canonical) and honored exogenously otherwise.

    Parameters
    ----------
    sigma_range : array
        Sigma values to sweep.
    C_range : array
        Reproductive cost values to sweep.
    [other params as in critical_threshold_sigma_star]

    Returns
    -------
    dict with keys:
        'sigma_grid', 'C_grid' : 2D meshgrid arrays
        'fitness_advantage' : 2D array (positive = signaling favored)
        'lambda_total' : 2D array of equilibrium lambda
        'alpha_eff' : 2D array of signaler vulnerability
        'B_lambda' : 2D array of signaling benefit
    """
    from signaling.layer3 import between_group_lambda_diagnostics, vulnerability_coefficient
    from signaling.layer2 import derived_conflict_reduction as _derived_r

    Sg, Cg = np.meshgrid(sigma_range, C_range)
    advantage = np.zeros_like(Sg)
    lam_grid = np.zeros_like(Sg)
    alpha_grid = np.zeros_like(Sg)
    B_grid = np.zeros_like(Sg)

    beta_eff = float(vulnerability_coefficient(k_0, gamma))

    for i in range(len(C_range)):
        for j in range(len(sigma_range)):
            s = float(Sg[i, j])
            c = float(Cg[i, j])

            # Compute equilibrium state (lambda_W-only canonical path: the
            # schedule and stock derive from lambda_W; lambda_C/lambda_X are
            # diagnostics at M_g(lambda_W)).
            eq = between_group_lambda_diagnostics(
                s, lambda_W, gamma, k_0, k_max, M_half,
                n, q_min, q_max, compute_lambda_C_func, delta=delta,
            )

            alpha = eq["alpha_eff"]
            M_eq = eq["M_g"]
            # War-avoidance conflict reduction: derived as r_bb at this cell's
            # equilibrium monument stock when r is None (canonical), else the
            # exogenous value; the passed r is honored, not silently
            # overridden by the derived value.
            r_cell = float(_derived_r(M_eq, M_eq)) if r is None else float(r)
            # Positional (sigma, C) advantage: C swept as the cost; survival/conflict
            # from the lambda_W equilibrium. No signaling reward (status nets out).
            advantage[i, j] = between_group_selection(
                s, c, alpha, beta_eff, r_cell, P_base, conflict_mortality,
            )
            lam_grid[i, j] = eq["lambda_composite"]
            alpha_grid[i, j] = alpha
            B_grid[i, j] = average_signaling_benefit(lambda_W, q_min, q_max)

    return {
        "sigma_grid": Sg,
        "C_grid": Cg,
        "fitness_advantage": advantage,
        "lambda_total": lam_grid,
        "alpha_eff": alpha_grid,
        "B_lambda": B_grid,
    }


def sigma_star_comparison(
    C_range: NDArray[np.float64],
    lambda_W: float = 0.3,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    r: float | None = None,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    alpha_initial: float = INITIAL_MODEL_PARAMS["alpha"],
    beta_initial: float = INITIAL_MODEL_PARAMS["beta"],
) -> dict[str, NDArray[np.float64]]:
    r"""Compare sigma* between the framework and the initial model across C values.

    For each reproductive cost C, computes:
    - sigma*_initial from the simplified initial model formula (rises with C)
    - sigma*_extended from the full feedback model

    Under the positional model the cost is endogenous (C = C_model(lambda_W)), so
    sigma*_extended is invariant to the exogenous C swept here; it is set by
    lambda_W. The function therefore contrasts the framework's endogenous-cost
    threshold against the initial model's assumed-cost baseline. To vary the
    framework's threshold, sweep lambda_W (see sigma_star_vs_param).

    Parameters
    ----------
    C_range : array
        Reproductive cost values (used for the initial-model baseline).
    [other params as in critical_threshold_sigma_star]

    Returns
    -------
    dict with keys:
        'C' : input cost values
        'sigma_star_initial' : initial model thresholds (rise with C)
        'sigma_star_extended' : framework thresholds (flat in C; set by lambda_W)
        'B_lambda' : legacy diagnostic (within-group net benefit at lambda_W)
    """
    n_pts = len(C_range)
    ss_initial = np.zeros(n_pts)
    ss_extended = np.zeros(n_pts)
    B_values = np.zeros(n_pts)

    for i, c_val in enumerate(C_range):
        ss_initial[i] = initial_model_sigma_star(float(c_val), alpha_initial, beta_initial)

        try:
            # r=None (default): canonical derived r. An explicit r switches
            # to the exogenous-r sensitivity mode; forwarding it under the
            # derived default would be silently ignored (guarded upstream).
            result = critical_threshold_sigma_star(
                C=float(c_val), lambda_W=lambda_W, gamma=gamma,
                k_0=k_0, k_max=k_max, M_half=M_half, n=n,
                q_min=q_min, q_max=q_max, r=r, P_base=P_base,
                conflict_mortality=conflict_mortality,
                use_derived_r=(r is None),
            )
            ss_extended[i] = result["sigma_star"]
            B_values[i] = result["B_lambda"]
        except ValueError:
            ss_extended[i] = np.nan
            B_values[i] = np.nan

    return {
        "C": C_range,
        "sigma_star_initial": ss_initial,
        "sigma_star_extended": ss_extended,
        "B_lambda": B_values,
    }


def sigma_star_vs_delta(
    delta_range: NDArray[np.float64],
    C: float = RAPA_NUI["C"],
    lambda_W: float = 0.3,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    r: float | None = None,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
) -> dict[str, NDArray[np.float64]]:
    r"""Sweep sigma* as a function of signal depreciation rate delta.

    Higher depreciation reduces effective monument stock (M_g* = I_g / delta).
    Within the delta>0 regime sigma* rises slightly with delta, but the
    overall effect on the threshold is negligible: sigma* stays near its
    static value across the plausible delta range (see the main-text depreciation results).
    This function produces the sigma*(delta) curve.

    Parameters
    ----------
    delta_range : array
        Depreciation rates to sweep, each in (0, 1].
    [other params as in critical_threshold_sigma_star]

    Returns
    -------
    dict with keys:
        'delta' : input depreciation rates
        'sigma_star' : critical threshold at each delta
        'lambda_total' : equilibrium lambda at each threshold
        'M_g' : effective monument stock at each threshold
    """
    n_pts = len(delta_range)
    ss = np.zeros(n_pts)
    lam_vals = np.zeros(n_pts)
    Mg_vals = np.zeros(n_pts)

    for i, d in enumerate(delta_range):
        try:
            result = critical_threshold_sigma_star(
                C=C, lambda_W=lambda_W, gamma=gamma,
                k_0=k_0, k_max=k_max, M_half=M_half, n=n,
                q_min=q_min, q_max=q_max, r=r, P_base=P_base,
                conflict_mortality=conflict_mortality, delta=float(d),
                use_derived_r=(r is None),
            )
            ss[i] = result["sigma_star"]
            lam_vals[i] = result["lambda_total"]
            # Compute M_g at sigma* with this delta (lambda_W-only schedule)
            from signaling.layer3 import between_group_lambda_diagnostics
            eq = between_group_lambda_diagnostics(
                result["sigma_star"], lambda_W, gamma, k_0, k_max,
                M_half, n, q_min, q_max, delta=float(d),
            )
            Mg_vals[i] = eq["M_g"]
        except (ValueError, RuntimeError):
            ss[i] = np.nan
            lam_vals[i] = np.nan
            Mg_vals[i] = np.nan

    return {
        "delta": delta_range,
        "sigma_star": ss,
        "lambda_total": lam_vals,
        "M_g": Mg_vals,
    }


# =====================================================================
# Parameter sensitivity analysis
# =====================================================================

# Parameters accepted by critical_threshold_sigma_star (for validation)
_VALID_PARAMS: set[str] = {
    "C", "C_exogenous", "lambda_W", "gamma", "k_0", "k_max", "M_half", "n",
    "q_min", "q_max", "r", "P_base", "conflict_mortality", "delta", "mode",
    "omega",
}


def sigma_star_vs_param(
    param_name: str,
    param_range: NDArray[np.float64],
    **kwargs: Any,
) -> dict[str, NDArray[np.float64]]:
    r"""Sweep sigma* as a function of any single parameter.

    Generalizes sigma_star_vs_delta to arbitrary parameters of
    critical_threshold_sigma_star. For each value in param_range,
    overrides that parameter while holding all others at the values
    specified in kwargs (or their defaults).

    Parameters
    ----------
    param_name : str
        Name of the parameter to sweep. Must be a keyword argument
        of critical_threshold_sigma_star.
    param_range : array
        Values to sweep.
    **kwargs
        Baseline values for all other parameters.

    Returns
    -------
    dict with keys:
        'param_values' : the swept values
        'sigma_star' : critical threshold at each value
        'sigma_star_initial' : initial model threshold at each value
        'lambda_total' : equilibrium lambda at each threshold
        'B_lambda' : signaling benefit at each threshold
    """
    if param_name not in _VALID_PARAMS:
        raise ValueError(
            f"Unknown parameter {param_name!r}. "
            f"Valid parameters: {sorted(_VALID_PARAMS)}"
        )

    n_pts = len(param_range)
    ss = np.zeros(n_pts)
    ss_init = np.zeros(n_pts)
    lam_vals = np.zeros(n_pts)
    B_vals = np.zeros(n_pts)

    for i, val in enumerate(param_range):
        kw = dict(kwargs)
        # Cast to int for group size
        kw[param_name] = int(val) if param_name == "n" else float(val)
        try:
            result = critical_threshold_sigma_star(**kw)
            ss[i] = result["sigma_star"]
            ss_init[i] = result["sigma_star_initial"]
            lam_vals[i] = result["lambda_total"]
            B_vals[i] = result["B_lambda"]
        except (ValueError, RuntimeError):
            ss[i] = np.nan
            ss_init[i] = np.nan
            lam_vals[i] = np.nan
            B_vals[i] = np.nan

    return {
        "param_values": np.asarray(param_range, dtype=np.float64),
        "sigma_star": ss,
        "sigma_star_initial": ss_init,
        "lambda_total": lam_vals,
        "B_lambda": B_vals,
    }


def sigma_star_vs_omega(
    omega_range: NDArray[np.float64],
    lambda_W: float = 0.68,
    **kwargs: Any,
) -> dict[str, NDArray[np.float64]]:
    r"""Sweep the self-consistent $\sigma^*$ across the convex weight $\omega$.

    For each $\omega \in [0, 1]$ in ``omega_range``, computes
    $\sigma^*(\omega)$ via :func:`sigma_star_self_consistent` with
    ``mode='convex'``. ``omega = 0`` recovers the multiplicative
    specification; ``omega = 1`` recovers the additive specification.

    Used to construct the supplementary convex-combination figure (multiplicative-
    additive sensitivity).

    Parameters
    ----------
    omega_range : array
        Values of $\omega$ to sweep; each must lie in [0, 1].
    lambda_W : float
        Within-group social reward at which to evaluate the sweep.
        Default 0.68 is the empirical anchor from Erasmus 1965.
    **kwargs
        Additional keyword arguments passed to
        :func:`sigma_star_self_consistent` (e.g., ``q_min``, ``q_max``,
        ``use_lambda_C``, ``use_derived_r``).

    Returns
    -------
    dict with keys:
        'omega' : the swept values
        'sigma_star' : critical threshold at each omega
        'sigma_star_base' : standard MLS baseline (same for all omega)
        'C_model' : internally consistent cost C_model(lambda_W)
        'lambda_total' : equilibrium lambda at threshold
        'B_lambda' : signaling benefit at threshold
        'converged' : convergence flag per omega
    """
    omega_arr = np.asarray(omega_range, dtype=np.float64)
    if np.any(omega_arr < 0.0) or np.any(omega_arr > 1.0):
        raise ValueError("All omega values must lie in [0, 1].")

    n_pts = len(omega_arr)
    ss = np.zeros(n_pts)
    lam_vals = np.zeros(n_pts)
    B_vals = np.zeros(n_pts)
    converged_flags = np.zeros(n_pts, dtype=bool)
    sigma_base = np.nan
    C_model = np.nan

    for i, w in enumerate(omega_arr):
        try:
            res = sigma_star_self_consistent(
                lambda_W=lambda_W, mode="convex", omega=float(w), **kwargs,
            )
            ss[i] = res["sigma_star"]
            lam_vals[i] = res["lambda_total"]
            B_vals[i] = res["B_lambda"]
            converged_flags[i] = bool(res["converged"])
            if i == 0:
                sigma_base = res["sigma_star_base"]
                C_model = res["C_model"]
        except (ValueError, RuntimeError):
            ss[i] = np.nan
            lam_vals[i] = np.nan
            B_vals[i] = np.nan
            converged_flags[i] = False

    return {
        "omega": omega_arr,
        "sigma_star": ss,
        "sigma_star_base": sigma_base,
        "C_model": C_model,
        "lambda_total": lam_vals,
        "B_lambda": B_vals,
        "converged": converged_flags,
    }


def sigma_star_bivariate(
    param1_name: str,
    param1_range: NDArray[np.float64],
    param2_name: str,
    param2_range: NDArray[np.float64],
    **kwargs: Any,
) -> dict[str, NDArray[np.float64]]:
    r"""Bivariate sweep of sigma* over two parameters.

    Computes sigma* on a 2D grid defined by param1_range x param2_range,
    holding all other parameters at their baseline values.

    Parameters
    ----------
    param1_name, param2_name : str
        Names of the two parameters to sweep.
    param1_range, param2_range : array
        Values to sweep for each parameter.
    **kwargs
        Baseline values for all other parameters.

    Returns
    -------
    dict with keys:
        'param1_grid', 'param2_grid' : 2D meshgrids (param1 on rows)
        'sigma_star' : 2D array of sigma* values
        'sigma_star_initial' : 2D array of initial model sigma*
    """
    for name in (param1_name, param2_name):
        if name not in _VALID_PARAMS:
            raise ValueError(
                f"Unknown parameter {name!r}. "
                f"Valid parameters: {sorted(_VALID_PARAMS)}"
            )

    n1, n2 = len(param1_range), len(param2_range)
    ss = np.full((n1, n2), np.nan)
    ss_init = np.full((n1, n2), np.nan)

    for i, v1 in enumerate(param1_range):
        for j, v2 in enumerate(param2_range):
            kw = dict(kwargs)
            kw[param1_name] = int(v1) if param1_name == "n" else float(v1)
            kw[param2_name] = int(v2) if param2_name == "n" else float(v2)
            try:
                result = critical_threshold_sigma_star(**kw)
                ss[i, j] = result["sigma_star"]
                ss_init[i, j] = result["sigma_star_initial"]
            except (ValueError, RuntimeError):
                pass  # remains nan

    p1_grid, p2_grid = np.meshgrid(param1_range, param2_range, indexing="ij")
    return {
        "param1_grid": p1_grid,
        "param2_grid": p2_grid,
        "sigma_star": ss,
        "sigma_star_initial": ss_init,
    }


def sensitivity_tornado(
    param_ranges: dict[str, tuple[float, float]],
    **baseline_kwargs: Any,
) -> dict[str, dict[str, float]]:
    r"""One-at-a-time sensitivity analysis for sigma*.

    For each parameter, computes sigma* at its low and high values
    while holding all others at baseline. Returns results sorted by
    sensitivity magnitude (swing), largest first.

    Parameters
    ----------
    param_ranges : dict
        Maps parameter names to (low, high) tuples defining the
        sensitivity range.
    **baseline_kwargs
        Baseline values for critical_threshold_sigma_star.

    Returns
    -------
    dict mapping parameter name to dict with keys:
        'sigma_star_low' : sigma* at the low parameter value
        'sigma_star_high' : sigma* at the high parameter value
        'sigma_star_baseline' : sigma* at baseline
        'swing' : abs(sigma_star_high - sigma_star_low)
    Ordered by swing (largest first).
    """
    # Compute baseline
    baseline_result = critical_threshold_sigma_star(**baseline_kwargs)
    sigma_baseline = baseline_result["sigma_star"]

    results: dict[str, dict[str, float]] = {}
    for param_name, (lo_val, hi_val) in param_ranges.items():
        kw_lo = dict(baseline_kwargs)
        kw_lo[param_name] = int(lo_val) if param_name == "n" else float(lo_val)
        kw_hi = dict(baseline_kwargs)
        kw_hi[param_name] = int(hi_val) if param_name == "n" else float(hi_val)

        try:
            ss_lo = critical_threshold_sigma_star(**kw_lo)["sigma_star"]
        except (ValueError, RuntimeError):
            ss_lo = np.nan

        try:
            ss_hi = critical_threshold_sigma_star(**kw_hi)["sigma_star"]
        except (ValueError, RuntimeError):
            ss_hi = np.nan

        swing = abs(ss_hi - ss_lo) if np.isfinite(ss_lo) and np.isfinite(ss_hi) else np.nan

        results[param_name] = {
            "sigma_star_low": ss_lo,
            "sigma_star_high": ss_hi,
            "sigma_star_baseline": sigma_baseline,
            "swing": swing,
        }

    # Sort by swing, largest first
    sorted_results = dict(
        sorted(results.items(), key=lambda x: -x[1]["swing"] if np.isfinite(x[1]["swing"]) else 0)
    )
    return sorted_results


# =====================================================================
# Full multilevel Price partition (positional within-group reward)
# =====================================================================
#
# This section assembles the two-level Price equation (Price 1972, Eq. A17;
# Hamilton 1975, Eq. 1/3/7; Frank 1998, relatedness-as-regression) with the
# framework's fitness components, treating the within-group signaling reward as
# a POSITIONAL good. Mechanism:
#
#   * Within-group: an honest signaler earns status s_W*(1-p_g) relative to a
#     non-signaler's -s_W*p_g; this is zero-sum, so it nets out of the group
#     mean but produces a positive within-group selection differential
#     (beta_0 > 0 whenever quality has mass above the floor q_min) -- so
#     monument signaling is NOT altruism in Hamilton's sense (the beta_0 < 0
#     case), and the free-rider problem is resolved within the group.
#   * Between-group: the group mean fitness is (1 - p_g*C_model)*S*K, with the
#     public goods S (networks) and K (deterrence) derived from the aggregate
#     stock M_g = p_g*M_full. This is a REINFORCEMENT term: its marginal value
#     at small M_g is sensitive to the network functional form (steep under
#     Michaelis-Menten, near-zero under threshold forms), so it is NOT a robust
#     emergence gate -- emergence is carried by the within-group term. sigma
#     governs the INTENSITY (interior equilibrium p*), not binary emergence.
#
# F (assortment/relatedness) enters only via the variance partition; it is a
# robustness variable, never a tested parameter (it maps to the unobservable
# fraction signaling p*). All defaults are Michaelis-Menten networks.


def within_group_status_differential(
    lam: float,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
) -> float:
    r"""Within-group positional status differential :math:`s_W`.

    .. math::
        s_W = \lambda \,(\mathbb{E}[q] - q_{\min})

    The within-group audience confers status for revealed quality: an honest
    signaler perceived as quality :math:`q` earns :math:`\lambda q`, while a
    non-signaler is perceived at the separating-equilibrium floor
    :math:`q_{\min}`. The differential between an average signaler and a
    non-signaler is :math:`s_W`. Status is a *positional* good (Henrich &
    Gil-White 2001; Bliege Bird & Smith 2005): within an all-signaling group it
    reshuffles status by quality but adds nothing to the group average, so it
    never enters the between-group (group-mean) fitness, only the within-group
    selection differential.
    """
    if lam <= 0:
        return 0.0
    return lam * ((q_min + q_max) / 2.0 - q_min)


def _member_fitnesses(
    p_g: float,
    sigma: float,
    lam: float,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    P_base: float = DEFAULT_P_BASE,
    conflict_mortality: float = DEFAULT_CONFLICT_MORTALITY,
    status_scale: float = 1.0,
    M_full: float | None = None,
    network_degree_spec: Any = None,
) -> tuple[float, float, float]:
    """Signaler, non-signaler, and mean fitness in a group with signaler fraction p_g.

    Public goods (survival S via networks, conflict K via deterrence) come from
    the aggregate stock M_g = p_g * M_full and are shared by all members. The
    positional within-group status is ``status_scale * s_W`` (set ``status_scale=0``
    for the pure-altruism / no-status limit). Returns ``(w_S, w_N, W_g)``.
    
    Numerical note: any tiny internal floor here guards literal
    division by zero only; the public entry points require
    lambda_W > 0 (ST-L0).
    """
    from signaling.layer1 import average_equilibrium_cost, expected_monument_stock
    from signaling.layer2 import derived_conflict_reduction
    from signaling.layer3 import network_degree, survival_probability

    if M_full is None:
        M_full = float(expected_monument_stock(n, q_min, q_max, max(lam, 1e-12)))
    M_g = p_g * M_full
    C = float(average_equilibrium_cost(lam, q_min, q_max))
    if network_degree_spec is None:
        k = float(network_degree(M_g, k_0, k_max, M_half))
    else:
        k = float(network_degree_spec.k(M_g))
    S = float(survival_probability(sigma, k, gamma))
    r = float(derived_conflict_reduction(max(M_g, 1e-9), max(M_g, 1e-9)))
    K = 1.0 - DEFAULT_WAR_COST * sigma * (1.0 - r)
    s_W = status_scale * within_group_status_differential(lam, q_min, q_max)

    # Positional status enters as a zero-sum ADDITIVE reallocation (not a
    # multiplicative modifier), so it nets out of the group mean exactly --
    # the "fully positional" requirement: within-group status competition
    # reshuffles fitness by quality without changing the group's total.
    # (A multiplicative form would leave a small cost-status deadweight residual
    # and yield Gintis-style bistability instead of graded intensity; that
    # variant is noted in the supplement, not used here.)
    w_S = S * K * ((1.0 - C) + s_W * (1.0 - p_g))        # signaler: cost + positional status
    w_N = S * K * (1.0 - s_W * p_g)                      # free-rider: no cost, loses status
    W_g = p_g * w_S + (1.0 - p_g) * w_N                  # group mean = (1 - p_g*C) * S * K exactly
    return w_S, w_N, W_g


def mixed_group_mean_fitness(p_g: float, sigma: float, lam: float, **kwargs) -> float:
    """Group mean fitness W_g(p_g) ~ (1 - p_g*C_model)*S*K (the positional reward nets out)."""
    return _member_fitnesses(p_g, sigma, lam, **kwargs)[2]


def within_group_regression(p: float, sigma: float, lam: float, **kwargs) -> float:
    r"""Within-group selection coefficient :math:`\beta_0 = w_S - w_N` at composition p.

    Positive means honest signalers out-reproduce free-riders within the group
    (free-rider problem resolved). With ``status_scale=0`` this returns
    :math:`-C_{\mathrm{model}}\,S K < 0`, recovering Hamilton (1975): without the
    status reward, within-group selection always opposes the costly trait.
    """
    w_S, w_N, _ = _member_fitnesses(p, sigma, lam, **kwargs)
    return w_S - w_N


def between_group_regression(
    p: float, sigma: float, lam: float, dp: float = 1e-4, **kwargs
) -> float:
    r"""Between-group selection coefficient :math:`\beta_1 = dW_g/dp` at composition p.

    The marginal value of a higher group signaling fraction. Positive when the
    added public-good benefit (networks, deterrence) outweighs the added cost.
    Its sign at small p is sensitive to the network functional form, so it is
    treated as a reinforcement term, not a robust emergence gate.
    """
    hi = min(p + dp, 1.0)
    lo = max(p - dp, 0.0)
    W_hi = _member_fitnesses(hi, sigma, lam, **kwargs)[2]
    W_lo = _member_fitnesses(lo, sigma, lam, **kwargs)[2]
    return (W_hi - W_lo) / (hi - lo)


def emergence_criterion(
    sigma: float, lam: float, F: float, p: float = 1e-4, **kwargs
) -> float:
    r"""Selection on signaling under assortment F (Hamilton 1975, Eq. 3, large-n limit).

    .. math::
        \mathrm{sign}(\Delta p) = \mathrm{sign}\big(F\,\beta_1 + (1-F)\,\beta_0\big)

    Positive means signaling increases. At low F the within-group status term
    :math:`\beta_0` carries it (network-form-free); at high F the between-group
    term :math:`\beta_1` governs (network-form-sensitive).
    """
    if lam <= 0.0:
        raise ValueError(
            "lam must be > 0: at lambda_W = 0 the schedule and stock vanish "
            "and the selection gradient is floating-point noise, which the "
            "root scan would classify with false confidence."
        )
    b1 = between_group_regression(p, sigma, lam, **kwargs)
    b0 = within_group_regression(p, sigma, lam, **kwargs)
    return F * b1 + (1.0 - F) * b0


def equilibrium_intensity_roots(
    sigma: float, lam: float, F: float, n_scan: int = 400, **kwargs
) -> dict[str, Any]:
    r"""All equilibria of the selection gradient :math:`g(p) = F\beta_1(p) + (1-F)\beta_0(p)`,
    with stability under the replicator flow :math:`\dot p \propto p(1-p)\,g(p)`.

    The gradient is not guaranteed monotone: under some network functional
    forms (e.g. the calibrated Hill :math:`n=2` alternative at low
    :math:`\sigma`) it has two interior zeros, so extinction and a positive
    interior state are simultaneously stable (bistability) and a single
    root-find would miss one of them. This function scans a dense grid
    (log-spaced near :math:`p = 0`, where roots can sit close to the
    boundary, then linear), brackets every sign change with ``brentq``, and
    classifies each interior root: a downward crossing (:math:`g` from + to
    -) is stable, an upward crossing unstable. The boundary states are
    classified from the gradient's sign beside them: :math:`p = 0` is stable
    iff :math:`g > 0` fails just above it, :math:`p = 1` stable iff
    :math:`g > 0` just below it.

    Returns
    -------
    dict with keys:
        'roots' : list of interior roots (ascending)
        'stable' : list of bools, stability of each interior root
        'zero_stable' : bool, stability of the extinction boundary p = 0
        'one_stable' : bool, stability of the saturation boundary p = 1
        'bistable' : bool, True when more than one attractor exists
    """
    p_lo, p_hi = 1e-4, 1.0 - 1e-4
    grid = np.unique(np.concatenate([
        np.geomspace(p_lo, 0.05, n_scan // 4),
        np.linspace(0.05, p_hi, n_scan),
    ]))
    g_vals = np.array([
        emergence_criterion(sigma, lam, F, float(p), **kwargs) for p in grid
    ])

    roots: list[float] = []
    stable: list[bool] = []
    for i in range(len(grid) - 1):
        a, b = g_vals[i], g_vals[i + 1]
        if a == 0.0:
            # Grid point exactly on a root: classify from neighbors.
            if 0 < i and np.sign(g_vals[i - 1]) != np.sign(b):
                roots.append(float(grid[i]))
                stable.append(bool(g_vals[i - 1] > 0 > b))
            continue
        if a * b < 0:
            root = float(brentq(
                lambda p: emergence_criterion(sigma, lam, F, p, **kwargs),
                float(grid[i]), float(grid[i + 1]),
            ))
            roots.append(root)
            stable.append(bool(a > 0 > b))

    zero_stable = bool(g_vals[0] < 0)
    one_stable = bool(g_vals[-1] > 0)
    n_attractors = int(zero_stable) + int(one_stable) + sum(stable)
    return {
        "roots": roots,
        "stable": stable,
        "zero_stable": zero_stable,
        "one_stable": one_stable,
        "bistable": n_attractors > 1,
    }


def equilibrium_intensity(
    sigma: float,
    lam: float,
    F: float,
    initial_fraction: float = 1e-3,
    **kwargs,
) -> float:
    r"""Equilibrium signaling fraction :math:`p^*(\sigma, F)` reached from a
    founding frequency.

    Convention (this is a basin-dependent quantity, not a global optimum):
    the returned value is the equilibrium the replicator flow
    :math:`\dot p \propto p(1-p)\,g(p)`, with
    :math:`g(p) = F\beta_1(p) + (1-F)\beta_0(p)`, reaches from
    ``initial_fraction`` (default: invasion from rarity, :math:`p_0 =
    10^{-3}`). From :math:`p_0` the one-dimensional flow moves in the
    direction of :math:`\mathrm{sign}(g(p_0))` and stops at the nearest
    equilibrium in that direction: the first interior root above
    :math:`p_0` (or the saturated value 1.0) when :math:`g(p_0) > 0`, the
    first interior root below (or extinction 0.0) when :math:`g(p_0) < 0`.

    The gradient need not be monotone: under some network forms (e.g. the
    calibrated Hill :math:`n = 2` alternative at low :math:`\sigma`) it has
    two interior zeros, extinction and a positive interior state are both
    stable, and the outcome genuinely depends on the founding frequency;
    use :func:`equilibrium_intensity_roots` for the full equilibrium
    structure. Under the default Michaelis-Menten network the gradient has
    a single crossing over the reported sweeps and the from-rarity value is
    the unique attractor. At moderate-to-high F the interior :math:`p^*`
    rises monotonically with :math:`\sigma`: environmental uncertainty
    governs the *intensity* of monument investment, not binary emergence
    (the framework's central result).
    """
    if not (0.0 < initial_fraction < 1.0):
        raise ValueError("initial_fraction must lie strictly inside (0, 1).")
    structure = equilibrium_intensity_roots(sigma, lam, F, **kwargs)
    roots = structure["roots"]
    g0 = emergence_criterion(sigma, lam, F, initial_fraction, **kwargs)
    if g0 > 0:
        above = [r for r in roots if r > initial_fraction]
        return float(above[0]) if above else 1.0
    if g0 < 0:
        below = [r for r in roots if r < initial_fraction]
        return float(below[-1]) if below else 0.0
    return float(initial_fraction)


def price_partition(
    p_g_values: NDArray[np.float64], sigma: float, lam: float, **kwargs
) -> dict[str, float]:
    r"""Two-level Price partition over an explicit ensemble of group compositions.

    Price (1972) Eq. A17 / Hamilton (1975) Eq. 1, for equal-sized groups:

    .. math::
        \bar w\, \Delta \bar p = \mathrm{Cov}_G(W_g, p_g)
                                 + \mathbb{E}_G[\mathrm{Cov}_g(w_i, p_i)]

    The within-group covariance for a binary trait in group g is
    :math:`p_g (1-p_g)(w_S - w_N)`. The transmission term is zero (individual
    strategies fixed within a generation). Returns the two terms, their sum, the
    directly computed :math:`\bar w \Delta \bar p` (selection change in mean
    signaling), and the identity residual (~0 validates the partition).
    """
    p_g = np.asarray(p_g_values, dtype=float)
    fits = [_member_fitnesses(float(p), sigma, lam, **kwargs) for p in p_g]
    wS = np.array([f[0] for f in fits])
    wN = np.array([f[1] for f in fits])
    Wg = np.array([f[2] for f in fits])
    w_bar = float(Wg.mean())
    if w_bar == 0.0:
        raise ValueError(
            "Ensemble mean fitness is exactly zero; the descendant-frequency "
            "form p-bar' = E[w p]/w-bar is undefined there (Price identity "
            "requires positive mean fitness). Reachable only at degenerate "
            "parameters, e.g. C_model = 1."
        )
    p_bar = float(p_g.mean())
    # selection change in mean signaling (transmission term = 0)
    dp_sel = float((p_g * wS).sum() / Wg.sum() - p_bar)
    cov_between = float(np.mean((Wg - w_bar) * (p_g - p_bar)))
    cov_within = float(np.mean(p_g * (1.0 - p_g) * (wS - wN)))
    return {
        "cov_between": cov_between,
        "cov_within": cov_within,
        "partition_sum": cov_between + cov_within,
        "w_bar_delta_p": w_bar * dp_sel,
        "identity_residual": (cov_between + cov_within) - w_bar * dp_sel,
    }


def collective_optimum_fraction(
    sigma: float, lam: float, n_grid: int = 200, **kwargs: Any
) -> float:
    r"""Builder fraction :math:`p_{\mathrm{opt}}` that maximizes group mean fitness.

    The group mean fitness :math:`W_g(p) = (1 - pC)\,S(\sigma, k(pM))\,K` is
    *non-monotone* in :math:`p`: a small builder minority already supplies most
    of the saturating public good (network survival :math:`S` and deterrence
    :math:`K` both saturate in the aggregate stock :math:`M_g = pM`), while
    each additional builder mainly adds the cost :math:`C`. The collectively
    optimal fraction is therefore *interior* (:math:`dW_g/dp = 0`), strictly
    below full adoption, and this holds for every network form tested
    (Michaelis-Menten, exponential, Hill, piecewise linear; pass the form via
    ``network_degree_spec``).

    Because within-group positional competition makes building individually
    favored (:math:`\beta_0 = SK(s_W - C) > 0`) up to full participation,
    groups are driven *past* :math:`p_{\mathrm{opt}}` toward full adoption: a
    reverse tragedy of the commons in which positional status overshoots the
    group's collective fitness optimum. This is the sense in which the
    between-group level enters the result: it does not favor maximal building,
    and the gap between :math:`p_{\mathrm{opt}}` and the individually-driven
    equilibrium is a between-group prediction (the within-group and between-group selection section).

    Returns :math:`p_{\mathrm{opt}} \in (0, 1)`, or a boundary value if
    :math:`W_g` is monotone on the interior. A grid scan locates the global
    maximum (some network forms, e.g. Hill, make :math:`W_g` dip near
    :math:`p=0` before rising to an interior peak), then a local root-find on
    :math:`dW_g/dp` refines it.
    """
    ps = np.linspace(0.0, 1.0, n_grid + 1)
    Wg = np.array(
        [_member_fitnesses(float(p), sigma, lam, **kwargs)[2] for p in ps]
    )
    i = int(np.argmax(Wg))
    if 0 < i < n_grid:
        a, b = float(ps[i - 1]), float(ps[i + 1])
        ga = between_group_regression(a, sigma, lam, **kwargs)
        gb = between_group_regression(b, sigma, lam, **kwargs)
        if ga > 0.0 > gb:
            return float(
                brentq(
                    lambda p: between_group_regression(p, sigma, lam, **kwargs), a, b
                )
            )
    return float(ps[i])
