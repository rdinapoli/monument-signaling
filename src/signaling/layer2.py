"""Intergroup assessment and conflict resolution.

Derives equilibrium conflict probability P_conflict(M_g, M_h) from
contest-theoretic assessment models, deriving the conflict reduction
parameter from strategic interaction rather than assuming it.

**Derived competitive return**: This module also produces lambda_C, the
marginal fitness value of conflict deterrence per unit of monument
investment, defined as the marginal REDUCTION in expected conflict losses
(a positive benefit):

    lambda_C = - d/dx_i [ sum_h f_h * P_conflict(M_g, M_h) * C_conflict ]

Since added stock lowers the conflict probability, the bracketed derivative
is negative and lambda_C > 0. Under the canonical lambda_W-only
specification, lambda_C is a diagnostic evaluated at M_g(lambda_W) and its
realized fitness effect is the war-avoidance advantage entering through K;
it informs (but does not feed back into) the channel-dominance comparison.

**Assessment noise**: The assessment noise parameter is constrained by the
signal fidelity rho_C from the channel selection analysis, not independently
specified. Monument signals have near-zero deception noise (physically
witnessed, hard to fake) but introduce lag noise (cumulative, reflecting
historical capacity) and dimension mismatch noise (productive capacity
correlates imperfectly with fighting capacity).

**Assessment mechanism**: Mutual assessment (Enquist-Leimar 1983) is
theoretically favored for monument-mediated conflict because monuments are
durable and visible across territorial boundaries, enabling comparison
without direct display. This is not just empirical fit; it follows from
the same signal properties that select for monuments as the channel.

Covers:
- Mutual assessment model (Enquist-Leimar): primary mechanism
- Self-assessment model (Taylor-Elwood): alternative yielding opposite predictions
- Conventional settlement (Bourgeois strategy): alternative
- War of attrition formulation (Hammerstein-Parker)
- lambda_C derivation from P_conflict and conflict cost parameters
- Calibration to reproduce initial model r = 0.75 as a special case

See the main-text intergroup assessment and conflict resolution section
(conflict probability equation; mutual assessment) and SI "War avoidance:
the conflict term as scarcity-scaled mutual deterrence".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

from signaling.calibration import (
    DEFAULT_ALPHA_CONFLICT,
    DEFAULT_BETA_CONFLICT,
    DEFAULT_C_RATE,
    DEFAULT_CONFLICT_COST,
    DEFAULT_D,
    DEFAULT_DELTA,
    DEFAULT_DISPUTE_FREQ,
    DEFAULT_KAPPA,
    DEFAULT_M_OWN,
    DEFAULT_M_SCALE,
    DEFAULT_M_THRESH,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_P_BASE,
    DEFAULT_RHO_MONUMENT,
    DEFAULT_RHO_PROD_FIGHT,
    DEFAULT_SIGMA_0,
    DEFAULT_SIGMA_Q,
    DEFAULT_STEEPNESS,
    DEFAULT_T_0,
    DEFAULT_V,
    DEFAULT_VAR_RHP,
)

# =====================================================================
# Signal-RHP mapping
# =====================================================================


def effective_noise(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    sigma_0: float = DEFAULT_SIGMA_0,
    kappa: float = DEFAULT_KAPPA,
) -> float | NDArray[np.float64]:
    r"""Compute effective assessment noise given monument investments.

    The effective noise decreases with total investment because larger
    monuments are more informative signals of group RHP:

    .. math::
        \sigma_{\text{eff}}(M_g, M_h)
            = \frac{\sigma_0}{\sqrt{1 + \kappa (M_g + M_h)}}

    Setting kappa = 0 recovers constant noise (pure SAM baseline).

    Parameters
    ----------
    M_g : float or array
        Monument stock of group g. Must be >= 0.
    M_h : float or array
        Monument stock of group h. Must be >= 0.
    sigma_0 : float
        Baseline assessment noise when total investment is zero.
        Must be > 0.
    kappa : float
        Information gain rate. Must be >= 0.

    Returns
    -------
    float or array
        Effective assessment noise sigma_eff > 0.

    See the mutual-assessment conflict probability equation (main-text,
    effective-noise definition) and SI "Assessment model parameter
    sensitivity" (information-gain rate kappa).
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    M_h = np.asarray(M_h, dtype=np.float64)
    total = M_g + M_h
    return sigma_0 / np.sqrt(1.0 + kappa * total)


# =====================================================================
# Constrained assessment noise
# =====================================================================


def constrained_sigma_0(
    rho_monument: float = DEFAULT_RHO_MONUMENT,
    V: float = DEFAULT_V,
    delta: float = DEFAULT_DELTA,
    sigma_q: float = DEFAULT_SIGMA_Q,
    rho_prod_fight: float = DEFAULT_RHO_PROD_FIGHT,
    var_RHP: float = DEFAULT_VAR_RHP,
) -> dict[str, float]:
    r"""Compose baseline assessment noise from a three-component variance decomposition.

    SCOPE: this is a CONSISTENCY CHECK on the assumed model value
    sigma_0 = 1.0 (SI parameter table), not a derivation of it. Under the
    stated observation model (three additive, mutually uncorrelated error
    components in common RHP units) the components compose at defaults to
    sigma_0 ~ 0.89, close to but below the assumed 1.0; the model equations
    use the assumed value. Decomposition:

    .. math::
        \sigma_0^2 = \sigma_{\text{deception}}^2
                    + \sigma_{\text{lag}}^2
                    + \sigma_{\text{mismatch}}^2

    **Deception** (near zero for monuments): how fakeable the signal is.
    Controlled by signal fidelity rho_monument from the channel selection
    analysis.

    **Lag**: monuments are cumulative signals. The observation model is the
    UNNORMALIZED depreciating stock M_t = (1 - delta) M_{t-1} + q_t (iid
    capacity innovations q_t with variance sigma_q^2), whose stationary
    variance is sigma_q^2 / (2 delta - delta^2); the lag component is the
    inherited-history part of the current-capacity error:

    .. math::
        \sigma_{\text{lag}}^2
            = \mathrm{Var}[(1-\delta) M_{t-1}]
            = \sigma_q^2 \cdot \frac{(1-\delta)^2}{2\delta - \delta^2}

    The stock-model choice is load-bearing: a NORMALIZED exponentially
    weighted signal M_t = (1 - delta) M_{t-1} + delta q_t would scale both
    the stationary variance and this inherited component by delta^2
    (inherited 0.0038 against 0.3837 at delta = 0.1, sigma_q = 0.3; SI
    "Assessment noise decomposition"). An earlier docstring described the
    stock as an EWMA while implementing the unnormalized formula; the
    unnormalized accumulating stock is the intended and implemented model.

    Higher delta (faster depreciation) reduces lag noise because the signal
    tracks current capacity more closely. This provides a second reason,
    beyond the Spence condition, why signal depreciation is structurally
    beneficial: it improves assessment accuracy.

    **Mismatch**: productive capacity (what monuments reveal) correlates
    imperfectly with fighting capacity (what matters for conflict). The
    unexplained variance is (1 - rho_prod_fight^2) * Var(RHP).

    Parameters
    ----------
    rho_monument : float
        Signal fidelity for monument channel, in [0, 1].
    V : float
        Contest value (normalized). Must be > 0.
    delta : float
        Signal depreciation rate, in (0, 1].
    sigma_q : float
        Standard deviation of group capacity fluctuations. Must be >= 0.
    rho_prod_fight : float
        Correlation between productive and fighting capacity, in [0, 1].
    var_RHP : float
        Variance of fighting capacity across groups. Must be >= 0.

    Returns
    -------
    dict with keys:
        'sigma_0' : derived baseline assessment noise
        'sigma_0_sq' : sigma_0 squared (total variance)
        'sigma_deception' : deception noise component (std dev)
        'sigma_lag' : lag noise component (std dev)
        'sigma_mismatch' : mismatch noise component (std dev)
        'sigma_deception_sq' : deception variance
        'sigma_lag_sq' : lag variance
        'sigma_mismatch_sq' : mismatch variance
        'ewma_multiplier' : the multiplier (1-delta)^2/(2 delta - delta^2)
            (key name retained for API stability; it is the inherited-history
            variance multiplier of the UNNORMALIZED stock, not an EWMA variance)

    Raises
    ------
    ValueError
        If parameters are outside valid ranges.

    See SI "Assessment noise decomposition".
    """
    if not 0.0 <= rho_monument <= 1.0:
        raise ValueError(f"rho_monument must be in [0, 1], got {rho_monument}")
    if not 0.0 < delta <= 1.0:
        raise ValueError(f"delta must be in (0, 1], got {delta}")
    if not 0.0 <= rho_prod_fight <= 1.0:
        raise ValueError(f"rho_prod_fight must be in [0, 1], got {rho_prod_fight}")
    if sigma_q < 0.0:
        raise ValueError(f"sigma_q must be >= 0, got {sigma_q}")
    if var_RHP < 0.0:
        raise ValueError(f"var_RHP must be >= 0, got {var_RHP}")
    if V <= 0.0:
        raise ValueError(f"V must be > 0, got {V}")

    # Component 1: deception noise
    # Low fidelity means the signal is fakeable, increasing noise.
    sigma_deception_sq = (1.0 - rho_monument) * V**2

    # Component 2: lag noise
    # The monument stock is an EWMA of past investment flows. If capacity
    # has variance sigma_q^2, the EWMA has this variance multiplier.
    # At delta=1, the multiplier is 0 (signal fully current each period).
    # As delta->0, the multiplier diverges (infinite history).
    if delta >= 1.0:
        ewma_mult = 0.0
    else:
        ewma_mult = (1.0 - delta) ** 2 / (2.0 * delta - delta**2)
    sigma_lag_sq = ewma_mult * sigma_q**2

    # Component 3: dimension mismatch noise
    # Productive capacity explains rho_prod_fight^2 of RHP variance;
    # the remainder is unexplained and contributes assessment noise.
    sigma_mismatch_sq = (1.0 - rho_prod_fight**2) * var_RHP

    # Total
    sigma_0_sq = sigma_deception_sq + sigma_lag_sq + sigma_mismatch_sq
    sigma_0 = float(np.sqrt(sigma_0_sq))

    return {
        "sigma_0": sigma_0,
        "sigma_0_sq": float(sigma_0_sq),
        "sigma_deception": float(np.sqrt(sigma_deception_sq)),
        "sigma_lag": float(np.sqrt(sigma_lag_sq)),
        "sigma_mismatch": float(np.sqrt(sigma_mismatch_sq)),
        "sigma_deception_sq": float(sigma_deception_sq),
        "sigma_lag_sq": float(sigma_lag_sq),
        "sigma_mismatch_sq": float(sigma_mismatch_sq),
        "ewma_multiplier": float(ewma_mult),
    }


# =====================================================================
# Mutual assessment (primary model, Enquist-Leimar 1983)
# =====================================================================


def war_of_attrition_deterrence(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    beta: float = DEFAULT_BETA_CONFLICT,
) -> float | NDArray[np.float64]:
    r"""Absolute-deterrence factor from a war of attrition (Maynard Smith 1974).

    This is the mechanism that averts war between two *symmetric, well-armed*
    groups, where relative assessment (Enquist & Leimar 1983) cannot help
    because neither side is the weaker that should concede. Two groups contest
    a resource of value :math:`V`. The expected destruction of escalation
    scales with their combined fighting capacity, which the monument stocks
    index: :math:`\text{war cost} = c\,(M_g + M_h)`. A risk-neutral group
    escalates only when the prize is worth the expected destruction,
    :math:`V > c\,(M_g + M_h)`. Treating the per-encounter stake :math:`V` as
    drawn from a log-logistic distribution (shape 1, scale :math:`s`) -- the
    standard heavy-tailed model for contested valuations -- the escalation
    (non-deterrence) probability is the survival function

    .. math::
        D(M_g, M_h) = \Pr\!\big[V > c\,(M_g+M_h)\big]
                    = \frac{1}{1 + \beta\,(M_g + M_h)},
        \qquad \beta = c/s .

    This is the absolute-deterrence factor that multiplies the assessment
    component in both regimes (:func:`mutual_assessment_conflict_prob`,
    :func:`self_assessment_conflict_prob`). It decreases monotonically from 1
    (no deterrence at zero stock) toward 0 (overwhelming mutual deterrence)
    and depends only on the *total* stock, not the asymmetry: deterrence is
    the symmetric-dyad complement to relative assessment. Because relative
    assessment alone predicts that precisely-assessed symmetric dyads fight
    *more* (the war-of-attrition parity limit), this derived deterrence factor
    -- not the assessment term -- is what produces the positive build-build
    conflict reduction :math:`r(M, M) > 0` that enters the threshold
    :math:`\sigma^*`. :math:`\beta = c/s` is the ratio of per-unit war cost to
    the stakes scale, the single calibrated parameter of the deterrence
    channel (the main-text war-avoidance section). An exponential stakes distribution
    instead yields :math:`D = \exp(-\beta(M_g+M_h))`; the two forms agree to
    first order in :math:`\beta(M_g+M_h)` but diverge at the operating anchor
    (:math:`r \approx 0.43` log-logistic vs :math:`\approx 0.77` exponential),
    yet give near-identical thresholds because the avoided war cost is far
    below the construction cost (the main-text war-avoidance section).

    Parameters
    ----------
    M_g, M_h : float or array
        Monument stocks (>= 0), indexing group fighting capacity.
    beta : float
        Deterrence coefficient :math:`\beta = c/s` (>= 0).

    Returns
    -------
    float or array
        Deterrence factor in (0, 1]; 1 at zero combined stock.

    See SI "War avoidance: the conflict term as scarcity-scaled mutual
    deterrence" and the main-text war-avoidance and competitive-feedback
    section.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    M_h = np.asarray(M_h, dtype=np.float64)
    return 1.0 / (1.0 + beta * (M_g + M_h))


def mutual_assessment_conflict_prob(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    sigma_0: float = DEFAULT_SIGMA_0,
    V: float = DEFAULT_V,
    D: float = DEFAULT_D,
    T_0: float = DEFAULT_T_0,
    beta: float = DEFAULT_BETA_CONFLICT,
    kappa: float = DEFAULT_KAPPA,
    P_base: float = DEFAULT_P_BASE,
) -> float | NDArray[np.float64]:
    r"""Conflict probability under mutual assessment (Enquist-Leimar 1983).

    The perceived RHP difference is

    .. math::
        \hat{\Delta R} \sim N(M_g - M_h, \; 2\sigma_{\text{eff}}^2)

    Conflict occurs when :math:`|\hat{\Delta R}| < T` (escalation
    threshold), combined with absolute deterrence:

    .. math::
        P_{\text{raw}}(M_g, M_h)
            = \left[\Phi\!\left(\frac{T - \delta}{\sqrt{2}\,\sigma_{\text{eff}}}\right)
              - \Phi\!\left(\frac{-T - \delta}{\sqrt{2}\,\sigma_{\text{eff}}}\right)\right]
              \cdot \frac{1}{1 + \beta(M_g + M_h)}

    where :math:`\delta = |M_g - M_h|` and :math:`T = T_0 V / D`.
    The result is normalized so that :math:`P(0,0) = P_{\text{base}}`.

    The framework decomposes the conflict probability into two
    structurally separate mechanisms operating in product
    (the main-text mutual-vs-self assessment section): an assessment-driven component
    (the normalized Gaussian-CDF ratio above) and an absolute-deterrence
    component (the :math:`1/(1 + \beta(M_g + M_h))` factor). Both apply
    identically across mutual and self-assessment regimes; see
    :func:`self_assessment_conflict_prob` for the parallel self-assessment
    construction. Pure mutual assessment (setting ``beta = 0``) predicts
    that well-informed symmetric dyads fight MORE because precise
    assessment confirms parity (the standard war-of-attrition limit);
    the absolute-deterrence factor captures the empirical observation
    that heavily-armed neighbors rarely escalate even at parity. The
    two mechanisms together produce the framework's prediction that
    :math:`r_{\mathrm{mutual}}(M, M)` rises monotonically with M.

    When deterrence is weak (small ``beta``), the parity amplification can
    push the symmetric-dyad probability ABOVE ``P_base`` (e.g. at
    ``beta = 0``, ``M_g = M_h = 10`` and defaults, P = 0.0166 = 1.66
    P_base), exactly as the main-text equation requires; the corresponding
    reduction :math:`r = 1 - P/P_{\mathrm{base}}` is then negative. The
    function bounds the result only as a probability, in [0, 1]; clipping
    at ``P_base`` would silently delete the amplification.

    Parameters
    ----------
    M_g, M_h : float or array
        Monument stocks. Must be >= 0.
    sigma_0 : float
        Baseline assessment noise. Must be > 0.
    V : float
        Contest value (normalized). Must be > 0.
    D : float
        Fighting cost. Must be > 0.
    T_0 : float
        Base escalation threshold scale. Must be > 0.
    beta : float
        Absolute deterrence coefficient. Must be >= 0.
    kappa : float
        Information gain rate. Must be >= 0.
    P_base : float
        Baseline conflict probability (at zero investment). Must be in (0, 1].

    Returns
    -------
    float or array
        Conflict probability in [0, 1]. Exceeds P_base at symmetric dyads
        when the deterrence factor is too weak to outrun the parity
        amplification.

    See the main-text mutual-assessment conflict probability equation and
    SI "Assessment model parameter sensitivity".
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    M_h = np.asarray(M_h, dtype=np.float64)

    delta = np.abs(M_g - M_h)
    total = M_g + M_h
    T = T_0 * V / D

    sig_eff = effective_noise(M_g, M_h, sigma_0, kappa)
    sqrt2_sig = np.sqrt(2.0) * sig_eff

    # Gaussian CDF component: probability that perceived difference is within T
    P_gauss = norm.cdf((T - delta) / sqrt2_sig) - norm.cdf((-T - delta) / sqrt2_sig)

    # Absolute-deterrence factor, derived from a war of attrition (the
    # symmetric-dyad mechanism; see war_of_attrition_deterrence). The
    # assessment term P_gauss carries the relative (asymmetric-dyad) mechanism.
    deterrence = war_of_attrition_deterrence(M_g, M_h, beta)

    P_raw = P_gauss * deterrence

    # Normalize so P(0, 0) = P_base
    # At M_g = M_h = 0: delta=0, total=0, sig_eff=sigma_0
    sqrt2_sig0 = np.sqrt(2.0) * sigma_0
    P_raw_00 = (norm.cdf(T / sqrt2_sig0) - norm.cdf(-T / sqrt2_sig0))
    # deterrence at (0,0) is 1.0

    # Guard against division by zero (P_raw_00 should always be positive
    # for T > 0 and sigma_0 > 0, but protect numerically)
    if np.any(P_raw_00 < 1e-15):
        return np.zeros_like(P_raw) + P_base

    P_conflict = P_base * P_raw / P_raw_00

    # Bound only as a probability. The displayed equation permits values
    # above P_base at parity when deterrence is weak (parity amplification);
    # clipping at P_base would delete that mechanism.
    return np.clip(P_conflict, 0.0, 1.0)


def mutual_assessment_conflict_prob_exponential_stake(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    sigma_0: float = DEFAULT_SIGMA_0,
    V: float = DEFAULT_V,
    D: float = DEFAULT_D,
    T_0: float = DEFAULT_T_0,
    beta: float = DEFAULT_BETA_CONFLICT,
    kappa: float = DEFAULT_KAPPA,
    P_base: float = DEFAULT_P_BASE,
) -> float | NDArray[np.float64]:
    r"""Mutual-assessment conflict probability under exponential-STAKE deterrence.

    Identical to :func:`mutual_assessment_conflict_prob` except that the
    absolute-deterrence factor is the exponential-stake form

    .. math::
        \exp(-\beta (M_g + M_h))

    in place of the log-logistic :math:`1/(1 + \beta(M_g + M_h))` (the two
    agree to first order in :math:`\beta (M_g + M_h)`). This is the
    functional-form robustness variant discussed in the SI's stake-threshold
    escalation section: at the anchor (M_g = M_h = 10, beta = 0.1) the
    deterrence factor is exp(-2) ~ 0.135 versus the log-logistic 1/3, so the
    derived reduction r = 1 - P/P_base is ~ 0.77 versus ~ 0.43.

    NOTE: distinct from :func:`mutual_assessment_exponential`, which replaces
    the Gaussian ASSESSMENT factor with an exponential approximation and
    keeps the log-logistic deterrence. The two "exponential" variants are
    different robustness axes.

    Parameters as in :func:`mutual_assessment_conflict_prob`.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    M_h = np.asarray(M_h, dtype=np.float64)

    delta = np.abs(M_g - M_h)
    T = T_0 * V / D

    sig_eff = effective_noise(M_g, M_h, sigma_0, kappa)
    sqrt2_sig = np.sqrt(2.0) * sig_eff
    P_gauss = norm.cdf((T - delta) / sqrt2_sig) - norm.cdf((-T - delta) / sqrt2_sig)

    deterrence = np.exp(-beta * (M_g + M_h))

    P_raw = P_gauss * deterrence

    sqrt2_sig0 = np.sqrt(2.0) * sigma_0
    P_raw_00 = (norm.cdf(T / sqrt2_sig0) - norm.cdf(-T / sqrt2_sig0))
    if np.any(P_raw_00 < 1e-15):
        return np.zeros_like(P_raw) + P_base

    P_conflict = P_base * P_raw / P_raw_00
    return np.clip(P_conflict, 0.0, 1.0)


def mutual_assessment_exponential(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    sigma_0: float = DEFAULT_SIGMA_0,
    alpha: float = DEFAULT_ALPHA_CONFLICT,
    beta: float = DEFAULT_BETA_CONFLICT,
    kappa: float = DEFAULT_KAPPA,
    P_base: float = DEFAULT_P_BASE,
) -> float | NDArray[np.float64]:
    r"""Tractable exponential approximation to the mutual assessment model.
    NOTE: this is the exponential approximation to the Gaussian ASSESSMENT
    factor (with the log-logistic deterrence retained); the SI's
    exponential-STAKE deterrence variant is
    :func:`mutual_assessment_conflict_prob_exponential_stake`. Different
    robustness axes despite the similar names.


    .. math::
        P_{\text{conflict}} = P_{\text{base}}
            \cdot \frac{\exp(-\alpha \cdot \delta / \sigma_{\text{eff}})}
                       {1 + \beta (M_g + M_h)}

    where :math:`\delta = |M_g - M_h|`. This is a closed-form
    approximation useful for calibration and sensitivity analysis
    where the Gaussian CDF form is less tractable.

    **Approximation quality**: With default alpha=1.0, the maximum
    absolute deviation from the Gaussian CDF form is ~0.004 over
    M in [0.1, 15], occurring at low M values near strong asymmetry.
    This is ~41% of P_base in relative terms but small in absolute
    fitness impact. The exponential form preserves all qualitative
    predictions (monotonicity in delta and total investment). For
    quantitative calibration, use the Gaussian CDF form
    (mutual_assessment_conflict_prob) directly.

    Parameters
    ----------
    M_g, M_h : float or array
        Monument stocks. Must be >= 0.
    sigma_0 : float
        Baseline assessment noise. Must be > 0.
    alpha : float
        Exponential decay rate. Must be >= 0.
    beta : float
        Absolute deterrence coefficient. Must be >= 0.
    kappa : float
        Information gain rate. Must be >= 0.
    P_base : float
        Baseline conflict probability. Must be in (0, 1].

    Returns
    -------
    float or array
        Conflict probability in [0, P_base].
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    M_h = np.asarray(M_h, dtype=np.float64)

    delta = np.abs(M_g - M_h)
    total = M_g + M_h
    sig_eff = effective_noise(M_g, M_h, sigma_0, kappa)

    P = P_base * np.exp(-alpha * delta / sig_eff) / (1.0 + beta * total)

    return np.clip(P, 0.0, P_base)


# =====================================================================
# Self-assessment (alternative, Taylor-Elwood 2003)
# =====================================================================


def self_assessment_conflict_prob(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    M_thresh: float = DEFAULT_M_THRESH,
    M_scale: float = DEFAULT_M_SCALE,
    beta: float = DEFAULT_BETA_CONFLICT,
    P_base: float = DEFAULT_P_BASE,
) -> float | NDArray[np.float64]:
    r"""Conflict probability under self-assessment (Taylor-Elwood 2003).

    The framework's conflict probability decomposes into two structurally
    separate mechanisms operating in product: an assessment-driven
    component and an absolute-deterrence component. The two mechanisms
    apply identically across assessment regimes (the main-text assessment section).
    Under self-assessment, the assessment-driven component is the product
    of per-group escalation probabilities, each based on the group's own
    monument investment independent of the opponent's signal:

    .. math::
        P_{\text{conflict}}^{\text{self}}(M_g, M_h)
            = \min\left\{1,\; P_{\text{base}} \cdot
              \underbrace{\frac{p(M_g) \cdot p(M_h)}{p(0)^2}}_{\text{assessment-driven (= 1 at } M=0)}
              \cdot \underbrace{\frac{1}{1 + \beta(M_g + M_h)}}_{\text{absolute deterrence}}\right\}

    where :math:`p(M) = \sigma((M - M_{\text{thresh}}) / M_{\text{scale}})`
    is the logistic sigmoid. The outer :math:`\min\{1, \cdot\}` is part of
    the model: the normalized assessment ratio is unbounded (the
    small-baseline normalization can push the product far above 1 at high
    stocks or high thresholds), and the probability saturates at certain
    conflict, the ceiling the SI self-assessment sensitivity section
    describes. Comparative statics are flat, and :math:`r_{\text{self}}`
    floors at :math:`1 - 1/P_{\text{base}}`, wherever the ceiling binds
    (the bound is displayed explicitly in the SI equation). The
    assessment term is normalized to 1 at
    :math:`M = 0` (dividing by :math:`p(0)^2`) and scaled by the common
    no-signaling baseline :math:`P_{\text{base}}`, so
    :math:`P_{\text{conflict}}^{\text{self}}(0, 0) = P_{\text{base}}`,
    matching :func:`mutual_assessment_conflict_prob`. At zero investment no
    signal exists, so both regimes reduce to the same baseline conflict rate
    by construction; the absolute-deterrence factor is likewise identical,
    so the regimes differ only in their assessment-driven component.

    The qualitative discriminating prediction at symmetric high-investment
    dyads is the sign of the conflict-reduction ratio: under mutual
    assessment, clear assessment combined with absolute deterrence yields
    :math:`r_{\text{mutual}}(M, M) > 0` (rising monotonically with M).
    Under self-assessment, the assessment-driven product :math:`p(M)^2`
    saturates near 1 once both parties exceed the escalation threshold,
    pushing the conflict probability well above :math:`P_{\text{base}}`
    despite the deterrence damping. :math:`r_{\text{self}}(M, M) < 0` for
    :math:`M > M_{\text{thresh}}`, with the minimum located near the
    intersection of the assessment-saturation regime and the deterrence
    half-saturation :math:`M \sim 1/(2\beta)`. For larger M, :math:`r_{\text{self}}`
    rises partially as deterrence dominates the saturated assessment
    term, but remains substantially below :math:`r_{\text{mutual}}` over
    the parameter range relevant to monument-building societies.

    The discriminating signature is therefore the sign and magnitude of
    :math:`r` at symmetric high-M dyads, not its monotonicity in M.

    Parameters
    ----------
    M_g, M_h : float or array
        Monument stocks. Must be >= 0.
    M_thresh : float
        Threshold monument stock for escalation confidence.
    M_scale : float
        Steepness of the sigmoid. Must be > 0.
    beta : float
        Absolute deterrence coefficient. Must be >= 0. Default matches
        :func:`mutual_assessment_conflict_prob`; setting ``beta = 0``
        recovers the pure-assessment self-assessment formulation used
        in some self-assessment expositions (e.g., Taylor-Elwood 2003).
    P_base : float
        Common no-signaling baseline conflict probability at M = 0,
        shared with :func:`mutual_assessment_conflict_prob`. Must be in
        (0, 1]. Normalizes the self-assessment expression to the same
        baseline as the mutual model so the two are directly comparable.

    Returns
    -------
    float or array
        Conflict probability in [0, 1].

    See the main-text self-assessment conflict probability.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    M_h = np.asarray(M_h, dtype=np.float64)

    # Anchor to the common no-signaling baseline P_base, with the assessment
    # term normalized to 1 at M = 0. At zero investment no signal exists, so
    # both assessment regimes must reduce to the same baseline conflict rate;
    # this makes the expression structurally parallel to
    # mutual_assessment_conflict_prob (P_base * [assessment factor = 1 at
    # M = 0] * deterrence), so any divergence at positive M is attributable
    # to the assessment mechanism rather than a baseline mismatch.
    #
    # Computed in LOG space: with the logistic p(M) = sigma((M - T)/s),
    # log p(M) = -logaddexp(0, (T - M)/s), so the normalized ratio
    # p(M_g) p(M_h) / p(0)^2 never forms a 0/0. The direct formula
    # underflowed p(0) to exactly 0 for M_thresh/M_scale beyond ~372 and
    # returned silent NaN, breaking the P(0,0) = P_base anchoring. The
    # outer min{1, .} is applied as a
    # log-domain cap, exp(min(log_raw, 0)), which is exact and cannot
    # overflow.
    z_g = (M_g - M_thresh) / M_scale
    z_h = (M_h - M_thresh) / M_scale
    log_assessment = (
        2.0 * np.logaddexp(0.0, M_thresh / M_scale)
        - np.logaddexp(0.0, -z_g)
        - np.logaddexp(0.0, -z_h)
    )

    # Absolute-deterrence factor, derived from a war of attrition and identical
    # across assessment regimes (see war_of_attrition_deterrence).
    deterrence = war_of_attrition_deterrence(M_g, M_h, beta)

    log_raw = np.log(P_base) + log_assessment + np.log(deterrence)
    return np.exp(np.minimum(log_raw, 0.0))


# =====================================================================
# Bourgeois convention (alternative, Maynard Smith 1982)
# =====================================================================


def bourgeois_conflict_prob(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    M_own: float = DEFAULT_M_OWN,
    P_base: float = DEFAULT_P_BASE,
    steepness: float = DEFAULT_STEEPNESS,
) -> float | NDArray[np.float64]:
    r"""Conflict probability under Bourgeois convention (Maynard Smith 1982).

    Monument investment establishes conventional ownership claims.
    Conflict is low when ownership is unambiguous (one group clearly
    above threshold, the other below) and high when ambiguous:

    .. math::
        P_{\text{conflict}} = P_{\text{base}} \cdot (1 - \text{ownership\_clarity})

    Ownership clarity is high when one group's M is well above M_own
    and the other's well below; low when both are near M_own or both
    far from it on the same side.

    Parameters
    ----------
    M_g, M_h : float or array
        Monument stocks. Must be >= 0.
    M_own : float
        Ownership threshold. Groups above this are "owners."
    P_base : float
        Baseline conflict probability when ownership is fully ambiguous.
    steepness : float
        Controls sharpness of the ownership classification. Must be > 0.

    Returns
    -------
    float or array
        Conflict probability in [0, P_base].

    See the main-text intergroup conflict-assessment figure (Bourgeois
    convention panel).
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    M_h = np.asarray(M_h, dtype=np.float64)

    # Ownership score: how clearly each group is classified as owner
    # Using tanh for smooth classification in [-1, 1]
    # Positive = owner, negative = intruder
    score_g = np.tanh(steepness * (M_g - M_own))
    score_h = np.tanh(steepness * (M_h - M_own))

    # Ownership clarity is high when scores have opposite signs
    # (one owner, one intruder) and each is confident.
    # clarity = |score_g - score_h| / 2, maximal when one is +1 and other -1
    clarity = np.abs(score_g - score_h) / 2.0

    P = P_base * (1.0 - clarity)

    return np.clip(P, 0.0, P_base)


# =====================================================================
# War of attrition (Hammerstein-Parker 1982)
# =====================================================================


def war_of_attrition_cost(
    M_g: float,
    M_h: float,
    V: float = DEFAULT_V,
    c_rate: float = DEFAULT_C_RATE,
) -> dict[str, float]:
    r"""War of attrition outcomes for a territorial dispute.

    Simplified competing-risks model inspired by Hammerstein and Parker
    (1982). Persistence is proportional to monument stock M; the group
    that persists longer wins the contested resource.

    .. math::
        E[\text{duration}] = \frac{M_g \cdot M_h}{c_{\text{rate}} (M_g + M_h)}

    .. math::
        P(g \text{ wins}) = \frac{M_g}{M_g + M_h}

    **Relationship to Hammerstein-Parker (1982)**: This is a simplified
    version of their framework, not a direct implementation of the full
    asymmetric ESS. The full H-P model derives role-dependent persistence
    densities (their Eqs. 8-9), a separation value s that determines
    which role can be the winning role (their Thm 1, Eq. 10), and shows
    that only the payoff-favored role can win in continuous WoA. In the
    discrete case, paradoxical ESS conventions are possible (their Thm 2).
    Our simplified model assumes: (a) persistence scales linearly with M
    rather than deriving it from ESS strategies, (b) no role asymmetry
    beyond M differences, (c) no sequential updating of beliefs during
    the contest. This simplification is appropriate for calibrating the
    expected cost of conflict given monument stocks, which is the input
    needed for lambda_C. For predictions about contest dynamics themselves
    (who initiates, how long they persist, whether paradoxical conventions
    emerge), the full H-P model would be needed.

    Parameters
    ----------
    M_g : float
        Monument stock of group g. Must be > 0.
    M_h : float
        Monument stock of group h. Must be > 0.
    V : float
        Value of the contested resource. Must be > 0.
    c_rate : float
        Cost accumulation rate per unit time. Must be > 0.

    Returns
    -------
    dict with keys:
        'duration' : expected contest duration
        'p_g_wins' : probability group g wins
        'p_h_wins' : probability group h wins
        'cost_g'   : expected cost to group g
        'cost_h'   : expected cost to group h
        'payoff_g' : expected net payoff to group g (V * p_win - cost)
        'payoff_h' : expected net payoff to group h

    See SI "War of attrition: a remark on escalation cost".
    See Hammerstein & Parker (1982), J. theor. Biol. 96:647-682.
    """
    if M_g <= 0 or M_h <= 0:
        raise ValueError("Monument stocks must be positive for war of attrition")

    total = M_g + M_h
    duration = (M_g * M_h) / (c_rate * total)
    p_g = M_g / total
    p_h = M_h / total

    # Expected cost is duration * cost rate for each group
    cost_g = duration * c_rate
    cost_h = duration * c_rate

    payoff_g = V * p_g - cost_g
    payoff_h = V * p_h - cost_h

    return {
        "duration": duration,
        "p_g_wins": p_g,
        "p_h_wins": p_h,
        "cost_g": cost_g,
        "cost_h": cost_h,
        "payoff_g": payoff_g,
        "payoff_h": payoff_h,
    }


# =====================================================================
# Derived conflict reduction
# =====================================================================


def derived_conflict_reduction(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    sigma_0: float = DEFAULT_SIGMA_0,
    V: float = DEFAULT_V,
    D: float = DEFAULT_D,
    T_0: float = DEFAULT_T_0,
    beta: float = DEFAULT_BETA_CONFLICT,
    kappa: float = DEFAULT_KAPPA,
    P_base: float = DEFAULT_P_BASE,
) -> float | NDArray[np.float64]:
    r"""Compute the derived conflict reduction r = 1 - P_conflict / P_base.

    This is the quantity that replaces the initial model's assumed
    r = 0.75. The mutual assessment model derives r as a function of
    both groups' monument investments, assessment precision, and
    contest parameters.

    Parameters
    ----------
    M_g, M_h : float or array
        Monument stocks.
    sigma_0, V, D, T_0, beta, kappa, P_base : float
        Assessment model parameters (see mutual_assessment_conflict_prob).

    Returns
    -------
    float or array
        Conflict reduction in [0, 1], where 0 means no reduction and
        1 means complete conflict elimination.
    """
    P = mutual_assessment_conflict_prob(
        M_g, M_h, sigma_0, V, D, T_0, beta, kappa, P_base,
    )
    return 1.0 - P / P_base


def derived_conflict_reduction_self_assessment(
    M_g: float | NDArray[np.float64],
    M_h: float | NDArray[np.float64],
    M_thresh: float = DEFAULT_M_THRESH,
    M_scale: float = DEFAULT_M_SCALE,
    P_base: float = DEFAULT_P_BASE,
    beta: float = DEFAULT_BETA_CONFLICT,
) -> float | NDArray[np.float64]:
    r"""Conflict reduction r_self under self-assessment.

    Symmetric parallel to :func:`derived_conflict_reduction` for the
    self-assessment case (Taylor-Elwood 2003; Maynard Smith 1982 war
    of attrition). The framework's conflict probability decomposes into
    an assessment-driven component and an absolute-deterrence component
    applying identically across regimes (the main-text assessment section):

    .. math::
        r_{\mathrm{self}}(M_g, M_h)
            = 1 - \frac{P_{\mathrm{conflict}}^{\mathrm{self}}(M_g, M_h)}{P_{\mathrm{base}}}

    where :math:`P_{\mathrm{conflict}}^{\mathrm{self}}` is computed by
    :func:`self_assessment_conflict_prob` and includes both the
    assessment-driven product :math:`p(M_g) p(M_h)` and the
    absolute-deterrence factor :math:`1/(1 + \beta(M_g + M_h))`.

    The qualitative contrast with mutual assessment at symmetric
    high-investment dyads is sharp: :math:`r_{\mathrm{mutual}}(M, M)`
    rises monotonically (clear assessment plus absolute deterrence
    avert conflict), while :math:`r_{\mathrm{self}}(M, M)` is sharply
    negative across the moderate-to-high symmetric M range because the
    assessment-driven product :math:`p(M)^2` saturates near 1 once
    both parties exceed :math:`M_{\mathrm{thresh}}`, pushing
    :math:`P_{\mathrm{self}}` well above :math:`P_{\mathrm{base}}`
    despite the deterrence damping. :math:`r_{\mathrm{self}}(M, M)`
    achieves its minimum near the intersection of the assessment-
    saturation regime and the deterrence half-saturation
    :math:`M \sim 1/(2\beta)`. For larger M, :math:`r_{\mathrm{self}}`
    rises partially as deterrence dominates the saturated assessment
    term, but remains substantially below :math:`r_{\mathrm{mutual}}`
    over the parameter range relevant to monument-building societies
    (the directional conflict-signature predictions in the main text).

    Parameters
    ----------
    M_g, M_h : float or array
        Monument stocks.
    M_thresh : float
        Escalation threshold for the self-assessment sigmoid.
    M_scale : float
        Steepness of the escalation sigmoid.
    P_base : float
        Baseline conflict probability used to normalize
        :math:`P_{\mathrm{conflict}}^{\mathrm{self}}` for the
        conflict-reduction definition :math:`r = 1 - P/P_{\mathrm{base}}`;
        :math:`r_{\mathrm{self}}` can be strongly negative when the
        assessment-driven product :math:`p(M_g) p(M_h)` exceeds
        :math:`P_{\mathrm{base}}` (i.e., both groups exceed the
        escalation threshold).
    beta : float
        Absolute-deterrence coefficient. Default matches the mutual-
        assessment specification; passed through to
        :func:`self_assessment_conflict_prob`. Setting ``beta = 0``
        recovers the pure-assessment self-assessment formulation.

    Returns
    -------
    float or array
        Conflict reduction r_self in [-inf, 1].

    Notes
    -----
    Unlike :math:`r_{\mathrm{mutual}}`, which is bounded above by 1 and
    typically positive, :math:`r_{\mathrm{self}}` can be negative for
    symmetric moderate-to-high M dyads. This reflects the framework's
    discriminating prediction at symmetric high-investment dyads:
    self-assessment generates more conflict at high symmetric
    investment than mutual assessment, even with absolute deterrence
    applied identically in both regimes.

    See the main-text self-assessment conflict probability.
    """
    P = self_assessment_conflict_prob(M_g, M_h, M_thresh, M_scale, beta=beta, P_base=P_base)
    return 1.0 - P / P_base


# =====================================================================
# lambda_C: competitive between-group feedback
# =====================================================================


def compute_lambda_C(
    M_g: float,
    M_h_neighbors: float | NDArray[np.float64] | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    dispute_freq: float = DEFAULT_DISPUTE_FREQ,
    conflict_cost: float = DEFAULT_CONFLICT_COST,
    sigma_0: float = DEFAULT_SIGMA_0,
    V: float = DEFAULT_V,
    D: float = DEFAULT_D,
    T_0: float = DEFAULT_T_0,
    beta: float = DEFAULT_BETA_CONFLICT,
    kappa: float = DEFAULT_KAPPA,
    P_base: float = DEFAULT_P_BASE,
    N: int | None = None,
    dx: float = 0.01,
) -> float:
    r"""Compute lambda_C, the competitive between-group component of lambda.

    lambda_C is the marginal fitness value of conflict deterrence per unit
    of monument investment: the marginal REDUCTION in expected conflict
    losses, summed over all neighboring groups. Writing the expected loss

    .. math::
        L(M_g) = \sum_h f_h \cdot P_{\text{conflict}}(M_g, M_h)
                 \cdot C_{\text{conflict}},

    the derived return is

    .. math::
        \lambda_C = -\frac{dL}{dM_g} \; > 0,

    where :math:`f_h` is the dispute frequency with neighbor h and
    C_conflict is the fitness cost of conflict when it occurs. Since added
    stock lowers the conflict probability, :math:`dL/dM_g < 0` and the
    negation makes lambda_C a positive benefit (the un-negated derivative is
    negative; the implementation negates it). The derivative is computed
    numerically via central differences on M_g, using
    :math:`dM_g / dx_i = 1` (each unit of individual investment adds one
    unit to group monument stock).

    When N (group size) is provided, lambda_C is divided by N to convert
    from the group-level marginal return to the individual-level marginal
    return, since one unit of individual investment adds 1/N units to the
    per-capita group signal (or equivalently, 1 unit to M_g but each
    member bears 1/N of the group-level return).

    Parameters
    ----------
    M_g : float
        Current monument stock of the focal group. Must be >= 0.
    M_h_neighbors : float or array or None
        Monument stocks of neighboring groups. If a scalar, all neighbors
        are assumed to have the same stock. If an array, its length is
        used as n_neighbors. If None, all neighbors are assumed to have
        M_g (symmetric default).
    n_neighbors : int
        Number of neighboring groups. Ignored if M_h_neighbors is an
        array. Must be >= 0.
    dispute_freq : float
        Per-neighbor-pair annual dispute probability. Must be >= 0.
    conflict_cost : float
        Fitness cost when conflict occurs. Must be >= 0.
    sigma_0, V, D, T_0, beta, kappa, P_base : float
        Mutual assessment model parameters.
    N : int or None
        Group size. If provided, lambda_C is divided by N to give the
        individual-level marginal return.
    dx : float
        Step size for central difference derivative. Must be > 0.

    Returns
    -------
    float
        lambda_C >= 0. The marginal fitness value of conflict deterrence.

    See the main-text war-avoidance and competitive-feedback (lambda_C)
    section.
    """
    if M_h_neighbors is None:
        M_h_arr = np.full(n_neighbors, M_g)
    else:
        M_h_arr = np.atleast_1d(np.asarray(M_h_neighbors, dtype=np.float64))
        if M_h_arr.ndim == 0:
            M_h_arr = np.full(n_neighbors, float(M_h_arr))
        # If an array was provided, n_neighbors is its length
        n_neighbors = len(M_h_arr)

    if n_neighbors == 0:
        return 0.0

    def _total_expected_loss(M_g_val: float) -> float:
        """Sum of expected conflict losses across all neighbors."""
        P_arr = mutual_assessment_conflict_prob(
            M_g_val, M_h_arr, sigma_0, V, D, T_0, beta, kappa, P_base,
        )
        # Expected loss = sum over neighbors of (dispute_freq * P_conflict * cost)
        return float(np.sum(dispute_freq * P_arr * conflict_cost))

    # Central difference: d(loss)/dM_g
    # M_g represents total group stock; dx_i = dx implies dM_g = dx
    loss_plus = _total_expected_loss(M_g + dx)
    loss_minus = _total_expected_loss(max(M_g - dx, 0.0))
    effective_dx = (M_g + dx) - max(M_g - dx, 0.0)

    # lambda_C is the marginal REDUCTION in loss, so negate the derivative
    d_loss = (loss_plus - loss_minus) / effective_dx
    lambda_C = -d_loss

    # Convert from group-level to individual-level if N provided
    if N is not None and N > 0:
        lambda_C = lambda_C / N

    return max(lambda_C, 0.0)


# =====================================================================
# Model comparison
# =====================================================================


def compare_assessment_models(
    M_range: NDArray[np.float64],
    sigma_0: float = DEFAULT_SIGMA_0,
    V: float = DEFAULT_V,
    D: float = DEFAULT_D,
    T_0: float = DEFAULT_T_0,
    beta: float = DEFAULT_BETA_CONFLICT,
    kappa: float = DEFAULT_KAPPA,
    P_base: float = DEFAULT_P_BASE,
    M_thresh: float = DEFAULT_M_THRESH,
    M_scale: float = DEFAULT_M_SCALE,
    M_own: float = DEFAULT_M_OWN,
    steepness: float = DEFAULT_STEEPNESS,
) -> dict[str, NDArray[np.float64]]:
    r"""Compute conflict probability surfaces for all three assessment models.

    Evaluates each model over a 2D grid of (M_g, M_h) values to enable
    visual comparison and identification of discriminating predictions.

    Parameters
    ----------
    M_range : array
        1D array of monument stock values to form the grid.
    sigma_0, V, D, T_0, beta, kappa, P_base : float
        Mutual assessment parameters.
    M_thresh, M_scale : float
        Self-assessment parameters.
    M_own, steepness : float
        Bourgeois parameters.

    Returns
    -------
    dict with keys:
        'M_g_grid', 'M_h_grid' : 2D meshgrid arrays
        'mutual' : 2D array of mutual assessment P_conflict
        'self' : 2D array of self-assessment P_conflict
        'bourgeois' : 2D array of Bourgeois P_conflict
    """
    Mg, Mh = np.meshgrid(M_range, M_range)

    P_mutual = mutual_assessment_conflict_prob(
        Mg, Mh, sigma_0, V, D, T_0, beta, kappa, P_base,
    )
    P_self = self_assessment_conflict_prob(Mg, Mh, M_thresh, M_scale, beta=beta, P_base=P_base)
    P_bourg = bourgeois_conflict_prob(Mg, Mh, M_own, P_base, steepness)

    return {
        "M_g_grid": Mg,
        "M_h_grid": Mh,
        "mutual": P_mutual,
        "self": P_self,
        "bourgeois": P_bourg,
    }


# =====================================================================
# Calibration
# =====================================================================


def calibrate_conflict_reduction(
    target_r: float = 0.75,
    M_high: float | None = None,
    P_base: float = DEFAULT_P_BASE,
    V: float = DEFAULT_V,
    D: float = DEFAULT_D,
    sigma_0_bounds: tuple[float, float] = (0.1, 5.0),
    T_0_bounds: tuple[float, float] = (0.01, 5.0),
    beta_bounds: tuple[float, float] = (0.0, 2.0),
    kappa_bounds: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    r"""Calibrate mutual assessment parameters to reproduce target conflict reduction.

    Uses scipy.optimize to find (sigma_0, T_0, beta, kappa) such that
    the derived conflict reduction for two high-investing groups matches
    target_r (default 0.75, i.e., the initial model's r).

    If M_high is not provided, it is computed from Layer 1 defaults using
    expected_monument_stock.

    Parameters
    ----------
    target_r : float
        Target conflict reduction (1 - P/P_base). Default 0.75.
    M_high : float or None
        Expected monument stock for a high-investing group. If None,
        computed from Layer 1 defaults.
    P_base : float
        Baseline conflict probability.
    V, D : float
        Contest value and fighting cost.
    sigma_0_bounds, T_0_bounds, beta_bounds, kappa_bounds : tuple
        Bounds for each parameter.

    Returns
    -------
    dict with keys:
        'sigma_0' : calibrated baseline noise
        'T_0' : calibrated threshold scale
        'beta' : calibrated deterrence coefficient
        'kappa' : calibrated information gain rate
        'r_achieved' : actual conflict reduction achieved
        'P_conflict' : calibrated conflict probability
        'M_high' : monument stock used
        'success' : whether optimization converged
    """
    if M_high is None:
        # Compute from Layer 1 defaults
        from signaling.calibration import (
            DEFAULT_LAMBDA,
            DEFAULT_N,
            DEFAULT_Q_MAX,
            DEFAULT_Q_MIN,
        )
        from signaling.layer1 import expected_monument_stock

        M_high = expected_monument_stock(
            DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX, DEFAULT_LAMBDA,
        )

    target_P = P_base * (1.0 - target_r)

    def objective(params: NDArray[np.float64]) -> float:
        s0, t0, b, k = params
        P = mutual_assessment_conflict_prob(
            M_high, M_high, s0, V, D, t0, b, k, P_base,
        )
        P_val = float(P)
        return (P_val - target_P) ** 2

    # Initial guess from defaults
    x0 = np.array([DEFAULT_SIGMA_0, DEFAULT_T_0, DEFAULT_BETA_CONFLICT, DEFAULT_KAPPA])
    bounds = [sigma_0_bounds, T_0_bounds, beta_bounds, kappa_bounds]

    result = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
    )

    s0_opt, t0_opt, b_opt, k_opt = result.x

    P_achieved = float(mutual_assessment_conflict_prob(
        M_high, M_high, s0_opt, V, D, t0_opt, b_opt, k_opt, P_base,
    ))
    r_achieved = 1.0 - P_achieved / P_base

    return {
        "sigma_0": float(s0_opt),
        "T_0": float(t0_opt),
        "beta": float(b_opt),
        "kappa": float(k_opt),
        "r_achieved": r_achieved,
        "P_conflict": P_achieved,
        "M_high": float(M_high),
        "success": result.success,
    }
