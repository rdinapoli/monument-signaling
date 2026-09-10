"""Cooperation networks and crisis buffering.

Models how individual signaling creates and maintains cooperation networks
that buffer against environmental crises, deriving vulnerability parameters
from network structure rather than assuming them.

Key results:
- Network degree k(M_g) is increasing in monument stock (main-text network-degree equation)
- Survival S(sigma, k) is increasing in k, decreasing in sigma (main-text survival equation)
- Vulnerability differential alpha_eff < beta_eff emerges from k_signal > k_nonsignal
- lambda_X is increasing in sigma, creating the positive feedback loop (main-text cooperative-feedback equation)
- Multiple equilibria can exist via channel selection: monument vs. alternative

**Channel selection constraint**: This module produces lambda_X, the cooperative
between-group component of lambda:

    lambda_X = (dk/dM_g) * (dS/dk)

where dk/dM_g is how additional monument investment increases network degree,
and dS/dk is the marginal survival benefit of an additional partner. Since
dS/dk is proportional to sigma, lambda_X is increasing in environmental
uncertainty, closing the feedback loop: higher sigma -> higher lambda_X ->
monument channel dominates -> more investment -> denser networks.

**Quality dimension**: Partner choice assesses productive/coordinative capacity
specifically. Monument construction reveals this dimension with high fidelity
(rho_X) because both the signal cost and the cooperation benefit depend on
the same underlying capacity.

See the main-text cooperation networks and crisis buffering section
(network-degree, survival, and cooperative-feedback equations).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from signaling.calibration import (
    DEFAULT_GAMMA,
    DEFAULT_K_0,
    DEFAULT_K_MAX,
    DEFAULT_LAMBDA,
    DEFAULT_M_HALF,
    DEFAULT_N,
    DEFAULT_P_CONNECT_BASE,
    DEFAULT_Q_MAX,
    DEFAULT_Q_MIN,
)

# =====================================================================
# Network degree: analytical model
# =====================================================================


def network_degree(
    M_g: float | NDArray[np.float64],
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
) -> float | NDArray[np.float64]:
    r"""Group intergroup network degree as a function of monument stock.

    Uses a saturating (Michaelis-Menten) form:

    .. math::
        k(M_g) = k_0 + k_{\max} \frac{M_g}{M_{1/2} + M_g}

    Properties:
    - k(0) = k_0 (baseline from kinship/proximity)
    - k -> k_0 + k_max as M_g -> infinity (bounded above)
    - dk/dM > 0 (more investment attracts more partners)
    - d^2k/dM^2 < 0 (diminishing returns)

    The saturating form reflects two constraints: (1) the pool of potential
    exchange partners is finite, and (2) maintaining relationships has costs
    that limit network size. This is the standard Michaelis-Menten functional
    response applied to partner acquisition.

    Parameters
    ----------
    M_g : float or array
        Group monument stock. Must be >= 0.
    k_0 : float
        Baseline network degree from kinship/proximity, without any
        signaling investment. Must be >= 0.
    k_max : float
        Maximum additional connections achievable through signaling.
        Must be >= 0.
    M_half : float
        Half-saturation monument stock: M_g at which half the
        signal-based connections are formed. Must be > 0.

    Returns
    -------
    float or array
        Network degree k >= k_0.

    See the main-text network-degree equation k(M_g).
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    signal_component = k_max * M_g / (M_half + M_g)
    return k_0 + signal_component


def network_degree_derivative(
    M_g: float | NDArray[np.float64],
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
) -> float | NDArray[np.float64]:
    r"""Derivative of network degree with respect to monument stock.

    .. math::
        \frac{dk}{dM_g} = k_{\max} \frac{M_{1/2}}{(M_{1/2} + M_g)^2}

    This is the marginal rate at which additional monument investment
    attracts new exchange partners. It is positive and decreasing
    (diminishing returns to signaling for partner acquisition).

    Parameters
    ----------
    M_g : float or array
        Group monument stock. Must be >= 0.
    k_max : float
        Maximum additional signal-based connections. Must be >= 0.
    M_half : float
        Half-saturation monument stock. Must be > 0.

    Returns
    -------
    float or array
        dk/dM_g > 0.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    return k_max * M_half / (M_half + M_g) ** 2


# =====================================================================
# Alternative network degree forms (robustness analysis; SI "Network degree functional-form robustness")
# =====================================================================


@dataclass(frozen=True)
class NetworkDegreeSpec:
    r"""Bundles a network-degree function $k(M)$ with its derivative $dk/dM$.

    Used to plumb alternative network-degree functional forms through the
    feedback-loop computations. Each spec must satisfy:

    - $k(0) = k_0$ (matches baseline degree from kinship/proximity)
    - $k(M) \to k_0 + k_{\max}$ as $M \to \infty$ (saturating upper bound)
    - $dk/dM > 0$ for $M \geq 0$ (monotonically increasing)

    For the robustness analysis (SI "Network degree functional-form
    robustness"), specs are calibrated by
    :func:`calibrate_alternative_network_forms` such that each form matches
    the Michaelis-Menten reference exactly at an empirical anchor $M_e$.
    """
    name: str
    k: Callable[[float], float]
    dk_dM: Callable[[float], float]


def network_degree_exponential(
    M_g: float | NDArray[np.float64],
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    tau: float = 6.0,
) -> float | NDArray[np.float64]:
    r"""Exponential-saturation network degree.

    .. math::
        k(M_g) = k_0 + k_{\max} \bigl(1 - e^{-M_g / \tau}\bigr)

    Concave throughout (no inflection); approaches saturation faster than
    Michaelis-Menten when calibrated to match at the empirical anchor.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    return k_0 + k_max * (1.0 - np.exp(-M_g / tau))


def network_degree_exponential_derivative(
    M_g: float | NDArray[np.float64],
    k_max: float = DEFAULT_K_MAX,
    tau: float = 6.0,
) -> float | NDArray[np.float64]:
    r"""Derivative of exponential-saturation network degree.

    .. math::
        \frac{dk}{dM_g} = \frac{k_{\max}}{\tau} e^{-M_g / \tau}
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    return (k_max / tau) * np.exp(-M_g / tau)


def network_degree_hill(
    M_g: float | NDArray[np.float64],
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    K: float = 5.0,
    hill_n: float = 2.0,
) -> float | NDArray[np.float64]:
    r"""Hill-function network degree.

    .. math::
        k(M_g) = k_0 + k_{\max} \frac{M_g^n}{K^n + M_g^n}

    With $n = 1$ this reduces to Michaelis-Menten with $K = M_{1/2}$. For
    $n = 2$ the function is sigmoid: convex below the inflection at
    $M = K / \sqrt{3}$ and concave above. The robustness analysis uses
    $n = 2$ to test a qualitatively different shape than the always-concave
    Michaelis-Menten.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    return k_0 + k_max * M_g ** hill_n / (K ** hill_n + M_g ** hill_n)


def network_degree_hill_derivative(
    M_g: float | NDArray[np.float64],
    k_max: float = DEFAULT_K_MAX,
    K: float = 5.0,
    hill_n: float = 2.0,
) -> float | NDArray[np.float64]:
    r"""Derivative of Hill-function network degree.

    .. math::
        \frac{dk}{dM_g} = k_{\max} \frac{n K^n M_g^{n-1}}{(K^n + M_g^n)^2}
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    # Guard the M^(n-1) term against M_g = 0 for n != 1; the limit is 0 for n > 1.
    M_safe = np.maximum(M_g, 1e-20)
    return (
        k_max
        * hill_n
        * K ** hill_n
        * M_safe ** (hill_n - 1.0)
        / (K ** hill_n + M_safe ** hill_n) ** 2
    )


def network_degree_piecewise(
    M_g: float | NDArray[np.float64],
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    slope: float = 1.0,
) -> float | NDArray[np.float64]:
    r"""Piecewise-linear-with-cap network degree.

    .. math::
        k(M_g) = k_0 + \min(\alpha M_g, k_{\max})

    Linear with slope $\alpha$ until $M = k_{\max}/\alpha$, then constant.
    Qualitatively different from the smooth saturating forms: has a sharp
    kink at the saturation point and zero curvature elsewhere. Tests
    robustness against a non-smooth alternative.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    return k_0 + np.minimum(slope * M_g, k_max)


def network_degree_piecewise_derivative(
    M_g: float | NDArray[np.float64],
    k_max: float = DEFAULT_K_MAX,
    slope: float = 1.0,
) -> float | NDArray[np.float64]:
    r"""Derivative of piecewise-linear-with-cap network degree.

    Returns ``slope`` for $M_g < k_{\max}/\mathrm{slope}$ and 0 thereafter.
    The derivative is discontinuous at the saturation point.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    return np.where(slope * M_g < k_max, slope, 0.0)


def calibrate_alternative_network_forms(
    M_anchor: float,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
) -> dict[str, NetworkDegreeSpec]:
    r"""Construct calibrated alternative network-degree specs.

    Each alternative form is parameterized so that it matches the
    Michaelis-Menten reference exactly at the anchor $M_{\mathrm{anchor}}$.
    The $k_0$ and $k_{\max}$ constraints are satisfied by construction in
    each functional form; one additional parameter per form is solved
    analytically to match the MM value at the anchor:

    - **Exponential**: $\tau = M_e / \ln(1 + M_e / M_{1/2})$.
    - **Hill, n=2**: $K = \sqrt{M_{1/2} M_e}$.
    - **Piecewise linear**: $\alpha = k_{\max} / (M_{1/2} + M_e)$,
      saturating at $M_g = M_{1/2} + M_e$.

    Parameters
    ----------
    M_anchor : float
        Empirical anchor $M_g$ at which all forms agree with Michaelis-Menten.
        Typically the equilibrium monument stock at the empirical $\lambda_W$.
    k_0, k_max, M_half : float
        MM reference parameters.

    Returns
    -------
    dict[str, NetworkDegreeSpec]
        Specs keyed by ``"michaelis_menten"``, ``"exponential"``,
        ``"hill_n2"``, ``"piecewise_linear"``.
    """
    tau = M_anchor / float(np.log(1.0 + M_anchor / M_half))
    K_hill = float(np.sqrt(M_half * M_anchor))
    slope_piecewise = k_max / (M_half + M_anchor)

    return {
        "michaelis_menten": NetworkDegreeSpec(
            name="Michaelis-Menten (reference)",
            k=lambda M: float(network_degree(M, k_0, k_max, M_half)),
            dk_dM=lambda M: float(network_degree_derivative(M, k_max, M_half)),
        ),
        "exponential": NetworkDegreeSpec(
            name="Exponential",
            k=lambda M, t=tau: float(network_degree_exponential(M, k_0, k_max, t)),
            dk_dM=lambda M, t=tau: float(
                network_degree_exponential_derivative(M, k_max, t)
            ),
        ),
        "hill_n2": NetworkDegreeSpec(
            name="Hill (n=2)",
            k=lambda M, kk=K_hill: float(network_degree_hill(M, k_0, k_max, kk, 2.0)),
            dk_dM=lambda M, kk=K_hill: float(
                network_degree_hill_derivative(M, k_max, kk, 2.0)
            ),
        ),
        "piecewise_linear": NetworkDegreeSpec(
            name="Piecewise linear",
            k=lambda M, s=slope_piecewise: float(
                network_degree_piecewise(M, k_0, k_max, s)
            ),
            dk_dM=lambda M, s=slope_piecewise: float(
                network_degree_piecewise_derivative(M, k_max, s)
            ),
        ),
    }


# =====================================================================
# Crisis survival
# =====================================================================


def survival_probability(
    sigma: float | NDArray[np.float64],
    k: float | NDArray[np.float64],
    gamma: float = DEFAULT_GAMMA,
) -> float | NDArray[np.float64]:
    r"""Crisis survival probability as a function of uncertainty and network degree.

    .. math::
        S(\sigma, k) = 1 - \frac{\sigma}{1 + \gamma k}

    Each exchange partner provides gamma units of buffering efficiency,
    reducing the effective crisis severity. Properties:

    - S(0, k) = 1 for all k (no crisis, certain survival)
    - S(sigma, 0) = 1 - sigma (maximum vulnerability without partners)
    - dS/dk = sigma * gamma / (1 + gamma * k)^2 > 0 (partners help)
    - dS/dsigma = -1 / (1 + gamma * k) < 0 (more stress hurts)
    - S > 0 requires sigma < 1 + gamma * k

    The form follows Winterhalder (1986) risk-pooling models: each
    additional partner reduces the effective variance of resource
    shortfalls, with diminishing returns as the network grows.

    Parameters
    ----------
    sigma : float or array
        Environmental uncertainty. Should be in [0, 1] for meaningful
        survival probabilities with small networks.
    k : float or array
        Network degree (number of exchange partners). Must be >= 0.
    gamma : float
        Buffering efficiency per exchange partner. Must be >= 0.

    Returns
    -------
    float or array
        Survival probability, clipped to [0, 1].

    See the main-text crisis-survival equation S(sigma, k).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    S = 1.0 - sigma / (1.0 + gamma * k)
    return np.clip(S, 0.0, 1.0)


# =====================================================================
# Vulnerability coefficients
# =====================================================================


def vulnerability_coefficient(
    k: float | NDArray[np.float64],
    gamma: float = DEFAULT_GAMMA,
) -> float | NDArray[np.float64]:
    r"""Vulnerability coefficient as a function of network degree.

    .. math::
        \alpha(k) = \frac{1}{1 + \gamma k}

    The survival function can be written S(sigma, k) = 1 - alpha(k) * sigma.
    In the initial model, signalers have alpha = 0.30 and non-signalers
    have beta = 0.90, but these are assumed. Here we derive them from
    network structure: denser networks yield lower vulnerability.

    Parameters
    ----------
    k : float or array
        Network degree. Must be >= 0.
    gamma : float
        Buffering efficiency per partner. Must be >= 0.

    Returns
    -------
    float or array
        Vulnerability coefficient in (0, 1].

    See the main-text derived vulnerability differential section.
    """
    if np.any(np.asarray(k) < 0):
        raise ValueError("network degree k must be >= 0")
    k = np.asarray(k, dtype=np.float64)
    return 1.0 / (1.0 + gamma * k)


def derive_vulnerability_differential(
    k_signal: float,
    k_nonsignal: float,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, float]:
    r"""Derive vulnerability parameters from network degree differential.

    Computes the effective vulnerability parameters alpha_eff (signalers)
    and beta_eff (non-signalers) from their respective network degrees.
    This replaces the initial model's assumed alpha = 0.30 and beta = 0.90
    with mechanistically derived values.

    The vulnerability ratio alpha/beta < 1 whenever k_signal > k_nonsignal,
    which holds when monument signaling attracts exchange partners (the
    central mechanism of Layer 3).

    Parameters
    ----------
    k_signal : float
        Network degree for signalers (monument builders). Must be >= 0.
    k_nonsignal : float
        Network degree for non-signalers. Must be >= 0.
    gamma : float
        Buffering efficiency per partner. Must be >= 0.

    Returns
    -------
    dict with keys:
        'alpha_eff' : signaler vulnerability
        'beta_eff' : non-signaler vulnerability
        'ratio' : alpha_eff / beta_eff (< 1 when signalers are less vulnerable)
        'k_signal' : input k_signal
        'k_nonsignal' : input k_nonsignal

    See the main-text derived vulnerability differential section.
    """
    alpha = float(vulnerability_coefficient(k_signal, gamma))
    beta = float(vulnerability_coefficient(k_nonsignal, gamma))
    ratio = alpha / beta if beta > 0 else 0.0
    return {
        "alpha_eff": alpha,
        "beta_eff": beta,
        "ratio": ratio,
        "k_signal": k_signal,
        "k_nonsignal": k_nonsignal,
    }


# =====================================================================
# lambda_X: cooperative between-group feedback
# =====================================================================


def compute_lambda_X(
    M_g: float,
    sigma: float,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    N: int | None = None,
    network_degree_spec: NetworkDegreeSpec | None = None,
) -> float:
    r"""Compute lambda_X, the cooperative between-group component of lambda.

    lambda_X is the marginal fitness value of exchange network access per
    unit of group monument investment:

    .. math::
        \lambda_X = \frac{dk}{dM_g} \cdot \frac{\partial S}{\partial k}
                   = \frac{dk}{dM_g} \cdot \frac{\sigma \gamma}{(1 + \gamma k)^2}

    This makes lambda_X explicitly a function of sigma (via dS/dk) and
    M_g (via dk/dM_g and k), creating the positive feedback loop:
    higher sigma -> higher dS/dk -> higher lambda_X -> more investment
    -> denser networks.

    lambda_X decreases with M_g (diminishing returns: both dk/dM and
    dS/dk decline as networks saturate), but increases with sigma.
    The key prediction is that lambda_X is proportional to sigma,
    making the cooperative channel more valuable in uncertain environments.

    When N (group size) is provided, lambda_X is divided by N to convert
    from group-level to individual-level marginal return, since one unit
    of individual investment adds 1 unit to M_g but the survival benefit
    is shared across N members. The self-consistent threshold and the lambda
    feedback loop use the undivided form (N=None), treating survival as a
    per-group private return; the /N option gives a per-capita reading but is
    not used in the threshold calculation.

    Parameters
    ----------
    M_g : float
        Group monument stock. Must be >= 0.
    sigma : float
        Environmental uncertainty. Must be >= 0.
    gamma : float
        Buffering efficiency per partner. Must be >= 0.
    k_0 : float
        Baseline network degree.
    k_max : float
        Maximum signal-based network degree.
    M_half : float
        Half-saturation monument stock. Must be > 0.
    N : int or None
        Group size. If provided, lambda_X is divided by N.

    Returns
    -------
    float
        lambda_X >= 0.

    See the main-text cooperative-feedback equation lambda_X.
    """
    if network_degree_spec is None:
        k = float(network_degree(M_g, k_0, k_max, M_half))
        dk_dM = float(network_degree_derivative(M_g, k_max, M_half))
    else:
        k = float(network_degree_spec.k(M_g))
        dk_dM = float(network_degree_spec.dk_dM(M_g))

    # dS/dk = sigma * gamma / (1 + gamma * k)^2
    dS_dk = sigma * gamma / (1.0 + gamma * k) ** 2

    lam_X = dk_dM * dS_dk

    if N is not None and N > 0:
        lam_X = lam_X / N

    return max(lam_X, 0.0)


# =====================================================================
# Lambda-sigma feedback and equilibrium
# =====================================================================


def between_group_lambda_diagnostics(
    sigma: float,
    lambda_W: float,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    compute_lambda_C_func: Any = None,
    delta: float = 0.0,
    network_degree_spec: NetworkDegreeSpec | None = None,
) -> dict[str, float]:
    r"""Canonical lambda_W-only evaluation of the between-group returns.

    NUMERICAL NOTE: tiny internal floors (e.g. 1e-10 on lambda in the
    stock computation) exist only to avoid literal division by zero and
    are unreachable through the guarded public entry points, which
    require lambda_W > 0; they are not a modeling device.

    Under the lambda_W-only first-order condition (main-text separating
    equilibrium), the individual schedule and hence the group stock
    :math:`M_g` derive from :math:`\lambda_W` alone, so no fixed-point
    iteration is required: :math:`\lambda_C` and :math:`\lambda_X` are
    evaluated once at :math:`M_g(\lambda_W)` as diagnostics. They are
    marginal returns on the physical stock, realized at the group level
    through the war-avoidance factor K and the survival factor S; they do
    not feed back into the individual schedule. The superseded
    composite-lambda variant, in which they did, is retained as
    :func:`lambda_total_at_sigma` for the SI Banach robustness comparison
    (its fixed point differs from :math:`\lambda_W` by under 1%).

    Parameters
    ----------
    sigma : float
        Environmental uncertainty in [0, 1] (enters lambda_X only).
    lambda_W : float
        Within-group informational reward; sole driver of the schedule.
    gamma, k_0, k_max, M_half : float
        Network and survival parameters.
    n : int
        Group size.
    q_min, q_max : float
        Quality range for the Layer 1 equilibrium.
    compute_lambda_C_func : callable(M_g) -> float, or None
        Function computing the marginal competitive return lambda_C given
        M_g. If None, lambda_C = 0.
    delta : float
        Signal depreciation rate in [0, 1]. When delta > 0, the effective
        stock is M_g = I_g / delta (steady state under depreciation).
    network_degree_spec : NetworkDegreeSpec or None
        Alternative network-degree functional form.

    Returns
    -------
    dict with keys:
        'lambda_W' : the input within-group reward
        'lambda_C' : marginal competitive return at M_g(lambda_W)
        'lambda_X' : marginal cooperative return at M_g(lambda_W)
        'lambda_composite' : lambda_W + lambda_C + lambda_X (diagnostic sum)
        'M_g' : effective monument stock at lambda_W (after depreciation)
        'I_g' : investment flow at lambda_W (before depreciation)
        'k' : network degree at M_g
        'alpha_eff' : signaler vulnerability coefficient at k
    """
    from signaling.layer1 import effective_monument_stock, expected_monument_stock

    I_g = expected_monument_stock(n, q_min, q_max, max(lambda_W, 1e-10))
    if delta > 0:
        M_g = float(effective_monument_stock(I_g, delta))
    else:
        M_g = I_g

    lam_C = float(compute_lambda_C_func(M_g)) if compute_lambda_C_func is not None else 0.0
    lam_X = float(compute_lambda_X(
        M_g, sigma, gamma, k_0, k_max, M_half,
        network_degree_spec=network_degree_spec,
    ))

    if network_degree_spec is None:
        k = float(network_degree(M_g, k_0, k_max, M_half))
    else:
        k = float(network_degree_spec.k(M_g))
    alpha_eff = float(vulnerability_coefficient(k, gamma))

    return {
        "lambda_W": float(lambda_W),
        "lambda_C": lam_C,
        "lambda_X": lam_X,
        "lambda_composite": float(lambda_W) + lam_C + lam_X,
        "M_g": M_g,
        "I_g": I_g,
        "k": k,
        "alpha_eff": alpha_eff,
    }


def lambda_total_at_sigma(
    sigma: float,
    lambda_W: float,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    compute_lambda_C_func: Any = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    damping: float = 0.5,
    delta: float = 0.0,
    lambda_init: float | None = None,
    network_degree_spec: NetworkDegreeSpec | None = None,
) -> dict[str, float]:
    r"""Composite-lambda fixed point (robustness variant, superseded).

    This is the superseded composite-lambda specification, in which the
    between-group returns lambda_C and lambda_X feed back into the
    individual schedule as if they were informational rewards captured
    through inferred quality. The canonical specification is the
    lambda_W-only first-order condition
    (:func:`between_group_lambda_diagnostics`): individuals optimize on
    lambda_W alone, and the between-group returns enter group fitness
    through the survival factor S and the war-avoidance factor K. This
    function is retained to quantify that the specification choice is
    immaterial: the fixed point differs from lambda_W by under 1%, moving
    the schedule x*(q) by under 0.5% (SI Banach robustness section).

    The feedback loop is:

    1. Start with lambda = lambda_W
    2. Compute I_g (investment flow) from equilibrium investment at current lambda
    3. Apply depreciation: M_g = I_g / delta if delta > 0, else M_g = I_g
    4. Compute lambda_C from Layer 2 (if provided)
    5. Compute lambda_X from network degree and sigma
    6. Update lambda = lambda_W + lambda_C + lambda_X
    7. Repeat until convergence

    Convergence is aided by damped iteration:
    lambda_new = damping * lambda_old + (1 - damping) * lambda_computed.

    Parameters
    ----------
    sigma : float
        Environmental uncertainty in [0, 1].
    lambda_W : float
        Within-group component (fixed, set by group social ecology).
    gamma, k_0, k_max, M_half : float
        Network and survival parameters.
    n : int
        Group size.
    q_min, q_max : float
        Quality range for Layer 1 equilibrium.
    compute_lambda_C_func : callable(M_g) -> float, or None
        Function computing lambda_C given current M_g. If None,
        lambda_C = 0 (no competitive feedback).
    tol : float
        Convergence tolerance for lambda.
    max_iter : int
        Maximum iterations.
    damping : float
        Damping factor in (0, 1). Higher values slow convergence
        but improve stability.
    delta : float
        Signal depreciation rate in [0, 1]. When delta > 0, the
        effective monument stock is M_g = I_g / delta (steady state
        under depreciation). When delta = 0, no depreciation is
        applied and M_g = I_g (the static model).

    Returns
    -------
    dict with keys:
        'lambda_total' : converged total lambda
        'lambda_W' : within-group component
        'lambda_C' : competitive component
        'lambda_X' : cooperative component
        'M_g' : equilibrium monument stock (effective, after depreciation)
        'I_g' : investment flow (before depreciation)
        'k' : network degree
        'alpha_eff' : signaler vulnerability coefficient
        'converged' : bool
        'iterations' : int
    """
    from signaling.layer1 import effective_monument_stock, expected_monument_stock

    lam = lambda_init if lambda_init is not None else lambda_W
    lam_C = 0.0
    lam_X = 0.0
    converged = False
    iterations = 0

    for i in range(max_iter):
        iterations = i + 1

        # Compute investment flow I_g at current lambda
        I_g = expected_monument_stock(n, q_min, q_max, max(lam, 1e-10))

        # Apply depreciation to get effective monument stock
        if delta > 0:
            M_g = float(effective_monument_stock(I_g, delta))
        else:
            M_g = I_g

        # Compute lambda_C from Layer 2
        if compute_lambda_C_func is not None:
            lam_C = compute_lambda_C_func(M_g)
        else:
            lam_C = 0.0

        # Compute lambda_X from network model
        lam_X = compute_lambda_X(
            M_g, sigma, gamma, k_0, k_max, M_half,
            network_degree_spec=network_degree_spec,
        )

        # Updated total lambda
        lam_new = lambda_W + lam_C + lam_X

        if abs(lam_new - lam) < tol:
            converged = True
            lam = lam_new
            break

        # Damped update for stability
        lam = damping * lam + (1.0 - damping) * lam_new

    # Final state computation
    I_g = expected_monument_stock(n, q_min, q_max, max(lam, 1e-10))
    if delta > 0:
        M_g = float(effective_monument_stock(I_g, delta))
    else:
        M_g = I_g
    if network_degree_spec is None:
        k = float(network_degree(M_g, k_0, k_max, M_half))
    else:
        k = float(network_degree_spec.k(M_g))
    alpha_eff = float(vulnerability_coefficient(k, gamma))

    return {
        "lambda_total": lam,
        "lambda_W": lambda_W,
        "lambda_C": lam_C,
        "lambda_X": lam_X,
        "M_g": M_g,
        "I_g": I_g,
        "k": k,
        "alpha_eff": alpha_eff,
        "converged": converged,
        "iterations": iterations,
    }


def lambda_sigma_sweep(
    sigma_range: NDArray[np.float64],
    lambda_W: float,
    gamma: float = DEFAULT_GAMMA,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    compute_lambda_C_func: Any = None,
    delta: float = 0.0,
) -> dict[str, NDArray[np.float64]]:
    r"""Compute equilibrium lambda, M_g, and k across a range of sigma values.

    This produces the lambda(sigma) curve that is the central output of the
    feedback loop analysis. The curve shows how environmental uncertainty
    drives signaling intensity through the cooperation network mechanism.

    Parameters
    ----------
    sigma_range : array
        Values of environmental uncertainty to sweep.
    lambda_W : float
        Within-group lambda component.
    delta : float
        Signal depreciation rate in [0, 1]. Passed through to
        lambda_total_at_sigma.
    [other params as in lambda_total_at_sigma]

    Returns
    -------
    dict with keys (all arrays matching sigma_range length):
        'sigma' : input sigma values
        'lambda_total' : equilibrium total lambda at each sigma
        'lambda_C' : competitive component at each sigma
        'lambda_X' : cooperative component at each sigma
        'M_g' : equilibrium monument stock
        'k' : network degree
        'alpha_eff' : vulnerability coefficient
    """
    results: dict[str, list[float]] = {
        "sigma": [], "lambda_total": [], "lambda_C": [], "lambda_X": [],
        "M_g": [], "k": [], "alpha_eff": [],
    }

    for sigma in sigma_range:
        eq = lambda_total_at_sigma(
            float(sigma), lambda_W, gamma, k_0, k_max, M_half,
            n, q_min, q_max, compute_lambda_C_func, delta=delta,
        )
        results["sigma"].append(float(sigma))
        results["lambda_total"].append(eq["lambda_total"])
        results["lambda_C"].append(eq["lambda_C"])
        results["lambda_X"].append(eq["lambda_X"])
        results["M_g"].append(eq["M_g"])
        results["k"].append(eq["k"])
        results["alpha_eff"].append(eq["alpha_eff"])

    return {key: np.array(val) for key, val in results.items()}


# =====================================================================
# Network simulation (NetworkX-based verification)
# =====================================================================


def form_network_signal_based(
    qualities: NDArray[np.float64],
    investments: NDArray[np.float64],
    rho: float = 0.95,
    threshold_quantile: float = 0.3,
    seed: int = 42,
) -> nx.Graph:
    r"""Form a cooperation network based on observed signal investments.

    Partners form connections when both parties assess each other as
    desirable based on observed investment. Higher signal fidelity (rho)
    means more reliable assessment of partner quality from signal.

    The model implements signal-based partner choice: perceived quality
    is rho * x_i + (1 - rho) * noise. Partners above a threshold
    form mutual connections. This produces assortative networks where
    high-quality signalers cluster together.

    Parameters
    ----------
    qualities : array of shape (n,)
        True qualities of individuals.
    investments : array of shape (n,)
        Observed signal investments x_i*.
    rho : float
        Signal fidelity in [0, 1]. Higher rho means more reliable
        assessment of partner quality from signal.
    threshold_quantile : float
        Minimum signal percentile for partnership consideration.
    seed : int
        Random seed for noise in perceived quality.

    Returns
    -------
    networkx.Graph
        Undirected graph where edges represent exchange partnerships.

    See the main-text signal-based partner choice section and SI
    "Network calibration and formation".
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(qualities)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    for i in range(n_nodes):
        G.nodes[i]["quality"] = float(qualities[i])
        G.nodes[i]["investment"] = float(investments[i])

    if n_nodes < 2:
        return G

    signal_thresh = np.quantile(investments, threshold_quantile)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Both must exceed threshold to be considered
            if investments[i] >= signal_thresh and investments[j] >= signal_thresh:
                # Perceived quality: signal + noise
                noise_scale = max(np.mean(investments), 0.1)
                perceived_i = rho * investments[i] + (1 - rho) * noise_scale * rng.standard_normal()
                perceived_j = rho * investments[j] + (1 - rho) * noise_scale * rng.standard_normal()

                # Connection forms when both perceive positive quality
                if perceived_i > 0 and perceived_j > 0:
                    G.add_edge(i, j)

    return G


def form_network_baseline(
    n: int,
    p_connect: float = DEFAULT_P_CONNECT_BASE,
    seed: int = 42,
) -> nx.Graph:
    """Form a baseline cooperation network without signaling.

    Uses Erdos-Renyi random graph to represent kinship/proximity-based
    network formation that does not depend on signaling investment.
    This provides the comparison case: cooperation networks that exist
    without costly signaling, maintained through kinship obligations
    and geographic proximity alone.

    Parameters
    ----------
    n : int
        Number of individuals.
    p_connect : float
        Base connection probability per pair. Must be in [0, 1].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    networkx.Graph

    See SI "Non-signaler baseline network sensitivity" and "Network
    calibration and formation".
    """
    return nx.erdos_renyi_graph(n, p_connect, seed=seed)


# =====================================================================
# Calibration
# =====================================================================


def calibrate_to_initial_vulnerability(
    alpha_target: float = 0.30,
    beta_target: float = 0.90,
    k_0: float = DEFAULT_K_0,
    k_max: float = DEFAULT_K_MAX,
    M_half: float = DEFAULT_M_HALF,
    n: int = DEFAULT_N,
    q_min: float = DEFAULT_Q_MIN,
    q_max: float = DEFAULT_Q_MAX,
    lam: float = DEFAULT_LAMBDA,
) -> dict[str, Any]:
    r"""Find gamma that reproduces initial model vulnerability parameters.

    Given target alpha_eff and beta_eff, finds the buffering efficiency
    gamma such that the vulnerability differential matches the initial
    model's assumed values (alpha = 0.30, beta = 0.90).

    The non-signaler has k = k_0, so:
    beta = 1 / (1 + gamma * k_0)  =>  gamma = (1/beta - 1) / k_0

    Then we verify that with signaler k = k(M_g), alpha matches the
    target. Since k_signal is typically much larger than k_0, exact
    matching of both alpha and beta simultaneously may require adjusting
    multiple parameters (gamma, k_0, k_max, M_half).

    Parameters
    ----------
    alpha_target : float
        Target signaler vulnerability (default 0.30 from initial model).
    beta_target : float
        Target non-signaler vulnerability (default 0.90).
    k_0 : float
        Baseline network degree (kinship/proximity).
    k_max : float
        Maximum signal-based degree.
    M_half : float
        Half-saturation monument stock.
    n : int
        Group size.
    q_min, q_max : float
        Quality range.
    lam : float
        Lambda for computing expected monument stock.

    Returns
    -------
    dict with keys:
        'gamma' : calibrated buffering efficiency
        'alpha_achieved' : vulnerability coefficient for signalers
        'beta_achieved' : vulnerability coefficient for non-signalers
        'alpha_target', 'beta_target' : input targets
        'k_signal', 'k_nonsignal' : network degrees
        'M_g' : expected monument stock

    See the main-text derived vulnerability differential section.
    """
    from signaling.layer1 import expected_monument_stock

    if k_0 <= 0:
        raise ValueError("k_0 must be > 0 for calibration from beta_target")

    # From beta: gamma = (1/beta - 1) / k_0
    gamma = (1.0 / beta_target - 1.0) / k_0

    # Compute M_g at given lambda
    M_g = expected_monument_stock(n, q_min, q_max, lam)

    # Signaler network degree
    k_signal = float(network_degree(M_g, k_0, k_max, M_half))

    # Achieved vulnerability
    alpha_achieved = float(vulnerability_coefficient(k_signal, gamma))
    beta_achieved = float(vulnerability_coefficient(k_0, gamma))

    return {
        "gamma": gamma,
        "alpha_achieved": alpha_achieved,
        "beta_achieved": beta_achieved,
        "alpha_target": alpha_target,
        "beta_target": beta_target,
        "k_signal": k_signal,
        "k_nonsignal": k_0,
        "M_g": M_g,
    }
