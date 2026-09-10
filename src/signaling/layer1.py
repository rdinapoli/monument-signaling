"""Individual signaling equilibrium.

Derives the separating equilibrium for costly monument investment,
where individual quality q determines optimal investment x*(q) through
the signaling mechanism.

Covers:
- Symbolic derivation of x*(q) = sqrt(lambda * (q^2 - q_min^2))
- Equilibrium fitness w*(q) and fitness gain Delta_w(q)
- Verification of the Spence (single-crossing) condition
- Aggregate group signal M_g = sum(x_i*)
- Sensitivity analysis across cost and benefit function forms

These results are reported in the main-text Layer 1 section, with derivations
in the supplementary "Layer 1 equilibrium-vs-baseline diagnostic" and "Functional form
robustness" sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.stats import beta as beta_dist
from scipy.stats import truncnorm

# =====================================================================
# Dataclasses
# =====================================================================


@dataclass(frozen=True)
class CostSpec:
    """Specification of a cost function c(x, q) and its partial derivatives.

    The cost function represents the fitness cost of investing x units of
    labor in monument construction for an individual of quality q.

    Attributes
    ----------
    c : callable (x, q) -> float
        Cost function c(x, q).
    c_x : callable (x, q) -> float
        Partial derivative dc/dx.
    c_xx : callable (x, q) -> float
        Second partial d^2c/dx^2 (convexity in x).
    c_xq : callable (x, q) -> float
        Cross partial d^2c/dxdq. The Spence condition requires this < 0.
    label : str
        Human-readable identifier for the cost specification.
    """

    c: Callable[[float, float], float]
    c_x: Callable[[float, float], float]
    c_xx: Callable[[float, float], float]
    c_xq: Callable[[float, float], float]
    label: str


@dataclass(frozen=True)
class BenefitSpec:
    """Specification of a benefit function b(q_hat, lam).

    The benefit function represents the social fitness reward from being
    perceived as quality q_hat, scaled by lambda.

    Attributes
    ----------
    b : callable (q_hat, lam) -> float
        Benefit function b(q_hat, lam).
    b_prime : callable (q_hat, lam) -> float
        Derivative db/dq_hat.
    label : str
        Human-readable identifier for the benefit specification.
    """

    b: Callable[[float, float], float]
    b_prime: Callable[[float, float], float]
    label: str


@dataclass(frozen=True)
class SpenceResult:
    """Results of Spence (single-crossing) condition verification.

    The Spence condition (d^2c/dxdq < 0) is necessary for a separating
    equilibrium: it ensures higher-quality types face lower marginal
    signaling costs.

    Attributes
    ----------
    quadratic_cross_partial : sympy.Expr
        Cross partial d^2c/dxdq for the quadratic cost c = x^2/(2q).
    power_cross_partial : sympy.Expr
        Cross partial d^2c/dxdq for the power cost c = x^a/(a*q^b).
    quadratic_is_negative : bool
        True if the quadratic cross partial is unconditionally negative.
    power_condition : sympy.Expr
        Condition on (a, b) ensuring the power cross partial is negative.
    symbols : dict mapping symbol names to sympy.Symbol objects.
    """

    quadratic_cross_partial: sp.Expr
    power_cross_partial: sp.Expr
    quadratic_is_negative: bool
    power_condition: sp.Expr
    symbols: dict[str, sp.Symbol]


@dataclass(frozen=True)
class EquilibriumResult:
    """Symbolic derivation of the separating equilibrium.

    The equilibrium investment function x*(q) is derived from the ODE
    x'(q) = lambda*q/x(q) with boundary condition x*(q_min) = 0.
    This is the separating-equilibrium schedule x*(q) of the main text (Layer 1).

    Attributes
    ----------
    x_star : sympy.Expr
        Equilibrium investment: sqrt(lam * (q^2 - q_min^2)).
    x_star_prime : sympy.Expr
        First derivative dx*/dq (positive for q > q_min).
    x_star_double_prime : sympy.Expr
        Second derivative d^2x*/dq^2 (negative: concavity).
    soc : sympy.Expr
        Second-order condition d^2w/dx^2 at equilibrium.
        Must be negative for all q > q_min.
    ode : sympy.Eq
        The differential equation defining the equilibrium.
    symbols : dict mapping symbol names to sympy.Symbol objects.
    """

    x_star: sp.Expr
    x_star_prime: sp.Expr
    x_star_double_prime: sp.Expr
    soc: sp.Expr
    ode: sp.Eq
    symbols: dict[str, sp.Symbol]


@dataclass(frozen=True)
class FitnessResult:
    """Symbolic derivation of equilibrium fitness and the Delta_w diagnostic.

    The equilibrium-vs-baseline difference Delta_w(q) = w*(q) - q is positive
    for all q > q_min. It is a partial-equilibrium diagnostic (the positional
    reward cannot deliver it groupwide); the load-bearing within-group result
    is the free-rider deterrent (free_rider_deterrent). See the supplementary
    "Layer 1 equilibrium-vs-baseline diagnostic".

    Attributes
    ----------
    w_star : sympy.Expr
        Equilibrium fitness: q*(1 + lam/2) + lam*q_min^2/(2*q).
    w_nosignal : sympy.Expr
        Fitness without signaling: q.
    delta_w : sympy.Expr
        Fitness gain: lam*q/2 + lam*q_min^2/(2*q).
    delta_w_prime : sympy.Expr
        Derivative d(Delta_w)/dq.
    delta_w_positive : bool
        Whether SymPy can verify Delta_w > 0 for q > q_min.
    symbols : dict mapping symbol names to sympy.Symbol objects.
    """

    w_star: sp.Expr
    w_nosignal: sp.Expr
    delta_w: sp.Expr
    delta_w_prime: sp.Expr
    delta_w_positive: bool
    symbols: dict[str, sp.Symbol]


# =====================================================================
# Cost function factories
# =====================================================================


def quadratic_cost() -> CostSpec:
    """Create the quadratic cost specification c(x, q) = x^2 / (2q).

    This is the default cost function satisfying:
      1. Increasing in x: dc/dx = x/q > 0
      2. Convex in x: d^2c/dx^2 = 1/q > 0
      3. Spence condition: d^2c/dxdq = -x/q^2 < 0

    This is the Spence condition of the main text (Layer 1, cost function).
    """
    return CostSpec(
        c=lambda x, q: x**2 / (2 * q),
        c_x=lambda x, q: x / q,
        c_xx=lambda x, q: 1.0 / q,
        c_xq=lambda x, q: -x / q**2,
        label="quadratic: x^2/(2q)",
    )


def power_cost(a: float = 2.0, b: float = 1.0) -> CostSpec:
    """Create a power cost specification c(x, q) = x^a / (a * q^b).

    Generalizes the quadratic cost. Reduces to quadratic when a=2, b=1.
    The Spence condition holds for all a > 1, b > 0.

    Parameters
    ----------
    a : float
        Exponent on investment (a > 1 required for convexity).
    b : float
        Exponent on quality (b > 0 required for Spence condition).
    """
    if a <= 1:
        raise ValueError(f"Power cost requires a > 1 for convexity, got a={a}")
    if b <= 0:
        raise ValueError(f"Power cost requires b > 0 for Spence condition, got b={b}")
    return CostSpec(
        c=lambda x, q: x**a / (a * q**b),
        c_x=lambda x, q: x ** (a - 1) / q**b,
        c_xx=lambda x, q: (a - 1) * x ** (a - 2) / q**b,
        c_xq=lambda x, q: -b * x ** (a - 1) / q ** (b + 1),
        label=f"power: x^{a}/({a}*q^{b})",
    )


def exponential_cost() -> CostSpec:
    """Create the exponential cost specification c(x, q) = (exp(x/q) - 1) / q.

    A strongly convex alternative to the quadratic cost. Satisfies the
    Spence condition because marginal cost exp(x/q)/q decreases in q
    (holding x fixed) due to the x/q ratio in the exponent.
    """
    return CostSpec(
        c=lambda x, q: (np.exp(x / q) - 1) / q,
        c_x=lambda x, q: np.exp(x / q) / q**2,
        c_xx=lambda x, q: np.exp(x / q) / q**3,
        c_xq=lambda x, q: -np.exp(x / q) * (2 * q + x) / q**4,
        label="exponential: (exp(x/q)-1)/q",
    )


# =====================================================================
# Benefit function factories
# =====================================================================


def linear_benefit() -> BenefitSpec:
    """Create the linear benefit specification b(q_hat, lam) = lam * q_hat.

    This is the signaling benefit b(q_hat) = lam*q_hat of the main text (Layer 1).
    """
    return BenefitSpec(
        b=lambda q_hat, lam: lam * q_hat,
        b_prime=lambda q_hat, lam: lam,
        label="linear: lam*q_hat",
    )


def concave_benefit(gamma: float = 0.5) -> BenefitSpec:
    """Create a concave benefit specification b(q_hat, lam) = lam * q_hat^gamma.

    Models diminishing social returns to perceived quality. The qualitative
    results (honest signaling, positive within-group selection) hold under
    concavity, but the equilibrium investment function changes form.
    The linear-benefit assumption and concave alternatives are examined in
    the supplementary "Functional form robustness".

    Parameters
    ----------
    gamma : float
        Concavity parameter, 0 < gamma < 1. Lower gamma means stronger
        diminishing returns.
    """
    if not 0 < gamma < 1:
        raise ValueError(f"Concave benefit requires 0 < gamma < 1, got gamma={gamma}")
    return BenefitSpec(
        b=lambda q_hat, lam: lam * q_hat**gamma,
        b_prime=lambda q_hat, lam: lam * gamma * q_hat ** (gamma - 1),
        label=f"concave: lam*q_hat^{gamma}",
    )


# =====================================================================
# Numerical core functions
# =====================================================================


def _validate_inputs(
    q: NDArray[np.float64] | float,
    q_min: float,
    lam: float,
) -> NDArray[np.float64]:
    """Validate and convert inputs for numerical functions.

    Raises ValueError for invalid parameter combinations.
    Returns q as a numpy array.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if q_min <= 0:
        raise ValueError(f"q_min must be positive, got {q_min}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if np.any(q_arr < q_min):
        raise ValueError(
            f"All q values must be >= q_min={q_min}, "
            f"got min(q)={np.min(q_arr)}"
        )
    return q_arr


def equilibrium_investment(
    q: NDArray[np.float64] | float,
    q_min: float,
    lam: float,
) -> NDArray[np.float64] | float:
    """Compute equilibrium investment x*(q) = sqrt(lam * (q^2 - q_min^2)).

    This is the separating equilibrium investment function derived from the
    ODE x'(q) = lam*q/x(q) with boundary condition x*(q_min) = 0.
    It is the separating-equilibrium schedule x*(q) of the main text (Layer 1).

    The reward parameter here is the within-group informational reward
    lambda_W alone. The between-group components lambda_C and lambda_X are
    marginal returns on the physical group stock M_g, realized at the group
    level through the war-avoidance factor K and the survival factor S; they
    are not informational rewards an individual captures through the
    inferred-quality channel q_hat(x), so they do not enter the individual
    first-order condition. (The superseded composite-lambda variant, in which
    they did, moves x* by under 0.5% at derived values; see the SI Banach
    robustness section and layer3.lambda_total_at_sigma.)

    Parameters
    ----------
    q : float or array
        Individual quality, must be >= q_min.
    q_min : float
        Minimum quality in the population, must be > 0.
    lam : float
        Within-group informational reward lambda_W, must be >= 0.

    Returns
    -------
    x_star : float or array
        Equilibrium investment level. Returns scalar if input q is scalar.
    """
    q_arr = _validate_inputs(q, q_min, lam)
    # Handle lam = 0: no signaling incentive, zero investment
    if lam == 0.0:
        result = np.zeros_like(q_arr)
        return float(result) if result.ndim == 0 else result
    # Factored form (q - q_min)(q + q_min) instead of q^2 - q_min^2: the
    # squared form mixes NumPy's vectorized power with a scalar power, which
    # can round 1 ulp apart, so at q == q_min the sqrt argument came out as
    # -1 ulp (silent NaN) or +1 ulp (spurious ~1e-8 investment) for ~0.07%
    # of floor values. In the factored form q - q_min is exactly 0.0 at the floor, enforcing
    # the boundary condition x*(q_min) = 0 exactly for every q_min.
    result = np.sqrt(lam * (q_arr - q_min) * (q_arr + q_min))
    return float(result) if result.ndim == 0 else result


def equilibrium_fitness(
    q: NDArray[np.float64] | float,
    q_min: float,
    lam: float,
) -> NDArray[np.float64] | float:
    """Compute equilibrium fitness w*(q) = q*(1 + lam/2) + lam*q_min^2/(2*q).

    This is the fitness of an individual of quality q who invests x*(q) in
    the signaling equilibrium, evaluated against a counterfactual no-signaling
    baseline w = q in which the reward lambda_W remains available. It is a
    diagnostic, not the free-rider resolution: the within-group reward is
    positional (zero-sum), so the group cannot all gain absolutely relative
    to no signaling. The load-bearing comparison is the deviation payoff
    (free_rider_deterrent), which conditions on the equilibrium.

    Parameters
    ----------
    q : float or array
        Individual quality, must be >= q_min.
    q_min : float
        Minimum quality in the population, must be > 0.
    lam : float
        Within-group informational reward lambda_W, must be >= 0.

    Returns
    -------
    w_star : float or array
        Fitness at equilibrium investment.
    """
    q_arr = _validate_inputs(q, q_min, lam)
    if lam == 0.0:
        result = q_arr.copy()
        return float(result) if result.ndim == 0 else result
    result = q_arr * (1 + lam / 2) + lam * q_min**2 / (2 * q_arr)
    return float(result) if result.ndim == 0 else result


def fitness_gain(
    q: NDArray[np.float64] | float,
    q_min: float,
    lam: float,
) -> NDArray[np.float64] | float:
    """Compute the equilibrium-vs-baseline diagnostic Delta_w = lam*q/2 + lam*q_min^2/(2*q).

    This quantity is positive for all q >= q_min when lam > 0. At q = q_min
    the two terms combine to give Delta_w = lam*q_min. It compares the
    equilibrium payoff to a counterfactual world with no signaling in which
    the reward lambda_W is nonetheless available, so it is a partial-
    equilibrium diagnostic. It is NOT the central within-group result:
    because the within-group reward is positional (zero-sum status), the
    group as a whole cannot gain Delta_w while paying the aggregate cost.
    The result that resolves free-riding is the deviation comparison
    Delta_fr(q) >= 0 (free_rider_deterrent), which conditions on the
    equilibrium and is consistent with positionality. See the supplementary
    "Layer 1 equilibrium-vs-baseline diagnostic" and "The
    within-group participation game and the resolution of free-riding".

    Parameters
    ----------
    q : float or array
        Individual quality, must be >= q_min.
    q_min : float
        Minimum quality in the population, must be > 0.
    lam : float
        Within-group informational reward lambda_W, must be >= 0.

    Returns
    -------
    delta_w : float or array
        Equilibrium-vs-baseline fitness difference (diagnostic).
    """
    q_arr = _validate_inputs(q, q_min, lam)
    if lam == 0.0:
        result = np.zeros_like(q_arr)
        return float(result) if result.ndim == 0 else result
    result = lam * q_arr / 2 + lam * q_min**2 / (2 * q_arr)
    return float(result) if result.ndim == 0 else result


def cost_at_equilibrium(
    q: NDArray[np.float64] | float,
    q_min: float,
    lam: float,
) -> NDArray[np.float64] | float:
    """Compute signaling cost at equilibrium: c(x*(q), q) = lam*(q^2 - q_min^2)/(2*q).

    This is the reproductive cost paid by quality-q individuals in the
    separating equilibrium, using the quadratic cost function. With the
    lambda_W-only first-order condition this closed form is exactly the cost
    priced by the free-rider deterrent (free_rider_deterrent), removing the
    former composite-lambda mismatch between the schedule and the deterrent.

    Parameters
    ----------
    q : float or array
        Individual quality, must be >= q_min.
    q_min : float
        Minimum quality in the population, must be > 0.
    lam : float
        Within-group informational reward lambda_W, must be >= 0.

    Returns
    -------
    cost : float or array
        Cost of equilibrium investment.
    """
    # ASSUMPTION: quadratic cost c(x,q) = x^2/(2q). Other cost functions satisfying
    # the Spence condition yield different closed forms; see cost_function_sensitivity().
    q_arr = _validate_inputs(q, q_min, lam)
    if lam == 0.0:
        result = np.zeros_like(q_arr)
        return float(result) if result.ndim == 0 else result
    result = lam * (q_arr**2 - q_min**2) / (2 * q_arr)
    return float(result) if result.ndim == 0 else result


def average_equilibrium_cost(
    lam: float,
    q_min: float,
    q_max: float,
) -> float:
    r"""Average reproductive cost at the separating equilibrium.

    Under the quadratic cost function :math:`c(x, q) = x^2/(2q)` and a uniform
    quality distribution on :math:`[q_{\min}, q_{\max}]`, the average of the
    equilibrium cost has a closed form:

    .. math::
        C_{\mathrm{model}}(\lambda) = E[c(x^*(q), q)]
            = \frac{\lambda}{q_{\max} - q_{\min}}
              \left[\frac{q_{\max}^2 - q_{\min}^2}{4}
                    - \frac{q_{\min}^2}{2}\ln\!\frac{q_{\max}}{q_{\min}}\right]

    This is the $C_{\mathrm{model}}(\lambda_W)$ that enters the assembled
    group fitness of the main text. It is the cost that the model's own
    equilibrium generates, as distinct from any exogenously calibrated cost.
    Using $C = C_{\mathrm{model}}(\lambda_W)$ in the threshold calculation
    enforces internal consistency between the cost function of Layer 1 and
    the per-capita cost appearing in the multilevel Price equation. Under the
    lambda_W-only first-order condition this is exact: the schedule, the cost,
    and the free-rider deterrent all carry the same lambda_W.

    Parameters
    ----------
    lam : float
        Fitness value of social rewards (:math:`\lambda`). Must be >= 0.
    q_min : float
        Minimum quality in the population. Must be > 0.
    q_max : float
        Maximum quality. Must be > q_min.

    Returns
    -------
    float
        Average equilibrium cost $C_{\mathrm{model}}(\lambda)$.

    See the supplementary "Self-consistency: $B(\lambda_W)$ and
    $C_{\mathrm{model}}(\lambda_W)$ derivations".
    """
    if lam <= 0.0:
        return 0.0
    q_range = q_max - q_min
    bracket = (q_max**2 - q_min**2) / 4.0 - (q_min**2 / 2.0) * np.log(q_max / q_min)
    return float(lam * bracket / q_range)


def lambda_W_for_C(
    target_C: float,
    q_min: float,
    q_max: float,
) -> float:
    r"""Invert $C_{\mathrm{model}}(\lambda) = C$ for :math:`\lambda`.

    Under the quadratic cost and uniform quality distribution, the average
    equilibrium cost is linear in $\lambda$, so the inverse is a single
    division. This function returns the within-group social reward
    :math:`\lambda_W` at which the model's endogenous cost matches a
    specified target (e.g., an empirical architectural-energetics estimate).

    Parameters
    ----------
    target_C : float
        Target average reproductive cost.
    q_min : float
        Minimum quality.
    q_max : float
        Maximum quality.

    Returns
    -------
    float
        $\lambda_W$ such that average_equilibrium_cost($\lambda_W$, q_min, q_max) = target_C.
    """
    q_range = q_max - q_min
    bracket = (q_max**2 - q_min**2) / 4.0 - (q_min**2 / 2.0) * np.log(q_max / q_min)
    return float(target_C * q_range / bracket)


# =====================================================================
# Free-rider deterrence and the binary participation game
# =====================================================================


def free_rider_deterrent(
    q: float | NDArray[np.float64],
    lam: float,
    q_min: float,
) -> float | NDArray[np.float64]:
    r"""Fitness loss a type-:math:`q` member incurs by free-riding (Layer 1).

    A free-rider declines to build and so invests :math:`x = 0`. Because the
    separating schedule has :math:`x^*(q_{\min}) = 0`, the action :math:`x = 0`
    is the equilibrium play of the lowest type and is ON PATH: inverting the
    schedule reads it as :math:`q_{\min}` (no off-path refinement is needed;
    refinements such as D1 govern only actions outside the schedule's range
    and have no bearing on an on-path action). A free-rider of true quality
    :math:`q` therefore pools
    with the floor type, is perceived at :math:`q_{\min}`, earning status
    :math:`\lambda q_{\min}` rather than :math:`\lambda q`, while saving
    the construction cost :math:`c(x^*(q)) = \lambda (q^2 - q_{\min}^2)/(2q)`.
    The deterrent is the incentive-compatibility margin against reporting the
    floor, already contained in the global-IC computation.

    The deterrent is the equilibrium payoff minus the free-ride payoff:

    .. math::
        \Delta_{\mathrm{fr}}(q)
            = \big[q + \lambda q - c(x^*(q))\big] - \big[q + \lambda q_{\min}\big]
            = \lambda (q - q_{\min}) - \frac{\lambda (q^2 - q_{\min}^2)}{2q}
            = \frac{\lambda (q - q_{\min})^2}{2q} \;\ge\; 0,

    strictly positive for every :math:`q > q_{\min}`, with
    :math:`\Delta_{\mathrm{fr}}(q_{\min}) = 0` exactly. Free-riding is thus
    individually deterred for all types above the floor, so building is each
    such type's unique best response; the floor type invests zero and is
    indifferent, so the within-group equilibrium is full participation in the
    ALMOST-SURE sense (all types above the measure-zero floor): the
    free-rider problem is resolved *within* the group, without recourse to
    between-group selection. The population mean of this
    deterrent equals the within-group selection differential
    :math:`s_W - C_{\mathrm{model}} = \beta_0/(SK)`
    (see :func:`participation_equilibrium`), which makes the sign structural
    (an expected square) with a magnitude that scales with quality
    heterogeneity. See the supplementary "The within-group participation game
    and the resolution of free-riding".

    Parameters
    ----------
    q : float or array
        True quality of the would-be free-rider. Must be > 0.
    lam : float
        Within-group social reward :math:`\lambda`.
    q_min : float
        Equilibrium floor at which a non-builder is inferred.

    Returns
    -------
    float or array
        The fitness loss :math:`\Delta_{\mathrm{fr}}(q) \ge 0`.
    """
    if lam < 0.0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if q_min <= 0.0:
        raise ValueError(f"q_min must be > 0, got {q_min}")
    q_arr_check = np.asarray(q, dtype=np.float64)
    if np.any(q_arr_check < q_min):
        raise ValueError(
            "free_rider_deterrent is defined for types q >= q_min (types "
            "below the floor do not exist in the separating equilibrium); "
            f"got min(q) = {float(np.min(q_arr_check))} < q_min = {q_min}"
        )
    qq = np.asarray(q, dtype=float)
    result = lam * (qq - q_min) ** 2 / (2.0 * qq)
    if np.ndim(q) == 0:
        return float(result)
    return result


def participation_equilibrium(
    lam: float,
    q_min: float,
    q_max: float,
) -> dict[str, float]:
    r"""Within-group participation result (full participation), uniform quality.

    The extensive-margin companion to the continuous intensity margin
    (:func:`derive_equilibrium`). In the separating equilibrium every type
    :math:`q > q_{\min}` strictly prefers to build, because the free-rider
    deterrent :math:`\Delta_{\mathrm{fr}}(q) = \lambda (q-q_{\min})^2/(2q)` is
    strictly positive (:func:`free_rider_deterrent`): deviating to :math:`x=0`
    is read by observers as the floor type :math:`q_{\min}` (the on-path
    schedule-inversion reading, since :math:`x=0` is the equilibrium action
    of :math:`q_{\min}`). The within-group equilibrium is therefore *full
    participation* in the almost-sure sense (the zero-investing floor type is
    indifferent); there is no interior build threshold.

    A free-rider (pooling with the floor type's on-path action) is inferred
    at :math:`q_{\min}`, while
    the average builder is perceived at :math:`\mathbb{E}[q]`, so the
    builder-vs-free-rider status differential is

    .. math::
        s_W = \lambda(\mathbb{E}[q] - q_{\min})
            = \lambda \tfrac{q_{\max}-q_{\min}}{2} \quad (\text{uniform}),

    matching :func:`signaling.price_equation.within_group_status_differential`.
    The within-group selection differential is
    :math:`\beta_0 = SK(s_W - C_{\mathrm{model}})`, and its value per unit
    :math:`SK` is the population mean of the per-type deterrent,

    .. math::
        \frac{\beta_0}{SK} = s_W - C_{\mathrm{model}}
            = \mathbb{E}\!\left[\Delta_{\mathrm{fr}}(q)\right]
            = \lambda\,\mathbb{E}\!\left[\frac{(q-q_{\min})^2}{2q}\right] \ge 0,

    an expected square that is positive for *every* quality distribution with
    mass above the floor and vanishes only as the population concentrates at
    :math:`q_{\min}`. It is *larger*, not smaller, for high-quality (e.g.
    right-skewed) populations, for which a free-rider has more to lose by being
    read at the floor (see :func:`participation_differential_sampled`).

    Scope: the full-participation result requires :math:`\lambda_W > 0`. At
    :math:`\lambda_W = 0` the schedule collapses (every type invests zero,
    :math:`\Delta_{\mathrm{fr}}(q) = 0` for all :math:`q`), there is no
    separation and no strict preference either way, so participation is
    *indeterminate*, not full: the function returns
    ``participation_rate = nan`` and ``full_participation = False`` there.
    For :math:`\lambda_W < 0` the square-root schedule is not real for
    :math:`q > q_{\min}` and the signaling model is outside its domain, so a
    ``ValueError`` is raised.

    Returns
    -------
    dict with keys ``s_W``, ``C_model``, ``mean_deterrent``
    (= :math:`s_W - C_{\mathrm{model}} = \beta_0/(SK)`), ``mu_N`` (free-rider
    inference, :math:`= q_{\min}`), ``mu_B`` (average builder,
    :math:`= \mathbb{E}[q]`), ``participation_rate`` (= 1.0 for
    :math:`\lambda_W > 0`; ``nan`` at :math:`\lambda_W = 0`, indeterminate),
    and ``full_participation`` (True only for :math:`\lambda_W > 0`).
    """
    if lam < 0.0:
        raise ValueError(
            "participation_equilibrium requires lambda_W >= 0: for negative "
            "lambda_W the separating schedule x*(q) is not real and the "
            "signaling model is outside its domain."
        )
    Eq = (q_min + q_max) / 2.0
    if lam == 0.0:
        # Degenerate boundary: x*(q) = 0 and Delta_fr(q) = 0 for every type.
        # All labels earn identical payoffs, so participation is
        # indeterminate (there is nothing to separate), not "full".
        return {
            "s_W": 0.0, "C_model": 0.0, "mean_deterrent": 0.0,
            "mu_N": float(q_min), "mu_B": float(Eq),
            "participation_rate": float("nan"),
            "full_participation": False,
        }
    C = average_equilibrium_cost(lam, q_min, q_max)
    s_W = lam * (Eq - q_min)
    return {
        "s_W": float(s_W),
        "C_model": float(C),
        "mean_deterrent": float(s_W - C),
        "mu_N": float(q_min),
        "mu_B": float(Eq),
        "participation_rate": 1.0,
        "full_participation": True,
    }


def participation_differential_sampled(
    lam: float,
    quality_samples: NDArray[np.float64],
    q_min: float,
    q_max: float,
) -> dict[str, float]:
    r"""Within-group differential :math:`\beta_0/(SK)` for an arbitrary sample.

    Computes the genuine within-group selection differential per unit
    :math:`SK` for an empirical or simulated quality distribution: the
    population mean of the per-type free-rider deterrent,

    .. math::
        \frac{\beta_0}{SK} = \lambda\,\mathbb{E}\!\left[\frac{(q-q_{\min})^2}{2q}\right]
        = s_W - C_{\mathrm{model}},

    where a free-rider is read at the floor :math:`q_{\min}`
    (:func:`free_rider_deterrent`). Being an expected square, it is positive
    for every distribution with mass above :math:`q_{\min}` and vanishes only
    as the sample concentrates at the floor. It is *larger* for high-quality
    (e.g. right-skewed) populations -- a free-rider has more to lose -- so the
    within-group resolution is strongest, not weakest, there. For uniform
    quality it reproduces the closed form in :func:`participation_equilibrium`.

    All three returned quantities are computed as SAMPLE means over
    ``quality_samples`` (not from the uniform closed forms), so the identity
    :math:`s_W - C = \mathbb{E}[\Delta_{\mathrm{fr}}]` holds exactly for the
    sample:

    .. math::
        C_{\mathrm{sample}} = \lambda\,\overline{\frac{q^2 - q_{\min}^2}{2q}},
        \qquad
        s_{W,\mathrm{sample}} = \lambda\,(\bar q - q_{\min}).

    (Returning the uniform closed-form cost for an arbitrary sample would
    break this identity off the uniform distribution.) The uniform
    closed-form cost is returned separately as
    ``C_model_uniform`` for comparison against the calibrated locus.

    Returns
    -------
    dict with keys ``mean_deterrent`` (= :math:`\beta_0/(SK) = s_W - C`,
    sample mean), ``C_model`` (sample mean equilibrium cost), ``s_W``
    (sample status differential :math:`\lambda(\bar q - q_{\min})`), and
    ``C_model_uniform`` (the uniform closed form, for reference).
    """
    q = np.asarray(quality_samples, dtype=float)
    if np.any(q <= 0):
        raise ValueError("quality_samples must be strictly positive.")
    C_sample = float(lam * np.mean((q ** 2 - q_min ** 2) / (2.0 * q)))
    s_W_sample = float(lam * (float(np.mean(q)) - q_min))
    mean_deterrent = float(lam * np.mean((q - q_min) ** 2 / (2.0 * q)))
    return {
        "mean_deterrent": mean_deterrent,
        "C_model": C_sample,
        "s_W": s_W_sample,
        "C_model_uniform": float(average_equilibrium_cost(lam, q_min, q_max)),
    }


# =====================================================================
# Deviation and pooling analysis
# =====================================================================


def receiver_inference(
    x: NDArray[np.float64] | float,
    q_min: float,
    lam: float,
    q_max: float | None = None,
) -> NDArray[np.float64] | float:
    r"""Compute the receiver's quality inference by inverting the equilibrium mapping.

    In the separating equilibrium, the receiver observes investment x and
    infers quality q_hat(x) = sqrt(x^2/lam + q_min^2). This is the
    standard least-cost separating equilibrium inference (Riley 1979).

    Clamps at q_min for x <= 0 and at q_max (if provided) for large x.

    Parameters
    ----------
    x : float or array
        Observed investment level(s).
    q_min : float
        Minimum quality in the population, must be > 0.
    lam : float
        Fitness value of social rewards, must be > 0.
    q_max : float or None
        Maximum quality. If provided, inferred quality is clamped at q_max.

    Returns
    -------
    q_hat : float or array
        Inferred quality.
    """
    if q_min <= 0:
        raise ValueError(f"q_min must be positive, got {q_min}")
    if lam <= 0:
        raise ValueError(f"lam must be positive for receiver inference, got {lam}")
    x_arr = np.asarray(x, dtype=np.float64)
    # Invert x*(q) = sqrt(lam*(q^2 - q_min^2)) to get q_hat = sqrt(x^2/lam + q_min^2)
    q_hat = np.sqrt(np.maximum(x_arr, 0.0) ** 2 / lam + q_min**2)
    # Clamp below at q_min (for x <= 0)
    q_hat = np.maximum(q_hat, q_min)
    if q_max is not None:
        q_hat = np.minimum(q_hat, q_max)
    return float(q_hat) if q_hat.ndim == 0 else q_hat


def fitness_at_deviation(
    x: NDArray[np.float64] | float,
    q: float,
    q_min: float,
    lam: float,
    q_max: float | None = None,
) -> NDArray[np.float64] | float:
    r"""Compute fitness w(x, q) = q - x^2/(2q) + lam * q_hat(x) for arbitrary x.

    At x = x*(q), this returns w*(q). The separating equilibrium is
    verified when this function is maximized at x = x*(q) for every q.

    Uses quadratic cost c(x, q) = x^2/(2q) and linear benefit lam * q_hat(x),
    where q_hat is the receiver inference from the separating equilibrium.

    Parameters
    ----------
    x : float or array
        Investment level(s) to evaluate.
    q : float
        True quality of the individual.
    q_min : float
        Minimum quality, must be > 0.
    lam : float
        Fitness value of social rewards, must be > 0.
    q_max : float or None
        Maximum quality for clamping receiver inference.

    Returns
    -------
    w : float or array
        Fitness at the given investment level(s).
    """
    if q_min <= 0:
        raise ValueError(f"q_min must be positive, got {q_min}")
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    if q < q_min:
        raise ValueError(f"q must be >= q_min={q_min}, got {q}")
    x_arr = np.asarray(x, dtype=np.float64)
    q_hat = receiver_inference(x_arr, q_min, lam, q_max)
    # w(x, q) = q - c(x, q) + lam * q_hat(x)
    # With quadratic cost: c(x, q) = x^2 / (2q)
    result = q - x_arr**2 / (2 * q) + lam * q_hat
    return float(result) if result.ndim == 0 else result


def pooling_fitness(
    q: NDArray[np.float64] | float,
    x_pool: float,
    q_mean: float,
    lam: float,
) -> NDArray[np.float64] | float:
    r"""Compute fitness under a pooling equilibrium.

    In a pooling equilibrium, all types invest x_pool and the receiver
    infers the prior mean quality E[q]:

        w_pool(q) = q - x_pool^2 / (2q) + lam * E[q]

    This is not an equilibrium under the Spence condition because high
    types can profitably deviate upward, which makes pooling unstable.

    Parameters
    ----------
    q : float or array
        True quality of the individual(s).
    x_pool : float
        Common investment level in the pooling outcome.
    q_mean : float
        Prior mean quality (receiver's inference under pooling).
    lam : float
        Fitness value of social rewards, must be >= 0.

    Returns
    -------
    w_pool : float or array
        Fitness under pooling.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    result = q_arr - x_pool**2 / (2 * q_arr) + lam * q_mean
    return float(result) if result.ndim == 0 else result


def expected_fitness_gain(
    q_min: float,
    q_max: float,
    lam: float,
    distribution: str = "uniform",
    **dist_params: float,
) -> float:
    r"""Compute E[Delta_w(q)] = B(lambda), the expected fitness gain from signaling.

    Integrates fitness_gain(q, q_min, lam) against the quality density
    over [q_min, q_max]. This is the within-group benefit of monument
    construction that offsets the reproductive cost C in the initial model
    comparison.

    Parameters
    ----------
    q_min : float
        Minimum quality, must be > 0.
    q_max : float
        Maximum quality, must be > q_min.
    lam : float
        Fitness value of social rewards, must be >= 0.
    distribution : str
        Quality distribution. One of 'uniform', 'truncated_normal', 'beta'.
    **dist_params :
        Additional parameters for the distribution:
        - truncated_normal: mu (mean), sigma (std dev) of the untruncated normal
        - beta: a (alpha shape), b (beta shape), rescaled to [q_min, q_max]

    Returns
    -------
    B_lambda : float
        Expected fitness gain E[Delta_w(q)], i.e. B(lambda).
    """
    if q_min <= 0:
        raise ValueError(f"q_min must be positive, got {q_min}")
    if q_max <= q_min:
        raise ValueError(f"q_max must exceed q_min, got q_max={q_max}, q_min={q_min}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if lam == 0.0:
        return 0.0

    # Build the density, reusing the same pattern as expected_monument_stock
    if distribution == "uniform":

        def integrand(q: float) -> float:
            return fitness_gain(q, q_min, lam) / (q_max - q_min)

    elif distribution == "truncated_normal":
        mu = dist_params.get("mu", (q_min + q_max) / 2)
        sigma = dist_params.get("sigma", (q_max - q_min) / 4)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        a_clip = (q_min - mu) / sigma
        b_clip = (q_max - mu) / sigma
        rv = truncnorm(a_clip, b_clip, loc=mu, scale=sigma)

        def integrand(q: float) -> float:
            return fitness_gain(q, q_min, lam) * rv.pdf(q)

    elif distribution == "beta":
        a_shape = dist_params.get("a", 2.0)
        b_shape = dist_params.get("b", 2.0)
        rv = beta_dist(a_shape, b_shape, loc=q_min, scale=q_max - q_min)

        def integrand(q: float) -> float:
            return fitness_gain(q, q_min, lam) * rv.pdf(q)

    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Supported: 'uniform', 'truncated_normal', 'beta'."
        )

    result, _ = quad(integrand, q_min, q_max, limit=100, epsrel=1e-10)
    return result


# =====================================================================
# Symbolic derivations
# =====================================================================

# Shared symbols used across symbolic functions
_q = sp.Symbol("q", positive=True)
_q_min = sp.Symbol("q_min", positive=True)
_q_max = sp.Symbol("q_max", positive=True)
_lam = sp.Symbol("lam", positive=True)
_x = sp.Symbol("x", positive=True)
_n = sp.Symbol("n", positive=True, integer=True)


def verify_spence_condition() -> SpenceResult:
    """Verify the single-crossing (Spence) condition symbolically.

    The Spence condition requires d^2c/dxdq < 0: the marginal cost of
    investment decreases with quality. This is the formal expression of
    Zahavi's handicap principle, the Spence condition of the main text (Layer 1).

    Verifies for:
      1. Quadratic cost: c = x^2/(2q) => d^2c/dxdq = -x/q^2 < 0
      2. General power cost: c = x^a/(a*q^b) => d^2c/dxdq = -b*x^(a-1)/q^(b+1)
    """
    x, q = _x, _q
    a = sp.Symbol("a", positive=True)
    b = sp.Symbol("b", positive=True)

    # Quadratic cost
    c_quad = x**2 / (2 * q)
    cross_quad = sp.diff(sp.diff(c_quad, x), q)
    cross_quad_simplified = sp.simplify(cross_quad)

    # Power cost
    c_power = x**a / (a * q**b)
    cross_power = sp.diff(sp.diff(c_power, x), q)
    cross_power_simplified = sp.simplify(cross_power)

    # The quadratic cross-partial is -x/q^2, which is negative for x > 0, q > 0
    quad_negative = sp.simplify(cross_quad_simplified + x / q**2) == 0

    # The power cross-partial is negative when b > 0 and a > 1 (since x > 0, q > 0)
    power_cond = sp.And(a > 1, b > 0)

    return SpenceResult(
        quadratic_cross_partial=cross_quad_simplified,
        power_cross_partial=cross_power_simplified,
        quadratic_is_negative=quad_negative,
        power_condition=power_cond,
        symbols={"x": x, "q": q, "a": a, "b": b},
    )


def derive_equilibrium() -> EquilibriumResult:
    """Derive the separating equilibrium symbolically.

    Solves the ODE x'(q) = lam*q/x(q) by separation of variables with
    boundary condition x*(q_min) = 0. The result is:

        x*(q) = sqrt(lam * (q^2 - q_min^2))

    Also computes first and second derivatives, and verifies the
    second-order condition. This yields the separating-equilibrium schedule
    x*(q) of the main text (Layer 1).

    The derivation proceeds by separation of variables rather than
    calling SymPy's dsolve, to make each step explicit and verifiable.
    """
    q, q_min, lam = _q, _q_min, _lam

    # The ODE from the first-order condition:
    # In equilibrium, FOC gives x'(q) = lam * q / x(q)
    x_func = sp.Function("x")
    ode = sp.Eq(x_func(q).diff(q), lam * q / x_func(q))

    # Solve by separation of variables:
    # x dx = lam q dq => x^2/2 = lam q^2/2 + K
    # Boundary condition x*(q_min) = 0: K = -lam q_min^2 / 2
    # => x* = sqrt(lam (q^2 - q_min^2))
    x_star = sp.sqrt(lam * (q**2 - q_min**2))

    # First derivative: verify this satisfies the ODE
    x_star_prime = sp.diff(x_star, q)
    x_star_prime_simplified = sp.simplify(x_star_prime)

    # Second derivative: shows concavity
    x_star_double_prime = sp.diff(x_star, q, 2)
    x_star_double_prime_simplified = sp.simplify(x_star_double_prime)

    # Second-order condition for fitness maximization at x*(q).
    # d^2w/dx^2 = -1/q + q_min^2/q^3 = -(q^2 - q_min^2)/q^3
    # This must be < 0 for q > q_min.
    # Derivation: q_hat(x) = sqrt(x^2/lam + q_min^2), so
    # q_hat''(x) = q_min^2 / (lam * q_hat(x)^3).
    # At equilibrium q_hat = q, so d^2w/dx^2 = -1/q + lam * q_min^2/(lam*q^3)
    # = -1/q + q_min^2/q^3 = -(q^2 - q_min^2)/q^3
    soc = -(q**2 - q_min**2) / q**3

    return EquilibriumResult(
        x_star=x_star,
        x_star_prime=x_star_prime_simplified,
        x_star_double_prime=x_star_double_prime_simplified,
        soc=soc,
        ode=ode,
        symbols={"q": q, "q_min": q_min, "lam": lam},
    )


def derive_equilibrium_fitness() -> FitnessResult:
    """Derive equilibrium fitness and the fitness gain from signaling.

    Substitutes x*(q) into the fitness function to obtain:
        w*(q) = q*(1 + lam/2) + lam*q_min^2/(2*q)

    The fitness gain Delta_w(q) = w*(q) - q = lam*q/2 + lam*q_min^2/(2*q)
    is a sum of two terms, each positive for q > 0 and lam > 0.
    This proves Delta_w > 0 for all q >= q_min.
    See the supplementary "Layer 1 equilibrium-vs-baseline diagnostic".
    """
    q, q_min, lam = _q, _q_min, _lam

    # Equilibrium investment
    x_star = sp.sqrt(lam * (q**2 - q_min**2))

    # Cost at equilibrium: x*^2 / (2q) = lam*(q^2 - q_min^2)/(2q)
    cost = x_star**2 / (2 * q)

    # Full fitness: w = q - cost + lam*q (benefit from signaling q_hat = q)
    w_star = q - cost + lam * q
    w_star_simplified = sp.simplify(sp.expand(w_star))

    # Fitness without signaling
    w_nosignal = q

    # Fitness gain
    delta_w = sp.simplify(sp.expand(w_star - q))
    delta_w_prime = sp.simplify(sp.diff(delta_w, q))

    # Verify positivity: Delta_w = lam*q/2 + lam*q_min^2/(2*q)
    # Both terms are positive for q > 0, lam > 0, q_min > 0
    # SymPy can verify this with assumptions
    term1 = lam * q / 2
    term2 = lam * q_min**2 / (2 * q)
    delta_w_positive = sp.ask(sp.Q.positive(term1)) and sp.ask(sp.Q.positive(term2))

    return FitnessResult(
        w_star=w_star_simplified,
        w_nosignal=w_nosignal,
        delta_w=delta_w,
        delta_w_prime=delta_w_prime,
        delta_w_positive=bool(delta_w_positive),
        symbols={"q": q, "q_min": q_min, "lam": lam},
    )


def derive_aggregate_signal() -> dict[str, Any]:
    """Derive properties of the group-level signal M_g = sum(x_i*).

    For a group of n individuals with quality drawn uniformly from
    [q_min, q_max], computes:
      - E[x*(q)]: expected individual investment
      - E[M_g] = n * E[x*(q)]: expected group monument stock
      - Var[M_g] = n * Var[x*(q)]: variance (for iid draws)

    Returns a dict with SymPy expressions. Uses sp.integrate for the
    uniform distribution case. The group monument stock M_g = sum(x_i*) is the
    aggregate assessed by neighboring groups (main text, Layer 1).
    """
    q, q_min, q_max, lam, n = _q, _q_min, _q_max, _lam, _n

    x_star = sp.sqrt(lam * (q**2 - q_min**2))

    # E[x*(q)] for uniform distribution on [q_min, q_max]
    density = sp.Rational(1) / (q_max - q_min)
    integrand = x_star * density

    expected_x_star = sp.integrate(integrand, (q, q_min, q_max))
    expected_x_star = sp.simplify(expected_x_star)

    # E[M_g] = n * E[x*(q)]
    expected_M_g = n * expected_x_star

    # E[x*^2] for variance computation
    expected_x_star_sq = sp.integrate(x_star**2 * density, (q, q_min, q_max))
    expected_x_star_sq = sp.simplify(expected_x_star_sq)

    # Var[x*] = E[x*^2] - (E[x*])^2
    var_x_star = sp.simplify(expected_x_star_sq - expected_x_star**2)

    # Var[M_g] = n * Var[x*] (for iid quality draws)
    var_M_g = n * var_x_star

    return {
        "expected_x_star": expected_x_star,
        "expected_M_g": expected_M_g,
        "var_x_star": var_x_star,
        "var_M_g": var_M_g,
        "symbols": {"q": q, "q_min": q_min, "q_max": q_max, "lam": lam, "n": n},
    }


# =====================================================================
# Aggregation functions
# =====================================================================


def group_monument_stock(
    qualities: NDArray[np.float64],
    q_min: float,
    lam: float,
) -> float:
    """Compute total group monument stock M_g = sum(x_i*(q_i)).

    The aggregate monument stock is the physical quantity observable by
    neighboring groups. It is the sum of individual equilibrium investments,
    each determined by that individual's quality. This aggregate is the group
    investment flow assessed by neighboring groups (main text, Layer 1).

    Parameters
    ----------
    qualities : array of float
        Individual quality values for each group member, all >= q_min.
    q_min : float
        Minimum quality, must be > 0.
    lam : float
        Fitness value of social rewards, must be >= 0.

    Returns
    -------
    M_g : float
        Total monument investment for the group.
    """
    x_stars = equilibrium_investment(np.asarray(qualities, dtype=np.float64), q_min, lam)
    return float(np.sum(x_stars))


def expected_monument_stock(
    n: int,
    q_min: float,
    q_max: float,
    lam: float,
    distribution: str = "uniform",
    **dist_params: float,
) -> float:
    """Compute E[M_g] for a group of size n via numerical integration.

    Integrates x*(q) against the specified quality distribution to obtain
    E[x*(q)], then multiplies by n. Under the lambda_W-only first-order
    condition the schedule input is lambda_W, so the group stock is set by
    the within-group reward alone (and is independent of sigma).

    Parameters
    ----------
    n : int
        Group size (number of individuals).
    q_min : float
        Minimum quality, must be > 0.
    q_max : float
        Maximum quality, must be > q_min.
    lam : float
        Within-group informational reward lambda_W, must be >= 0.
    distribution : str
        Quality distribution. One of 'uniform', 'truncated_normal', 'beta'.
    **dist_params :
        Additional parameters for the distribution:
        - truncated_normal: mu (mean), sigma (std dev) of the untruncated normal
        - beta: a (alpha shape), b (beta shape) for the Beta distribution,
          rescaled to [q_min, q_max]

    Returns
    -------
    E_M_g : float
        Expected group monument stock.

    Raises
    ------
    ValueError
        If distribution is not recognized or parameters are invalid.
    """
    if q_min <= 0:
        raise ValueError(f"q_min must be positive, got {q_min}")
    if q_max <= q_min:
        raise ValueError(f"q_max must exceed q_min, got q_max={q_max}, q_min={q_min}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if lam == 0.0:
        return 0.0

    # Integrand: x*(q) * f(q) where f is the quality density
    if distribution == "uniform":

        def integrand(q: float) -> float:
            return np.sqrt(lam * (q**2 - q_min**2)) / (q_max - q_min)

    elif distribution == "truncated_normal":
        mu = dist_params.get("mu", (q_min + q_max) / 2)
        sigma = dist_params.get("sigma", (q_max - q_min) / 4)
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        a_clip = (q_min - mu) / sigma
        b_clip = (q_max - mu) / sigma
        rv = truncnorm(a_clip, b_clip, loc=mu, scale=sigma)

        def integrand(q: float) -> float:
            return np.sqrt(lam * (q**2 - q_min**2)) * rv.pdf(q)

    elif distribution == "beta":
        a_shape = dist_params.get("a", 2.0)
        b_shape = dist_params.get("b", 2.0)
        rv = beta_dist(a_shape, b_shape, loc=q_min, scale=q_max - q_min)

        def integrand(q: float) -> float:
            return np.sqrt(lam * (q**2 - q_min**2)) * rv.pdf(q)

    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Supported: 'uniform', 'truncated_normal', 'beta'."
        )

    # Numerical integration with documented tolerance
    E_x_star, _ = quad(integrand, q_min, q_max, limit=100, epsrel=1e-10)
    return n * E_x_star


# =====================================================================
# Signal depreciation and reinvestment
# =====================================================================


def effective_monument_stock(
    I_g: float | NDArray[np.float64],
    delta: float,
) -> float | NDArray[np.float64]:
    r"""Steady-state effective monument stock under signal depreciation.

    Monuments persist physically but their informational value as signals
    of current group capacity depreciates at rate delta per period.
    Under constant investment flow I_g, the dynamics are:

    .. math::
        M_g(t+1) = (1 - \delta) M_g(t) + I_g

    At steady state, M_g^* = I_g / delta.

    For delta -> 0 (no depreciation), M_g^* -> infinity, reflecting that
    a permanent monument is an indefinitely lasting signal. For delta = 1,
    M_g^* = I_g, meaning the signal fully depreciates each period and only
    the current flow matters (equivalent to a feast or other ephemeral signal).

    Parameters
    ----------
    I_g : float or array
        Investment flow per period (e.g., from expected_monument_stock).
        Must be >= 0.
    delta : float
        Signal depreciation rate in (0, 1]. delta = 0 is disallowed
        because steady state is undefined (stock grows without bound).

    Returns
    -------
    float or array
        Steady-state effective monument stock M_g* = I_g / delta.

    Raises
    ------
    ValueError
        If delta is not in (0, 1].
    """
    if delta <= 0 or delta > 1:
        raise ValueError(f"delta must be in (0, 1], got {delta}")
    I_g = np.asarray(I_g, dtype=np.float64)
    return I_g / delta


def monument_stock_trajectory(
    I_g: float,
    delta: float,
    T: int,
    M_0: float = 0.0,
) -> NDArray[np.float64]:
    r"""Monument stock trajectory under signal depreciation.

    Computes the time series M_g(t) for t = 0, 1, ..., T given:

    .. math::
        M_g(t+1) = (1 - \delta) M_g(t) + I_g

    The trajectory converges to M_g^* = I_g / delta as t -> infinity.

    Parameters
    ----------
    I_g : float
        Constant investment flow per period. Must be >= 0.
    delta : float
        Signal depreciation rate in (0, 1].
    T : int
        Number of time periods.
    M_0 : float
        Initial monument stock at t = 0.

    Returns
    -------
    array of shape (T + 1,)
        Monument stock at each time step from t = 0 to t = T.
    """
    if delta <= 0 or delta > 1:
        raise ValueError(f"delta must be in (0, 1], got {delta}")
    M = np.zeros(T + 1)
    M[0] = M_0
    for t in range(T):
        M[t + 1] = (1.0 - delta) * M[t] + I_g
    return M


def signal_half_life(delta: float) -> float:
    r"""Signal half-life: periods until monument stock halves without reinvestment.

    If investment ceases (I_g = 0), the stock evolves as
    M_g(t) = M_g(0) * (1 - delta)^t. The half-life is:

    .. math::
        t_{1/2} = \frac{\ln 2}{\ln(1 / (1 - \delta))}
                = \frac{\ln 2}{-\ln(1 - \delta)}

    Parameters
    ----------
    delta : float
        Signal depreciation rate in (0, 1).

    Returns
    -------
    float
        Number of periods until signal strength halves. Always > 0.
    """
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return np.log(2.0) / (-np.log(1.0 - delta))


def maintenance_to_new_ratio(
    M_g: float | NDArray[np.float64],
    I_g: float,
    delta: float,
) -> float | NDArray[np.float64]:
    r"""Ratio of maintenance (signal-renewal) to new-construction effort.

    Implements the renovation-ratio R(M_g) of the main-text signal-depreciation
    section. Decomposing each period's investment flow under the geometric
    depreciation recurrence :math:`M_g(t+1) = (1 - \delta) M_g(t) + I_g`: the
    share that offsets depreciation is :math:`\delta M_g` (maintenance / signal
    renewal), and the remainder :math:`I_g - \delta M_g` is net new
    construction. Their ratio is

    .. math::
        R(M_g) = \frac{\delta M_g}{I_g - \delta M_g},

    which rises monotonically with accumulated stock :math:`M_g` and diverges
    as :math:`M_g \to M_g^* = I_g / \delta` (at steady state all investment is
    maintenance and net new construction vanishes). Defined on the growth
    regime :math:`0 \le \delta M_g < I_g`; at or beyond steady state the ratio
    is infinite and ``np.inf`` is returned.

    The rising direction is derived within the geometric depreciation form;
    positive depreciation (:math:`\delta > 0`) itself follows from the
    multi-period Spence condition. Mapping the model's maintenance flow to
    archaeologically observed renovation labor (and net growth to initial
    construction) is an interpretive assumption, not part of this derivation.

    Parameters
    ----------
    M_g : float or array
        Accumulated effective monument stock. Must be >= 0.
    I_g : float
        Per-period investment flow. Must be > 0.
    delta : float
        Signal depreciation rate in (0, 1].

    Returns
    -------
    float or array
        Maintenance-to-new-construction ratio R(M_g) >= 0; np.inf at or
        beyond steady state (delta * M_g >= I_g).
    """
    if delta <= 0 or delta > 1:
        raise ValueError(f"delta must be in (0, 1], got {delta}")
    if I_g <= 0:
        raise ValueError(f"I_g must be > 0, got {I_g}")
    M_g = np.asarray(M_g, dtype=np.float64)
    if np.any(M_g < 0):
        raise ValueError("M_g must be >= 0")
    maintenance = delta * M_g
    new_construction = I_g - maintenance
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(new_construction > 0, maintenance / new_construction, np.inf)
    return R


def maintenance_cost(
    M_g: float | NDArray[np.float64],
    q: float,
    delta: float,
) -> float | NDArray[np.float64]:
    r"""Per-period cost to HOLD a monument stock against depreciation.

    To keep observed stock at :math:`M_g` when it decays at rate
    :math:`\delta`, the current holder must reinvest the depreciated increment
    :math:`I = \delta M_g` each period, at the quadratic construction cost
    :math:`c(I, q) = I^2/(2q)`:

    .. math::
        c_{\mathrm{maint}}(M_g, q; \delta) = \frac{(\delta M_g)^2}{2 q}.

    This maintenance cost, not the (predecessor-paid, sunk) construction cost,
    is what keeps an inherited monument an honest signal of the *current*
    holder's quality: letting the stock stand is free but lets the signal
    decay, while holding it at :math:`M_g` costs :math:`(\delta M_g)^2/(2q)`,
    which is type-dependent (cheaper for high :math:`q`). See
    :func:`verify_multi_period_spence_condition`.
    """
    M_g = np.asarray(M_g, dtype=np.float64)
    out = (delta * M_g) ** 2 / (2.0 * q)
    return float(out) if out.ndim == 0 else out


def verify_multi_period_spence_condition() -> dict[str, Any]:
    r"""Multi-period (inheritance) Spence condition: :math:`\delta>0` is necessary.

    Single-period signaling uses the construction cost
    :math:`c(x,q)=x^2/(2q)`, whose cross-partial
    :math:`\partial^2 c/\partial x\,\partial q = -x/q^2 < 0` delivers
    single-crossing (:func:`verify_spence_condition`). For a *durable* signal
    assessed as accumulated stock :math:`M`, a successor inherits
    :math:`(1-\delta)M_1` from a predecessor at zero cost; the only ongoing
    cost is maintenance against depreciation,
    :math:`c_{\mathrm{maint}}(M,q;\delta)=(\delta M)^2/(2q)`
    (:func:`maintenance_cost`). Its cross-partial in the signal :math:`M` and
    the current holder's quality :math:`q` is

    .. math::
        \frac{\partial^2 c_{\mathrm{maint}}}{\partial M\,\partial q}
            = -\frac{\delta^2 M}{q^2},

    strictly negative iff :math:`\delta>0` and exactly zero at
    :math:`\delta=0`. Hence:

    - :math:`\delta=0`: maintenance is free, the cross-partial vanishes, and
      single-crossing fails. An inherited stock is a cost-free pool any
      successor can display, so observed stock cannot separate the successor's
      quality: the signal goes stale across generations.
    - :math:`\delta>0`: maintenance is costly and type-dependent, single-
      crossing is restored, and observed stock again separates the *current*
      holder's quality.

    This is the formal "Spence condition extended to a durable, inherited
    medium": positive informational depreciation is necessary for accumulated
    stock to remain an honest signal across generations (the multi-period
    Spence condition of the main-text signal-depreciation section). It is the
    derivation underlying the geometric depreciation form used
    by :func:`effective_monument_stock` and :func:`maintenance_to_new_ratio`.

    Returns
    -------
    dict
        ``cross_partial`` (:math:`-\delta^2 M/q^2`),
        ``cross_partial_at_delta_zero`` (:math:`0`),
        ``is_negative_for_positive_delta`` (bool),
        ``single_period_cross_partial`` (:math:`-x/q^2`, for comparison).
    """
    M, q, delta, x = sp.symbols("M q delta x", positive=True)
    c_maint = (delta * M) ** 2 / (2 * q)
    cross = sp.simplify(sp.diff(c_maint, M, q))
    cross_at_zero = sp.simplify(cross.subs(delta, 0))
    c_build = x**2 / (2 * q)
    single = sp.simplify(sp.diff(c_build, x, q))
    is_neg = bool(cross.subs({M: 1, q: 1, delta: sp.Rational(1, 10)}) < 0)
    return {
        "cross_partial": cross,
        "cross_partial_at_delta_zero": cross_at_zero,
        "is_negative_for_positive_delta": is_neg,
        "single_period_cross_partial": single,
    }


# =====================================================================
# Sensitivity analysis
# =====================================================================


def _local_expansion_initial_condition(
    q_min: float,
    epsilon: float,
    lam: float,
    cost_type: str,
    benefit_type: str = "linear",
    **params: float,
) -> float:
    """Compute initial condition for ODE integration near the singular boundary.

    Near q_min, x*(q) -> 0 and the ODE x' = f(q)/g(x) is singular.
    We use a local Taylor expansion to start the integration at
    q_min + epsilon with a non-zero initial condition.

    Parameters
    ----------
    q_min : float
        Minimum quality.
    epsilon : float
        Distance from q_min at which to start integration.
    lam : float
        Lambda parameter.
    cost_type : str
        Type of cost function.
    benefit_type : str
        Type of benefit function.
    **params :
        Additional parameters (e.g., a, b for power cost, gamma for concave benefit).

    Returns
    -------
    x0 : float
        Initial condition x*(q_min + epsilon).
    """
    if benefit_type == "concave":
        gamma = params.get("gamma", 0.5)
        # ODE: x' = lam*gamma*q^(gamma-1) / c_x(x, q)
        # For quadratic cost c_x = x/q: x' = lam*gamma*q^gamma/x
        # Near q_min: x ~ sqrt(2*lam*gamma*q_min^gamma * epsilon)
        return np.sqrt(2 * lam * gamma * q_min**gamma * epsilon)

    # Linear benefit: x' = lam / c_x(x, q)
    if cost_type == "quadratic":
        # x ~ sqrt(2*lam*q_min*epsilon)
        return np.sqrt(2 * lam * q_min * epsilon)
    elif cost_type == "power":
        a = params.get("a", 2.0)
        b = params.get("b", 1.0)
        # x^(a-1) dx = lam * q_min^b * dε => x = (a*lam*q_min^b*ε)^(1/a)
        return (a * lam * q_min**b * epsilon) ** (1.0 / a)
    elif cost_type == "exponential":
        # Exponential cost c = (exp(x/q)-1)/q has c_x = exp(x/q)/q^2.
        # Near x=0: exp(x/q) ~ 1, so x' = lam/c_x ~ lam*q_min^2.
        return lam * q_min**2 * epsilon
    else:
        raise ValueError(f"Unknown cost_type '{cost_type}'")


def cost_function_sensitivity(
    q_range: NDArray[np.float64],
    q_min: float,
    lam: float,
    cost_type: str = "quadratic",
    **params: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute equilibrium investment under different cost functions.

    For the quadratic cost, uses the closed-form solution. For non-quadratic
    costs satisfying the Spence condition, solves the ODE x'(q) = lam/c_x(x,q)
    numerically via solve_ivp. Uses linear benefit throughout.

    Parameters
    ----------
    q_range : array of float
        Quality values at which to evaluate x*(q). Must start at or above q_min.
    q_min : float
        Minimum quality, must be > 0.
    lam : float
        Lambda parameter, must be > 0.
    cost_type : str
        One of 'quadratic', 'power', 'exponential'.
    **params :
        Additional parameters:
        - power: a (float, default 2.0), b (float, default 1.0)

    Returns
    -------
    q_array : ndarray
        Quality values (may differ slightly from q_range for ODE solutions).
    x_star_array : ndarray
        Equilibrium investment at each quality value.
    """
    q_arr = np.asarray(q_range, dtype=np.float64)
    if lam <= 0:
        raise ValueError(f"lam must be positive for sensitivity analysis, got {lam}")

    if cost_type == "quadratic":
        # Closed-form solution
        q_valid = q_arr[q_arr >= q_min]
        x_star = np.sqrt(lam * (q_valid**2 - q_min**2))
        return q_valid, x_star

    # For non-quadratic costs, solve the ODE numerically
    if cost_type == "power":
        spec = power_cost(a=params.get("a", 2.0), b=params.get("b", 1.0))
    elif cost_type == "exponential":
        spec = exponential_cost()
    else:
        raise ValueError(f"Unknown cost_type '{cost_type}'")

    q_end = float(q_arr[-1])
    # Start slightly above q_min to avoid the singularity
    eps = 1e-8 * (q_end - q_min)
    q_start = q_min + eps
    x0 = _local_expansion_initial_condition(q_min, eps, lam, cost_type, **params)

    def ode_rhs(q: float, x: NDArray) -> NDArray:
        # x' = lam / c_x(x, q) for linear benefit
        cx = spec.c_x(x[0], q)
        if cx <= 0:
            return np.array([0.0])
        return np.array([lam / cx])

    sol = solve_ivp(
        ode_rhs,
        [q_start, q_end],
        [x0],
        t_eval=q_arr[q_arr >= q_start],
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return sol.t, sol.y[0]


def benefit_function_sensitivity(
    q_range: NDArray[np.float64],
    q_min: float,
    lam: float,
    benefit_type: str = "linear",
    **params: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute equilibrium investment under different benefit functions.

    Uses quadratic cost throughout, varying the benefit specification.
    For linear benefit, uses the closed-form solution. For concave benefit
    b = lam*q_hat^gamma, solves the ODE x'(q) = lam*gamma*q^(gamma-1)/c_x(x,q)
    numerically. See the supplementary "Functional form robustness".

    Parameters
    ----------
    q_range : array of float
        Quality values at which to evaluate x*(q).
    q_min : float
        Minimum quality, must be > 0.
    lam : float
        Lambda parameter, must be > 0.
    benefit_type : str
        One of 'linear', 'concave'.
    **params :
        Additional parameters:
        - concave: gamma (float, default 0.5), 0 < gamma < 1.

    Returns
    -------
    q_array : ndarray
        Quality values.
    x_star_array : ndarray
        Equilibrium investment at each quality value.
    """
    q_arr = np.asarray(q_range, dtype=np.float64)
    if lam <= 0:
        raise ValueError(f"lam must be positive for sensitivity analysis, got {lam}")

    if benefit_type == "linear":
        q_valid = q_arr[q_arr >= q_min]
        x_star = np.sqrt(lam * (q_valid**2 - q_min**2))
        return q_valid, x_star

    if benefit_type != "concave":
        raise ValueError(f"Unknown benefit_type '{benefit_type}'")

    gamma = params.get("gamma", 0.5)
    if not 0 < gamma < 1:
        raise ValueError(f"Concave benefit requires 0 < gamma < 1, got {gamma}")

    q_end = float(q_arr[-1])
    eps = 1e-8 * (q_end - q_min)
    q_start = q_min + eps
    x0 = _local_expansion_initial_condition(
        q_min, eps, lam, "quadratic", benefit_type="concave", gamma=gamma
    )

    def ode_rhs(q: float, x: NDArray) -> NDArray:
        # ODE: x' = lam*gamma*q^(gamma-1) / c_x(x, q)
        # With quadratic cost c_x = x/q: x' = lam*gamma*q^gamma / x
        if x[0] <= 0:
            return np.array([0.0])
        return np.array([lam * gamma * q**gamma / x[0]])

    sol = solve_ivp(
        ode_rhs,
        [q_start, q_end],
        [x0],
        t_eval=q_arr[q_arr >= q_start],
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return sol.t, sol.y[0]


# =====================================================================
# Channel selection analysis
# =====================================================================
# The functions below extend the single-channel equilibrium to a
# multi-channel setting where individuals choose among alternative
# signaling channels. The analysis decomposes lambda into
# audience-specific components and compares signal fidelity across
# channels to establish when monument construction dominates.
# See the main-text channel-selection subsection ("why monuments
# specifically") and the supplementary "Channel selection analysis".


def mean_payoff_multiplier(
    q_min: float = 0.1,
    q_max: float = 2.0,
) -> float:
    r"""Community-mean payoff multiplier :math:`\bar A = E[A(q)]` for channel choice.

    At the separating equilibrium with channel-specific effective reward
    :math:`e_s = \lambda_s \rho_s`, a type-:math:`q` member's equilibrium
    net payoff above baseline is

    .. math::
        U_s^*(q) - q = e_s\,A(q) - f_s, \qquad
        A(q) = \frac{q^2 + q_{\min}^2}{2q},

    (the per-unit cost shape :math:`\kappa_s` cancels; see
    :func:`channel_dominance_condition`). Under community-level channel
    choice with the mean-payoff aggregation rule, the criterion is
    :math:`\bar V_s = e_s\,\bar A - f_s` with

    .. math::
        \bar A = \mathbb{E}[A(q)]
            = \frac{\mathbb{E}[q]}{2}
              + \frac{q_{\min}^2}{2(q_{\max} - q_{\min})}
                \ln\frac{q_{\max}}{q_{\min}}

    under uniform quality. At the default range (0.1, 2.0),
    :math:`\bar A = 0.5329`. The former criterion V_s = e_s - f_s omitted
    this multiplier, effectively doubling the weight of the (uncalibrated)
    fixed costs relative to the derived payoff; fixed costs are now stated
    in payoff units. Because A(q) is increasing in q,
    individual rankings are type-dependent (lower-capacity members weight
    fixed costs more heavily); the community-level norm choice suppresses
    that heterogeneity, which is why the aggregation rule must be stated.

    Parameters
    ----------
    q_min, q_max : float
        Quality range (uniform distribution). Defaults mirror
        calibration.DEFAULT_Q_MIN / DEFAULT_Q_MAX.

    Returns
    -------
    float
        The multiplier :math:`\bar A > 0`.
    """
    if q_min <= 0 or q_max <= q_min:
        raise ValueError(f"Need 0 < q_min < q_max, got ({q_min}, {q_max})")
    Eq = (q_min + q_max) / 2.0
    return float(
        Eq / 2.0
        + q_min**2 / (2.0 * (q_max - q_min)) * np.log(q_max / q_min)
    )


@dataclass(frozen=True)
class SignalChannel:
    """Specification of a signaling channel and its properties.

    Each channel is characterized by its audience reach (what fraction of
    each audience's potential reward it captures) and its signal fidelity
    (how reliably the signal maps to productive/coordinative capacity).

    The audience activation values a_s^j in [0, 1] determine the effective
    lambda for the channel: lambda_s = a_W * lam_W + a_C * lam_C + a_X * lam_X.
    The signal fidelity rho_s in [0, 1] captures windfall substitution
    vulnerability and quality-dimension mismatch.

    See the supplementary "Channel selection analysis" for the audience
    decomposition and fidelity analysis.

    Attributes
    ----------
    name : str
        Human-readable identifier for the channel.
    audience_W : float
        Fraction of within-group reward (lam_W) this channel captures.
    audience_C : float
        Fraction of competitive between-group reward (lam_C) this channel captures.
    audience_X : float
        Fraction of cooperative between-group reward (lam_X) this channel captures.
    rho : float
        Signal fidelity: reliability of mapping to productive capacity.
    fixed_cost : float
        Scale-independent fixed (access) cost of using the channel as a
        signal, in PAYOFF units (the units of the community-mean equilibrium
        payoff). Enters channel selection as the net return
        V_s = lambda_s * rho * A_bar - fixed_cost, where A_bar is the
        community-mean payoff multiplier (mean_payoff_multiplier). Monuments
        are lumpy (high fixed cost: minimum recognizable scale +
        collective-labor overhead); cheaper channels scale down. Illustrative
        defaults were re-expressed in payoff units (old reward-unit
        values x A_bar ~ 0.533), preserving the dominance frontier. See
        channel_dominance_condition.
    quality_dimension : str
        What quality dimension this channel reveals.
    description : str
        Brief explanation of the channel's signaling properties.
    """

    name: str
    audience_W: float
    audience_C: float
    audience_X: float
    rho: float
    fixed_cost: float

    def __post_init__(self) -> None:
        # Domain validation: an
        # unvalidated channel (negative fixed cost, out-of-range fidelity or
        # audience weight) would silently win or lose the channel ranking.
        for label, v in (
            ("audience_W", self.audience_W),
            ("audience_C", self.audience_C),
            ("audience_X", self.audience_X),
            ("rho", self.rho),
        ):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"SignalChannel.{label} must be in [0, 1], got {v}")
        if self.fixed_cost < 0.0:
            raise ValueError(
                f"SignalChannel.fixed_cost must be >= 0, got {self.fixed_cost}"
            )
    quality_dimension: str
    description: str


# Module-level channel constants with audience activation values matching the
# supplementary "Channel selection analysis". Fidelity and fixed-cost defaults
# mirror calibration.py (DEFAULT_RHO_*, DEFAULT_FIXED_COST_*). These are
# structural properties of the channels, not free parameters. The fixed cost
# f_s is the scale-independent access cost, in payoff units, and enters
# channel selection via the net return V_s = lambda_s*rho*A_bar - f_s
# (see channel_dominance_condition and mean_payoff_multiplier).
# f_monument is an ILLUSTRATIVE default (swept in the manuscript), not a
# calibrated value; the payoff-unit defaults (0.16, 0.011) equal the former
# reward-unit values (0.30, 0.02) times A_bar ~ 0.533, preserving the
# dominance frontier under the payoff-unit criterion.

MONUMENT_CHANNEL = SignalChannel(
    name="monument",
    audience_W=1.0,
    audience_C=1.0,
    audience_X=1.0,
    rho=0.95,
    fixed_cost=0.16,
    quality_dimension="productive/coordinative capacity",
    description=(
        "Durable, visible across territorial boundaries, collective "
        "production witnessed by all audiences. No windfall substitution."
    ),
)

FEAST_CHANNEL = SignalChannel(
    name="feast",
    audience_W=0.9,
    audience_C=0.2,
    audience_X=0.2,
    rho=0.5,
    fixed_cost=0.011,
    quality_dimension="resource accumulation (noisy)",
    description=(
        "High within-group visibility but ephemeral. Rival groups and "
        "distant partners rarely witness the event. Vulnerable to "
        "windfall substitution."
    ),
)

RITUAL_CHANNEL = SignalChannel(
    name="ritual",
    audience_W=0.7,
    audience_C=0.1,
    audience_X=0.1,
    rho=0.3,
    fixed_cost=0.011,
    quality_dimension="commitment/pain tolerance",
    description=(
        "Signals cooperative commitment to in-group, not productive "
        "capacity. Not visible between groups. Quality dimension "
        "mismatch with what audiences value."
    ),
)

HUNTING_CHANNEL = SignalChannel(
    name="hunting",
    audience_W=0.6,
    audience_C=0.05,
    audience_X=0.15,
    rho=0.4,
    fixed_cost=0.011,
    quality_dimension="individual physical prowess",
    description=(
        "Signals individual skill, not collective capacity. Limited "
        "between-group observability. Some exchange partner relevance "
        "through demonstrated resource acquisition ability."
    ),
)

VERBAL_CHANNEL = SignalChannel(
    name="verbal",
    audience_W=0.3,
    audience_C=0.0,
    audience_X=0.0,
    rho=0.05,
    fixed_cost=0.0,
    quality_dimension="none (cheap talk)",
    description=(
        "Minimal information content. No between-group reach. Near-zero "
        "fidelity because claims are costless to produce."
    ),
)

# Convenience list of all standard channels for iteration
ALL_CHANNELS: list[SignalChannel] = [
    MONUMENT_CHANNEL,
    FEAST_CHANNEL,
    RITUAL_CHANNEL,
    HUNTING_CHANNEL,
    VERBAL_CHANNEL,
]


def lambda_decomposition(
    lam_W: float,
    lam_C: float,
    lam_X: float,
) -> dict[str, float]:
    """Compute effective lambda from audience-specific components.

    The total lambda is the sum of within-group, competitive between-group,
    and cooperative between-group reward components:

        lambda = lam_W + lam_C + lam_X

    This is the audience decomposition of the main-text channel-selection
    subsection.

    Parameters
    ----------
    lam_W : float
        Within-group social reward component (>= 0).
    lam_C : float
        Competitive between-group component (>= 0).
    lam_X : float
        Cooperative between-group component (>= 0).

    Returns
    -------
    dict with keys:
        'total': float, the total lambda
        'lam_W': float, the within-group component
        'lam_C': float, the competitive component
        'lam_X': float, the cooperative component
    """
    if lam_W < 0 or lam_C < 0 or lam_X < 0:
        raise ValueError(
            f"All lambda components must be non-negative, got "
            f"lam_W={lam_W}, lam_C={lam_C}, lam_X={lam_X}"
        )
    return {
        "total": lam_W + lam_C + lam_X,
        "lam_W": lam_W,
        "lam_C": lam_C,
        "lam_X": lam_X,
    }


def channel_effective_lambda(
    channel: SignalChannel,
    lam_W: float,
    lam_C: float,
    lam_X: float,
) -> float:
    """Compute effective lambda for a specific signal channel.

    The effective lambda is the audience-weighted sum:

        lambda_s = a_s^W * lam_W + a_s^C * lam_C + a_s^X * lam_X

    where a_s^j are the channel's audience activation fractions.
    See the supplementary "Channel selection analysis".

    Parameters
    ----------
    channel : SignalChannel
        The signal channel specification.
    lam_W : float
        Within-group social reward component (>= 0).
    lam_C : float
        Competitive between-group component (>= 0).
    lam_X : float
        Cooperative between-group component (>= 0).

    Returns
    -------
    lambda_s : float
        Effective lambda for this channel.
    """
    if lam_W < 0 or lam_C < 0 or lam_X < 0:
        raise ValueError(
            f"All lambda components must be non-negative, got "
            f"lam_W={lam_W}, lam_C={lam_C}, lam_X={lam_X}"
        )
    return (
        channel.audience_W * lam_W
        + channel.audience_C * lam_C
        + channel.audience_X * lam_X
    )


def signal_fidelity(
    channel: SignalChannel,
    windfall_prob: float = 0.0,
    quality_correlation: float | None = None,
) -> float:
    """Compute adjusted signal fidelity for a given channel.

    Base fidelity comes from the channel's rho attribute. Two adjustment
    mechanisms can modify it:

    1. Windfall substitution (primarily affects feasting): if windfall_prob > 0,
       the effective fidelity is rho * (1 - windfall_prob). This models the
       probability that a high signal is funded by transient windfall rather
       than sustained productive capacity.

    2. Quality correlation override: if quality_correlation is provided, it
       replaces the channel's rho entirely. This models the correlation
       between the channel's revealed quality dimension and the fitness-relevant
       dimension (productive capacity).

    See the supplementary "Channel selection analysis".

    Parameters
    ----------
    channel : SignalChannel
        The signal channel specification.
    windfall_prob : float
        Probability of windfall substitution, in [0, 1]. Default 0.
    quality_correlation : float or None
        If provided, overrides rho with this correlation value.

    Returns
    -------
    rho_adjusted : float
        Adjusted signal fidelity in [0, 1].
    """
    if not 0 <= windfall_prob <= 1:
        raise ValueError(f"windfall_prob must be in [0, 1], got {windfall_prob}")
    if quality_correlation is not None:
        if not 0 <= quality_correlation <= 1:
            raise ValueError(
                f"quality_correlation must be in [0, 1], got {quality_correlation}"
            )
        return quality_correlation * (1 - windfall_prob)

    rho = channel.rho * (1 - windfall_prob)
    # Clamp to [0, 1] for numerical safety
    return max(0.0, min(1.0, rho))


def channel_dominance_condition(
    channels: list[SignalChannel],
    lam_W: float,
    lam_C: float,
    lam_X: float,
    windfall_prob: float = 0.0,
    q_min: float = 0.1,
    q_max: float = 2.0,
) -> list[tuple[str, float, float, float]]:
    """Rank signal channels by the community-mean net return
    V_s = lambda_s * rho_s * A_bar - f_s.

    Computes the effective lambda and signal fidelity for each channel, then
    ranks by the community-mean equilibrium net payoff
    V_s = lambda_s * rho_s * A_bar - f_s, where A_bar = E[(q^2+q_min^2)/(2q)]
    is the mean payoff multiplier (mean_payoff_multiplier; 0.5329 at the
    default quality range) and f_s is the channel's fixed (access) cost in
    payoff units, PER CONTRIBUTOR.

    SCOPE (emergence-stage incidence assumption): this comparison is
    scoped to the channel-adoption
    (emergence) stage, at which the between-group audience returns
    (tilde-lambda_C, tilde-lambda_X) are assumed to accrue to the displaying
    contributor (partner access to the household; competitor reading of its
    capacity), so the channel-specific reward lambda_s legitimately prices
    the within-channel schedule. In the ESTABLISHED practice the canonical
    model prices the schedule with lambda_W alone and realizes between-group
    returns at group level through S and K; an architecture-consistent
    established-practice channel comparison (channel-wise group mean fitness
    with audience-read stocks) requires additional modeling choices and is
    deferred to future work. Channel dominance under it need not coincide
    with the emergence-stage result. The decision maker is the community's contribution norm
    (main-text channel-selection subsection), and the aggregation rule is
    the mean of members' equilibrium payoffs; because the multiplier A(q)
    increases with quality, individual rankings are type-dependent
    (lower-capacity members weight fixed costs more heavily), and the
    community norm suppresses that heterogeneity. The dominant channel
    (highest V_s) is the one the community's norm routes signaling through.

    (The simpler criterion V_s = lambda_s * rho_s - f_s omits A_bar and
    is not the equilibrium payoff derived in the SI channel section; with
    fixed costs expressed in payoff units, the two criteria give the same
    dominance frontier.)

    Why a fixed cost, not a per-unit cost: in the Spence separating
    equilibrium with marginal cost kappa_s * x^2/(2q), the equilibrium net
    payoff is lambda_s*rho_s*(q^2+q_min^2)/(2q), INDEPENDENT of kappa_s (a
    signaler facing a higher marginal cost invests proportionally less,
    exactly offsetting). The per-unit cost therefore cannot drive channel
    choice; only the scale-independent fixed cost f_s can. Monuments carry a
    substantial f_s (minimum recognizable scale + collective-labor overhead),
    making monument dominance conditional: monuments win when the
    multi-audience return overcomes their fixed cost, and a cheaper channel
    (typically feasting) wins the within-group-only / low-stakes regime.

    This is the community-mean channel net return V_s = lambda_s*rho_s*A_bar
    - f_s of the main-text channel-selection subsection (derivation:
    supplementary "Channel selection analysis").

    Parameters
    ----------
    channels : list of SignalChannel
        Signal channels to compare.
    lam_W : float
        Within-group social reward component (>= 0).
    lam_C : float
        Competitive between-group component (>= 0).
    lam_X : float
        Cooperative between-group component (>= 0).
    windfall_prob : float
        Windfall probability applied to feasting fidelity. Default 0.

    Returns
    -------
    ranking : list of (name, lambda_s, rho_s, V_s) tuples
        Sorted descending by net return V_s = lambda_s * rho_s * A_bar - f_s.
        The 4th element is the NET community-mean return (not the gross
        lambda_s * rho_s).
    """
    A_bar = mean_payoff_multiplier(q_min, q_max)
    results = []
    for ch in channels:
        lam_s = channel_effective_lambda(ch, lam_W, lam_C, lam_X)
        # Apply windfall adjustment only to feast channel
        wp = windfall_prob if ch.name == "feast" else 0.0
        rho_s = signal_fidelity(ch, windfall_prob=wp)
        # Net return: community-mean equilibrium payoff minus the channel's
        # fixed access cost (payoff units). The per-unit (marginal) cost
        # cancels at the Spence separating equilibrium, so beyond the
        # audience-weighted gross return only the fixed cost distinguishes
        # channels (see docstring).
        V_s = lam_s * rho_s * A_bar - ch.fixed_cost
        results.append((ch.name, lam_s, rho_s, V_s))
    # Sort descending by net return V_s
    results.sort(key=lambda t: t[3], reverse=True)
    return results


def monument_dominance_threshold(
    lam_W: float,
    lam_C_range: NDArray[np.float64],
    lam_X_range: NDArray[np.float64],
    channels: list[SignalChannel] | None = None,
    windfall_prob: float = 0.0,
) -> NDArray[np.float64]:
    """Compute whether monuments dominate in (lam_C, lam_X) parameter space.

    For each point in the grid defined by lam_C_range x lam_X_range,
    determines whether monument construction has the highest community-mean
    net return V_s = lambda_s * rho_s * A_bar - f_s among all channels
    (see channel_dominance_condition).

    See the supplementary "Channel-selection sensitivity: monument fixed cost
    and audience weights".

    Parameters
    ----------
    lam_W : float
        Within-group social reward component (fixed).
    lam_C_range : array of float
        Values of competitive between-group lambda to evaluate.
    lam_X_range : array of float
        Values of cooperative between-group lambda to evaluate.
    channels : list of SignalChannel or None
        Channels to compare. Defaults to ALL_CHANNELS.
    windfall_prob : float
        Windfall probability for feasting. Default 0.

    Returns
    -------
    dominance_grid : 2D array of bool
        Shape (len(lam_X_range), len(lam_C_range)). True where monuments
        have the highest net return. Rows correspond to lam_X values,
        columns to lam_C values, following meshgrid 'ij' convention.
    """
    if channels is None:
        channels = ALL_CHANNELS
    lam_C_arr = np.asarray(lam_C_range, dtype=np.float64)
    lam_X_arr = np.asarray(lam_X_range, dtype=np.float64)
    grid = np.zeros((len(lam_X_arr), len(lam_C_arr)), dtype=bool)

    for i, lx in enumerate(lam_X_arr):
        for j, lc in enumerate(lam_C_arr):
            ranking = channel_dominance_condition(
                channels, lam_W, lc, lx, windfall_prob=windfall_prob
            )
            # Monument dominates if it is first in the ranking
            grid[i, j] = ranking[0][0] == "monument"
    return grid


def incremental_lambda(
    channel_ranking: list[tuple[str, float, float, float]],
) -> float:
    """Compute incremental lambda for the dominant channel over the best alternative.

    The incremental lambda is V_dominant - V_second_best (community-mean net
    returns, after subtracting fixed costs). It is a DESCRIPTIVE diagnostic
    of channel competition: the margin by which the dominant channel's norm
    out-earns the runner-up, i.e., the payoff buffer sustaining the channel
    norm against drift to the best alternative. It is NOT a schedule input:
    a difference of net community-mean payoffs is not a marginal
    inferred-quality reward and must not be substituted for lambda in the
    Spence first-order condition.

    When the dominant channel is not monuments, returns the incremental
    lambda for whatever channel dominates (which may be useful for
    comparison purposes).

    Parameters
    ----------
    channel_ranking : list of (name, lambda_s, rho_s, V_s) tuples
        Output of channel_dominance_condition, sorted descending by net V_s.

    Returns
    -------
    lambda_incr : float
        Net return of the dominant channel minus the best alternative.
        Always >= 0 by construction (the dominant channel has the highest V_s).
    """
    if len(channel_ranking) < 2:
        raise ValueError("Need at least two channels to compute incremental lambda")
    R_best = channel_ranking[0][3]
    R_second = channel_ranking[1][3]
    return R_best - R_second
