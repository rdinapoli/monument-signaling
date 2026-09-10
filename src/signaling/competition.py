"""Competitive monument investment: the parity-hump prediction (M11 + M2).

Monument labor is modeled as effort in a dyadic Tullock contest-success
function. Two neighboring groups with capacities q_i, q_j compete; the prize
is a common deterrence/standing value V; capacity enters as cost c = 1/q
(Spence-consistent: higher capacity lowers the cost of a given labor
diversion). Win/deterrence probability p_i = x_i/(x_i+x_j); payoff
p_i V - c_i x_i. The unique interior Nash equilibrium is

    x_i* = V q_i^2 q_j / (q_i + q_j)^2          (group i's labor toward j)
    X*   = x_i* + x_j* = V q_i q_j/(q_i+q_j)
         = (V qbar/2)(1 - delta_q^2)              (qbar = mean, delta_q = asymmetry)

Both groups are always active (the disadvantaged group's equilibrium payoff
is V p_j^2 > 0), so the total-investment hump is the exact smooth quadratic
in delta_q = (q_i-q_j)/(q_i+q_j): at fixed mean capacity qbar, X* peaks at
parity (delta_q = 0) and falls as delta_q^2 (prediction M11). Directed effort
x_i*(q_j) peaks at the matched rival q_j = q_i (prediction M2). Aggregated
over a region's neighbor pairs, the regional result is SCOPED to fixed,
approximately degree-regular interaction structures: there, at fixed total
capacity, investment is highest when capacities are evenly distributed (the
graph objective sum q_i q_j/(q_i+q_j) is concave with an equal-split
critical point on a regular graph). On degree-heterogeneous adjacency the
ordering can reverse, because capacity at a high-degree community enters
more contests: on a three-community star with total capacity 12 the sum is
maximized at a moderately unequal allocation (hub 12(sqrt(2)-1) ~ 4.97,
total ~ 4.118 > 4.0 equal; pinned in test_competition.py). The pairwise
parity form is the clean prediction; either form is sign-opposite to
monotone power/wealth-display (SI section "Further derived predictions",
regional-corollary paragraph).

Symbol note: delta_q denotes capacity asymmetry (q_i-q_j)/(q_i+q_j), local
to this module. It is distinct from the global signal-depreciation rate
delta (the per-period rate at which an unmaintained monument's signaling
value decays), which is used in layer1.py and price_equation.py.

Decoupled, isolated module: consumes capacities and the prize V; its output
is a standalone result and is NEVER wired into lambda_C, lambda, x*(q), or
sigma*. The manuscript reports this prediction in the supplementary section
"Further derived predictions: placement, competitive investment, and size
inequality". Contest-theory sources: Tullock (1980); Parker (1974);
Enquist & Leimar (1983); Hammerstein & Parker (1982).
"""
from __future__ import annotations

from signaling.calibration import COMPETITION


def dyadic_effort(q_i: float, q_j: float, V: float = COMPETITION["V"]) -> float:
    """Group i's equilibrium monument labor directed at neighbor j:
    x_i* = V q_i^2 q_j / (q_i + q_j)^2."""
    return V * q_i**2 * q_j / (q_i + q_j) ** 2


def dyadic_total(q_i: float, q_j: float, V: float = COMPETITION["V"]) -> float:
    """Total dyadic monument labor X* = V q_i q_j/(q_i+q_j)."""
    return V * q_i * q_j / (q_i + q_j)


def dyadic_asymmetry(q_i: float, q_j: float) -> float:
    """Capacity asymmetry delta_q = (q_i - q_j)/(q_i + q_j) in [-1, 1].

    Named delta_q locally to avoid clash with the global signal-depreciation
    rate delta (the per-period decay rate of an unmaintained monument's
    signaling value).
    """
    return (q_i - q_j) / (q_i + q_j)


def verify_dyadic_equilibrium() -> tuple:
    """Symbolically verify the closed-form efforts satisfy both FOCs.

    Substitutes x_i*, x_j* into the first-order conditions
    V x_j/X^2 - c_i and V x_i/X^2 - c_j (with c = 1/q) and simplifies.
    Returns the two residuals, which are both 0 for the true equilibrium.
    Mirrors the symbolic-verification pattern of verify_spence_condition
    in layer1.py.
    """
    import sympy as sp

    q_i, q_j, V = sp.symbols("q_i q_j V", positive=True)
    c_i, c_j = 1 / q_i, 1 / q_j
    x_i = V * q_i**2 * q_j / (q_i + q_j) ** 2
    x_j = V * q_i * q_j**2 / (q_i + q_j) ** 2
    X = x_i + x_j
    res_i = sp.simplify(V * x_j / X**2 - c_i)
    res_j = sp.simplify(V * x_i / X**2 - c_j)
    return res_i, res_j


def regional_investment(
    capacities,
    pairs,
    V: float = COMPETITION["V"],
) -> float:
    """Total regional monument labor = sum of dyadic totals over neighbor
    pairs (the peer-polity aggregation).

    Parameters
    ----------
    capacities : sequence of float
        Group capacities, indexed 0..n-1.
    pairs : list of (int, int)
        Neighbor relations; each (i, j) is a competing dyad.
    V : float
        Common contest prize.

    Returns
    -------
    float
        Sum over pairs of dyadic_total(q_i, q_j, V). At fixed total
        capacity this is highest at the equal allocation on complete or
        degree-regular ``pairs`` structures; on degree-heterogeneous
        adjacency the ordering can reverse (star counterexample in the
        module docstring and test_competition.py), so the parity claim is
        scoped to approximately degree-regular interaction structures.
    """
    return sum(dyadic_total(capacities[i], capacities[j], V) for i, j in pairs)
