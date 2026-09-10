"""Monument placement: the interior-optimum siting prediction.

A group that has decided to build chooses a location, reduced to a scalar
exposure coordinate z in [0, z_max] (z=0 the community/resource core,
z=z_max the most exposed/peripheral terrain). The placement payoff is

    Pi(z) = R * [ lam_W eta_W(z) + lam_C eta_C(z) + lam_X eta_X(z) ] - G(z)

with within-group visibility eta_W non-increasing (A3), between-group
visibility eta_C = eta_X saturating-increasing (A1), and resource cost G
convex (A2). Under A1-A3 Pi is strictly concave, so z* is interior and
unique, and two sign-robust comparative statics hold:
dz*/d lam_C > 0 and dz*/d gamma < 0.

Isolated downstream module: consumes (lam_W, lam_C, lam_X) and the
per-unit return R as fixed inputs; feeds back into nothing. The audience
decomposition lambda = lam_W + lam_C + lam_X is the one used throughout the
framework (Layer 1 and the channel-selection analysis); this placement
prediction is reported in the supplementary section "Further derived
predictions: placement, competitive investment, and size inequality".
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar


def visibility_between(z: ArrayLike, z_half: float) -> np.ndarray:
    """Between-group visibility eta_C(z) = eta_X(z) = z / (z_half + z).

    Michaelis-Menten saturation (A1): eta' > 0, eta'' <= 0. Reuses the
    saturating form of k(M_g) in layer3.py for cross-framework consistency.
    """
    z = np.asarray(z, dtype=float)
    return z / (z_half + z)


def visibility_within(z: ArrayLike, b: float) -> np.ndarray:
    """Within-group visibility eta_W(z) = exp(-b z), non-increasing (A3)."""
    z = np.asarray(z, dtype=float)
    return np.exp(-b * z)


def resource_cost(z: ArrayLike, gamma: float) -> np.ndarray:
    """Resource/logistics cost G(z; gamma) = gamma * z**2, convex (A2)."""
    z = np.asarray(z, dtype=float)
    return gamma * z**2


def placement_payoff(
    z: ArrayLike,
    lam_W: float,
    lam_C: float,
    lam_X: float,
    R: float,
    z_half: float,
    b: float,
    gamma: float,
) -> np.ndarray:
    """Placement payoff Pi(z). See module docstring for the specification."""
    eta_b = visibility_between(z, z_half)
    eta_w = visibility_within(z, b)
    benefit = R * (lam_W * eta_w + lam_C * eta_b + lam_X * eta_b)
    return benefit - resource_cost(z, gamma)


def optimal_placement(
    lam_W: float,
    lam_C: float,
    lam_X: float,
    R: float,
    z_half: float,
    b: float,
    gamma: float,
    z_max: float,
) -> float:
    """Return z* = argmax_{z in [0, z_max]} Pi(z).

    Uses bounded scalar minimization on -Pi (scipy.optimize, per the
    modeling rules: no hand-rolled solvers). Pi is strictly concave under
    A1-A3, so the bounded optimum is the unique global maximizer.
    """
    res = minimize_scalar(
        lambda z: -float(
            placement_payoff(z, lam_W, lam_C, lam_X, R, z_half, b, gamma)
        ),
        bounds=(0.0, z_max),
        method="bounded",
    )
    return float(res.x)


def comparative_static(
    param: str,
    *,
    lam_W: float,
    lam_C: float,
    lam_X: float,
    R: float,
    z_half: float,
    b: float,
    gamma: float,
    z_max: float,
    eps: float = 1e-4,
) -> float:
    """Central-difference d z* / d param for param in {lam_C, lam_X, gamma}.

    Returns the numerical derivative of the optimal exposure with respect to
    the named parameter. Signs (theory): dz*/d lam_C > 0, dz*/d lam_X > 0,
    dz*/d gamma < 0.
    """
    if param not in {"lam_C", "lam_X", "gamma"}:
        raise ValueError(f"unsupported comparative-static parameter: {param}")
    base = dict(
        lam_W=lam_W, lam_C=lam_C, lam_X=lam_X, R=R,
        z_half=z_half, b=b, gamma=gamma, z_max=z_max,
    )
    up = dict(base)
    down = dict(base)
    up[param] = base[param] + eps
    down[param] = base[param] - eps
    return (optimal_placement(**up) - optimal_placement(**down)) / (2 * eps)
