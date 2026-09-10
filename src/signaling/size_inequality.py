"""Size-inequality prediction (M1): monument-size inequality as the
pushforward of the capacity distribution.

The separating equilibrium makes individual monument investment x*(q) a
strictly increasing image of capacity q. The distribution of contribution
sizes is therefore the pushforward of the capacity distribution through
x*. Because monument size and wealth (~q) are *different* monotone images
of q, the size-Gini is a predictable function of the capacity-Gini, not
equal to it. Since x*(q_min) = 0, contributors near the threshold build
almost nothing, so the pushforward amplifies low-end inequality: the
predicted size-Gini exceeds the capacity-Gini for distributions reaching
toward q_min. This discriminates against functional-threshold accounts
(sizes clustered at a requirement) and stochastic accretion (size
distribution unrelated to wealth).

Isolated module: reuses x*(q) (equilibrium_investment) read-only; its
output is a standalone result and is never wired into sigma*. The
manuscript reports this prediction in the supplementary section "Further
derived predictions: placement, competitive investment, and size
inequality".
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from signaling.layer1 import equilibrium_investment


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of a non-negative distribution (0 = equality).

    Uses the sorted-rank formula G = 2 * sum(i x_(i)) / (n sum x) - (n+1)/n,
    with x sorted ascending and i = 1..n. Returns 0 for an empty, single,
    or all-zero input.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0.0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(2.0 * np.sum(index * x) / (n * total) - (n + 1) / n)


def monument_sizes(
    capacities: Sequence[float],
    q_min: float,
    lam: float,
) -> NDArray[np.float64]:
    """Pushforward of the capacities through x*(q): the predicted monument
    sizes. Capacities must be >= q_min (only above-threshold individuals
    build)."""
    return np.array(
        [equilibrium_investment(q, q_min, lam) for q in capacities], dtype=float
    )


def predicted_size_gini(
    capacities: Sequence[float],
    q_min: float,
    lam: float,
) -> float:
    """Predicted monument-size Gini = Gini of the x*(q) pushforward of the
    capacity distribution."""
    return gini(monument_sizes(capacities, q_min, lam))
