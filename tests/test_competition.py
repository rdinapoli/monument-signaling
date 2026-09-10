"""Tests for the competitive-investment prediction (M11 parity hump + M2
directed scale) from a dyadic Tullock contest. Encodes the theoretical
properties from the SI "Competitive investment: the parity hump and the
directed-scale peak": the closed-form equilibrium (verified
symbolically against the FOC), the parity hump in total investment, the
parity-peaked directed effort, the no-dropout property, the regional
inequality result, and isolation from the sigma* pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from signaling.competition import (
    dyadic_effort,
    dyadic_total,
    dyadic_asymmetry,
    verify_dyadic_equilibrium,
)
from signaling.calibration import COMPETITION


class TestDyadicEquilibrium:
    """Closed-form equilibrium and its symbolic verification."""

    def test_symbolic_foc_residuals_zero(self) -> None:
        """The closed-form efforts satisfy both first-order conditions
        exactly: substituting x_i*, x_j* into V x_j/X^2 - c_i and
        V x_i/X^2 - c_j simplifies to 0."""
        res_i, res_j = verify_dyadic_equilibrium()
        assert res_i == 0
        assert res_j == 0

    def test_numeric_foc_satisfied(self) -> None:
        """Numerically, V x_j/X^2 = c_i = 1/q_i at the equilibrium."""
        q_i, q_j, V = 5.0, 3.0, 1.0
        x_i = dyadic_effort(q_i, q_j, V)
        x_j = dyadic_effort(q_j, q_i, V)
        X = x_i + x_j
        assert V * x_j / X**2 == pytest.approx(1.0 / q_i)
        assert V * x_i / X**2 == pytest.approx(1.0 / q_j)

    def test_total_equals_asymmetry_form(self) -> None:
        """X* = V q_i q_j/(q_i+q_j) = (V qbar/2)(1 - delta_q^2)."""
        q_i, q_j, V = 7.0, 2.0, 1.3
        qbar = (q_i + q_j) / 2.0
        delta_q = dyadic_asymmetry(q_i, q_j)
        assert dyadic_total(q_i, q_j, V) == pytest.approx((V * qbar / 2.0) * (1 - delta_q**2))

    def test_effort_closed_form(self) -> None:
        """x_i* = V q_i^2 q_j/(q_i+q_j)^2 at a sample point."""
        q_i, q_j, V = 5.0, 3.0, 1.0
        assert dyadic_effort(q_i, q_j, V) == pytest.approx(V * q_i**2 * q_j / (q_i + q_j) ** 2)

    def test_total_is_sum_of_directed_efforts(self) -> None:
        q_i, q_j, V = 4.0, 6.0, 0.9
        assert dyadic_total(q_i, q_j, V) == pytest.approx(
            dyadic_effort(q_i, q_j, V) + dyadic_effort(q_j, q_i, V)
        )


class TestHumpAndDirectedPeak:
    """M11 (aggregate hump at parity) and M2 (directed peak at matched rival)."""

    def test_total_peaks_at_parity_fixed_mean(self) -> None:
        """At fixed mean capacity, X* is maximized at delta_q = 0 (parity)."""
        qbar, V = 5.0, 1.0
        deltas = np.linspace(-0.95, 0.95, 391)
        totals = np.array([
            dyadic_total(qbar * (1 + d), qbar * (1 - d), V) for d in deltas
        ])
        peak = deltas[int(np.argmax(totals))]
        assert peak == pytest.approx(0.0, abs=1e-2)
        # falls as delta_q^2: ends are strictly below the center
        assert totals[0] < totals[len(totals) // 2]
        assert totals[-1] < totals[len(totals) // 2]

    def test_directed_effort_peaks_at_matched_rival(self) -> None:
        """Holding q_i fixed, x_i*(q_j) is maximized at q_j = q_i (M2)."""
        q_i, V = 5.0, 1.0
        q_js = np.linspace(0.2, 20.0, 500)
        efforts = np.array([dyadic_effort(q_i, qj, V) for qj in q_js])
        peak_qj = q_js[int(np.argmax(efforts))]
        assert peak_qj == pytest.approx(q_i, abs=0.1)

    def test_directed_effort_not_monotone_in_rival_strength(self) -> None:
        """Rules out monotone power/wealth-display: x_i* rises then falls in
        q_j, so it is NOT monotone increasing."""
        q_i, V = 5.0, 1.0
        x_at_matched = dyadic_effort(q_i, q_i, V)
        x_at_much_stronger = dyadic_effort(q_i, 50.0, V)
        assert x_at_much_stronger < x_at_matched

    def test_disadvantaged_group_stays_active(self) -> None:
        """No dropout: the weak group's equilibrium payoff equals V p_j^2 > 0."""
        q_i, q_j, V = 10.0, 1.0, 1.0
        x_i = dyadic_effort(q_i, q_j, V)
        x_j = dyadic_effort(q_j, q_i, V)
        X = x_i + x_j
        p_j = x_j / X
        payoff_j = V * p_j - (1.0 / q_j) * x_j
        assert payoff_j == pytest.approx(V * p_j**2)
        assert payoff_j > 0.0


class TestEdgeCases:
    def test_equal_capacities(self) -> None:
        """At parity, X* = V q/2 and each group invests V q/4."""
        q, V = 6.0, 1.0
        assert dyadic_total(q, q, V) == pytest.approx(V * q / 2.0)
        assert dyadic_effort(q, q, V) == pytest.approx(V * q / 4.0)

    def test_total_to_zero_as_rival_vanishes(self) -> None:
        """As q_j -> 0, total investment -> 0."""
        assert dyadic_total(10.0, 1e-6, 1.0) == pytest.approx(0.0, abs=1e-5)

    def test_asymmetry_bounds(self) -> None:
        assert dyadic_asymmetry(5.0, 5.0) == pytest.approx(0.0)
        assert dyadic_asymmetry(9.0, 1.0) == pytest.approx(0.8)


from signaling.competition import regional_investment


class TestRegional:
    """Regional aggregate over neighbor pairs. Investment falls with
    capacity inequality at fixed total capacity on symmetric (complete or
    degree-regular) interaction structures; on degree-heterogeneous
    adjacency the ordering can reverse (see the star test), which is why
    the SI scopes the regional corollary to approximately degree-regular
    structures."""

    def test_regional_sums_dyads(self) -> None:
        caps = [2.0, 2.0, 2.0]
        pairs = [(0, 1), (0, 2), (1, 2)]
        expected = sum(dyadic_total(caps[i], caps[j], 1.0) for i, j in pairs)
        assert regional_investment(caps, pairs, 1.0) == pytest.approx(expected)

    def test_even_region_invests_more_than_uneven(self) -> None:
        """Same total capacity (6), complete triangle: equal distribution
        invests strictly more than an unequal one."""
        pairs = [(0, 1), (0, 2), (1, 2)]
        even = regional_investment([2.0, 2.0, 2.0], pairs, 1.0)
        uneven = regional_investment([4.0, 1.0, 1.0], pairs, 1.0)
        assert even > uneven

    def test_star_adjacency_reverses_the_inequality_ordering(self) -> None:
        """Scope boundary of the regional corollary: on a three-community
        STAR (hub of degree 2, leaves of
        degree 1) with total capacity 12, the equal allocation gives 4.0
        while the optimal allocation is moderately unequal, hub at
        12(sqrt(2)-1) ~ 4.97 giving ~ 4.118: capacity at a high-degree
        community enters more contests, so equality is optimal only on
        degree-regular structures."""
        pairs = [(0, 1), (0, 2)]  # community 0 is the hub
        equal = regional_investment([4.0, 4.0, 4.0], pairs, 1.0)
        hub = 12.0 * (np.sqrt(2.0) - 1.0)
        leaf = (12.0 - hub) / 2.0
        unequal = regional_investment([hub, leaf, leaf], pairs, 1.0)
        assert equal == pytest.approx(4.0, abs=1e-12)
        assert unequal == pytest.approx(4.11775, abs=1e-4)
        assert unequal > equal

    def test_regional_empty_pairs_zero(self) -> None:
        assert regional_investment([2.0, 3.0], [], 1.0) == pytest.approx(0.0)


class TestIsolationAndPins:
    def test_exported_at_package_level(self) -> None:
        import signaling
        for name in (
            "dyadic_effort", "dyadic_total", "dyadic_asymmetry",
            "verify_dyadic_equilibrium", "regional_investment",
        ):
            assert hasattr(signaling, name), name

    def test_headline_sigma_star_unchanged(self) -> None:
        """Regression guard: importing competition has no side effect on the
        headline self-consistent threshold (~0.4771, lambda_W-only FOC)."""
        import signaling.competition  # noqa: F401
        from signaling.price_equation import sigma_star_self_consistent
        result = sigma_star_self_consistent(lambda_W=0.68, mode="multiplicative")
        assert result["sigma_star"] == pytest.approx(0.4771, abs=1e-3)

    def test_pin_dyadic_total(self) -> None:
        """Exact closed-form pin: dyadic_total(5,3,1) = 15/8."""
        assert dyadic_total(5.0, 3.0, 1.0) == pytest.approx(1.875)

    def test_pin_dyadic_effort(self) -> None:
        """Exact closed-form pin: dyadic_effort(5,3,1) = 75/64."""
        assert dyadic_effort(5.0, 3.0, 1.0) == pytest.approx(1.171875)
