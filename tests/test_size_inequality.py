"""Tests for the size-inequality prediction (M1). Monument-size inequality
is the pushforward of the capacity distribution through x*(q); because
x*(q_min) = 0, the pushforward amplifies low-end inequality, so the
predicted size-Gini exceeds the capacity-Gini for distributions reaching
toward q_min. See the SI "Size inequality: the pushforward of capacity
through the equilibrium".
"""
from __future__ import annotations

import numpy as np
import pytest

from signaling.size_inequality import gini, monument_sizes, predicted_size_gini
from signaling.calibration import DEFAULT_Q_MIN, DEFAULT_LAMBDA
from signaling.layer1 import equilibrium_investment


class TestGini:
    def test_perfect_equality_is_zero(self) -> None:
        assert gini([3.0, 3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """Gini of [1,2,3,4,5] = 0.26667 (standard reference value)."""
        assert gini([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.26666667, abs=1e-6)

    def test_single_value_is_zero(self) -> None:
        assert gini([5.0]) == pytest.approx(0.0)

    def test_all_zero_is_zero(self) -> None:
        assert gini([0.0, 0.0, 0.0]) == pytest.approx(0.0)


class TestPushforward:
    def test_monument_sizes_match_equilibrium_investment(self) -> None:
        """monument_sizes is the x*(q) image of the capacities."""
        caps = [0.5, 1.0, 2.0, 3.0]
        sizes = monument_sizes(caps, DEFAULT_Q_MIN, DEFAULT_LAMBDA)
        expected = [equilibrium_investment(q, DEFAULT_Q_MIN, DEFAULT_LAMBDA) for q in caps]
        assert np.allclose(sizes, expected)

    def test_threshold_contributor_builds_zero(self) -> None:
        """x*(q_min) = 0: the marginal contributor builds nothing."""
        sizes = monument_sizes([DEFAULT_Q_MIN, 2.0], DEFAULT_Q_MIN, DEFAULT_LAMBDA)
        assert sizes[0] == pytest.approx(0.0)

    def test_predicted_size_gini_is_gini_of_sizes(self) -> None:
        caps = [0.5, 1.0, 2.0, 3.0, 5.0]
        sizes = monument_sizes(caps, DEFAULT_Q_MIN, DEFAULT_LAMBDA)
        assert predicted_size_gini(caps, DEFAULT_Q_MIN, DEFAULT_LAMBDA) == pytest.approx(gini(sizes))


class TestAmplification:
    def test_size_gini_exceeds_capacity_gini_near_threshold(self) -> None:
        """The distinctive M1 result (NOT an identity): with a contributor at
        the threshold, size inequality exceeds wealth inequality."""
        caps = [DEFAULT_Q_MIN, 1.0, 2.0, 3.0, 5.0]
        size_g = predicted_size_gini(caps, DEFAULT_Q_MIN, DEFAULT_LAMBDA)
        cap_g = gini(caps)
        assert size_g > cap_g

    def test_equal_capacities_zero_both(self) -> None:
        caps = [2.0, 2.0, 2.0]
        assert predicted_size_gini(caps, DEFAULT_Q_MIN, DEFAULT_LAMBDA) == pytest.approx(0.0)
        assert gini(caps) == pytest.approx(0.0)


class TestIsolationAndPin:
    def test_exported_at_package_level(self) -> None:
        import signaling
        for name in ("gini", "monument_sizes", "predicted_size_gini"):
            assert hasattr(signaling, name), name

    def test_headline_sigma_star_unchanged(self) -> None:
        import signaling.size_inequality  # noqa: F401
        from signaling.price_equation import sigma_star_self_consistent
        result = sigma_star_self_consistent(lambda_W=0.68, mode="multiplicative")
        assert result["sigma_star"] == pytest.approx(0.4771, abs=1e-3)

    def test_gini_pin(self) -> None:
        """Exact reference pin."""
        assert gini([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.26666667, abs=1e-6)
