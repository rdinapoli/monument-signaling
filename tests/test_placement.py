"""Tests for the monument-placement prediction (interior optimum of
audience visibility vs resource access). Encodes the theoretical
properties from the SI "Placement: the interior optimum of audience
visibility and resource access": saturating between-group
visibility (A1), non-increasing within-group visibility (A3), convex
cost (A2), a unique interior optimum, and two sign-robust comparative
statics.
"""
from __future__ import annotations

import numpy as np
import pytest

from signaling.placement import (
    visibility_between,
    visibility_within,
    resource_cost,
)
from signaling.calibration import PLACEMENT


from signaling.placement import placement_payoff
from signaling.calibration import DEFAULT_LAM_W, DEFAULT_LAM_C, DEFAULT_LAM_X


def _payoff_kwargs():
    return dict(
        lam_W=DEFAULT_LAM_W, lam_C=DEFAULT_LAM_C, lam_X=DEFAULT_LAM_X,
        R=PLACEMENT["R"], z_half=PLACEMENT["z_half"], b=PLACEMENT["b"],
        gamma=PLACEMENT["gamma"],
    )


class TestFormFunctions:
    """A1-A3 shape properties of the visibility and cost forms."""

    def test_between_visibility_increasing_and_concave(self) -> None:
        """A1: eta_C' > 0 and eta_C'' <= 0 (saturating)."""
        z = np.linspace(0.0, PLACEMENT["z_max"], 200)
        eta = visibility_between(z, PLACEMENT["z_half"])
        d1 = np.gradient(eta, z)
        d2 = np.gradient(d1, z)
        assert np.all(d1[1:-1] > 0.0)
        assert np.all(d2[1:-1] <= 1e-9)

    def test_within_visibility_nonincreasing(self) -> None:
        """A3: eta_W' <= 0 (peers concentrated at the core)."""
        z = np.linspace(0.0, PLACEMENT["z_max"], 200)
        eta = visibility_within(z, PLACEMENT["b"])
        d1 = np.gradient(eta, z)
        assert np.all(d1[1:-1] <= 1e-9)

    def test_cost_convex_increasing(self) -> None:
        """A2: G' > 0, G'' > 0."""
        z = np.linspace(0.05, PLACEMENT["z_max"], 200)
        g = resource_cost(z, PLACEMENT["gamma"])
        d1 = np.gradient(g, z)
        d2 = np.gradient(d1, z)
        assert np.all(d1 > 0.0)
        assert np.all(d2[1:-1] > 0.0)


class TestPayoff:
    """Pi(z) shape: strictly concave under A1-A3."""

    def test_payoff_strictly_concave(self) -> None:
        """Pi'' < 0 on (0, z_max)."""
        z = np.linspace(1e-3, PLACEMENT["z_max"], 400)
        pi = placement_payoff(z, **_payoff_kwargs())
        d2 = np.gradient(np.gradient(pi, z), z)
        assert np.all(d2[2:-2] < 0.0)

    def test_payoff_matches_definition_at_point(self) -> None:
        """Pi(z) equals R*(lam-weighted visibility) - cost at a sample z."""
        kw = _payoff_kwargs()
        z = 0.4
        eta_b = z / (kw["z_half"] + z)
        eta_w = np.exp(-kw["b"] * z)
        expected = kw["R"] * (
            kw["lam_W"] * eta_w + kw["lam_C"] * eta_b + kw["lam_X"] * eta_b
        ) - kw["gamma"] * z**2
        assert placement_payoff(z, **kw) == pytest.approx(expected)


from signaling.placement import optimal_placement


class TestOptimum:
    """z* = argmax Pi(z): interior, corner limits, edge cases."""

    def test_optimum_interior(self) -> None:
        """0 < z* < z_max when between-group audiences are salient."""
        kw = _payoff_kwargs()
        zstar = optimal_placement(z_max=PLACEMENT["z_max"], **kw)
        assert 0.0 < zstar < PLACEMENT["z_max"]

    def test_optimum_at_core_without_between_group(self) -> None:
        """lam_C = lam_X = 0 => z* = 0 (only within-group pull + cost)."""
        kw = _payoff_kwargs()
        kw["lam_C"] = 0.0
        kw["lam_X"] = 0.0
        zstar = optimal_placement(z_max=PLACEMENT["z_max"], **kw)
        assert zstar == pytest.approx(0.0, abs=1e-3)

    def test_pure_display_corner(self) -> None:
        """gamma -> 0 with no within-group pull => z* -> z_max (most visible)."""
        kw = _payoff_kwargs()
        kw["lam_W"] = 0.0
        kw["gamma"] = 1e-6
        zstar = optimal_placement(z_max=PLACEMENT["z_max"], **kw)
        assert zstar == pytest.approx(PLACEMENT["z_max"], abs=1e-2)

    def test_all_lambda_zero_goes_to_core(self) -> None:
        """No signaling benefit => only cost matters => z* = 0."""
        kw = _payoff_kwargs()
        kw["lam_W"] = kw["lam_C"] = kw["lam_X"] = 0.0
        zstar = optimal_placement(z_max=PLACEMENT["z_max"], **kw)
        assert zstar == pytest.approx(0.0, abs=1e-3)


from signaling.placement import comparative_static


class TestComparativeStatics:
    """The two sign-robust results: exposure rises with between-group
    salience, contracts under scarcity."""

    def test_optimum_increases_with_lambda_C(self) -> None:
        """dz*/d lam_C > 0."""
        kw = _payoff_kwargs()
        assert comparative_static("lam_C", z_max=PLACEMENT["z_max"], **kw) > 0.0

    def test_optimum_increases_with_lambda_X(self) -> None:
        """dz*/d lam_X > 0."""
        kw = _payoff_kwargs()
        assert comparative_static("lam_X", z_max=PLACEMENT["z_max"], **kw) > 0.0

    def test_optimum_decreases_with_scarcity(self) -> None:
        """dz*/d gamma < 0."""
        kw = _payoff_kwargs()
        assert comparative_static("gamma", z_max=PLACEMENT["z_max"], **kw) < 0.0

    def test_lambda_C_sign_matches_ift_formula(self) -> None:
        """Symbolic-vs-numeric: the numeric dz*/d lam_C has the same sign as
        the implicit-function-theorem value -(R eta_C'(z*)) / Pi''(z*), which
        is > 0 because eta_C' > 0 (A1) and Pi'' < 0 (concavity)."""
        kw = _payoff_kwargs()
        zstar = optimal_placement(z_max=PLACEMENT["z_max"], **kw)
        # eta_C'(z) = z_half / (z_half + z)^2 > 0
        eta_c_prime = kw["z_half"] / (kw["z_half"] + zstar) ** 2
        # Pi''(z*) via central second difference
        h = 1e-4
        pi = lambda z: float(placement_payoff(z, **kw))
        pi_pp = (pi(zstar + h) - 2 * pi(zstar) + pi(zstar - h)) / h**2
        ift = -(kw["R"] * eta_c_prime) / pi_pp
        numeric = comparative_static("lam_C", z_max=PLACEMENT["z_max"], **kw)
        assert ift > 0.0
        assert np.sign(numeric) == np.sign(ift)


Z_STAR_ANCHOR = 0.018261180773735272  # measured regression anchor


class TestRegressionPin:
    def test_zstar_anchor_value(self) -> None:
        """Pin z* at the default anchor so a parameter/formula change flags."""
        kw = _payoff_kwargs()
        zstar = optimal_placement(z_max=PLACEMENT["z_max"], **kw)
        assert zstar == pytest.approx(Z_STAR_ANCHOR, abs=1e-3)  # measured


class TestIsolation:
    """Placement is a downstream module: importing it changes no headline
    number, and its symbols are exported at package level."""

    def test_exported_at_package_level(self) -> None:
        import signaling
        for name in (
            "placement_payoff", "optimal_placement", "comparative_static",
            "visibility_between", "visibility_within", "resource_cost",
        ):
            assert hasattr(signaling, name), name

    def test_headline_sigma_star_unchanged(self) -> None:
        """MOD4 regression guard: with placement importable, the headline
        self-consistent threshold is unchanged (~0.4771, lambda_W-only FOC)."""
        import signaling.placement  # noqa: F401  (import has no side effects)
        from signaling.price_equation import sigma_star_self_consistent
        from signaling.calibration import DEFAULT_LAMBDA_W_ANCHOR
        res = sigma_star_self_consistent(
            lambda_W=DEFAULT_LAMBDA_W_ANCHOR, mode="multiplicative"
        )
        assert res["sigma_star"] == pytest.approx(0.4771, abs=1e-3)


class TestEndpointDerivative:
    """Interior optimum needs Pi'(0) > 0 AND Pi'(z_max) < 0 (A1-A2 alone
    do not exclude the upper corner)."""

    def test_endpoint_derivatives_at_calibrated_params(self):
        kw = _payoff_kwargs()
        z_max = PLACEMENT["z_max"]
        eps = 1e-5
        d0 = (placement_payoff(eps, **kw) - placement_payoff(0.0, **kw)) / eps
        dz = (placement_payoff(z_max, **kw) - placement_payoff(z_max - eps, **kw)) / eps
        assert d0 > 0.0, "Pi'(0) > 0 fails at calibrated params"
        assert dz < 0.0, "Pi'(z_max) < 0 fails at calibrated params"
