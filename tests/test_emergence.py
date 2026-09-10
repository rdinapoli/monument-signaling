"""Tests for src/signaling/emergence.py (bistable-emergence analysis).

Encodes the three structural properties of the bistable-emergence result
reported in the main-text bistable-emergence section:

- (a) Rare builder cannot invade at phi = 0 under multiplicative fitness
  with self-consistent cost (within-group benefit does not overcome cost
  without any network or deterrence advantage).
- (b) Invasion threshold sigma*_inv(phi) is monotonically decreasing in
  phi (more neighbor-adopters make invasion easier).
- (c) In the limit phi -> 1, sigma*_inv converges to the maintenance
  threshold sigma* computed by the threshold calculation.
"""
from __future__ import annotations

import numpy as np
import pytest

from signaling.emergence import (
    phi_star,
    rare_builder_fitness_advantage,
    replicator_dynamics,
    sigma_star_invasion,
)


class TestRareBuilderFitnessAdvantage:
    """Structural properties of the rare-builder fitness advantage."""

    def test_lone_builder_cannot_invade_multiplicative(self):
        """At phi = 0, advantage is negative for all sigma under multiplicative
        fitness with self-consistent cost (lambda_W = 0.30)."""
        for sigma in (0.01, 0.1, 0.3, 0.5, 0.8):
            res = rare_builder_fitness_advantage(
                sigma, lambda_W=0.30, frac_signalers=0.0,
                mode="multiplicative", use_self_consistent_cost=True,
            )
            assert res["advantage"] < 0, (
                f"Rare builder gained advantage at sigma={sigma}, phi=0, "
                f"which contradicts the bistability result"
            )

    def test_full_adoption_gives_network_saturation(self):
        """At phi = 1, network degree k_rare matches the full equilibrium k(M_g)."""
        res = rare_builder_fitness_advantage(
            0.30, lambda_W=0.30, frac_signalers=1.0, mode="multiplicative"
        )
        # k_rare should equal k(M_g) at full adoption
        from signaling.calibration import DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF
        from signaling.layer3 import network_degree
        k_full = float(network_degree(res["M_g"], DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF))
        assert res["k_rare"] == pytest.approx(k_full, rel=1e-10)

    def test_advantage_monotonic_in_phi(self):
        """For fixed sigma, advantage increases as phi rises (more neighbors signal)."""
        advs = [
            rare_builder_fitness_advantage(
                0.30, lambda_W=0.30, frac_signalers=phi, mode="multiplicative"
            )["advantage"]
            for phi in (0.0, 0.25, 0.50, 0.75, 1.0)
        ]
        # Strictly increasing
        for a, b in zip(advs[:-1], advs[1:]):
            assert b > a, f"Advantage non-monotonic in phi: {advs}"

    def test_M_g_matches_equilibrium_monument_stock(self):
        """M_g in the dict equals the Layer 1 equilibrium monument stock."""
        from signaling.layer1 import expected_monument_stock
        from signaling.calibration import DEFAULT_N, DEFAULT_Q_MAX, DEFAULT_Q_MIN
        res = rare_builder_fitness_advantage(
            0.30, lambda_W=0.30, frac_signalers=0.5, mode="multiplicative"
        )
        M_expected = expected_monument_stock(DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX, 0.30)
        assert res["M_g"] == pytest.approx(M_expected, rel=1e-10)

    def test_mixed_mode_coincides_with_multiplicative(self):
        """Mixed and multiplicative modes COINCIDE under the positional model
        (no separate within-group reward term to leave undiscounted)."""
        mult = rare_builder_fitness_advantage(
            0.30, lambda_W=0.30, frac_signalers=0.1, mode="multiplicative"
        )["advantage"]
        mix = rare_builder_fitness_advantage(
            0.30, lambda_W=0.30, frac_signalers=0.1, mode="mixed"
        )["advantage"]
        assert mult == pytest.approx(mix), (
            "Mixed and multiplicative coincide under the positional model "
            "(no separate within-group reward term)"
        )

    def test_invalid_mode_raises(self):
        """Unknown mode string raises ValueError."""
        with pytest.raises(ValueError):
            rare_builder_fitness_advantage(
                0.30, lambda_W=0.30, frac_signalers=0.1, mode="oops"
            )


class TestSigmaStarInvasion:
    """Bistability structure: sigma*_inv(phi)."""

    def test_phi_zero_never_invades_multiplicative(self):
        """Under multiplicative with self-consistent C, phi=0 yields infinite invasion
        threshold (bistability)."""
        s = sigma_star_invasion(lambda_W=0.30, frac_signalers=0.0, mode="multiplicative")
        assert not np.isfinite(s), f"expected inf, got {s}"

    def test_phi_values_monument_manuscript(self):
        """Invasion thresholds at the empirical anchor lambda_W = 0.68 (positional model).

        sigma*_inv(phi) decreases from infinity (no invasion at very low phi, where a
        rare builder has too few signaling neighbors) toward the maintenance value
        sigma* ~ 0.477 as phi -> 1 (the manuscript's ~0.48).
        """
        assert not np.isfinite(
            sigma_star_invasion(lambda_W=0.68, frac_signalers=0.05, mode="multiplicative")
        )
        expected = {
            0.25: 0.728,
            0.50: 0.591,
            0.75: 0.521,
            1.00: 0.477,
        }
        for phi, expected_val in expected.items():
            s = sigma_star_invasion(lambda_W=0.68, frac_signalers=phi, mode="multiplicative")
            assert s == pytest.approx(expected_val, abs=0.02), (
                f"sigma*_inv(phi={phi}) = {s:.3f} at lambda_W=0.68, expected ~{expected_val}"
            )

    def test_phi_values_legacy_lambda_03(self):
        """Invasion thresholds at the lower-lambda_W regime (lambda_W = 0.30).

        Regression test for the lower-lambda_W regime, below the empirical
        anchor (lambda_W = 0.68). Expected values are generated from the
        function itself, not transcribed from manuscript prose.
        """
        expected = {
            0.05: 0.832,
            0.25: 0.444,
            0.50: 0.320,
            0.75: 0.266,
        }
        for phi, expected_val in expected.items():
            s = sigma_star_invasion(lambda_W=0.30, frac_signalers=phi, mode="multiplicative")
            assert s == pytest.approx(expected_val, abs=0.015), (
                f"sigma*_inv(phi={phi}) = {s:.3f} at lambda_W=0.30, expected ~{expected_val}"
            )

    def test_monotonic_decreasing_in_phi(self):
        """sigma*_inv is monotonically decreasing in phi."""
        phis = [0.05, 0.1, 0.25, 0.5, 1.0]
        values = [sigma_star_invasion(lambda_W=0.30, frac_signalers=p, mode="multiplicative")
                  for p in phis]
        # All finite for these phi's
        assert all(np.isfinite(v) for v in values)
        for a, b in zip(values[:-1], values[1:]):
            assert b <= a, f"sigma*_inv not monotonic decreasing in phi: {values}"

    def test_phi_one_approaches_maintenance_threshold(self):
        """At phi = 1, sigma*_inv approximately matches the maintenance sigma*. They
        differ by ~0.03 because the war-avoidance makes the conflict reduction r
        matter, and emergence uses the no-feedback M_g while critical_threshold
        uses the feedback-equilibrium M_g (slightly different r)."""
        # Library maintenance threshold using self-consistent C and multiplicative
        from signaling.layer1 import average_equilibrium_cost
        from signaling.price_equation import critical_threshold_sigma_star
        lam_W = 0.30
        C_m = average_equilibrium_cost(lam_W, 0.1, 2.0)
        maint = critical_threshold_sigma_star(
            C=C_m, lambda_W=lam_W, mode="multiplicative"
        )["sigma_star"]
        inv = sigma_star_invasion(lambda_W=lam_W, frac_signalers=1.0, mode="multiplicative")
        assert inv == pytest.approx(maint, abs=0.04), (
            f"Invasion at phi=1 ({inv:.4f}) should match maintenance ({maint:.4f})"
        )

    def test_mixed_equals_multiplicative(self):
        """Under the positional model, mixed and multiplicative coincide (no separate
        reward term), so the invasion threshold is identical, not over-permissive."""
        s_mix = sigma_star_invasion(lambda_W=0.30, frac_signalers=0.5, mode="mixed")
        s_mult = sigma_star_invasion(lambda_W=0.30, frac_signalers=0.5, mode="multiplicative")
        assert s_mix == pytest.approx(s_mult)


class TestPhiStar:
    """Interior saddle phi*(sigma) of the replicator dynamics."""

    def test_no_saddle_below_maintenance_threshold(self):
        """Below the maintenance sigma* (~0.477 at lambda_W=0.68), no interior
        saddle: rare builder is disadvantaged at all phi <= 1."""
        ps = phi_star(0.10, lambda_W=0.68)
        assert not np.isfinite(ps), f"Expected NaN at sigma below threshold, got {ps}"

    def test_saddle_exists_above_maintenance_threshold(self):
        """Above sigma*, an interior saddle separates the two basins."""
        for sigma in (0.60, 0.70, 0.80, 0.90):
            ps = phi_star(sigma, lambda_W=0.68)
            assert np.isfinite(ps), f"phi*({sigma}) should be finite, got {ps}"
            assert 0.0 < ps < 1.0, f"phi*({sigma}) = {ps} should lie in (0, 1)"

    def test_saddle_decreasing_in_sigma(self):
        """Higher environmental uncertainty lowers the adoption threshold for
        invasion: phi*(sigma) is strictly decreasing."""
        sigmas = [0.60, 0.70, 0.75, 0.80, 0.90]
        ps_values = [phi_star(s, lambda_W=0.68) for s in sigmas]
        for a, b in zip(ps_values[:-1], ps_values[1:]):
            assert b < a, (
                f"phi*(sigma) not strictly decreasing: {list(zip(sigmas, ps_values))}"
            )

    def test_saddle_is_inverse_of_sigma_star_invasion(self):
        """phi*(sigma_inv(phi)) = phi: the two functions parameterize the
        same locus in (sigma, phi) space."""
        for phi in (0.25, 0.50, 0.75):
            sigma_inv = sigma_star_invasion(lambda_W=0.68, frac_signalers=phi)
            assert np.isfinite(sigma_inv)
            ps_back = phi_star(sigma_inv, lambda_W=0.68)
            assert ps_back == pytest.approx(phi, abs=0.02), (
                f"phi*({sigma_inv}) = {ps_back}, expected {phi}"
            )

    def test_saddle_zero_advantage(self):
        """By construction, the rare-builder advantage at phi* vanishes."""
        sigma = 0.70
        ps = phi_star(sigma, lambda_W=0.68)
        adv = rare_builder_fitness_advantage(
            sigma, lambda_W=0.68, frac_signalers=ps, mode="multiplicative",
        )["advantage"]
        assert abs(adv) < 1e-5, f"Advantage at phi* should be ~0, got {adv}"


class TestReplicatorDynamics:
    """Bistable emergence demonstrated via replicator trajectories.

    The replicator equation phidot = phi(1-phi)[w_MB - w_NB] has phi = 0
    and phi = 1 as absorbing boundaries and phi* as an unstable interior
    saddle. The defining bistability signature is that initial conditions
    below phi* flow to 0 while initial conditions above phi* flow to 1,
    giving two distinct basins of attraction.
    """

    def test_phi_below_saddle_converges_to_zero(self):
        """phi_0 well below the saddle flows to non-signaling."""
        sigma = 0.70
        ps = phi_star(sigma, lambda_W=0.68)  # ~0.287 at sigma=0.70
        traj = replicator_dynamics(
            sigma, lambda_W=0.68, phi_0=ps * 0.4, t_max=400,
        )
        assert traj["success"]
        assert traj["absorbed_to_0"], (
            f"Expected convergence to 0 from phi_0={ps * 0.4:.3f} (saddle={ps:.3f}); "
            f"terminal phi = {traj['terminal_phi']}"
        )
        assert traj["terminal_phi"] < 1e-2

    def test_phi_above_saddle_converges_to_one(self):
        """phi_0 well above the saddle flows to full adoption."""
        sigma = 0.70
        ps = phi_star(sigma, lambda_W=0.68)  # ~0.287 at sigma=0.70
        phi_0 = ps + 0.6 * (1.0 - ps)
        traj = replicator_dynamics(
            sigma, lambda_W=0.68, phi_0=phi_0, t_max=400,
        )
        assert traj["success"]
        assert traj["absorbed_to_1"], (
            f"Expected convergence to 1 from phi_0={phi_0:.3f} (saddle={ps:.3f}); "
            f"terminal phi = {traj['terminal_phi']}"
        )
        assert traj["terminal_phi"] > 1.0 - 1e-2

    def test_both_basins_distinguishable(self):
        """A grid of initial conditions partitions into two basins separated by phi*."""
        sigma = 0.70
        ps = phi_star(sigma, lambda_W=0.68)
        # phi_0 strictly below the saddle should flow to 0
        for phi_0 in (0.02, 0.05, 0.10):
            if phi_0 >= ps:
                continue
            traj = replicator_dynamics(
                sigma, lambda_W=0.68, phi_0=phi_0, t_max=400,
            )
            assert traj["absorbed_to_0"], (
                f"phi_0={phi_0} < phi*={ps:.3f} should flow to 0; "
                f"got terminal = {traj['terminal_phi']}"
            )
        # phi_0 strictly above the saddle should flow to 1
        for phi_0 in (0.40, 0.60, 0.85):
            if phi_0 <= ps:
                continue
            traj = replicator_dynamics(
                sigma, lambda_W=0.68, phi_0=phi_0, t_max=400,
            )
            assert traj["absorbed_to_1"], (
                f"phi_0={phi_0} > phi*={ps:.3f} should flow to 1; "
                f"got terminal = {traj['terminal_phi']}"
            )

    def test_boundary_phi_zero_stays_zero(self):
        """The non-signaling boundary is absorbing."""
        traj = replicator_dynamics(0.40, lambda_W=0.68, phi_0=0.0, t_max=50)
        assert traj["terminal_phi"] == 0.0

    def test_boundary_phi_one_stays_one(self):
        """The full-adoption boundary is absorbing."""
        traj = replicator_dynamics(0.40, lambda_W=0.68, phi_0=1.0, t_max=50)
        assert traj["terminal_phi"] == 1.0

    def test_invalid_phi_0_raises(self):
        """phi_0 outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            replicator_dynamics(0.40, lambda_W=0.68, phi_0=-0.1)
        with pytest.raises(ValueError):
            replicator_dynamics(0.40, lambda_W=0.68, phi_0=1.1)

    def test_below_maintenance_threshold_no_invasion(self):
        """Below the maintenance threshold (no interior saddle), any interior
        phi_0 < 1 flows to 0 because the advantage is negative for all phi < 1."""
        # sigma = 0.10 is well below sigma* ~ 0.477 at lambda_W = 0.68
        for phi_0 in (0.10, 0.50, 0.90):
            traj = replicator_dynamics(
                0.10, lambda_W=0.68, phi_0=phi_0, t_max=400,
            )
            assert traj["terminal_phi"] < phi_0 + 1e-3, (
                f"Below threshold, phi_0={phi_0} should not grow; "
                f"terminal = {traj['terminal_phi']}"
            )

    def test_trajectory_at_saddle_is_slow(self):
        """Initial condition exactly at the saddle stays near phi* for a
        meaningful time before numerical noise pushes it off. The hallmark
        of an unstable fixed point is slow departure rather than no motion."""
        sigma = 0.70
        ps = phi_star(sigma, lambda_W=0.68)
        traj = replicator_dynamics(
            sigma, lambda_W=0.68, phi_0=ps, t_max=100, n_steps=200,
        )
        # Within the first 25% of the integration, phi should be near ps
        early = len(traj["t"]) // 4
        assert abs(traj["phi"][early] - ps) < 0.05, (
            f"Trajectory at saddle drifted too quickly: phi[t/4] = "
            f"{traj['phi'][early]}, ps = {ps}"
        )

    def test_replicator_accepts_network_degree_spec(self):
        """Replicator dynamics accept and use an alternative network-degree spec.
        At the empirical anchor, the calibrated alternatives agree with MM
        at the equilibrium M_g, so trajectories should match closely."""
        from signaling.layer1 import expected_monument_stock
        from signaling.layer3 import calibrate_alternative_network_forms
        from signaling.calibration import (
            DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF,
            DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX,
        )

        M_e = expected_monument_stock(
            DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX, 0.68,
        )
        specs = calibrate_alternative_network_forms(
            M_e, DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF,
        )
        # Trajectory under MM should match trajectory under exponential at anchor
        sigma, phi_0 = 0.40, 0.25
        traj_mm = replicator_dynamics(
            sigma, lambda_W=0.68, phi_0=phi_0, t_max=200,
            network_degree_spec=specs["michaelis_menten"],
        )
        traj_exp = replicator_dynamics(
            sigma, lambda_W=0.68, phi_0=phi_0, t_max=200,
            network_degree_spec=specs["exponential"],
        )
        # Both should converge to phi = 1 (above saddle); terminal values
        # should be very close
        assert traj_mm["absorbed_to_1"] == traj_exp["absorbed_to_1"]
        assert abs(traj_mm["terminal_phi"] - traj_exp["terminal_phi"]) < 1e-3
