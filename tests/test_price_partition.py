"""Tests for the full multilevel Price partition (positional within-group reward).

These encode the theoretical properties established and verified during the model
derivation: the Price identity, the positional
net-out, the within-group free-rider resolution (beta_0 > 0) and its Hamilton-
altruism limit, the F-robustness of favorability, and the central result
that environmental uncertainty governs the equilibrium *intensity* of signaling.

Grounded in Price (1972) Eq. A17, Hamilton (1975) Eq. 1/3/7, Frank (1998).
"""
import numpy as np
import pytest

from signaling.price_equation import (
    within_group_status_differential,
    mixed_group_mean_fitness,
    within_group_regression,
    between_group_regression,
    emergence_criterion,
    equilibrium_intensity,
    price_partition,
    collective_optimum_fraction,
)
from signaling.calibration import (
    DEFAULT_Q_MIN,
    DEFAULT_Q_MAX,
    DEFAULT_N,
    DEFAULT_GAMMA,
    DEFAULT_K_0,
    DEFAULT_K_MAX,
    DEFAULT_M_HALF,
    DEFAULT_P_BASE,
    DEFAULT_CONFLICT_MORTALITY,
    DEFAULT_WAR_COST,
)
from signaling.layer1 import average_equilibrium_cost, expected_monument_stock
from signaling.layer2 import derived_conflict_reduction
from signaling.layer3 import (
    calibrate_alternative_network_forms,
    network_degree,
    survival_probability,
)

LAM = 0.68  # empirical calibration (lambda_W anchored to C_emp ~ 0.35)


class TestStatusDifferential:
    def test_formula(self):
        """s_W = lambda * (E[q] - q_min): the within-group positional status differential."""
        Eq = (DEFAULT_Q_MIN + DEFAULT_Q_MAX) / 2.0
        assert within_group_status_differential(LAM) == pytest.approx(LAM * (Eq - DEFAULT_Q_MIN))

    def test_zero_at_zero_lambda(self):
        assert within_group_status_differential(0.0) == 0.0


class TestPriceIdentity:
    def test_identity_holds(self):
        """Price (1972) A17: w_bar*Delta_p == Cov_between(W_g, p_g) + E[Cov_within]."""
        rng = np.random.default_rng(0)
        p_g = np.clip(rng.beta(1.5, 3.0, 300), 1e-4, 1 - 1e-4)
        res = price_partition(p_g, sigma=0.5, lam=LAM)
        assert res["identity_residual"] == pytest.approx(0.0, abs=1e-12)

    def test_identity_holds_across_sigma(self):
        rng = np.random.default_rng(1)
        p_g = np.clip(rng.uniform(0.05, 0.95, 200), 1e-4, 1 - 1e-4)
        for sigma in (0.1, 0.5, 0.9):
            res = price_partition(p_g, sigma=sigma, lam=LAM)
            assert res["identity_residual"] == pytest.approx(0.0, abs=1e-12)


class TestPositionalNetsOut:
    def test_positional_cancellation_symbolic(self):
        """INDEPENDENT algebra (sympy): p*wS + (1-p)*wN = (1 - p*C)*S*K for
        symbolic S, K, C, s_W, p -- the positional transfers cancel exactly
        as an identity, not merely at sampled parameters (the numeric test
        below builds its expectation from the same
        S/K/r calls as the implementation, so this symbolic check is the
        non-circular content)."""
        import sympy as sp

        S_, K_, C_, sW_, p_ = sp.symbols("S K C s_W p")
        wS = S_ * K_ * ((1 - C_) + sW_ * (1 - p_))
        wN = S_ * K_ * (1 - sW_ * p_)
        assert sp.simplify(p_ * wS + (1 - p_) * wN - (1 - p_ * C_) * S_ * K_) == 0

    def test_group_mean_equals_cost_times_public_goods(self):
        """Fully positional: W_g(p) = (1 - p*C_model)*S*K at sampled p
        (numeric consistency; the identity itself is proven symbolically in
        test_positional_cancellation_symbolic)."""
        C = float(average_equilibrium_cost(LAM, DEFAULT_Q_MIN, DEFAULT_Q_MAX))
        M_full = float(expected_monument_stock(DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX, LAM))
        for p in (0.2, 0.5, 0.8):
            M_g = p * M_full
            S = float(survival_probability(
                0.5, network_degree(M_g, DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF), DEFAULT_GAMMA
            ))
            r = float(derived_conflict_reduction(M_g, M_g))
            K = 1.0 - DEFAULT_WAR_COST * 0.5 * (1.0 - r)  # war-avoidance conflict factor at sigma=0.5
            assert mixed_group_mean_fitness(p, 0.5, LAM) == pytest.approx((1.0 - p * C) * S * K)


class TestFreeRiderResolution:
    def test_within_group_favors_signaling_at_calibration(self):
        """beta_0 = w_S - w_N > 0: honest signalers out-reproduce free-riders within the group."""
        assert within_group_regression(1e-3, 0.5, LAM) > 0

    def test_within_group_favors_signaling_all_costs(self):
        """Under the positional (additive zero-sum) reward, beta_0 > 0 across the empirical
        lambda interval -- the free-rider problem is resolved at every monument cost."""
        for lam in (0.39, 0.68, 0.97):
            assert within_group_regression(1e-3, 0.5, lam) > 0

    def test_hamilton_altruism_limit(self):
        """Zeroing the status reward (status_scale=0) recovers Hamilton (1975): within-group
        selection always opposes a costly group-beneficial trait (beta_0 < 0)."""
        assert within_group_regression(1e-3, 0.5, LAM, status_scale=0.0) < 0

    def test_status_reward_reverses_sign(self):
        """The within-group status reward is what flips beta_0 from negative (altruism) to positive
        (signaling) -- the framework's central departure from Hamilton's altruism case."""
        b0_signaling = within_group_regression(1e-3, 0.5, LAM)
        b0_altruism = within_group_regression(1e-3, 0.5, LAM, status_scale=0.0)
        assert b0_altruism < 0 < b0_signaling


class TestFRobustFavorability:
    def test_favored_across_relatedness(self):
        """At realistic sigma, signaling is favored across the full relatedness range F in [0,1];
        no prediction hinges on the unobservable F."""
        for F in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert emergence_criterion(0.3, LAM, F) > 0


class TestEquilibriumIntensity:
    def test_intensity_rises_with_sigma(self):
        """Central result (framing B): at moderate relatedness, the equilibrium signaling
        intensity p*(sigma) increases with environmental uncertainty -- sigma governs intensity."""
        F = 0.5
        sigmas = [0.1, 0.3, 0.5, 0.7, 0.9]
        pstars = [equilibrium_intensity(s, LAM, F) for s in sigmas]
        # p* rises with sigma, then saturates at full adoption at high sigma (within-group
        # positional + war-avoidance coordination drive everyone to build).
        assert all(pstars[i] <= pstars[i + 1] for i in range(len(pstars) - 1))
        assert pstars[0] < pstars[-1]
        assert 0.0 < pstars[2] < 1.0  # interior at the calibration band

    def test_low_relatedness_full_arms_race(self):
        """At low relatedness, within-group status competition drives full investment (p*=1)
        regardless of sigma -- the pure positional arms race."""
        assert equilibrium_intensity(0.2, LAM, F=0.25) == pytest.approx(1.0)

    def test_no_establishment_at_high_F_low_sigma(self):
        """At high relatedness and near-zero uncertainty, the between-group term cannot carry
        signaling and the (weak) within-group term does not establish it: p*=0."""
        assert equilibrium_intensity(0.001, LAM, F=0.9) == pytest.approx(0.0)

    def test_mm_gradient_single_crossing_over_reported_sweeps(self):
        """Under the default Michaelis-Menten network the selection gradient has
        at most one interior zero across the reported (sigma, F) sweeps, so the
        from-rarity p* is the unique attractor and the plotted intensity curves
        are basin-independent (measured over sigma in [0.05, 0.95] at F = 0.5
        and 0.65)."""
        from signaling.price_equation import equilibrium_intensity_roots

        for F in (0.5, 0.65):
            for sigma in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9):
                structure = equilibrium_intensity_roots(sigma, LAM, F)
                assert len(structure["roots"]) <= 1, (
                    f"MM gradient not single-crossing at sigma={sigma}, F={F}: "
                    f"{structure['roots']}"
                )
                assert not structure["bistable"]

    def test_hill_alternative_bistable_at_low_sigma(self):
        """The solver must not assume a monotone gradient: under the calibrated
        Hill n=2 network alternative at sigma ~ 0.119, F = 0.5, the gradient
        has two interior zeros (~0.0124 unstable, ~0.5089 stable), so
        extinction and a positive interior state are simultaneously stable.
        The from-rarity convention returns 0.0 (correct: the founding
        frequency sits in the extinction basin), while a founding fraction
        above the unstable root reaches the positive stable state. This is
        the counterexample that a single-root implementation would miss."""
        from signaling.layer1 import expected_monument_stock
        from signaling.layer3 import calibrate_alternative_network_forms
        from signaling.price_equation import equilibrium_intensity_roots

        M_anchor = float(expected_monument_stock(12, 0.1, 2.0, LAM))
        hill = calibrate_alternative_network_forms(M_anchor)["hill_n2"]
        sigma, F = 0.11888888888888889, 0.5

        structure = equilibrium_intensity_roots(
            sigma, LAM, F, network_degree_spec=hill
        )
        assert len(structure["roots"]) == 2
        assert structure["roots"][0] == pytest.approx(0.012414, abs=1e-4)
        assert structure["roots"][1] == pytest.approx(0.508869, abs=1e-4)
        assert structure["stable"] == [False, True]
        assert structure["zero_stable"] is True
        assert structure["bistable"] is True

        assert equilibrium_intensity(
            sigma, LAM, F, network_degree_spec=hill
        ) == pytest.approx(0.0)
        assert equilibrium_intensity(
            sigma, LAM, F, initial_fraction=0.6, network_degree_spec=hill
        ) == pytest.approx(0.508869, abs=1e-4)


class TestReverseTragedyOfCommons:
    """The between-group level enters the result: W_g(p) is non-monotone with an
    INTERIOR maximum (the collective optimum is a builder minority), robust
    across network functional forms, while individual positional competition
    drives groups PAST that optimum. The Price partition locates selection at
    the within-group level (cov_within > 0 dominant; cov_between small).
    """

    def _specs(self):
        M_full = float(
            expected_monument_stock(DEFAULT_N, DEFAULT_Q_MIN, DEFAULT_Q_MAX, LAM)
        )
        return calibrate_alternative_network_forms(M_full)

    def test_Wg_nonmonotone_interior_optimum(self):
        """W_g(p) is maximized at an interior p (< full adoption) at the anchor."""
        p_opt = collective_optimum_fraction(0.478, LAM)
        assert 0.0 < p_opt < 1.0

    def test_interior_optimum_robust_across_network_forms(self):
        """The interior collective optimum holds for MM, exponential, Hill, piecewise."""
        for name, spec in self._specs().items():
            for sigma in (0.30, 0.478, 0.60):
                p_opt = collective_optimum_fraction(sigma, LAM, network_degree_spec=spec)
                assert 0.0 < p_opt < 1.0, f"{name} at sigma={sigma}: p_opt={p_opt}"

    def test_positional_overshoot(self):
        """Individuals build past the collective optimum: participation rate > p_opt."""
        from signaling.layer1 import participation_equilibrium

        p_star = participation_equilibrium(LAM, DEFAULT_Q_MIN, DEFAULT_Q_MAX)[
            "participation_rate"
        ]
        p_opt = collective_optimum_fraction(0.478, LAM)
        assert p_star > p_opt

    def test_within_group_dominates_partition_across_forms(self):
        """cov_within > 0 and exceeds |cov_between| near the threshold, every form."""
        ens = np.linspace(0.01, 0.99, 99)
        for name, spec in self._specs().items():
            part = price_partition(ens, 0.478, LAM, network_degree_spec=spec)
            assert part["cov_within"] > 0.0, name
            assert part["cov_within"] > abs(part["cov_between"]), name

    def test_partition_identity_holds_for_all_forms(self):
        """The Price identity closes for every network form (residual ~ 0)."""
        ens = np.linspace(0.01, 0.99, 99)
        for name, spec in self._specs().items():
            part = price_partition(ens, 0.478, LAM, network_degree_spec=spec)
            assert abs(part["identity_residual"]) < 1e-12, name
