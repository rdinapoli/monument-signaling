"""Tests for Layer 1: individual signaling equilibrium.

Verifies theoretical properties of the separating equilibrium:
- Boundary condition: x*(q_min) == 0
- Positivity: x*(q) > 0 for all q > q_min
- Monotonicity: x*(q) strictly increasing in q
- Fitness gain: Delta_w(q) > 0 for all q > q_min
- Differential benefit: Delta_w increasing in q
- First-order and second-order conditions hold
- M_g increases with group size and mean quality

Each test encodes a theoretical prediction from the framework,
not merely that the code runs without error.
"""

import numpy as np
import pytest
import sympy as sp

from signaling.calibration import DEFAULT_LAMBDA, DEFAULT_Q_MAX, DEFAULT_Q_MIN
from signaling.layer1 import (
    BenefitSpec,
    CostSpec,
    EquilibriumResult,
    FitnessResult,
    SpenceResult,
    benefit_function_sensitivity,
    concave_benefit,
    average_equilibrium_cost,
    cost_at_equilibrium,
    cost_function_sensitivity,
    lambda_W_for_C,
    derive_aggregate_signal,
    derive_equilibrium,
    derive_equilibrium_fitness,
    equilibrium_fitness,
    equilibrium_investment,
    expected_fitness_gain,
    expected_monument_stock,
    exponential_cost,
    fitness_at_deviation,
    fitness_gain,
    free_rider_deterrent,
    group_monument_stock,
    linear_benefit,
    maintenance_cost,
    maintenance_to_new_ratio,
    participation_differential_sampled,
    participation_equilibrium,
    pooling_fitness,
    power_cost,
    quadratic_cost,
    receiver_inference,
    verify_multi_period_spence_condition,
    verify_spence_condition,
)

# Shared test parameters
Q_MIN = DEFAULT_Q_MIN
Q_MAX = DEFAULT_Q_MAX
LAM = DEFAULT_LAMBDA
Q_RANGE = np.linspace(Q_MIN + 1e-6, Q_MAX, 200)


# =====================================================================
# TestSpenceCondition
# =====================================================================


class TestSpenceCondition:
    """Verify the single-crossing (Spence/handicap) condition."""

    def test_quadratic_cross_partial_negative(self) -> None:
        """Quadratic cost d^2c/dxdq = -x/q^2 is negative for x > 0, q > 0."""
        result = verify_spence_condition()
        assert isinstance(result, SpenceResult)
        assert result.quadratic_is_negative

    def test_power_cross_partial_negative(self) -> None:
        """Power cost cross-partial is negative for a > 1, b > 0."""
        result = verify_spence_condition()
        x = result.symbols["x"]
        q = result.symbols["q"]
        a = result.symbols["a"]
        b = result.symbols["b"]
        # Evaluate at specific values to verify negativity
        val = result.power_cross_partial.subs({x: 1, q: 1, a: 2, b: 1})
        assert float(val) < 0
        # Check a different valid parameter set
        val2 = result.power_cross_partial.subs({x: 0.5, q: 2, a: 3, b: 0.5})
        assert float(val2) < 0

    def test_spence_fails_without_quality_dependence(self) -> None:
        """A cost function c(x) = x^2/2 (no q dependence) has zero cross-partial."""
        x, q = sp.symbols("x q", positive=True)
        c_no_q = x**2 / 2
        cross = sp.diff(sp.diff(c_no_q, x), q)
        assert cross == 0


# =====================================================================
# TestEquilibriumInvestment
# =====================================================================


class TestEquilibriumInvestment:
    """Verify properties of the equilibrium investment function x*(q)."""

    def test_boundary_condition_zero(self) -> None:
        """x*(q_min) = 0: lowest-quality type does not invest."""
        assert equilibrium_investment(Q_MIN, Q_MIN, LAM) == pytest.approx(0.0, abs=1e-15)

    def test_boundary_condition_zero_for_arbitrary_floor_values(self) -> None:
        """x*(q_min) = 0 EXACTLY for every q_min, not only the calibrated 0.1.

        Computing sqrt(lam*(q**2 - q_min**2)) with mixed vectorized/scalar
        powers that round 1 ulp apart returns silent NaN (argument -1 ulp)
        or a spurious ~1e-8 investment (+1 ulp) for ~0.07% of floor values.
        The factored form (q - q_min)(q + q_min) makes the boundary exact."""
        rng = np.random.default_rng(0)
        floors = rng.uniform(0.05, 2.0, 50_000)
        for qm in floors:
            x = equilibrium_investment(float(qm), float(qm), LAM)
            assert x == 0.0, f"x*(q_min) != 0 at q_min={qm!r}: {x!r}"
        # Away from the floor the factored form agrees with the algebraic one.
        q = rng.uniform(0.1, 2.0, 100_000)
        a = equilibrium_investment(q, 0.1, LAM)
        b = np.sqrt(LAM * (q**2 - 0.1**2))
        assert np.nanmax(np.abs(a - b)) < 1e-12

    def test_positivity_above_q_min(self) -> None:
        """x*(q) > 0 for all q > q_min."""
        x_star = equilibrium_investment(Q_RANGE, Q_MIN, LAM)
        assert np.all(x_star > 0)

    def test_monotonically_increasing(self) -> None:
        """x*(q) is strictly increasing in q: higher quality invests more."""
        x_star = equilibrium_investment(Q_RANGE, Q_MIN, LAM)
        diffs = np.diff(x_star)
        assert np.all(diffs > 0)

    def test_concavity(self) -> None:
        """x*(q) is concave: incremental investment narrows at higher q."""
        x_star = equilibrium_investment(Q_RANGE, Q_MIN, LAM)
        diffs = np.diff(x_star)
        second_diffs = np.diff(diffs)
        assert np.all(second_diffs < 0)

    def test_lambda_zero_returns_zero(self) -> None:
        """When lam = 0, no signaling incentive: x*(q) = 0 for all q."""
        x_star = equilibrium_investment(Q_RANGE, Q_MIN, 0.0)
        assert np.all(x_star == 0.0)

    def test_increases_with_lambda(self) -> None:
        """Higher lambda produces higher investment at every quality level."""
        q_test = np.linspace(Q_MIN + 0.01, Q_MAX, 50)
        x_low = equilibrium_investment(q_test, Q_MIN, 0.3)
        x_high = equilibrium_investment(q_test, Q_MIN, 0.7)
        assert np.all(x_high > x_low)

    def test_vectorization(self) -> None:
        """Function handles arrays and returns correct shapes."""
        q_arr = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
        x_star = equilibrium_investment(q_arr, Q_MIN, LAM)
        assert x_star.shape == q_arr.shape
        # Each element should match scalar call
        for i, q_val in enumerate(q_arr):
            expected = equilibrium_investment(q_val, Q_MIN, LAM)
            assert x_star[i] == pytest.approx(expected, rel=1e-14)

    def test_invalid_q_below_q_min(self) -> None:
        """Raises ValueError when any q < q_min."""
        with pytest.raises(ValueError, match="q_min"):
            equilibrium_investment(0.05, Q_MIN, LAM)

    def test_invalid_negative_lambda(self) -> None:
        """Raises ValueError for negative lambda."""
        with pytest.raises(ValueError, match="lam"):
            equilibrium_investment(1.0, Q_MIN, -0.1)

    def test_invalid_nonpositive_q_min(self) -> None:
        """Raises ValueError for q_min <= 0."""
        with pytest.raises(ValueError, match="q_min"):
            equilibrium_investment(1.0, 0.0, LAM)


# =====================================================================
# TestFreeRiderDeterrence (participation game; main-text within-group free-rider resolution)
# =====================================================================


class TestFreeRiderDeterrence:
    """The free-rider problem is resolved within the group by derivation.

    Encodes the central result: free-riding (deviating to x=0,
    inferred at q_min via the on-path schedule inversion) is deterred for every type
    q > q_min; the within-group differential beta_0/(SK) = s_W - C equals the
    population-mean deterrent; the sign is structural; and its magnitude scales
    with quality heterogeneity (the scope condition).
    """

    def test_deterrent_positive_above_q_min(self) -> None:
        """Delta_fr(q) = lam(q-q_min)^2/(2q) > 0 for all q > q_min."""
        d = free_rider_deterrent(Q_RANGE, LAM, Q_MIN)
        assert np.all(d > 0)

    def test_deterrent_zero_at_floor(self) -> None:
        """The lowest type is indifferent: Delta_fr(q_min) = 0."""
        assert free_rider_deterrent(Q_MIN, LAM, Q_MIN) == pytest.approx(0.0, abs=1e-15)

    def test_deterrent_closed_form(self) -> None:
        """Matches lam(q-q_min)^2/(2q) exactly (scalar)."""
        q = 1.3
        assert free_rider_deterrent(q, LAM, Q_MIN) == pytest.approx(
            LAM * (q - Q_MIN) ** 2 / (2.0 * q), rel=1e-12
        )

    def test_mean_deterrent_equals_beta0_over_SK(self) -> None:
        """E[Delta_fr(q)] over uniform quality equals s_W - C_model = beta_0/(SK)."""
        from signaling.price_equation import within_group_status_differential

        q = np.linspace(Q_MIN, Q_MAX, 400_001)
        mean_det = float(np.mean(free_rider_deterrent(q, LAM, Q_MIN)))
        s_W = within_group_status_differential(LAM, Q_MIN, Q_MAX)
        C = average_equilibrium_cost(LAM, Q_MIN, Q_MAX)
        assert mean_det == pytest.approx(s_W - C, rel=1e-4)
        assert s_W - C > 0  # within-group selection favours building

    def test_participation_recovers_sW_uniform(self) -> None:
        """Participation-game s_W equals the closed-form s_W at every lambda (uniform)."""
        from signaling.price_equation import within_group_status_differential

        for lam in (0.3, 0.5, 0.68, 1.0):
            eq = participation_equilibrium(lam, Q_MIN, Q_MAX)
            assert eq["s_W"] == pytest.approx(
                within_group_status_differential(lam, Q_MIN, Q_MAX), rel=1e-12
            )

    def test_full_participation_no_interior_threshold(self) -> None:
        """The within-group equilibrium is full participation: every type above the
        floor builds (the free-rider deterrent is positive), so there is no interior
        build threshold; a free-rider is inferred at q_min."""
        eq = participation_equilibrium(0.68, Q_MIN, Q_MAX)
        assert eq["full_participation"] is True
        assert eq["participation_rate"] == pytest.approx(1.0)
        assert eq["mu_N"] == pytest.approx(Q_MIN)
        assert eq["mean_deterrent"] > 0.0

    def test_lambda_zero_participation_indeterminate(self) -> None:
        """Exact boundary scope of the full-participation result: at
        lambda_W = 0 every type invests zero and Delta_fr(q) = 0 for all q,
        so there is no separation and no strict preference either way.
        Participation is indeterminate, not full."""
        eq = participation_equilibrium(0.0, Q_MIN, Q_MAX)
        assert eq["full_participation"] is False
        assert np.isnan(eq["participation_rate"])
        assert eq["mean_deterrent"] == pytest.approx(0.0)
        assert eq["s_W"] == pytest.approx(0.0)

    def test_lambda_negative_raises(self) -> None:
        """For lambda_W < 0 the separating schedule sqrt(lambda(q^2 - q_min^2))
        is not real for q > q_min: the signaling model is outside its domain
        and the function must refuse rather than report full participation."""
        with pytest.raises(ValueError, match="lambda_W >= 0"):
            participation_equilibrium(-0.1, Q_MIN, Q_MAX)

    def test_scope_vanishes_only_at_floor(self) -> None:
        """beta_0/(SK) is an expected square: positive for any population with mass
        above the floor, vanishing only as the population concentrates at q_min. A
        high-but-narrow population still has a large deterrent -- it is the gap from
        the floor, not heterogeneity per se, that matters."""
        rng = np.random.default_rng(2)
        at_floor = Q_MIN + 0.02 * rng.random(1_000_000)      # concentrated at q_min
        high_narrow = 1.0 + 0.05 * rng.random(1_000_000)     # homogeneous, high quality
        d_floor = participation_differential_sampled(0.68, at_floor, Q_MIN, Q_MAX)["mean_deterrent"]
        d_high = participation_differential_sampled(0.68, high_narrow, Q_MIN, Q_MAX)["mean_deterrent"]
        assert d_floor == pytest.approx(0.0, abs=5e-3)  # vanishes at the floor
        assert d_high > 0.2  # large despite low spread

    def test_sampled_uniform_matches_closed_form(self) -> None:
        """Sampled uniform quality reproduces the closed-form within-group differential."""
        rng = np.random.default_rng(0)
        q = Q_MIN + (Q_MAX - Q_MIN) * rng.random(2_000_000)
        sampled = participation_differential_sampled(0.68, q, Q_MIN, Q_MAX)
        closed = participation_equilibrium(0.68, Q_MIN, Q_MAX)
        assert sampled["mean_deterrent"] == pytest.approx(closed["mean_deterrent"], rel=3e-3)
        assert sampled["s_W"] == pytest.approx(closed["s_W"], rel=3e-3)

    def test_sampled_right_skew_increases_differential(self) -> None:
        """Right-skewed quality (most members high) STRENGTHENS the resolution: the
        free-rider differential is larger than uniform, not smaller (a free-rider has
        more to lose by being read at the floor)."""
        rng = np.random.default_rng(1)
        q_skew = Q_MIN + (Q_MAX - Q_MIN) * rng.beta(5, 2, 2_000_000)
        skew = participation_differential_sampled(0.68, q_skew, Q_MIN, Q_MAX)
        uniform = participation_equilibrium(0.68, Q_MIN, Q_MAX)
        assert skew["mean_deterrent"] > uniform["mean_deterrent"]

    def test_sampled_C_and_sW_are_sample_means(self) -> None:
        """For a nonuniform sample the returned cost and status differential
        must be the SAMPLE means, not the uniform closed forms, so the exact
        identity s_W - C = E[Delta_fr] holds for the sample. Counterexample:
        q = (0.2, 0.2, 2, 2, 2) at lambda_W =
        0.68 gives sample-correct C = 0.42738 and s_W = 0.8024 (the uniform
        closed form would return C = 0.35164, breaking the identity)."""
        q = np.array([0.2, 0.2, 2.0, 2.0, 2.0])
        res = participation_differential_sampled(0.68, q, Q_MIN, Q_MAX)
        C_expected = float(0.68 * np.mean((q ** 2 - Q_MIN ** 2) / (2.0 * q)))
        sW_expected = float(0.68 * (q.mean() - Q_MIN))
        assert res["C_model"] == pytest.approx(C_expected, abs=1e-12)
        assert res["C_model"] == pytest.approx(0.42738, abs=1e-5)
        assert res["s_W"] == pytest.approx(sW_expected, abs=1e-12)
        assert res["s_W"] == pytest.approx(0.8024, abs=1e-5)
        # Exact identity for the sample: s_W - C = mean deterrent.
        assert res["s_W"] - res["C_model"] == pytest.approx(
            res["mean_deterrent"], abs=1e-12
        )
        # The uniform closed form is still available, explicitly labeled.
        assert res["C_model_uniform"] == pytest.approx(0.351639, abs=1e-5)


# =====================================================================
# TestMultiPeriodSpence (depreciation necessity; main-text signal-depreciation section)
# =====================================================================


class TestMultiPeriodSpence:
    """Positive informational depreciation delta>0 is necessary for an inherited
    monument to remain an honest signal of the current holder's quality. The
    maintenance cost (delta M)^2/(2q) supplies the single-crossing the inherited
    (sunk) construction cost cannot.
    """

    def test_cross_partial_negative_for_positive_delta(self) -> None:
        """d2 c_maint/dM dq = -delta^2 M/q^2 < 0 when delta>0 (single-crossing holds)."""
        result = verify_multi_period_spence_condition()
        assert result["is_negative_for_positive_delta"]

    def test_cross_partial_vanishes_at_delta_zero(self) -> None:
        """At delta=0 maintenance is free and the cross-partial is exactly 0:
        single-crossing fails, so an inherited stock cannot separate the holder."""
        result = verify_multi_period_spence_condition()
        assert result["cross_partial_at_delta_zero"] == 0

    def test_cross_partial_symbolic_form(self) -> None:
        """Cross-partial equals -delta^2 M / q^2 symbolically."""
        result = verify_multi_period_spence_condition()
        M, q, delta = sp.symbols("M q delta", positive=True)
        expected = -(delta**2) * M / q**2
        assert sp.simplify(result["cross_partial"] - expected) == 0

    def test_maintenance_cost_increases_with_delta(self) -> None:
        """Holding stock against faster decay costs more."""
        c_low = maintenance_cost(10.0, 1.0, 0.05)
        c_high = maintenance_cost(10.0, 1.0, 0.20)
        assert c_high > c_low > 0

    def test_maintenance_cost_decreases_with_quality(self) -> None:
        """Higher-quality holders maintain the signal more cheaply (single-crossing)."""
        assert maintenance_cost(10.0, 2.0, 0.1) < maintenance_cost(10.0, 0.5, 0.1)

    def test_maintenance_cost_zero_at_delta_zero(self) -> None:
        """No depreciation -> free to hold an inherited stock (the failure mode)."""
        assert maintenance_cost(10.0, 1.0, 0.0) == pytest.approx(0.0, abs=1e-15)

    def test_maintenance_cost_closed_form(self) -> None:
        """c_maint = (delta M)^2 / (2 q)."""
        assert maintenance_cost(8.0, 1.5, 0.1) == pytest.approx(
            (0.1 * 8.0) ** 2 / (2 * 1.5), rel=1e-12
        )


# =====================================================================
# TestEquilibriumFitness
# =====================================================================


class TestEquilibriumFitness:
    """Verify properties of equilibrium fitness w*(q)."""

    def test_exceeds_baseline_quality(self) -> None:
        """w*(q) > q for all q >= q_min when lam > 0."""
        w_star = equilibrium_fitness(Q_RANGE, Q_MIN, LAM)
        assert np.all(w_star > Q_RANGE)

    def test_equals_quality_at_lambda_zero(self) -> None:
        """w*(q) = q when lam = 0 (no signaling benefits)."""
        w_star = equilibrium_fitness(Q_RANGE, Q_MIN, 0.0)
        np.testing.assert_allclose(w_star, Q_RANGE, rtol=1e-14)

    def test_consistency_with_components(self) -> None:
        """w*(q) = q - cost + lam*q (benefit) matches the closed form."""
        q_test = np.linspace(Q_MIN + 0.01, Q_MAX, 100)
        w_star = equilibrium_fitness(q_test, Q_MIN, LAM)
        cost = cost_at_equilibrium(q_test, Q_MIN, LAM)
        # Fitness should equal: q - cost + lam*q
        w_manual = q_test - cost + LAM * q_test
        np.testing.assert_allclose(w_star, w_manual, rtol=1e-12)


# =====================================================================
# TestFitnessGain
# =====================================================================


class TestFitnessGain:
    """Verify the equilibrium-vs-baseline DIAGNOSTIC Delta_w(q) > 0 for all
    q > q_min. (The manuscript demoted Delta_w to a partial-equilibrium
    diagnostic: the load-bearing Layer-1 result is the free-rider deterrent
    Delta_fr and the positional accounting, SI fitness-gain section; a
    positional reward cannot deliver Delta_w groupwide.)"""

    def test_positive_for_all_q_above_q_min(self) -> None:
        """Delta_w(q) > 0 for all q > q_min (diagnostic property; not the
        free-rider resolution, which rests on Delta_fr)."""
        dw = fitness_gain(Q_RANGE, Q_MIN, LAM)
        assert np.all(dw > 0)

    def test_increasing_in_q(self) -> None:
        """Higher-quality individuals benefit more from signaling."""
        dw = fitness_gain(Q_RANGE, Q_MIN, LAM)
        diffs = np.diff(dw)
        assert np.all(diffs > 0)

    def test_at_q_min_equals_lambda_q_min(self) -> None:
        """Delta_w(q_min) = lam*q_min/2 + lam*q_min/2 = lam*q_min."""
        dw = fitness_gain(Q_MIN, Q_MIN, LAM)
        expected = LAM * Q_MIN
        assert dw == pytest.approx(expected, rel=1e-12)

    def test_zero_at_lambda_zero(self) -> None:
        """No fitness gain when lam = 0."""
        dw = fitness_gain(Q_RANGE, Q_MIN, 0.0)
        assert np.all(dw == 0.0)


# =====================================================================
# TestFirstAndSecondOrder
# =====================================================================


class TestFirstAndSecondOrder:
    """Verify first-order and second-order conditions of the equilibrium."""

    def test_foc_holds_at_equilibrium(self) -> None:
        """The first-order condition x'(q) = lam*q/x(q) holds numerically."""
        # Start well above q_min to avoid the high-curvature region where
        # finite-difference approximation of the derivative is poor
        q_test = np.linspace(Q_MIN + 0.2, Q_MAX, 500)
        x_star = equilibrium_investment(q_test, Q_MIN, LAM)
        # Numerical derivative via central differences on a fine grid
        dq = q_test[1] - q_test[0]
        x_prime_numerical = np.gradient(x_star, dq)
        # Theoretical: x'(q) = lam*q / x(q)
        x_prime_theoretical = LAM * q_test / x_star
        # Exclude endpoints where gradient uses forward/backward differences
        np.testing.assert_allclose(
            x_prime_numerical[5:-5],
            x_prime_theoretical[5:-5],
            rtol=1e-4,
        )

    def test_soc_negative_at_equilibrium(self) -> None:
        """Second-order condition, derived INDEPENDENTLY via sympy from the
        reporting payoff U(r; q) = lam*r - x*(r)^2/(2q) rather than by
        restating the implementation's formula (a test-local copy of the
        same expression could not catch a shared error). Sympy gives
        U_rr = -lam/q < 0 (global concavity in the report), and the
        x-space curvature d2U/dx2 = -(q^2 - q_min^2)/q^3 matches the
        docstring formula on substitution."""
        import sympy as sp

        r_, q_, lam_, qmin_ = sp.symbols("r q lam q_min", positive=True)
        x_of_r = sp.sqrt(lam_ * (r_**2 - qmin_**2))
        U = lam_ * r_ - x_of_r**2 / (2 * q_)
        U_rr = sp.diff(U, r_, 2)
        assert sp.simplify(U_rr + lam_ / q_) == 0  # U_rr = -lam/q exactly
        q_test = np.linspace(Q_MIN + 0.01, Q_MAX, 200)
        soc = -(q_test**2 - Q_MIN**2) / q_test**3
        assert np.all(soc < 0)


# =====================================================================
# TestSymbolicNumericalCross
# =====================================================================


class TestSymbolicNumericalCross:
    """Cross-validate symbolic SymPy derivations against NumPy numerics."""

    def test_symbolic_matches_numerical(self) -> None:
        """SymPy expressions match NumPy at 20 random parameter triples."""
        eq_result = derive_equilibrium()
        fit_result = derive_equilibrium_fitness()
        syms = eq_result.symbols

        rng = np.random.default_rng(seed=42)
        for _ in range(20):
            q_val = rng.uniform(0.2, 3.0)
            qm_val = rng.uniform(0.01, q_val * 0.9)
            lam_val = rng.uniform(0.1, 2.0)

            subs = {syms["q"]: q_val, syms["q_min"]: qm_val, syms["lam"]: lam_val}

            # x*(q) comparison
            x_sympy = float(eq_result.x_star.subs(subs))
            x_numpy = equilibrium_investment(q_val, qm_val, lam_val)
            assert x_sympy == pytest.approx(x_numpy, rel=1e-12)

            # w*(q) comparison
            w_sympy = float(fit_result.w_star.subs(subs))
            w_numpy = equilibrium_fitness(q_val, qm_val, lam_val)
            assert w_sympy == pytest.approx(w_numpy, rel=1e-12)

            # Delta_w comparison
            dw_sympy = float(fit_result.delta_w.subs(subs))
            dw_numpy = fitness_gain(q_val, qm_val, lam_val)
            assert dw_sympy == pytest.approx(dw_numpy, rel=1e-12)


# =====================================================================
# TestGroupMonumentStock
# =====================================================================


class TestGroupMonumentStock:
    """Verify properties of the aggregate group signal M_g."""

    def test_single_member(self) -> None:
        """M_g for a single member equals that individual's x*(q)."""
        q_val = 1.0
        M_g = group_monument_stock(np.array([q_val]), Q_MIN, LAM)
        x_star = equilibrium_investment(q_val, Q_MIN, LAM)
        assert M_g == pytest.approx(x_star, rel=1e-14)

    def test_increases_with_group_size(self) -> None:
        """Larger groups produce higher M_g, holding mean quality fixed."""
        rng = np.random.default_rng(seed=123)
        q_small = rng.uniform(Q_MIN, Q_MAX, size=5)
        q_large = np.concatenate([q_small, rng.uniform(Q_MIN, Q_MAX, size=5)])
        M_small = group_monument_stock(q_small, Q_MIN, LAM)
        M_large = group_monument_stock(q_large, Q_MIN, LAM)
        assert M_large > M_small

    def test_increases_with_mean_quality(self) -> None:
        """Higher mean quality produces higher M_g for same group size."""
        q_low = np.array([0.3, 0.4, 0.5])
        q_high = np.array([1.0, 1.2, 1.5])
        M_low = group_monument_stock(q_low, Q_MIN, LAM)
        M_high = group_monument_stock(q_high, Q_MIN, LAM)
        assert M_high > M_low

    def test_zero_when_all_at_q_min(self) -> None:
        """M_g = 0 when all members have quality q_min."""
        qualities = np.full(5, Q_MIN)
        M_g = group_monument_stock(qualities, Q_MIN, LAM)
        assert M_g == pytest.approx(0.0, abs=1e-14)


# =====================================================================
# TestExpectedMonumentStock
# =====================================================================


class TestExpectedMonumentStock:
    """Verify properties of E[M_g] under different quality distributions."""

    def test_positive(self) -> None:
        """E[M_g] > 0 for lam > 0 and n > 0."""
        E_M = expected_monument_stock(10, Q_MIN, Q_MAX, LAM)
        assert E_M > 0

    def test_increases_with_n(self) -> None:
        """E[M_g] is proportional to group size n."""
        E_small = expected_monument_stock(5, Q_MIN, Q_MAX, LAM)
        E_large = expected_monument_stock(10, Q_MIN, Q_MAX, LAM)
        # Should be exactly 2x for iid draws
        assert E_large == pytest.approx(2 * E_small, rel=1e-10)

    def test_increases_with_lambda(self) -> None:
        """Higher lambda produces higher E[M_g]."""
        E_low = expected_monument_stock(10, Q_MIN, Q_MAX, 0.3)
        E_high = expected_monument_stock(10, Q_MIN, Q_MAX, 0.7)
        assert E_high > E_low

    def test_matches_monte_carlo(self) -> None:
        """E[M_g] from integration matches Monte Carlo estimate (rel=0.02)."""
        n = 12
        E_analytic = expected_monument_stock(n, Q_MIN, Q_MAX, LAM, "uniform")
        # Monte Carlo with many samples
        rng = np.random.default_rng(seed=2024)
        n_trials = 50000
        mc_M_g = np.zeros(n_trials)
        for i in range(n_trials):
            qualities = rng.uniform(Q_MIN, Q_MAX, size=n)
            mc_M_g[i] = group_monument_stock(qualities, Q_MIN, LAM)
        mc_mean = np.mean(mc_M_g)
        assert mc_mean == pytest.approx(E_analytic, rel=0.02)

    def test_zero_at_lambda_zero(self) -> None:
        """E[M_g] = 0 when lam = 0."""
        E_M = expected_monument_stock(10, Q_MIN, Q_MAX, 0.0)
        assert E_M == 0.0


# =====================================================================
# TestCostSensitivity
# =====================================================================


class TestCostSensitivity:
    """Verify sensitivity analysis across cost function forms."""

    def test_quadratic_ode_matches_closed_form(self) -> None:
        """ODE solution for quadratic cost matches closed form (rtol=1e-6)."""
        q_arr = np.linspace(Q_MIN + 0.01, Q_MAX, 100)
        q_ode, x_ode = cost_function_sensitivity(q_arr, Q_MIN, LAM, "quadratic")
        x_closed = equilibrium_investment(q_ode, Q_MIN, LAM)
        np.testing.assert_allclose(x_ode, x_closed, rtol=1e-10)

    def test_power_2_1_reduces_to_quadratic(self) -> None:
        """Power cost with a=2, b=1 matches quadratic (they are identical)."""
        q_arr = np.linspace(Q_MIN + 0.01, Q_MAX, 100)
        q_power, x_power = cost_function_sensitivity(
            q_arr, Q_MIN, LAM, "power", a=2.0, b=1.0
        )
        x_quad = equilibrium_investment(q_power, Q_MIN, LAM)
        np.testing.assert_allclose(x_power, x_quad, rtol=1e-6)

    def test_all_costs_give_monotonic_investment(self) -> None:
        """Equilibrium investment is monotonically increasing for all valid costs."""
        q_arr = np.linspace(Q_MIN + 0.01, Q_MAX, 200)
        for cost_type, params in [
            ("quadratic", {}),
            ("power", {"a": 3.0, "b": 0.5}),
            ("exponential", {}),
        ]:
            _, x_star = cost_function_sensitivity(q_arr, Q_MIN, LAM, cost_type, **params)
            diffs = np.diff(x_star)
            assert np.all(diffs > 0), f"Monotonicity failed for {cost_type}"


# =====================================================================
# TestBenefitSensitivity
# =====================================================================


class TestBenefitSensitivity:
    """Verify sensitivity analysis across benefit function forms."""

    def test_linear_matches_closed_form(self) -> None:
        """Linear benefit gives closed-form result."""
        q_arr = np.linspace(Q_MIN + 0.01, Q_MAX, 100)
        q_lin, x_lin = benefit_function_sensitivity(q_arr, Q_MIN, LAM, "linear")
        x_closed = equilibrium_investment(q_lin, Q_MIN, LAM)
        np.testing.assert_allclose(x_lin, x_closed, rtol=1e-10)

    def test_concave_lower_than_linear_at_high_q(self) -> None:
        """Concave benefit produces lower investment at high quality."""
        q_arr = np.linspace(Q_MIN + 0.01, Q_MAX, 200)
        _, x_lin = benefit_function_sensitivity(q_arr, Q_MIN, LAM, "linear")
        _, x_conc = benefit_function_sensitivity(
            q_arr, Q_MIN, LAM, "concave", gamma=0.5
        )
        # At high q, concave benefit should give lower investment
        # (diminishing returns compress investment at the top)
        assert x_conc[-1] < x_lin[-1]

    def test_concave_monotonic(self) -> None:
        """Concave benefit still produces monotonically increasing investment."""
        q_arr = np.linspace(Q_MIN + 0.01, Q_MAX, 200)
        _, x_conc = benefit_function_sensitivity(
            q_arr, Q_MIN, LAM, "concave", gamma=0.5
        )
        diffs = np.diff(x_conc)
        assert np.all(diffs > 0)


# =====================================================================
# TestEdgeCases
# =====================================================================


class TestEdgeCases:
    """Test boundary conditions and extreme parameter values."""

    def test_q_equals_q_min(self) -> None:
        """At q = q_min: x* = 0, w* = q_min*(1 + lam), Delta_w = lam*q_min."""
        x = equilibrium_investment(Q_MIN, Q_MIN, LAM)
        w = equilibrium_fitness(Q_MIN, Q_MIN, LAM)
        dw = fitness_gain(Q_MIN, Q_MIN, LAM)
        assert x == pytest.approx(0.0, abs=1e-15)
        # w*(q_min) = q_min*(1 + lam/2) + lam*q_min/2 = q_min*(1 + lam)
        assert w == pytest.approx(Q_MIN * (1 + LAM), rel=1e-12)
        assert dw == pytest.approx(LAM * Q_MIN, rel=1e-12)

    def test_q_equals_q_max(self) -> None:
        """At q = q_max, all functions return finite positive values."""
        x = equilibrium_investment(Q_MAX, Q_MIN, LAM)
        w = equilibrium_fitness(Q_MAX, Q_MIN, LAM)
        dw = fitness_gain(Q_MAX, Q_MIN, LAM)
        assert np.isfinite(x) and x > 0
        assert np.isfinite(w) and w > Q_MAX
        assert np.isfinite(dw) and dw > 0

    def test_tiny_lambda(self) -> None:
        """Very small lambda produces very small but positive investment."""
        tiny_lam = 1e-10
        x = equilibrium_investment(1.0, Q_MIN, tiny_lam)
        assert x > 0
        assert x < 1e-4  # Should be very small

    def test_large_lambda(self) -> None:
        """Large lambda produces large investment; functions remain finite."""
        big_lam = 100.0
        q_test = np.linspace(Q_MIN + 0.01, Q_MAX, 50)
        x = equilibrium_investment(q_test, Q_MIN, big_lam)
        w = equilibrium_fitness(q_test, Q_MIN, big_lam)
        dw = fitness_gain(q_test, Q_MIN, big_lam)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(w))
        assert np.all(np.isfinite(dw))

    def test_q_min_near_zero(self) -> None:
        """Small q_min does not cause numerical issues."""
        tiny_qmin = 1e-6
        q_test = np.array([1e-5, 0.01, 0.1, 1.0])
        x = equilibrium_investment(q_test, tiny_qmin, LAM)
        assert np.all(np.isfinite(x))
        assert np.all(x >= 0)

    def test_wide_quality_range(self) -> None:
        """Large q_max/q_min ratio does not cause numerical issues."""
        q_test = np.linspace(Q_MIN + 0.001, 100.0, 500)
        x = equilibrium_investment(q_test, Q_MIN, LAM)
        w = equilibrium_fitness(q_test, Q_MIN, LAM)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(w))
        # Monotonicity should still hold
        assert np.all(np.diff(x) > 0)


# =====================================================================
# TestSymbolicDerivations
# =====================================================================


class TestSymbolicDerivations:
    """Verify internal consistency of symbolic derivation results."""

    def test_equilibrium_ode_satisfied(self) -> None:
        """The derived x*(q) satisfies the ODE x' = lam*q/x."""
        result = derive_equilibrium()
        q = result.symbols["q"]
        q_min = result.symbols["q_min"]
        lam = result.symbols["lam"]
        # x' should equal lam*q/x*
        ratio = sp.simplify(result.x_star_prime - lam * q / result.x_star)
        assert ratio == 0

    def test_fitness_gain_sum_of_positive_terms(self) -> None:
        """Delta_w = lam*q/2 + lam*q_min^2/(2*q) is a sum of positive terms."""
        result = derive_equilibrium_fitness()
        assert result.delta_w_positive

    def test_aggregate_expected_M_g_proportional_to_n(self) -> None:
        """E[M_g] is proportional to n."""
        agg = derive_aggregate_signal()
        n = agg["symbols"]["n"]
        # E[M_g] / n should not contain n
        ratio = sp.simplify(agg["expected_M_g"] / n)
        assert n not in ratio.free_symbols


# =====================================================================
# TestReceiverInference
# =====================================================================


class TestReceiverInference:
    """Verify the receiver inference function q_hat(x) = sqrt(x^2/lam + q_min^2)."""

    def test_roundtrip_with_equilibrium_investment(self) -> None:
        """receiver_inference(x*(q), ...) == q for all q > q_min."""
        q_test = np.linspace(Q_MIN + 0.01, Q_MAX, 100)
        x_star = equilibrium_investment(q_test, Q_MIN, LAM)
        q_hat = receiver_inference(x_star, Q_MIN, LAM)
        np.testing.assert_allclose(q_hat, q_test, rtol=1e-12)

    def test_zero_investment_gives_q_min(self) -> None:
        """receiver_inference(0, ...) == q_min."""
        q_hat = receiver_inference(0.0, Q_MIN, LAM)
        assert q_hat == pytest.approx(Q_MIN, rel=1e-14)

    def test_clamping_at_q_max(self) -> None:
        """Inferred quality is clamped at q_max for large x."""
        # Investment above x*(q_max) should clamp at q_max
        x_above = equilibrium_investment(Q_MAX, Q_MIN, LAM) * 1.5
        q_hat = receiver_inference(x_above, Q_MIN, LAM, q_max=Q_MAX)
        assert q_hat == pytest.approx(Q_MAX, rel=1e-14)

    def test_monotonically_increasing(self) -> None:
        """q_hat(x) is monotonically increasing in x."""
        x_range = np.linspace(0.0, 3.0, 200)
        q_hat = receiver_inference(x_range, Q_MIN, LAM)
        diffs = np.diff(q_hat)
        assert np.all(diffs >= 0)


# =====================================================================
# TestFitnessAtDeviation
# =====================================================================


class TestFitnessAtDeviation:
    """Verify fitness w(x, q) for arbitrary investment x."""

    def test_at_equilibrium_matches_equilibrium_fitness(self) -> None:
        """w(x*(q), q) == w*(q) for all q."""
        q_test = np.linspace(Q_MIN + 0.01, Q_MAX, 50)
        for q_val in q_test:
            x_star = equilibrium_investment(q_val, Q_MIN, LAM)
            w_dev = fitness_at_deviation(x_star, q_val, Q_MIN, LAM)
            w_eq = equilibrium_fitness(q_val, Q_MIN, LAM)
            assert w_dev == pytest.approx(w_eq, rel=1e-12)

    def test_equilibrium_is_global_maximum(self) -> None:
        """w(x, q) <= w(x*(q), q) for all x, i.e. equilibrium is optimal."""
        q_test = [0.3, 0.7, 1.2, 1.8]
        for q_val in q_test:
            x_star = equilibrium_investment(q_val, Q_MIN, LAM)
            w_star = fitness_at_deviation(x_star, q_val, Q_MIN, LAM)
            # Dense grid of deviations
            x_grid = np.linspace(0.0, x_star * 3 + 0.5, 1000)
            w_grid = fitness_at_deviation(x_grid, q_val, Q_MIN, LAM)
            assert np.all(w_grid <= w_star + 1e-10), (
                f"Found deviation exceeding equilibrium for q={q_val}"
            )

    def test_no_profitable_mimicry(self) -> None:
        """w(x*(q'), q) < w(x*(q), q) for q != q': no type benefits from mimicry."""
        q_values = np.array([0.3, 0.7, 1.2, 1.8])
        for q_true in q_values:
            w_honest = fitness_at_deviation(
                equilibrium_investment(q_true, Q_MIN, LAM),
                q_true, Q_MIN, LAM,
            )
            for q_mimic in q_values:
                if q_mimic == q_true:
                    continue
                w_mimic = fitness_at_deviation(
                    equilibrium_investment(q_mimic, Q_MIN, LAM),
                    q_true, Q_MIN, LAM,
                )
                assert w_honest > w_mimic, (
                    f"Profitable mimicry: q={q_true} benefits from mimicking q={q_mimic}"
                )

    def test_fitness_decreasing_away_from_equilibrium(self) -> None:
        """Fitness decreases in both directions away from x*(q)."""
        q_val = 1.0
        x_star = equilibrium_investment(q_val, Q_MIN, LAM)
        w_star = fitness_at_deviation(x_star, q_val, Q_MIN, LAM)
        # Below equilibrium
        x_below = np.linspace(0.0, x_star * 0.99, 50)
        w_below = fitness_at_deviation(x_below, q_val, Q_MIN, LAM)
        assert np.all(w_below < w_star)
        # Above equilibrium
        x_above = np.linspace(x_star * 1.01, x_star * 3, 50)
        w_above = fitness_at_deviation(x_above, q_val, Q_MIN, LAM)
        assert np.all(w_above < w_star)


# =====================================================================
# TestPoolingFitness
# =====================================================================


class TestPoolingFitness:
    """Verify pooling equilibrium fitness computation."""

    def test_basic_computation(self) -> None:
        """w_pool = q - x_pool^2/(2q) + lam*q_mean."""
        q_val = 1.0
        x_pool = 0.5
        q_mean = 0.8
        lam_val = 0.5
        expected = q_val - x_pool**2 / (2 * q_val) + lam_val * q_mean
        result = pooling_fitness(q_val, x_pool, q_mean, lam_val)
        assert result == pytest.approx(expected, rel=1e-14)

    def test_high_types_prefer_separating(self) -> None:
        """w*(q) > w_pool(q) for q well above E[q]."""
        q_mean = (Q_MIN + Q_MAX) / 2
        x_pool = equilibrium_investment(q_mean, Q_MIN, LAM)
        # High-quality types: upper third of the range
        q_high = np.linspace(q_mean + 0.3, Q_MAX, 50)
        w_sep = equilibrium_fitness(q_high, Q_MIN, LAM)
        w_pool = pooling_fitness(q_high, x_pool, q_mean, LAM)
        assert np.all(w_sep > w_pool), (
            "High types should prefer separating over pooling"
        )


# =====================================================================
# TestExpectedFitnessGain
# =====================================================================


class TestExpectedFitnessGain:
    """Verify E[Delta_w(q)] = B(lambda)."""

    def test_positive_for_positive_lambda(self) -> None:
        """E[Delta_w] > 0 for lam > 0."""
        B = expected_fitness_gain(Q_MIN, Q_MAX, LAM)
        assert B > 0

    def test_increases_with_lambda(self) -> None:
        """B(lambda) is increasing in lambda."""
        B_low = expected_fitness_gain(Q_MIN, Q_MAX, 0.3)
        B_high = expected_fitness_gain(Q_MIN, Q_MAX, 0.7)
        assert B_high > B_low

    def test_uniform_matches_analytical(self) -> None:
        r"""E[Delta_w] for uniform matches lam/2 * E[q] + lam*q_min^2/2 * E[1/q].

        Delta_w(q) = lam*q/2 + lam*q_min^2/(2*q), so
        E[Delta_w] = lam/2 * E[q] + lam*q_min^2/2 * E[1/q]

        For uniform on [q_min, q_max]:
          E[q] = (q_min + q_max) / 2
          E[1/q] = ln(q_max/q_min) / (q_max - q_min)
        """
        lam_val = LAM
        E_q = (Q_MIN + Q_MAX) / 2
        E_inv_q = np.log(Q_MAX / Q_MIN) / (Q_MAX - Q_MIN)
        analytical = lam_val / 2 * E_q + lam_val * Q_MIN**2 / 2 * E_inv_q
        numerical = expected_fitness_gain(Q_MIN, Q_MAX, lam_val, "uniform")
        assert numerical == pytest.approx(analytical, rel=1e-10)


# =====================================================================
# Signal depreciation and reinvestment
# =====================================================================


from signaling.layer1 import (
    effective_monument_stock,
    monument_stock_trajectory,
    signal_half_life,
)


class TestEffectiveMonumentStock:
    """Tests for steady-state effective monument stock under depreciation."""

    def test_inverse_delta(self):
        """M_g* = I_g / delta: effective stock is decreasing in delta."""
        I_g = 5.0
        M_low_delta = effective_monument_stock(I_g, 0.05)
        M_high_delta = effective_monument_stock(I_g, 0.20)
        assert M_low_delta > M_high_delta

    def test_recovers_flow_at_delta_one(self):
        """M_g* = I_g when delta = 1: full depreciation means only flow matters."""
        I_g = 3.7
        M = effective_monument_stock(I_g, 1.0)
        assert M == pytest.approx(I_g)

    def test_exact_value(self):
        """M_g* = I_g / delta for specific values."""
        assert effective_monument_stock(10.0, 0.1) == pytest.approx(100.0)
        assert effective_monument_stock(5.0, 0.5) == pytest.approx(10.0)

    def test_array_input(self):
        """Works with array inputs."""
        I_g = np.array([1.0, 2.0, 5.0])
        result = effective_monument_stock(I_g, 0.1)
        expected = np.array([10.0, 20.0, 50.0])
        np.testing.assert_allclose(result, expected)

    def test_invalid_delta_zero(self):
        """delta = 0 raises ValueError (undefined steady state)."""
        with pytest.raises(ValueError):
            effective_monument_stock(5.0, 0.0)

    def test_invalid_delta_negative(self):
        """Negative delta raises ValueError."""
        with pytest.raises(ValueError):
            effective_monument_stock(5.0, -0.1)

    def test_invalid_delta_above_one(self):
        """delta > 1 raises ValueError."""
        with pytest.raises(ValueError):
            effective_monument_stock(5.0, 1.5)


class TestMonumentStockTrajectory:
    """Tests for monument stock trajectory M_g(t)."""

    def test_approaches_steady_state(self):
        """M_g(T) -> I_g / delta for large T."""
        I_g = 5.0
        delta = 0.1
        T = 200
        traj = monument_stock_trajectory(I_g, delta, T)
        steady_state = I_g / delta
        assert traj[-1] == pytest.approx(steady_state, rel=1e-4)

    def test_starts_at_M_0(self):
        """Trajectory starts at the specified initial condition."""
        traj = monument_stock_trajectory(5.0, 0.1, 10, M_0=3.0)
        assert traj[0] == 3.0

    def test_monotonically_increasing_from_zero(self):
        """When M_0 = 0 and I_g > 0, trajectory is monotonically increasing."""
        traj = monument_stock_trajectory(5.0, 0.1, 50)
        diffs = np.diff(traj)
        assert np.all(diffs > 0)

    def test_length(self):
        """Output has T + 1 elements."""
        T = 25
        traj = monument_stock_trajectory(5.0, 0.1, T)
        assert len(traj) == T + 1


class TestSignalHalfLife:
    """Tests for signal half-life computation."""

    def test_positive(self):
        """Half-life is positive for delta in (0, 1)."""
        for d in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]:
            assert signal_half_life(d) > 0

    def test_decreasing_in_delta(self):
        """Higher depreciation gives shorter half-life."""
        assert signal_half_life(0.05) > signal_half_life(0.10)
        assert signal_half_life(0.10) > signal_half_life(0.20)

    def test_known_value(self):
        """For delta = 0.5, half-life = 1 period (stock halves each period)."""
        assert signal_half_life(0.5) == pytest.approx(1.0, rel=1e-10)

    def test_invalid_delta_zero(self):
        """delta = 0 raises ValueError."""
        with pytest.raises(ValueError):
            signal_half_life(0.0)

    def test_invalid_delta_one(self):
        """delta = 1 raises ValueError (ln(0) undefined)."""
        with pytest.raises(ValueError):
            signal_half_life(1.0)


class TestMaintenanceToNewRatio:
    """Tests for the maintenance-to-new-construction ratio R(M_g).

    R(M_g) = delta*M_g / (I_g - delta*M_g) is derived within the geometric
    depreciation model (the main-text renovation-ratio equation): it rises monotonically
    with accumulated stock and diverges at steady state, where all investment
    is maintenance and net new construction vanishes.
    """

    def test_zero_stock_zero_ratio(self):
        """At M_g = 0 all investment is new construction, so R = 0."""
        assert float(maintenance_to_new_ratio(0.0, 5.0, 0.1)) == pytest.approx(0.0)

    def test_rises_monotonically_with_stock(self):
        """R(M_g) increases as accumulated stock grows."""
        I_g, delta = 5.0, 0.1
        M_vals = np.linspace(0.0, 0.9 * I_g / delta, 50)  # within the growth regime
        R = np.array([float(maintenance_to_new_ratio(m, I_g, delta)) for m in M_vals])
        assert np.all(np.diff(R) > 0)

    def test_known_value(self):
        """R = delta*M_g/(I_g - delta*M_g): I_g=5, delta=0.1, M_g=20 -> 2/3."""
        assert float(maintenance_to_new_ratio(20.0, 5.0, 0.1)) == pytest.approx(2.0 / 3.0)

    def test_diverges_at_steady_state(self):
        """At M_g = M_g* = I_g/delta, net new construction vanishes, R -> inf."""
        I_g, delta = 5.0, 0.1
        assert np.isinf(float(maintenance_to_new_ratio(I_g / delta, I_g, delta)))

    def test_invalid_params(self):
        """delta out of range or I_g <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            maintenance_to_new_ratio(1.0, 5.0, 0.0)
        with pytest.raises(ValueError):
            maintenance_to_new_ratio(1.0, 0.0, 0.1)


# =====================================================================
# TestAverageEquilibriumCost
# =====================================================================
class TestAverageEquilibriumCost:
    """Tests for C_model(lambda) = E[c(x*(q), q)] (manuscript Eq. 21).

    Verifies that the analytical closed form matches numerical integration,
    and that the lambda_W inverse returns the correct value.
    """

    def test_zero_lambda_zero_cost(self):
        """At lambda = 0 no one signals, so average cost is zero."""
        assert average_equilibrium_cost(0.0, DEFAULT_Q_MIN, DEFAULT_Q_MAX) == 0.0

    def test_linear_in_lambda(self):
        """C_model is linear in lambda under quadratic cost."""
        c1 = average_equilibrium_cost(0.3, DEFAULT_Q_MIN, DEFAULT_Q_MAX)
        c2 = average_equilibrium_cost(0.6, DEFAULT_Q_MIN, DEFAULT_Q_MAX)
        assert c2 == pytest.approx(2 * c1, rel=1e-10)

    def test_matches_numerical_integration(self):
        """Analytical closed form matches quad integration of cost_at_equilibrium."""
        from scipy.integrate import quad
        for lam in (0.10, 0.30, 0.68, 1.00):
            analytical = average_equilibrium_cost(lam, DEFAULT_Q_MIN, DEFAULT_Q_MAX)
            numerical, _ = quad(
                lambda q: float(cost_at_equilibrium(q, DEFAULT_Q_MIN, lam)),
                DEFAULT_Q_MIN, DEFAULT_Q_MAX,
            )
            numerical /= DEFAULT_Q_MAX - DEFAULT_Q_MIN
            assert analytical == pytest.approx(numerical, rel=1e-8), f"mismatch at lambda={lam}"

    def test_reproduces_manuscript_values(self):
        """Verifies C_model(0.30) ~ 0.155 and C_model(0.68) ~ 0.352 claimed in the main text."""
        assert average_equilibrium_cost(0.30, 0.1, 2.0) == pytest.approx(0.1551, abs=1e-3)
        assert average_equilibrium_cost(0.68, 0.1, 2.0) == pytest.approx(0.3516, abs=1e-3)

    def test_B_minus_Cmodel_equals_informational_rent(self):
        """B(lambda) - C_model(lambda) = lambda * q_min^2 * ln(q_max/q_min) / (q_max - q_min).

        This identity is stated in the manuscript just after Eq. 21.
        """
        from signaling.price_equation import average_signaling_benefit
        q_min, q_max = 0.1, 2.0
        for lam in (0.1, 0.3, 0.68, 1.0):
            B = average_signaling_benefit(lam, q_min, q_max)
            C_m = average_equilibrium_cost(lam, q_min, q_max)
            rent = lam * q_min**2 * np.log(q_max / q_min) / (q_max - q_min)
            assert B - C_m == pytest.approx(rent, rel=1e-10)


class TestLambdaWforC:
    """Tests for the inverse: lambda_W such that C_model(lambda_W) = target_C."""

    def test_round_trip(self):
        """lambda_W_for_C(average_equilibrium_cost(lam)) == lam."""
        for lam in (0.1, 0.3, 0.68, 1.2):
            C = average_equilibrium_cost(lam, DEFAULT_Q_MIN, DEFAULT_Q_MAX)
            recovered = lambda_W_for_C(C, DEFAULT_Q_MIN, DEFAULT_Q_MAX)
            assert recovered == pytest.approx(lam, rel=1e-10)

    def test_empirical_anchor(self):
        """Manuscript claims lambda_W ~ 0.68 reproduces empirical C = 0.35."""
        lam = lambda_W_for_C(0.35, 0.1, 2.0)
        assert lam == pytest.approx(0.68, abs=0.01)


class TestBeliefRobustnessCutoff:
    """Exact sensitivity of the mean free-rider deterrent to the perceived
    non-builder quality mu_N (positivity does not extend up to E[q]; the
    exact cutoff is mu_N* = E[q] - C_model/lambda_W)."""

    def test_mean_deterrent_cutoff_exact(self):
        from signaling.layer1 import average_equilibrium_cost
        lam, q_min, q_max = 0.68, 0.1, 2.0
        Eq = (q_min + q_max) / 2.0
        C = average_equilibrium_cost(lam, q_min, q_max)
        mu_star = Eq - C / lam
        # The literal pins below are the independent anchors (hand-derived
        # from the closed forms); the local mean_det() is convenience only.
        assert mu_star == pytest.approx(0.5328835, abs=1e-6)
        # mean deterrent E[Delta_fr(q; mu_N)] = lam*(E[q] - mu_N) - C
        def mean_det(mu_N):
            return lam * (Eq - mu_N) - C
        assert mean_det(q_min) == pytest.approx(0.2944, abs=1e-3)   # floor reading
        assert mean_det(mu_star) == pytest.approx(0.0, abs=1e-12)   # exact zero
        assert mean_det(0.8) == pytest.approx(-0.1816, abs=1e-3)    # negative above

    def test_type_level_partial_participation_above_floor(self):
        """Any mu_N > q_min makes the deterrent negative for low types:
        full participation is exactly the floor-reading result."""
        from signaling.layer1 import cost_at_equilibrium
        lam, q_min = 0.68, 0.1
        mu_N = 0.5
        q = 0.3  # a type below mu_N
        det = lam * (q - mu_N) - float(cost_at_equilibrium(q, q_min, lam))
        assert det < 0.0
