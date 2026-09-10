"""Tests for the assembled multilevel Price equation.

Verifies theoretical properties of the reassembled framework:
- At the matched anchor the derived threshold sits just below the same-C
  baseline (0.477 vs 0.498), holding close to and straddling the standard
  ~0.50 baseline over the (C, n) range rather than lying far below it
- B(lambda) > 0 (a Layer-1 diagnostic; it does not enter the threshold)
- The within-group term is positive when s_W > C (positional rule)
- Fitness advantage changes sign at sigma*
- Components reduce to initial model values under appropriate choices
- Phase space produces correct structure
"""

import numpy as np
import pytest

from signaling.price_equation import (
    average_signaling_benefit,
    average_signaling_benefit_numerical,
    between_group_selection,
    classify_equilibrium,
    critical_threshold_sigma_star,
    fitness_advantage,
    group_fitness_nonsignaler,
    group_fitness_signaler,
    individual_fitness,
    initial_model_sigma_star,
    phase_space,
    sigma_star_comparison,
    sigma_star_self_consistent,
    stability_analysis,
    stability_analysis_self_consistent,
    within_group_selection,
)


class TestSigmaStarSelfConsistentDelta:
    """Verify the delta parameter exposure on sigma_star_self_consistent.
    At delta = 0, must match the no-delta call. Within positive delta,
    sigma* should be monotonically increasing (less accumulation ->
    weaker feedback -> higher threshold)."""

    def test_delta_zero_matches_default(self):
        """sigma_star_self_consistent(..., delta=0.0) equals the no-delta
        default (delta defaults to 0.0)."""
        r_default = sigma_star_self_consistent(
            lambda_W=0.68, mode="multiplicative",
        )
        r_zero = sigma_star_self_consistent(
            lambda_W=0.68, mode="multiplicative", delta=0.0,
        )
        assert r_zero["sigma_star"] == pytest.approx(
            r_default["sigma_star"], rel=1e-6,
        )

    def test_delta_monotonic_within_positive_range(self):
        """Within the positive delta regime, sigma* is monotonically
        increasing in delta (the manuscript's depreciation-raises-threshold
        claim, applied at the empirical anchor lambda_W = 0.68)."""
        deltas = [0.05, 0.10, 0.15, 0.20, 0.30]
        sigma_stars = []
        for d in deltas:
            r = sigma_star_self_consistent(
                lambda_W=0.68, mode="multiplicative", delta=d,
            )
            sigma_stars.append(r["sigma_star"])
        for a, b in zip(sigma_stars[:-1], sigma_stars[1:]):
            assert b > a, (
                f"sigma* should increase with delta within positive range; "
                f"got {list(zip(deltas, sigma_stars))}"
            )


# =====================================================================
# Average signaling benefit B(lambda)
# =====================================================================


class TestAverageSignalingBenefit:
    """Tests for B(lambda) = E[Delta_w(q)]."""

    def test_positive(self):
        """B(lambda) > 0 for lambda > 0.

        This is the central result of Layer 1: within-group selection
        can favor signaling because social rewards exceed costs.
        """
        B = average_signaling_benefit(0.5)
        assert B > 0

    def test_zero_for_zero_lambda(self):
        """B(0) = 0: no signaling benefit without social rewards."""
        B = average_signaling_benefit(0.0)
        assert B == pytest.approx(0.0)

    def test_increasing_in_lambda(self):
        """dB/dlambda > 0: more social reward means more benefit."""
        B1 = average_signaling_benefit(0.3)
        B2 = average_signaling_benefit(0.5)
        B3 = average_signaling_benefit(1.0)
        assert B3 > B2 > B1

    def test_linear_in_lambda(self):
        """B(lambda) is exactly proportional to lambda."""
        B1 = average_signaling_benefit(0.5)
        B2 = average_signaling_benefit(1.0)
        assert B2 == pytest.approx(2.0 * B1, rel=1e-8)

    def test_analytical_matches_numerical(self):
        """Analytical formula matches numerical integration.

        This verifies the derivation: the analytical formula for E[Delta_w]
        over a uniform distribution agrees with direct numerical integration
        of the fitness gain function.
        """
        for lam in [0.1, 0.5, 1.0, 2.0]:
            B_analytical = average_signaling_benefit(lam)
            B_numerical = average_signaling_benefit_numerical(lam)
            assert B_analytical == pytest.approx(B_numerical, rel=1e-4)

    def test_sensitivity_to_q_max(self):
        """B increases with q_max (higher quality ceiling raises average benefit).

        The dominant term in B is (lambda/2)*E[q], which increases with q_max.
        """
        B_low = average_signaling_benefit(0.5, q_min=0.1, q_max=1.0)
        B_high = average_signaling_benefit(0.5, q_min=0.1, q_max=3.0)
        assert B_high > B_low


# =====================================================================
# Group fitness
# =====================================================================


class TestGroupFitness:
    """Tests for group-level fitness comparison."""

    def test_signaler_fitness_basic(self):
        """Signaler fitness is positive for moderate parameters."""
        w = group_fitness_signaler(0.2, lam=0.68, alpha_eff=0.35, r=0.75)
        assert w > 0

    def test_nonsignaler_fitness_basic(self):
        """Non-signaler fitness is positive for moderate sigma."""
        w = group_fitness_nonsignaler(0.2, beta_eff=0.87)
        assert w > 0

    def test_signaler_advantage_high_sigma(self):
        """Monument builders have higher fitness above the threshold (high sigma).

        When environmental uncertainty is high, the survival advantage from
        denser networks (lower alpha) outweighs the reproductive cost C.
        """
        w_MB = group_fitness_signaler(0.8, lam=0.68, alpha_eff=0.35, r=0.75)
        w_NB = group_fitness_nonsignaler(0.8, beta_eff=0.87)
        assert w_MB > w_NB

    def test_nonsignaler_advantage_low_sigma(self):
        """Non-builders have higher fitness at very low sigma.

        When crises are rare, the survival advantage is small,
        and the reproductive cost C dominates.
        """
        w_MB = group_fitness_signaler(0.01, lam=0.68, alpha_eff=0.35, r=0.75)
        w_NB = group_fitness_nonsignaler(0.01, beta_eff=0.87)
        assert w_NB > w_MB

    # NOTE: the former "B(lambda) increases signaler fitness" / "B shifts the
    # threshold down" tests are removed: under the positional model the
    # within-group reward nets out of the group fitness and does not move the
    # between-group threshold. The within-group behaviour (beta_0 > 0, the
    # free-rider resolution) is tested in tests/test_price_partition.py.


# =====================================================================
# Fitness advantage and sigma*
# =====================================================================


class TestFitnessAdvantage:
    """Tests for the fitness advantage function."""

    def test_sign_changes_with_sigma(self):
        """Fitness advantage changes from negative to positive as sigma increases.

        At low sigma: NB advantage (cost C dominates)
        At high sigma: MB advantage (survival + signaling benefit dominates)
        """
        adv_low = fitness_advantage(0.01, lam=0.68, alpha_eff=0.35, beta_eff=0.87, r=0.75)
        adv_high = fitness_advantage(0.8, lam=0.68, alpha_eff=0.35, beta_eff=0.87, r=0.75)
        assert adv_low < 0
        assert adv_high > 0

    # NOTE: "B shifts the threshold down" is removed -- the within-group reward is
    # positional (nets out of the group fitness) and does not move the between-group
    # crossover. See tests/test_price_partition.py for the within-group dynamics.


class TestInitialModelSigmaStar:
    """Tests for the initial model threshold."""

    def test_simplified_formula(self):
        """sigma*_initial = C / [beta - (1-C)*alpha] ~ 0.496."""
        ss = initial_model_sigma_star()
        expected = 0.35 / (0.90 - 0.65 * 0.30)
        assert ss == pytest.approx(expected)
        assert ss == pytest.approx(0.4965, rel=0.01)

    def test_increases_with_C(self):
        """Higher cost raises the threshold."""
        ss1 = initial_model_sigma_star(C=0.20)
        ss2 = initial_model_sigma_star(C=0.50)
        assert ss2 > ss1

    def test_decreases_with_beta(self):
        """Higher non-signaler vulnerability lowers the threshold."""
        ss1 = initial_model_sigma_star(beta=0.70)
        ss2 = initial_model_sigma_star(beta=0.95)
        assert ss2 < ss1


class TestCriticalThreshold:
    """Tests for the extended model sigma* derivation."""

    def test_sigma_star_lower_than_initial(self):
        """At defaults, sigma*_extended < sigma*_initial: an UNLIKE-for-like
        comparison (the framework's endogenous C_model(0.3) = 0.155 against
        the baseline's exogenous C = 0.35), retained as a regression pin of
        the default call, not as a mechanism claim. The superseded "B(lambda)
        reduces the effective cost" reading is gone from the model: B does
        not enter the threshold. The like-for-like comparison is
        test_matched_anchor_gap_small_and_negative below."""
        result = critical_threshold_sigma_star()
        assert result["sigma_star"] < result["sigma_star_initial"]

    def test_matched_anchor_gap_small_and_negative(self):
        """Like-for-like comparison at matched cost: with the baseline
        evaluated at the SAME C_model(lambda_W), the derived threshold sits
        just below it, by 0.021-0.023, uniformly over lambda_W in
        [0.68, 1.0] (measured 0.4771 vs 0.4984 at the anchor). This encodes
        the manuscript's actual claim ("close to and straddling the
        standard baseline rather than lowering it well below": the straddle
        is the (C, n) grid against the fixed ~0.50 baseline, while this
        matched comparison is uniformly just below)."""
        from signaling.layer1 import average_equilibrium_cost

        for lw in (0.68, 0.8, 1.0):
            cm = float(average_equilibrium_cost(lw, 0.1, 2.0))
            ext = sigma_star_self_consistent(lw)["sigma_star"]
            base = initial_model_sigma_star(cm)
            gap = ext - base
            assert -0.03 < gap < 0.0, (
                f"lambda_W={lw}: matched gap {gap:+.4f} outside (-0.03, 0)"
            )

    def test_converged(self):
        """Root-finding converges."""
        result = critical_threshold_sigma_star()
        assert result["converged"]

    def test_B_lambda_positive(self):
        """B(lambda) > 0 at the threshold."""
        result = critical_threshold_sigma_star()
        assert result["B_lambda"] > 0

    def test_alpha_less_than_beta(self):
        """alpha_eff < beta_eff at the threshold."""
        result = critical_threshold_sigma_star()
        assert result["alpha_eff"] < result["beta_eff"]

    def test_sigma_star_in_valid_range(self):
        """sigma* is between 0 and 1."""
        result = critical_threshold_sigma_star()
        assert 0 < result["sigma_star"] < 1

    def test_default_sigma_star_below_standard_baseline(self):
        """Default-call sigma* sits below the standard ~0.50 baseline.

        (Formerly named test_rapa_nui_above_threshold with a narrative that
        the framework might classify Rapa Nui's sigma = 0.20 as
        above-threshold; that classification is FALSE under the current
        model, whose threshold is 0.236 at the defaults and 0.477 at the
        anchor, both above 0.20, and the manuscript makes no such
        classification claim. Retitled to what the assertion checks.)"""
        result = critical_threshold_sigma_star()
        assert result["sigma_star"] < 0.50

    def test_higher_lambda_W_raises_threshold(self):
        """Higher within-group reward lambda_W raises sigma*.

        Cost is endogenous (C = C_model(lambda_W)), so the threshold is driven by
        lambda_W, not by an exogenous C. A larger lambda_W raises the equilibrium
        monument cost and thus the collective-optimum threshold.
        """
        r1 = critical_threshold_sigma_star(lambda_W=0.30)
        r2 = critical_threshold_sigma_star(lambda_W=0.60)
        if r1["converged"] and r2["converged"]:
            assert r2["sigma_star"] > r1["sigma_star"]


# =====================================================================
# Within-group and between-group selection
# =====================================================================


class TestSelectionTerms:
    """Tests for Price equation selection components."""

    def test_within_group_positive_when_sW_exceeds_C(self):
        """Within-group selection favors signaling when s_W > C, the
        POSITIONAL rule (beta_0 = SK(s_W - C) > 0). The signaling framework
        resolves the free-rider problem through the private positional
        return, contradicting the standard assumption that within-group
        selection always opposes group-beneficial traits."""
        wg = within_group_selection(0.757, C=0.35)
        assert wg > 0

    def test_within_group_sign_follows_sW_not_B(self):
        """Discriminating pin: the sign
        rule is s_W - C, NOT the superseded B(lambda) - C. The two
        rules disagree on lambda in (0.368, 0.657): at lambda = 0.45,
        B = 0.240 < C = 0.35 (the old rule would say "opposes") yet
        s_W - C = +0.078 > 0. This test fails if anyone reintroduces the
        B-based rule."""
        lam, C = 0.45, 0.35
        B = average_signaling_benefit(lam)
        wg = within_group_selection(lam, C=C)
        assert B < C  # the superseded rule's premise holds here...
        assert wg == pytest.approx(0.0775, abs=1e-3)
        assert wg > 0  # ...yet the positional rule says favors

    def test_within_group_negative_low_lambda(self):
        """Within-group selection opposes signaling when s_W < C (tiny
        lambda: the positional differential cannot cover the cost)."""
        wg = within_group_selection(0.01, C=0.35)
        assert wg < 0

    def test_between_group_deltaw_positive_high_sigma(self):
        """The two-homogeneous-group difference Delta w(sigma) is positive
        at high sigma. NOTE the object: this is Delta w of Eqs. 12-13, not
        the Price between-group covariance, which stays slightly negative
        until sigma ~ 0.73 (tested in test_price_partition.py); the two
        must not be conflated near the threshold."""
        bg = between_group_selection(
            sigma=0.8, C=0.35, alpha_eff=0.35, beta_eff=0.87, r=0.75,
        )
        assert bg > 0

    def test_between_group_negative_low_sigma(self):
        """Between-group selection opposes signaling at very low sigma."""
        bg = between_group_selection(
            sigma=0.01, C=0.35, alpha_eff=0.35, beta_eff=0.87, r=0.75,
        )
        assert bg < 0


# =====================================================================
# Equilibrium classification
# =====================================================================


class TestEquilibriumClassification:
    """Tests for the three-type equilibrium classification."""

    def test_monument_equilibrium(self):
        """High sigma + monument dominance -> monument equilibrium."""
        eq = classify_equilibrium(0.5, sigma_star=0.3, monument_dominates=True)
        assert eq == "monument"

    def test_alternative_equilibrium(self):
        """High sigma + alternative dominance -> alternative equilibrium."""
        eq = classify_equilibrium(0.5, sigma_star=0.3, monument_dominates=False)
        assert eq == "alternative"

    def test_no_signaling_equilibrium(self):
        """Low sigma -> no signaling."""
        eq = classify_equilibrium(0.1, sigma_star=0.3, monument_dominates=True)
        assert eq == "no_signaling"

    def test_boundary(self):
        """At sigma = sigma*, classified as monument (or alternative)."""
        eq = classify_equilibrium(0.3, sigma_star=0.3, monument_dominates=True)
        assert eq == "monument"


# =====================================================================
# Individual fitness
# =====================================================================


class TestIndividualFitness:
    """Tests for the full individual fitness function."""

    def test_positive_for_moderate_params(self):
        """Fitness is positive for reasonable parameter values."""
        w = individual_fitness(
            x_i=0.5, q_i=1.0, sigma=0.2, k=5.0,
            P_conflict=0.005, lam=0.5,
        )
        assert float(w) > 0

    def test_higher_quality_higher_fitness(self):
        """Higher quality individuals have higher fitness at the equilibrium
        under the mixed specification.

        Under the mixed fitness form, the signaling reward lam*q is added to
        the multiplicative cost-survival product, so the reward increases
        with quality independent of the cost rise and the relationship is
        cleanly monotonic. Under the multiplicative form, high-quality
        individuals incur substantial costs (c approaches 1/2 at q_max for
        large lam), so monotonicity is not guaranteed at all parameter values.
        """
        q_vals = np.array([0.5, 1.0, 1.5, 2.0])
        x_vals = np.sqrt(0.5 * (q_vals ** 2 - 0.1 ** 2))
        w = individual_fitness(x_vals, q_vals, sigma=0.2, k=5.0,
                               P_conflict=0.005, lam=0.5, mode="mixed")
        diffs = np.diff(w)
        assert np.all(diffs > 0)

    def test_modes_give_different_values(self):
        """Different fitness combination modes produce different values."""
        kwargs = dict(x_i=0.5, q_i=1.0, sigma=0.2, k=5.0, P_conflict=0.005, lam=0.5)
        w_mixed = float(individual_fitness(**kwargs, mode="mixed"))
        w_mult = float(individual_fitness(**kwargs, mode="multiplicative"))
        w_add = float(individual_fitness(**kwargs, mode="additive"))
        # All should be positive but different
        assert w_mixed > 0
        assert w_mult > 0
        assert w_add > 0
        # They shouldn't all be exactly equal (different combination rules)
        assert not (w_mixed == pytest.approx(w_mult) and w_mult == pytest.approx(w_add))

    def test_invalid_mode_raises(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            individual_fitness(0.5, 1.0, 0.2, 5.0, 0.005, 0.5, mode="invalid")


# =====================================================================
# Sigma* comparison
# =====================================================================


class TestSigmaStarComparison:
    """Tests for the sigma* comparison across cost values."""

    def test_extended_invariant_to_exogenous_C(self):
        """The framework's sigma* is invariant to the initial model's exogenous C.

        Cost is endogenous in the framework (C = C_model(lambda_W)), so sweeping
        the initial model's assumed reproductive cost leaves the extended threshold
        unchanged; only the initial-model baseline moves with C. This is the
        positional-model replacement for the (incorrect) earlier claim that
        B(lambda) made the extended threshold "consistently lower."
        """
        C_range = np.linspace(0.1, 0.5, 5)
        result = sigma_star_comparison(C_range)
        ext = result["sigma_star_extended"]
        assert np.all(np.isfinite(ext))
        assert np.ptp(ext) < 1e-9, f"extended sigma* should be C-invariant, got {ext}"
        # The initial-model baseline, by contrast, rises with the assumed cost.
        assert np.all(np.diff(result["sigma_star_initial"]) > 0)

    def test_initial_baseline_increases_with_C(self):
        """The initial-model baseline rises with reproductive cost; the framework's
        endogenous-cost threshold does not respond to the exogenous C."""
        C_range = np.linspace(0.1, 0.4, 5)
        result = sigma_star_comparison(C_range)

        diffs_init = np.diff(result["sigma_star_initial"])
        assert np.all(diffs_init > 0)

        # Extended threshold is endogenous-cost-driven, hence flat in exogenous C.
        ext = result["sigma_star_extended"]
        assert np.all(np.isfinite(ext))
        assert np.ptp(ext) < 1e-9


# =====================================================================
# Phase space
# =====================================================================


class TestPhaseSpace:
    """Tests for the (sigma, C) phase space computation."""

    def test_returns_correct_shape(self):
        """Output grids have correct shape."""
        sigma_range = np.linspace(0.1, 0.5, 5)
        C_range = np.linspace(0.1, 0.5, 4)
        result = phase_space(sigma_range, C_range)
        assert result["fitness_advantage"].shape == (4, 5)
        assert result["sigma_grid"].shape == (4, 5)

    def test_high_sigma_low_C_favors_signaling(self):
        """High sigma, low C region should favor monument building."""
        sigma_range = np.array([0.6, 0.8])
        C_range = np.array([0.1, 0.15])
        result = phase_space(sigma_range, C_range)
        # At least some cells should be positive
        assert np.any(result["fitness_advantage"] > 0)

    def test_low_sigma_high_C_opposes_signaling(self):
        """Low sigma, high C region should oppose monument building.

        Passes r=0.0 exogenously (no war avoidance), which phase_space
        honors rather than silently deriving r. The assertion holds under
        either r, but the exercised path is the one the call names."""
        sigma_range = np.array([0.01, 0.02])
        C_range = np.array([0.6, 0.8])
        result = phase_space(sigma_range, C_range, r=0.0)
        # All cells should be negative (high cost, low benefit)
        assert np.all(result["fitness_advantage"] < 0)


# =====================================================================
# Regression tests
# =====================================================================


class TestRegression:
    """Regression tests: lock in key quantitative results."""

    def test_B_lambda_default(self):
        """B(lambda=0.5) at default q range is stable."""
        B = average_signaling_benefit(0.5, 0.1, 2.0)
        assert B == pytest.approx(0.2664, rel=0.01)

    def test_initial_sigma_star_simplified(self):
        """Initial model sigma* ~ 0.496."""
        ss = initial_model_sigma_star()
        assert ss == pytest.approx(0.4965, rel=0.01)

    def test_extended_sigma_star_range(self):
        """Default-call sigma* regression range. (The default call uses
        lambda_W = 0.3, whose endogenous C_model = 0.155 is small, hence the
        low value 0.236; this is a regression pin of the default, not a
        claim that the framework's threshold is "substantially lower" than
        the baseline, which the manuscript does not make.)"""
        result = critical_threshold_sigma_star()
        assert 0.1 < result["sigma_star"] < 0.45
        assert result["sigma_star"] > 0


# =====================================================================
# Signal depreciation
# =====================================================================

from signaling.price_equation import sigma_star_vs_delta


class TestSigmaStarWithDepreciation:
    """Tests for sigma* behavior under signal depreciation."""

    def test_sigma_star_increases_with_delta(self):
        """Higher depreciation raises the critical threshold.

        When signals depreciate faster, the effective monument stock is lower,
        the feedback loop is weaker, and more environmental uncertainty is
        needed to sustain monument building.
        """
        result_low = critical_threshold_sigma_star(delta=0.05)
        result_high = critical_threshold_sigma_star(delta=0.20)
        assert result_high["sigma_star"] > result_low["sigma_star"]

    def test_sigma_star_recovers_base_at_zero_delta(self):
        """delta = 0 reproduces the static model threshold."""
        result_static = critical_threshold_sigma_star(delta=0.0)
        result_base = critical_threshold_sigma_star()
        assert result_static["sigma_star"] == pytest.approx(
            result_base["sigma_star"], rel=1e-6
        )

    def test_sigma_star_vs_delta_sweep(self):
        """sigma_star_vs_delta produces valid sweep results."""
        import numpy as np
        delta_range = np.array([0.05, 0.10, 0.20])
        result = sigma_star_vs_delta(delta_range)
        assert len(result["sigma_star"]) == 3
        assert np.all(np.isfinite(result["sigma_star"]))
        # Should be monotonically increasing
        assert result["sigma_star"][0] < result["sigma_star"][1]
        assert result["sigma_star"][1] < result["sigma_star"][2]

    def test_phase_space_with_delta(self):
        """Phase space computation works with depreciation."""
        import numpy as np
        sigma_range = np.linspace(0.1, 0.5, 5)
        C_range = np.linspace(0.2, 0.5, 5)
        result = phase_space(sigma_range, C_range, delta=0.1)
        assert result["fitness_advantage"].shape == (5, 5)


# =====================================================================
# Parameter sweep: sigma* vs single parameter
# =====================================================================

from signaling.price_equation import sigma_star_vs_param, sensitivity_tornado


class TestSigmaStarVsParam:
    """Tests for sigma_star_vs_param: one-at-a-time parameter sweeps."""

    def test_lambda_W_sweep_increasing(self):
        """sigma* increases as lambda_W increases.

        Higher within-group lambda_W raises the equilibrium monument cost
        C_model(lambda_W), so a larger between-group survival/deterrence benefit
        (higher environmental uncertainty) is needed to favor building. The
        within-group reward is positional and does not lower the between-group
        threshold.
        """
        values = np.array([0.1, 0.3, 0.5])
        result = sigma_star_vs_param("lambda_W", values)
        ss = result["sigma_star"]
        assert np.all(np.isfinite(ss))
        assert ss[0] < ss[1] < ss[2]

    def test_n_sweep_decreasing(self):
        """sigma* should decrease as group size n increases.

        Larger groups produce more aggregate monument investment,
        strengthening the feedback loop and lowering the threshold.
        """
        values = np.array([6, 12, 24])
        result = sigma_star_vs_param("n", values)
        ss = result["sigma_star"]
        assert np.all(np.isfinite(ss))
        assert ss[0] > ss[1] > ss[2]

    def test_gamma_sweep_decreasing(self):
        """sigma* should decrease as buffering efficiency gamma increases.

        Higher gamma means each network partner provides more crisis
        buffering, making monument building more valuable at lower sigma.
        Use a wider range to see the effect clearly; the sensitivity is
        small relative to lambda_W and C.
        """
        values = np.array([0.05, 0.3, 0.8])
        result = sigma_star_vs_param("gamma", values)
        ss = result["sigma_star"]
        assert np.all(np.isfinite(ss))
        # Overall trend: higher gamma lowers threshold
        assert ss[0] > ss[2]

    def test_baseline_C_sweep_flat_lambda_W_sweep_increases(self):
        """Sweeping the baseline-comparison cost argument ``C`` leaves the
        framework's sigma* flat by design (on the canonical path the framework's
        cost is the endogenous C_model(lambda_W); ``C`` feeds only the
        assumed-coefficient baseline). The lambda_W sweep, which actually moves
        the equilibrium cost, raises sigma* monotonically.
        """
        ss_C = sigma_star_vs_param("C", np.array([0.2, 0.35, 0.5]))["sigma_star"]
        assert np.all(np.isfinite(ss_C))
        assert np.ptp(ss_C) < 1e-9, f"baseline-C sweep should leave sigma* flat, got {ss_C}"

        ss_lam = sigma_star_vs_param("lambda_W", np.array([0.3, 0.5, 0.7]))["sigma_star"]
        assert np.all(np.isfinite(ss_lam))
        assert ss_lam[0] < ss_lam[1] < ss_lam[2]

    def test_off_locus_C_exogenous_moves_sigma_star(self):
        """The off-locus mode (C_exogenous) decouples the framework's
        reproductive cost from lambda_W: sigma* rises monotonically in the
        exogenous cost at fixed lambda_W. Pins the measured off-locus values at
        the anchor coefficients (alpha_eff, beta_eff, r from lambda_W = 0.68),
        independently cross-checked against the closed-form quadratic roots of
        Eqs. 12--13 (0.282912 and 0.652006).
        """
        ss = sigma_star_vs_param(
            "C_exogenous", np.array([0.2, 0.35, 0.5]), lambda_W=0.68
        )["sigma_star"]
        assert np.all(np.isfinite(ss))
        assert ss[0] < ss[1] < ss[2]
        assert ss[0] == pytest.approx(0.282912, abs=1e-4)
        assert ss[2] == pytest.approx(0.652006, abs=1e-4)

    def test_off_locus_at_locus_reproduces_endogenous(self):
        """Setting C_exogenous to C_model(lambda_W) must reproduce the
        endogenous (self-consistent) threshold exactly: the off-locus mode is a
        strict generalization whose on-locus slice is the canonical path.
        """
        from signaling.layer1 import average_equilibrium_cost

        lam = 0.68
        cm = float(average_equilibrium_cost(lam, 0.1, 2.0))
        ss_end = critical_threshold_sigma_star(lambda_W=lam)["sigma_star"]
        ss_exo = critical_threshold_sigma_star(C_exogenous=cm, lambda_W=lam)["sigma_star"]
        assert ss_exo == pytest.approx(ss_end, abs=1e-12)


class TestThresholdRootStructure:
    """Delta w(sigma) is an exact quadratic under the multiplicative
    specification, so it can in principle have zero, one, or two roots in
    [0, 1]. The solver brackets every crossing with a dense sign-scan (an
    endpoint-only check would misclassify a profile favored only on an
    interior band as 'never favored'). Existence and uniqueness of 'the'
    threshold are an empirical property of the calibrated coefficients, so
    these tests pin them where the paper asserts them."""

    def test_anchor_has_exactly_one_root(self):
        """At the calibrated anchor the quadratic has algebraic roots 0.477
        and 3.286, so exactly one lies in [0, 1]: 'the' threshold is
        well-defined there."""
        r = critical_threshold_sigma_star(lambda_W=0.68)
        assert r["n_roots"] == 1
        assert r["sigma_roots"][0] == pytest.approx(0.4771, abs=1e-3)
        assert r["converged"]

    def test_Cn_rectangle_single_root_everywhere(self):
        """Across the reported (C, n) rectangle (C mapped to lambda_W by the
        self-consistent locus), the threshold quadratic has exactly one root
        in [0, 1] at every corner and midpoint: the headline range
        [0.24, 0.65] is a range of unique thresholds, not lowest-of-several."""
        for C in (0.20, 0.35, 0.50):
            for n in (10, 60, 250):
                res = sigma_star_self_consistent(lambda_W=1.93 * C, n=n)
                assert res["n_roots"] == 1, (
                    f"(C={C}, n={n}): expected one root, got {res['sigma_roots']}"
                )

    def test_depreciation_on_coefficients_disclosed_values(self):
        """The depreciation-on threshold (delta = 0.10) evaluates the derived
        coefficients at the steady-state stock I_g/delta ~ 103, where they
        differ from the static-reference values quoted elsewhere: r_bb shifts
        from 0.4345 to ~0.81 and alpha_eff from 0.3323 to ~0.29. These are
        the numbers reported in the main text's depreciation-results
        section and Table 1."""
        from signaling.layer2 import derived_conflict_reduction
        from signaling.layer3 import between_group_lambda_diagnostics

        res = sigma_star_self_consistent(0.68, delta=0.10)
        assert res["sigma_star"] == pytest.approx(0.4181, abs=1e-3)
        diag = between_group_lambda_diagnostics(res["sigma_star"], 0.68, delta=0.10)
        M_star = diag["M_g"]
        assert M_star == pytest.approx(103.1, abs=0.2)
        assert float(derived_conflict_reduction(M_star, M_star)) == pytest.approx(
            0.806, abs=2e-3
        )
        assert diag["alpha_eff"] == pytest.approx(0.287, abs=2e-3)

    def test_never_favored_classified_from_full_scan(self):
        """At k_0 = 5 (baseline network parity) the survival differential
        vanishes and building is never favored: the scan finds no crossing
        anywhere in [0, 1] (not merely matching endpoint signs) and reports
        sigma* = inf."""
        r = critical_threshold_sigma_star(lambda_W=0.68, k_0=5.0)
        assert r["n_roots"] == 0
        assert np.isinf(r["sigma_star"])
        assert not r["converged"]

    def test_explicit_r_with_derived_mode_raises(self):
        """Silent-override guard (same hazard class as an exogenous-C sweep
        that ignores the passed C): on the canonical
        path r is derived at the equilibrium stock, so an explicitly passed r
        would be silently ignored. The function must refuse instead, so that
        sigma_star_vs_param('r', ...) under defaults yields visible NaN
        rather than a silently flat sweep."""
        with pytest.raises(ValueError, match="use_derived_r"):
            critical_threshold_sigma_star(lambda_W=0.68, r=0.5)
        ss = sigma_star_vs_param("r", np.array([0.0, 0.5]), lambda_W=0.68)["sigma_star"]
        assert np.all(np.isnan(ss)), (
            "defaults r-sweep should surface as NaN, not silently flat"
        )

    def test_exogenous_r_sweep_pins_manuscript_range(self):
        """Pins the manuscript sensitivity claim (assumptions section):
        varying r exogenously over [0, 0.95] moves sigma* only within
        [0.42, 0.54]. Measured: 0.5376 at r = 0 (no war avoidance anywhere)
        down to 0.4206 at r = 0.95, monotone decreasing (more avoided war
        cost favors building earlier)."""
        ss_lo = critical_threshold_sigma_star(
            lambda_W=0.68, r=0.0, use_derived_r=False
        )["sigma_star"]
        ss_hi = critical_threshold_sigma_star(
            lambda_W=0.68, r=0.95, use_derived_r=False
        )["sigma_star"]
        assert ss_lo == pytest.approx(0.5376, abs=1e-3)
        assert ss_hi == pytest.approx(0.4206, abs=1e-3)
        ss_sweep = sigma_star_vs_param(
            "r", np.array([0.0, 0.475, 0.95]), lambda_W=0.68, use_derived_r=False
        )["sigma_star"]
        assert ss_sweep[0] > ss_sweep[1] > ss_sweep[2]

    def test_phase_space_honors_exogenous_r(self):
        """phase_space's r parameter is honored, not a dead parameter that
        is always overridden by the internally derived value. With r = 0 the
        builder
        keeps no war-avoidance and the advantage must be strictly lower than
        under the derived r at a favorable cell."""
        sigma_range = np.array([0.7])
        C_range = np.array([0.2])
        adv_derived = phase_space(sigma_range, C_range, lambda_W=0.68)[
            "fitness_advantage"][0, 0]
        adv_r0 = phase_space(sigma_range, C_range, lambda_W=0.68, r=0.0)[
            "fitness_advantage"][0, 0]
        assert adv_r0 < adv_derived

    def test_interior_band_shape_exists_for_unconstrained_coefficients(self):
        """The two-interior-root profile motivating the scan: with
        coefficients alpha = 0.959596, beta = 0.9,
        r = 1, C = 0.01, Delta w(sigma) is negative at both ends of [0, 1]
        yet positive on an interior band (algebraic roots 0.042 and 0.884),
        so an endpoint-only sign check would misread it as never-favored.
        Such coefficient sets require alpha > beta and are NOT reachable
        through the derived path (k_signal >= k_0 forces
        alpha_eff <= beta_eff), which is why the derived grid above is
        single-rooted; the scan makes the solver correct without relying on
        that structural bound."""
        from signaling.price_equation import fitness_advantage

        kw = dict(alpha_eff=0.959596, beta_eff=0.9, r=1.0, C_exogenous=0.01)
        assert fitness_advantage(0.02, 0.68, **kw) < 0
        assert fitness_advantage(0.50, 0.68, **kw) > 0
        assert fitness_advantage(0.95, 0.68, **kw) < 0

    def test_invalid_param_raises(self):
        """Unknown parameter name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            sigma_star_vs_param("bogus", np.array([0.1, 0.2]))

    def test_returns_correct_keys(self):
        """Return dict contains the expected keys."""
        result = sigma_star_vs_param("C", np.array([0.3]))
        expected_keys = {"param_values", "sigma_star", "sigma_star_initial",
                         "lambda_total", "B_lambda"}
        assert set(result.keys()) == expected_keys


# =====================================================================
# Sensitivity tornado analysis
# =====================================================================


class TestSensitivityTornado:
    """Tests for sensitivity_tornado: one-at-a-time sensitivity analysis."""

    @pytest.fixture()
    def default_ranges(self):
        """Standard parameter ranges for tornado analysis."""
        return {
            "lambda_W": (0.1, 0.5),
            "C": (0.2, 0.5),
            "gamma": (0.1, 0.5),
            "n": (6, 24),
            "r": (0.5, 0.9),
        }

    def test_returns_all_params(self, default_ranges):
        """Output has an entry for each input parameter."""
        result = sensitivity_tornado(default_ranges)
        assert set(result.keys()) == set(default_ranges.keys())

    def test_swing_non_negative(self, default_ranges):
        """All swing values are non-negative (or NaN)."""
        result = sensitivity_tornado(default_ranges)
        for param, info in result.items():
            swing = info["swing"]
            if np.isfinite(swing):
                assert swing >= 0.0, f"Negative swing for {param}: {swing}"

    def test_lambda_W_or_C_high_sensitivity(self, default_ranges):
        """lambda_W or C should be among the most sensitive parameters.

        These directly determine the cost-benefit tradeoff for monument
        building, so they should produce a swing > 0.1 in sigma*.
        """
        result = sensitivity_tornado(default_ranges)
        lam_swing = result["lambda_W"]["swing"]
        C_swing = result["C"]["swing"]
        assert lam_swing > 0.1 or C_swing > 0.1, (
            f"Expected lambda_W swing ({lam_swing}) or C swing ({C_swing}) > 0.1"
        )

    def test_baseline_matches_default(self):
        """Baseline sigma* in tornado output matches critical_threshold_sigma_star defaults."""
        result = sensitivity_tornado({"C": (0.2, 0.5)})
        baseline_from_tornado = result["C"]["sigma_star_baseline"]
        default_result = critical_threshold_sigma_star()
        assert baseline_from_tornado == pytest.approx(
            default_result["sigma_star"], rel=1e-6
        )


# =====================================================================
# Fitness mode variants
# =====================================================================


class TestFitnessMode:
    """Tests for fitness combination mode parameter in group fitness and sigma*."""

    def test_mode_parameter_accepted(self):
        """group_fitness_signaler accepts mode='multiplicative' without error."""
        w = group_fitness_signaler(0.3, 0.35, 0.35, 0.75, mode="multiplicative")
        assert np.isfinite(w)

    def test_mixed_equals_multiplicative(self):
        """Mixed and multiplicative coincide under the positional model.

        With the within-group reward positional (no separate additive reward term
        in the group fitness), 'mixed' and 'multiplicative' are identical.
        """
        kwargs = dict(sigma=0.3, lam=0.68, alpha_eff=0.35, r=0.75)
        w_mixed = group_fitness_signaler(**kwargs, mode="mixed")
        w_mult = group_fitness_signaler(**kwargs, mode="multiplicative")
        assert w_mixed == pytest.approx(w_mult)

    def test_sigma_star_multiplicative_converges(self):
        """critical_threshold_sigma_star with mode='multiplicative' converges."""
        result = critical_threshold_sigma_star(mode="multiplicative")
        assert result["converged"]

    def test_sigma_star_ext_lower_all_modes(self):
        """Default-call sigma*_ext < sigma*_initial for both mixed and
        multiplicative modes: the unlike-for-like default comparison (see
        test_sigma_star_lower_than_initial) is mode-robust. (The superseded
        "signaling benefits lower the threshold" mechanism is not the
        claim; B does not enter the threshold in the v6 model.)"""
        for mode in ("mixed", "multiplicative"):
            result = critical_threshold_sigma_star(mode=mode)
            assert result["sigma_star"] < result["sigma_star_initial"], (
                f"sigma*_ext ({result['sigma_star']}) >= sigma*_initial "
                f"({result['sigma_star_initial']}) for mode={mode}"
            )

    def test_additive_mode_works(self):
        """critical_threshold_sigma_star with mode='additive' returns finite sigma*."""
        result = critical_threshold_sigma_star(mode="additive")
        assert np.isfinite(result["sigma_star"])


class TestConvexMode:
    """Tests for the convex-combination fitness specification.

    Encodes the convex-combination fitness robustness claim (main-text
    assumptions section; Supp. convex-combination figure): omega = 0 recovers
    multiplicative; omega = 1 recovers additive; sigma*(omega) interpolates
    monotonically between the endpoints and sits near the standard MLS
    baseline across omega in [0, 1].
    """

    def test_convex_omega_zero_matches_multiplicative(self):
        """sigma*(omega=0) equals the pure multiplicative result."""
        from signaling.price_equation import sigma_star_self_consistent

        r_mult = sigma_star_self_consistent(lambda_W=0.68, mode="multiplicative")
        r_conv = sigma_star_self_consistent(lambda_W=0.68, mode="convex", omega=0.0)
        assert r_conv["sigma_star"] == pytest.approx(r_mult["sigma_star"], rel=1e-6)

    def test_convex_omega_one_matches_additive(self):
        """sigma*(omega=1) is the additive limit: finite and ABOVE the multiplicative value.

        Under the positional model the additive specification does NOT collapse
        (there is no separate reward to over-credit); sigma* at omega=1 is finite
        and higher than the multiplicative (omega=0) threshold.
        """
        from signaling.price_equation import sigma_star_self_consistent

        r_conv = sigma_star_self_consistent(lambda_W=0.68, mode="convex", omega=1.0)
        r_mult = sigma_star_self_consistent(lambda_W=0.68, mode="multiplicative")
        assert np.isfinite(r_conv["sigma_star"])
        assert r_conv["sigma_star"] > r_mult["sigma_star"]

    def test_convex_sigma_star_monotonic_in_omega(self):
        """sigma*(omega) is non-decreasing in omega across [0, 1] (mult <= additive)."""
        from signaling.price_equation import sigma_star_self_consistent

        omegas = [0.0, 0.25, 0.5, 0.75, 1.0]
        sigmas = [
            sigma_star_self_consistent(
                lambda_W=0.68, mode="convex", omega=w,
            )["sigma_star"]
            for w in omegas
        ]
        for i in range(len(sigmas) - 1):
            assert sigmas[i] <= sigmas[i + 1] + 1e-8, (
                f"Non-monotone: sigma*({omegas[i]})={sigmas[i]:.4f} > "
                f"sigma*({omegas[i+1]})={sigmas[i+1]:.4f}"
            )

    def test_convex_rejects_invalid_omega(self):
        """mode='convex' raises ValueError for omega outside [0, 1] or None."""
        with pytest.raises(ValueError, match="omega in"):
            group_fitness_signaler(0.3, 0.68, 0.35, 0.75, mode="convex", omega=-0.1)
        with pytest.raises(ValueError, match="omega in"):
            group_fitness_signaler(0.3, 0.68, 0.35, 0.75, mode="convex", omega=1.1)
        with pytest.raises(ValueError, match="omega in"):
            group_fitness_signaler(0.3, 0.68, 0.35, 0.75, mode="convex", omega=None)

    def test_convex_robustness_finite_near_baseline(self):
        """sigma*(omega) is finite and sits near the MLS baseline for all omega in [0, 1].

        The derived between-group parameters (network buffering plus the sigma-scaled
        war-avoidance, lambda_C) bring the threshold close to the MLS baseline across the
        multiplicative-additive range. Whether it lands just above or just below baseline
        is w0-dependent (the war-cost scale is a soft, tabled calibration); at w0 = 0.30 it
        is marginally below. The robust claim is proximity, not a sign.
        """
        from signaling.price_equation import sigma_star_vs_omega

        omegas = np.linspace(0.0, 1.0, 11)
        sweep = sigma_star_vs_omega(omegas, lambda_W=0.68)
        assert np.all(np.isfinite(sweep["sigma_star"]))
        assert np.all((sweep["sigma_star"] > 0.40) & (sweep["sigma_star"] < 0.80)), (
            f"sigma*(omega) out of the near-baseline band: "
            f"range [{np.min(sweep['sigma_star']):.4f}, {np.max(sweep['sigma_star']):.4f}]"
        )

    def test_convex_omega_half_regression(self):
        """REGRESSION: sigma*(omega=0.5) at lambda_W=0.68 locked at ~0.50.

        Locks the convex-combination result against future drift.
        Computed value (positional + war-avoidance, w0=0.30): 0.5042 (w0-provisional).
        """
        from signaling.price_equation import sigma_star_self_consistent

        res = sigma_star_self_consistent(lambda_W=0.68, mode="convex", omega=0.5)
        assert res["sigma_star"] == pytest.approx(0.504, abs=0.005)


class TestStabilityAnalysis:
    """Tests for equilibrium stability properties.

    Verifies that sigma* is a proper transcritical bifurcation point:
    the gradient is positive (crossing sigma* favors monument building),
    the feedback loop fixed point is unique, and invasion is feasible.
    """

    def test_gradient_positive_mixed(self):
        """d(fitness_advantage)/d_sigma > 0 at sigma* (mixed mode).

        A positive gradient confirms that increasing environmental
        uncertainty past the threshold tips selection toward monument
        building, as the framework predicts.
        """
        result = stability_analysis(mode="mixed")
        assert result["gradient_positive"]
        assert result["gradient_at_threshold"] > 0

    def test_gradient_positive_multiplicative(self):
        """d(fitness_advantage)/d_sigma > 0 at sigma* (multiplicative)."""
        result = stability_analysis(mode="multiplicative")
        assert result["gradient_positive"]

    def test_fixed_point_unique_mixed(self):
        """The lambda feedback loop converges to a unique fixed point.

        Multiple initial conditions should yield the same equilibrium
        lambda, confirming no bistability in the feedback loop.
        """
        result = stability_analysis(mode="mixed")
        assert result["fixed_point_unique"]
        assert result["fixed_point_spread"] < 1e-4

    def test_fixed_point_unique_multiplicative(self):
        """Uniqueness holds under multiplicative fitness."""
        result = stability_analysis(mode="multiplicative")
        assert result["fixed_point_unique"]

    def test_advantage_positive_above_threshold_mixed(self):
        """Consistency check: the fitness advantage is positive just above
        sigma*.

        Since sigma_star is defined as the root of the advantage function,
        this property is structural rather than an independent dynamical
        claim. The test confirms that the numerical root-finding produced
        a sigma* such that the advantage flips sign in the expected
        direction.
        """
        result = stability_analysis(mode="mixed")
        assert result["advantage_positive_above_threshold"]
        assert result["invasion_advantage"] > 0

    def test_advantage_positive_above_threshold_multiplicative(self):
        """Same consistency check under multiplicative fitness."""
        result = stability_analysis(mode="multiplicative")
        assert result["advantage_positive_above_threshold"]

    def test_fixed_point_unique_across_sigma_sweep(self):
        """Uniqueness of the lambda feedback fixed point at five sigma
        values spanning [0.1, 0.9].

        Backs the manuscript claim that the feedback loop converges to
        the same fixed point from initial conditions spanning lambda in
        [0.05, 2.0] at each of five sigma values (main text, self-consistent
        lambda feedback loop).
        """
        sigmas = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = stability_analysis(
            mode="multiplicative", sigma_check_values=sigmas,
            n_initial_conditions=20,
        )
        # All sigma values should have a unique fixed point.
        assert all(result["fixed_point_unique_per_sigma"]), (
            f"Fixed-point uniqueness failed at one or more sigmas: "
            f"{list(zip(sigmas, result['fixed_point_unique_per_sigma']))}"
        )
        # Spreads should all be very tight (< 1e-4).
        max_spread = max(result["fixed_point_spread_per_sigma"])
        assert max_spread < 1e-4, f"max spread {max_spread:.2e} >= 1e-4"


class TestFeedbackMapContraction:
    """The lambda feedback map T(lambda) = lambda_W + lambda_C + lambda_X
    is a contraction on R+ at calibrated parameters. Backs the
    Banach-fixed-point argument in the main-text lambda-feedback-loop derivation."""

    def test_lipschitz_bound_below_one(self):
        """The numerical Lipschitz constant of T is well below 1 across
        the operating range sigma in [0.1, 0.9], lambda in [0.3, 1.5] at
        lambda_W = 0.68.

        This is the contraction property required for the Banach
        fixed-point theorem to guarantee existence and uniqueness.
        """
        from signaling.layer1 import expected_monument_stock
        from signaling.layer3 import compute_lambda_X

        gamma = 0.3
        k_0 = 0.5
        k_max = 8.0
        M_half = 3.0
        n = 12
        q_min = 0.1
        q_max = 2.0
        lambda_W = 0.68

        def T_iterate(lam, sigma):
            # T(lambda) = lambda_W + lambda_C(M) + lambda_X(M; sigma).
            # lambda_C is omitted here because at calibrated parameters
            # (lambda_W = 0.68, default network params) it contributes
            # essentially zero to the total: the manuscript reports
            # lambda_C + lambda_X combined are < 1% of total lambda
            # (main-text lambda-feedback-loop derivation). Including lambda_C would change
            # |T'| by < 1e-6 and does not affect the contraction claim.
            M = expected_monument_stock(
                n=n, q_min=q_min, q_max=q_max, lam=lam,
            )
            lam_X = compute_lambda_X(
                M, sigma, gamma=gamma, k_0=k_0,
                k_max=k_max, M_half=M_half,
            )
            return lambda_W + lam_X

        eps = 1e-6
        max_L = 0.0
        for sigma in np.linspace(0.1, 0.9, 9):
            for lam_test in np.linspace(0.3, 1.5, 25):
                T_plus = T_iterate(lam_test + eps, sigma)
                T_minus = T_iterate(lam_test - eps, sigma)
                L = abs((T_plus - T_minus) / (2 * eps))
                max_L = max(max_L, L)

        # Operating-range bound: enforce the manuscript claim |T'| <= 0.025
        # tightly. Current actual max is ~0.0246; threshold 0.03 leaves a
        # small safety margin while protecting against drift that would
        # invalidate the manuscript claim.
        assert max_L < 0.03, f"max Lipschitz constant {max_L:.4f} >= 0.03"
        # Sanity: contraction at all.
        assert max_L < 1.0, f"map is not a contraction at calibrated params"


class TestFeedbackMapContractionAcrossLambdaW:
    """Extend the Lipschitz-bound verification across the full lambda_W
    range over which the manuscript reports sigma*(lambda_W)."""

    def test_lipschitz_bound_lambda_W_structure(self):
        """Non-vacuous lambda_W-sweep test: the map
        T(lambda) = lambda_W + lambda_C + lambda_X has lambda_W only as an
        additive constant, so T' is INDEPENDENT of lambda_W and re-sweeping
        the [0.3, 1.5] box per lambda_W would only re-verify the same
        numbers. The substantive content is three separate claims, each
        asserted here:

        1. T' is lambda_W-independent (asserted exactly).
        2. |T'| <= 0.03 on the invariant box lambda in [0.3, 1.5], which
           contains the fixed point for lambda_W >= 0.3.
        3. At the low-reward endpoint lambda_W = 0.10 the fixed point
           (~0.10-0.12) lies BELOW the box, where |T'| reaches ~0.17:
           above the 0.03 box bound (the previous claim was false there)
           but still well below 1, so contraction and uniqueness hold.
        """
        from signaling.layer1 import expected_monument_stock
        from signaling.layer3 import compute_lambda_X, lambda_total_at_sigma

        gamma, k_0, k_max, M_half = 0.3, 0.5, 8.0, 3.0
        n, q_min, q_max = 12, 0.1, 2.0
        eps = 1e-6

        def T_iterate(lam, sigma, lw):
            M = expected_monument_stock(n=n, q_min=q_min, q_max=q_max, lam=lam)
            lam_X = compute_lambda_X(
                M, sigma, gamma=gamma, k_0=k_0, k_max=k_max, M_half=M_half,
            )
            return lw + lam_X

        def Tprime(lam, sigma, lw):
            return abs(
                (T_iterate(lam + eps, sigma, lw) - T_iterate(lam - eps, sigma, lw))
                / (2 * eps)
            )

        # 1. lambda_W-independence: identical derivative at the extremes.
        for lam in (0.15, 0.5, 1.2):
            for sigma in (0.1, 0.5, 0.9):
                # rel tolerance covers central-difference cancellation
                # noise (~1e-10); a real lambda_W dependence would show at
                # the 1e-2 scale.
                assert Tprime(lam, sigma, 0.10) == pytest.approx(
                    Tprime(lam, sigma, 1.00), rel=1e-6, abs=1e-9
                )

        # 2. Box bound, ONE pass (not per lambda_W).
        max_L_box = max(
            Tprime(lam, sigma, 0.68)
            for sigma in np.linspace(0.1, 0.9, 5)
            for lam in np.linspace(0.3, 1.5, 15)
        )
        assert max_L_box < 0.03

        # 2b. The fixed point sits inside the box for calibrated rewards.
        for lw in (0.30, 0.50, 0.68, 1.00):
            for sigma in (0.1, 0.5, 0.9):
                fp = lambda_total_at_sigma(sigma, lw)["lambda_total"]
                assert 0.3 <= fp <= 1.5, (lw, sigma, fp)

        # 3. The lambda_W = 0.10 endpoint: fixed point BELOW the box,
        #    derivative above the box bound yet contractive.
        fps = [lambda_total_at_sigma(sigma, 0.10)["lambda_total"]
               for sigma in (0.1, 0.5, 0.9)]
        assert all(fp < 0.3 for fp in fps)
        max_L_fp = max(
            Tprime(lam, sigma, 0.10)
            for sigma in np.linspace(0.1, 0.9, 5)
            for lam in np.linspace(min(fps) - 0.01, max(fps) + 0.02, 9)
        )
        assert 0.03 < max_L_fp < 0.2, max_L_fp


class TestLipschitzDerivativeAnalytical:
    """Tests for the analytical closed-form Lipschitz bound (Supp.
    the Banach contraction analysis). Verifies that ``lipschitz_derivative_analytical``
    reproduces the central-difference bound used by
    ``test_lipschitz_bound_below_one`` to within 5% at the empirical
    anchor and across the lambda_W sweep.
    """

    def test_analytical_matches_numerical_across_operating_range(self):
        """The closed-form expression for |T'(lambda)| matches the
        central-difference value at every (lambda, sigma) on the
        operating-range grid, for every manuscript-reported lambda_W.

        Tolerance: 5% relative error (the analytical and finite-difference
        derivatives agree to within numerical noise, so this is loose by
        design; the empirical agreement is below 0.01% at most grid points).
        """
        from signaling.layer1 import expected_monument_stock
        from signaling.layer3 import compute_lambda_X
        from signaling.price_equation import lipschitz_derivative_analytical

        gamma = 0.3
        k_0 = 0.5
        k_max = 8.0
        M_half = 3.0
        n = 12
        q_min = 0.1
        q_max = 2.0

        eps = 1e-6
        for lambda_W in (0.10, 0.30, 0.50, 0.68, 1.00):
            def T_iterate(lam, sigma, lw=lambda_W):
                M = expected_monument_stock(
                    n=n, q_min=q_min, q_max=q_max, lam=lam,
                )
                lam_X = compute_lambda_X(
                    M, sigma, gamma=gamma, k_0=k_0,
                    k_max=k_max, M_half=M_half,
                )
                return lw + lam_X

            for sigma in np.linspace(0.1, 0.9, 5):
                for lam_test in np.linspace(0.3, 1.5, 13):
                    L_analytic = lipschitz_derivative_analytical(
                        lam=lam_test, sigma=sigma, lambda_W=lambda_W,
                        gamma=gamma, k_0=k_0, k_max=k_max, M_half=M_half,
                        n=n, q_min=q_min, q_max=q_max,
                    )
                    L_numerical = abs(
                        (T_iterate(lam_test + eps, sigma)
                         - T_iterate(lam_test - eps, sigma))
                        / (2 * eps)
                    )
                    if max(L_analytic, L_numerical) > 1e-6:
                        relerr = (
                            abs(L_analytic - L_numerical)
                            / max(L_analytic, L_numerical)
                        )
                        assert relerr < 0.05, (
                            f"Analytical/numerical mismatch at "
                            f"lambda_W={lambda_W}, sigma={sigma}, "
                            f"lam={lam_test}: analytic={L_analytic:.6e}, "
                            f"numerical={L_numerical:.6e}, "
                            f"rel.err.={relerr:.4f}"
                        )

    def test_analytical_bound_at_anchor_matches_manuscript(self):
        """The analytical maximum over the operating range matches the
        manuscript-reported numerical bound |T'(lambda)| <= 0.025.

        At the empirical anchor (lambda_W = 0.68), the supremum is
        attained at the corner (sigma, lambda) = (0.9, 0.3); the
        analytical value at that corner is 0.0246 (matches the numerical
        bound to four decimal places).
        """
        from signaling.price_equation import lipschitz_derivative_analytical

        max_L = 0.0
        for sigma in np.linspace(0.1, 0.9, 9):
            for lam in np.linspace(0.3, 1.5, 25):
                L = lipschitz_derivative_analytical(
                    lam=lam, sigma=sigma, lambda_W=0.68,
                )
                max_L = max(max_L, L)

        assert max_L < 0.025, (
            f"Analytical max |T'(lambda)| = {max_L:.4f} >= 0.025; "
            f"breaks the manuscript bound."
        )
        # Reproduce the corner value 0.0246 to three decimals.
        L_corner = lipschitz_derivative_analytical(
            lam=0.3, sigma=0.9, lambda_W=0.68,
        )
        assert L_corner == pytest.approx(0.0246, abs=1e-3), (
            f"Analytical corner value {L_corner:.4f} does not match "
            f"numerical 0.0246."
        )

    def test_lambda_C_contribution_is_negligible_at_anchor(self):
        """Verify the manuscript's claim that omitting the lambda_C term
        from |T'(lambda)| changes the bound by < 0.5% at the empirical
        anchor (the Banach contraction analysis, paragraph on the symmetric-neighbor
        approximation).
        """
        from signaling.price_equation import lipschitz_derivative_analytical

        L_without_C = lipschitz_derivative_analytical(
            lam=0.68, sigma=0.5, lambda_W=0.68,
        )
        L_with_C = lipschitz_derivative_analytical(
            lam=0.68, sigma=0.5, lambda_W=0.68, include_lambda_C=True,
        )
        relerr = abs(L_with_C - L_without_C) / max(L_without_C, 1e-12)
        assert relerr < 0.005, (
            f"lambda_C contribution {relerr:.4f} >= 0.5% at anchor; "
            f"breaks the 'negligible' approximation."
        )


class TestStabilityAnalysisSelfConsistent:
    """Tests for the canonical self-consistent wrapper. Verifies that the
    three stability properties hold at the self-consistent threshold
    rather than at the legacy exogenous-C parameterization."""

    def test_runs_at_empirical_anchor(self):
        """Wrapper produces standard stability output at lambda_W = 0.68."""
        result = stability_analysis_self_consistent(
            lambda_W=0.68, mode="multiplicative",
        )
        assert "C_model" in result
        assert "r_used" in result
        assert "alpha_eff" in result
        assert result["C_model"] == pytest.approx(0.35, abs=0.02)
        assert result["sigma_star"] == pytest.approx(0.478, abs=0.02)

    def test_gradient_positive_at_empirical_anchor(self):
        """Gradient of fitness advantage is positive at the self-consistent
        sigma*, across lambda_W in {0.3, 0.68, 1.0}."""
        for lw in (0.30, 0.68, 1.0):
            result = stability_analysis_self_consistent(
                lambda_W=lw, mode="multiplicative",
            )
            assert result["gradient_positive"], (
                f"Gradient negative at self-consistent sigma* for "
                f"lambda_W = {lw}"
            )

    def test_fixed_point_unique_self_consistent(self):
        """Uniqueness holds at the self-consistent threshold."""
        result = stability_analysis_self_consistent(
            lambda_W=0.68, mode="multiplicative",
        )
        assert result["fixed_point_unique"]

    def test_multi_sigma_uniqueness_self_consistent(self):
        """Uniqueness sweep at the canonical specification, 5 sigma values."""
        result = stability_analysis_self_consistent(
            lambda_W=0.68, mode="multiplicative",
            sigma_check_values=[0.1, 0.3, 0.5, 0.7, 0.9],
        )
        assert all(result["fixed_point_unique_per_sigma"])


class TestSigmaStarSelfConsistent:
    """Tests for the self-consistent threshold calculation (main-text threshold
    derivation and results).

    The self-consistent threshold enforces:
      (1) C = C_model(lambda_W) from the quadratic cost equilibrium.
      (2) r = r(M_g(lambda_W)) derived from mutual assessment.
      (3) lambda_C and lambda_X evaluated as diagnostics at M_g(lambda_W);
          the schedule carries lambda_W alone (lambda_W-only FOC).

    These tests verify the canonical numbers reported in the manuscript.
    """

    def test_empirical_anchor_reproduces_manuscript(self):
        """At lambda_W ~ 0.68, sigma* ~ 0.477 (positional + war-avoidance), near the MLS
        baseline (marginally below at w0=0.30; the sign is w0-dependent, w0 tabled)."""
        from signaling.price_equation import sigma_star_self_consistent
        res = sigma_star_self_consistent(lambda_W=0.68, mode="multiplicative")
        assert res["sigma_star"] == pytest.approx(0.4771, abs=1e-3)
        assert res["C_model"] == pytest.approx(0.35, abs=0.01)

    def test_threshold_family(self):
        """Reproduces the sigma*(lambda_W) threshold-family values reported in the main text."""
        from signaling.price_equation import sigma_star_self_consistent
        expected = {
            0.10: 0.092,
            0.30: 0.236,
            0.50: 0.366,
            0.68: 0.477,
            1.00: 0.663,
        }
        for lam_W, exp_sigma in expected.items():
            res = sigma_star_self_consistent(lambda_W=lam_W, mode="multiplicative")
            assert res["sigma_star"] == pytest.approx(exp_sigma, abs=0.02), (
                f"sigma*({lam_W}) = {res['sigma_star']:.4f}, expected ~{exp_sigma}"
            )

    def test_thresholds_finite_and_increasing(self):
        """sigma*_ext is finite, positive, and increases with lambda_W in [0.05, 1.0].

        Under the positional model the threshold is derived (not lowered): it rises
        from near zero at low lambda_W to ~0.73 at lambda_W = 1.0, crossing the MLS
        baseline (~0.50) near lambda_W ~ 0.65.
        """
        from signaling.price_equation import sigma_star_self_consistent
        sigs = []
        for lam_W in np.linspace(0.05, 1.0, 10):
            res = sigma_star_self_consistent(lambda_W=float(lam_W), mode="multiplicative")
            sig = res["sigma_star"]
            assert np.isfinite(sig) and 0.0 < sig < 1.0, f"sigma*({lam_W}) = {sig}"
            sigs.append(sig)
        assert all(sigs[i] < sigs[i + 1] for i in range(len(sigs) - 1)), "sigma* not increasing in lambda_W"

    def test_mixed_equals_multiplicative_self_consistent(self):
        """Under the positional model, mixed and multiplicative give identical sigma*.

        The within-group reward is positional (no separate additive reward term),
        so the two specifications coincide; mixed does not collapse the threshold."""
        from signaling.price_equation import sigma_star_self_consistent
        for lam_W in (0.10, 0.30, 0.68):
            res_mix = sigma_star_self_consistent(lambda_W=lam_W, mode="mixed")
            res_mult = sigma_star_self_consistent(lambda_W=lam_W, mode="multiplicative")
            assert res_mix["sigma_star"] == pytest.approx(res_mult["sigma_star"], rel=1e-6)

    def test_derived_r_converges(self):
        """With use_derived_r=True, r converges to r(M_eq) ~ 0.36 at lambda_W=0.30."""
        from signaling.price_equation import sigma_star_self_consistent
        res = sigma_star_self_consistent(
            lambda_W=0.30, mode="multiplicative", use_derived_r=True,
        )
        assert res["r"] == pytest.approx(0.355, abs=0.01)

    def test_derived_r_modest_impact(self):
        """Closing the r feedback vs leaving r=0.75 changes sigma* modestly (~0.03).

        With the sigma-scaled war-avoidance the conflict reduction r matters (it
        sits in the war cost W(sigma)*(1-r)), so the derived-vs-stipulated r choice is no
        longer negligible as in the no-war-avoidance model, though still small.
        """
        from signaling.price_equation import sigma_star_self_consistent
        res_derived = sigma_star_self_consistent(
            lambda_W=0.30, mode="multiplicative", use_derived_r=True,
        )
        res_stipulated = sigma_star_self_consistent(
            lambda_W=0.30, mode="multiplicative", use_derived_r=False,
        )
        assert abs(res_derived["sigma_star"] - res_stipulated["sigma_star"]) < 0.05

    def test_C_model_matches_layer1(self):
        """C_model in the result dict equals average_equilibrium_cost(lambda_W)."""
        from signaling.layer1 import average_equilibrium_cost
        from signaling.price_equation import sigma_star_self_consistent
        for lam_W in (0.10, 0.30, 0.68):
            res = sigma_star_self_consistent(lambda_W=lam_W, mode="multiplicative")
            expected = average_equilibrium_cost(lam_W, 0.1, 2.0)
            assert res["C_model"] == pytest.approx(expected, rel=1e-10)


class TestLambdaWOnlySpecification:
    """The lambda_W-only FOC: the schedule, cost, and stock derive
    from lambda_W alone; the composite-lambda fixed point is a robustness
    variant whose threshold differs by under 0.5%."""

    def test_lambda_W_only_vs_composite_within_half_percent(self):
        """The superseded composite-lambda variant (fixed point via
        lambda_total_at_sigma) moves sigma* by less than 0.5% relative to
        the canonical lambda_W-only threshold, the quantitative basis for
        the SI Banach robustness claim."""
        from scipy.optimize import brentq
        from signaling.layer3 import lambda_total_at_sigma
        from signaling.price_equation import (
            _default_lambda_C_factory,
            fitness_advantage,
            sigma_star_self_consistent,
        )
        from signaling.layer3 import vulnerability_coefficient

        res_canonical = sigma_star_self_consistent(lambda_W=0.68)
        beta_eff = vulnerability_coefficient(0.5, 0.3)
        lam_C_func = _default_lambda_C_factory()

        def _composite_diff(sigma: float) -> float:
            eq = lambda_total_at_sigma(
                sigma, 0.68, compute_lambda_C_func=lam_C_func,
            )
            return fitness_advantage(
                sigma, eq["lambda_total"], eq["alpha_eff"], beta_eff,
                res_canonical["r"],
            )

        sigma_composite = brentq(_composite_diff, 0.01, 0.99, xtol=1e-8)
        rel_diff = abs(res_canonical["sigma_star"] - sigma_composite) / sigma_composite
        assert rel_diff < 0.005, (
            f"composite vs lambda_W-only sigma*: {sigma_composite:.4f} vs "
            f"{res_canonical['sigma_star']:.4f} (rel diff {rel_diff:.4%})"
        )

    def test_monument_stock_sigma_independent(self):
        """Under the lambda_W-only schedule, M_g, k, and alpha_eff from the
        canonical diagnostics are independent of sigma (lambda_X enters
        group fitness through S, not the schedule)."""
        from signaling.layer3 import between_group_lambda_diagnostics
        lo = between_group_lambda_diagnostics(0.1, 0.68)
        hi = between_group_lambda_diagnostics(0.9, 0.68)
        assert lo["M_g"] == pytest.approx(hi["M_g"], rel=1e-12)
        assert lo["k"] == pytest.approx(hi["k"], rel=1e-12)
        assert lo["alpha_eff"] == pytest.approx(hi["alpha_eff"], rel=1e-12)
        # lambda_X itself is the sigma-dependent diagnostic
        assert hi["lambda_X"] > lo["lambda_X"]

    def test_between_group_share_below_one_percent(self):
        """At the anchor stock the diagnostic between-group components are
        under 1% of lambda_W (the manuscript's negligibility claim)."""
        from signaling.layer3 import between_group_lambda_diagnostics
        from signaling.price_equation import _default_lambda_C_factory
        d = between_group_lambda_diagnostics(
            0.9, 0.68, compute_lambda_C_func=_default_lambda_C_factory(),
        )
        share = (d["lambda_C"] + d["lambda_X"]) / d["lambda_composite"]
        assert share < 0.01


class TestSigmaStarHelpersCanonical:
    """Regression guard: the lower-level sigma* helpers derive the
    war-avoidance conflict reduction r = r_bb by default (use_derived_r=True), so
    they agree with the canonical self-consistent threshold (~0.478 at the anchor),
    not the stipulated baseline r = 0.75 value (~0.44). This guards against silent
    staleness in the supplement sigma* figures (phase_space, sigma_star_vs_param,
    sigma_star_bivariate, sensitivity_tornado)."""

    def test_critical_threshold_derives_r_by_default(self):
        from signaling.layer1 import average_equilibrium_cost
        Cm = average_equilibrium_cost(0.68, 0.1, 2.0)
        res = critical_threshold_sigma_star(C=Cm, lambda_W=0.68)
        assert res["sigma_star"] == pytest.approx(0.478, abs=0.01)
        assert res["r"] == pytest.approx(0.435, abs=0.03)  # derived r_bb, not the legacy 0.75

    def test_critical_threshold_matches_self_consistent_at_anchor(self):
        from signaling.layer1 import average_equilibrium_cost
        Cm = average_equilibrium_cost(0.68, 0.1, 2.0)
        ct = critical_threshold_sigma_star(C=Cm, lambda_W=0.68)["sigma_star"]
        sc = sigma_star_self_consistent(lambda_W=0.68)["sigma_star"]
        assert ct == pytest.approx(sc, abs=1e-3)

    def test_legacy_fixed_r_still_available(self):
        from signaling.layer1 import average_equilibrium_cost
        Cm = average_equilibrium_cost(0.68, 0.1, 2.0)
        res = critical_threshold_sigma_star(C=Cm, lambda_W=0.68, use_derived_r=False, r=0.75)
        assert res["sigma_star"] == pytest.approx(0.442, abs=0.01)

    def test_sigma_star_vs_param_canonical_at_anchor(self):
        sweep = sigma_star_vs_param("lambda_W", np.array([0.68]))
        assert float(sweep["sigma_star"][0]) == pytest.approx(0.478, abs=0.01)


class TestSigmaStarVsN:
    """Regression-pin sigma*(n) across the face-to-face audience range.

    The framework drops the single-source Roscoe 2009 n = 12 anchor and treats n
    as a sweep across n in [10, 250] (the face-to-face audience range; lower
    bound where Layer-3 saturation starts, upper bound at the face-to-face
    witnessing constraint). Headline sigma* therefore becomes a RANGE over the
    in-scope (C, n) rectangle rather than a point.

    These tests pin sigma*(n) at fixed lambda_W = 0.68 (per-builder C ~ 0.35
    central anchor) and sigma*(C) at fixed n = 12 (the legacy reference), plus
    the four corners of the in-scope rectangle and the central tendency. The
    pinned values are produced by sigma_star_self_consistent under the
    multiplicative specification and lambda_W = LAMBDA_W_PER_C * C with
    LAMBDA_W_PER_C = 1.93 (the C_model calibration relation, q_min = 0.1, q_max = 2.0). Any
    change to the threshold pipeline that affects these values should be
    reflected in the main-text threshold-results section and the SI (C, n)
    joint-sensitivity figure (sec:si-Cn-joint; generate_figure_Cn_joint.py).

    See:
      - generate_figure_Cn_joint.py (the figure that visualizes these values)
      - main-text threshold-results section (headline-range presentation)
      - Supp. sec:si-Cn-joint (the SI subsection)
    """

    LAMBDA_W_PER_C: float = 1.93  # Matches generate_figure_Cn_joint.py

    def test_sigma_star_decreases_with_n_at_central_C(self):
        """At fixed lambda_W = 0.68 (per-builder C ~ 0.35 central anchor),
        sigma* decreases monotonically with n: Layer-3 network saturation
        becomes more complete at larger n, so the network-buffering
        advantage at signaling is larger and a lower sigma* suffices."""
        sigs = []
        for n_val in (10, 25, 50, 100, 250):
            res = sigma_star_self_consistent(lambda_W=0.68, n=int(n_val))
            sigs.append(res["sigma_star"])
        assert all(sigs[i] >= sigs[i + 1] - 1e-6
                   for i in range(len(sigs) - 1)), \
            f"sigma*(n) not non-increasing: {sigs}"

    def test_sigma_star_values_at_n_anchors(self):
        """Pin sigma*(n) at fixed lambda_W = 0.68 (per-builder C ~ 0.35)
        across the face-to-face range. Reference values produced by
        generate_figure_Cn_joint.py and reported in the SI subsection
        sec:si-Cn-joint."""
        expected = {
            10: 0.488,
            25: 0.450,
            50: 0.432,
            100: 0.421,
            250: 0.410,
        }
        for n_val, exp_sigma in expected.items():
            res = sigma_star_self_consistent(lambda_W=0.68, n=int(n_val))
            assert res["sigma_star"] == pytest.approx(exp_sigma, abs=0.015), (
                f"sigma*(lambda_W=0.68, n={n_val}) = {res['sigma_star']:.4f}, "
                f"expected ~{exp_sigma}"
            )

    def test_sigma_star_rectangle_corners(self):
        """Pin the four corners of the in-scope (C, n) rectangle, applying
        the C -> lambda_W coupling lambda_W = 1.93 * C (the C_model calibration relation).
        These values define the headline sigma* range reported in
        the main-text threshold-results section."""
        expected = {
            (0.20, 10):  0.303,
            (0.20, 250): 0.236,
            (0.50, 10):  0.653,
            (0.50, 250): 0.578,
        }
        for (c, n_val), exp_sigma in expected.items():
            lambda_W = self.LAMBDA_W_PER_C * c
            res = sigma_star_self_consistent(lambda_W=lambda_W, n=int(n_val))
            assert res["sigma_star"] == pytest.approx(exp_sigma, abs=0.020), (
                f"sigma*(C={c}, n={n_val}) = {res['sigma_star']:.4f}, "
                f"expected ~{exp_sigma}"
            )

    def test_sigma_star_central_tendency_in_range(self):
        """At the rectangle midpoint (C = 0.35, n = 60), sigma* ~ 0.43, the
        central-tendency value reported in the main-text threshold-results
        section as a reading aid (not a privileged anchor)."""
        lambda_W = self.LAMBDA_W_PER_C * 0.35  # ~0.676
        res = sigma_star_self_consistent(lambda_W=lambda_W, n=60)
        assert res["sigma_star"] == pytest.approx(0.426, abs=0.020), (
            f"central sigma*(C=0.35, n=60) = {res['sigma_star']:.4f}, "
            f"expected ~0.426"
        )

    def test_sigma_star_increases_with_C_at_fixed_n(self):
        """At fixed n = 60, sigma* increases monotonically with C: the
        positional reward s_W ~ lambda_W (~1.93 * C) grows faster than the
        cost; the threshold rises as required for the manuscript's
        headline-range claim ([0.236, 0.653] over (C, n))."""
        n_val = 60
        sigs = []
        for c in (0.20, 0.30, 0.40, 0.50):
            lambda_W = self.LAMBDA_W_PER_C * c
            res = sigma_star_self_consistent(lambda_W=lambda_W, n=n_val)
            sigs.append(res["sigma_star"])
        assert all(sigs[i] <= sigs[i + 1] + 1e-6
                   for i in range(len(sigs) - 1)), \
            f"sigma*(C) not non-decreasing at n=60: {sigs}"

    def test_sigma_star_n_12_anchor_still_valid(self):
        """The legacy n = 12 anchor (for regression/backward compatibility; CALIBRATED UNDER n=12 ILLUSTRATION
        per the calibration.py annotation) still produces ~0.478 at
        lambda_W = 0.68. This test guards backward consistency: the
        framework no longer fixes n = 12 as the default, but the value at
        that point must be preserved as a legacy reference."""
        res = sigma_star_self_consistent(lambda_W=0.68, n=12)
        assert res["sigma_star"] == pytest.approx(0.478, abs=0.015), (
            f"sigma*(lambda_W=0.68, n=12) = {res['sigma_star']:.4f}, "
            f"expected ~0.478 (legacy reference anchor)"
        )


class TestIndividualFitnessRewardContract:
    """The signaling reward depends on OBSERVED investment through the
    schedule inverse q_hat(x) = sqrt(q_min^2 + x^2/lambda), not lambda*q
    at every x."""

    def test_zero_investment_gets_boundary_belief(self):
        w = individual_fitness(0.0, 1.0, sigma=0.0, k=0.0, P_conflict=0.0,
                               lam=0.5, mode="additive")
        # reproduction 1 (no cost) + S-1 = 0 + reward lam*q_min = 0.05
        assert float(w) == pytest.approx(1.05, abs=1e-12)

    def test_reward_tracks_investment_not_quality(self):
        """Two types making the same investment receive the same reward."""
        kwargs = dict(sigma=0.0, k=0.0, P_conflict=0.0, lam=0.5, mode="additive")
        w_lo = float(individual_fitness(0.4, 0.5, **kwargs))
        w_hi = float(individual_fitness(0.4, 1.5, **kwargs))
        # rewards equal; fitness differs only through the cost term c(x, q)
        c_lo, c_hi = 0.4**2 / (2 * 0.5), 0.4**2 / (2 * 1.5)
        assert (w_lo + c_lo) == pytest.approx(w_hi + c_hi, abs=1e-12)

    def test_equilibrium_path_reward_is_lambda_q(self):
        """On the equilibrium path q_hat(x*(q)) = q exactly."""
        from signaling.layer1 import equilibrium_investment
        q, q_min, lam = 1.3, 0.1, 0.5
        x = float(equilibrium_investment(q, q_min, lam))
        w = float(individual_fitness(x, q, sigma=0.0, k=0.0, P_conflict=0.0,
                                     lam=lam, mode="additive"))
        cost = x**2 / (2 * q)
        assert w == pytest.approx(1.0 - cost + lam * q, abs=1e-12)

class TestOmegaGuard:
    """omega must not be silently ignored (the same silent-override guard
    pattern as for r). Passing omega without mode='convex' would otherwise
    produce a flat sigma*(omega) = 0.4771 sweep with no error."""

    def test_omega_without_convex_mode_raises(self):
        with pytest.raises(ValueError, match="convex"):
            sigma_star_self_consistent(0.68, omega=0.5)
        with pytest.raises(ValueError, match="convex"):
            critical_threshold_sigma_star(C=0.35, omega=0.5)

    def test_convex_mode_consumes_omega(self):
        """The convex path moves the threshold: omega = 1 (additive limit)
        gives sigma* ~ 0.5267 vs the multiplicative 0.4771, matching the
        SI convex-combination section."""
        s1 = sigma_star_self_consistent(0.68, mode="convex", omega=1.0)["sigma_star"]
        s0 = sigma_star_self_consistent(0.68, mode="convex", omega=0.0)["sigma_star"]
        assert s0 == pytest.approx(0.4771, abs=1e-3)
        assert s1 == pytest.approx(0.5267, abs=1e-3)
        assert s1 - s0 > 0.03



class TestDeltaOneIsReference:
    """The manuscript's reference threshold
    (sigma* ~ 0.48, r ~ 0.43, alpha_eff ~ 0.33 at M_g = I_g ~ 10.3) is
    reported as the delta = 1 flow-assessment steady state, not as a
    "delta = 0 accounting" (which has no steady state). These tests pin the
    identity delta = 1 == reference path, the monotone sigma*(delta), and
    the central-point depreciation-on value that the text previously
    borrowed from the anchor (0.42) instead of measuring (0.4043)."""

    def test_delta_one_reproduces_reference_path_exactly(self):
        from signaling.layer3 import between_group_lambda_diagnostics
        ref = critical_threshold_sigma_star(lambda_W=0.68, delta=0.0)
        one = critical_threshold_sigma_star(lambda_W=0.68, delta=1.0)
        assert one["sigma_star"] == pytest.approx(ref["sigma_star"], abs=1e-9)
        assert one["sigma_star"] == pytest.approx(0.4771, abs=1e-3)
        d_ref = between_group_lambda_diagnostics(ref["sigma_star"], 0.68)
        d_one = between_group_lambda_diagnostics(one["sigma_star"], 0.68, delta=1.0)
        assert d_one["M_g"] == pytest.approx(d_ref["M_g"], abs=1e-9)
        assert d_one["M_g"] == pytest.approx(10.307, abs=2e-3)
        assert d_one["alpha_eff"] == pytest.approx(d_ref["alpha_eff"], abs=1e-9)

    def test_sigma_star_monotone_increasing_in_delta_at_anchor(self):
        deltas = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
        ss = [critical_threshold_sigma_star(lambda_W=0.68, delta=d)["sigma_star"] for d in deltas]
        assert all(b > a for a, b in zip(ss, ss[1:]))
        assert ss[0] == pytest.approx(0.4107, abs=1e-3)
        assert ss[-1] == pytest.approx(0.4771, abs=1e-3)
        assert ss[-1] - ss[0] < 0.07

    def test_central_point_depreciation_on_value(self):
        static = sigma_star_self_consistent(0.68, n=60, delta=1.0)["sigma_star"]
        dep = sigma_star_self_consistent(0.68, n=60, delta=0.10)["sigma_star"]
        assert static == pytest.approx(0.4288, abs=1e-3)
        assert dep == pytest.approx(0.4043, abs=1e-3)
        assert dep < static


class TestAssortmentCrossoverMagnitudes:
    """The F ~ 0.06 crossover quoted in main
    SS4.1 implies beta_1 ~ 16 beta_0 at the rare-builder limit. Pins the ratio,
    the crossover at both quoted sigma values, and the units reconciliation
    (int_0^1 beta_1 dp = W_g(1) - W_g(0); zero at sigma*; hump ~ 0.13) that
    shows a steep local slope is consistent with O(0.1) fitness differences."""

    @staticmethod
    def _vals(sigma):
        from signaling.price_equation import within_group_regression, between_group_regression
        b0 = within_group_regression(1e-3, sigma, 0.68)
        b1 = between_group_regression(1e-3, sigma, 0.68)
        return b0, b1

    def test_ratio_and_crossover_at_threshold(self):
        b0, b1 = self._vals(0.477)
        assert b0 == pytest.approx(0.148, abs=3e-3)
        assert b1 == pytest.approx(2.40, abs=0.02)
        assert b1 / b0 == pytest.approx(16.2, abs=0.3)
        assert b0 / (b0 + b1) == pytest.approx(0.058, abs=2e-3)

    def test_crossover_at_sigma_068(self):
        b0, b1 = self._vals(0.68)
        assert b0 / (b0 + b1) == pytest.approx(0.029, abs=2e-3)

    def test_beta1_integrates_to_delta_w(self):
        from signaling.price_equation import between_group_regression, mixed_group_mean_fitness
        ps = np.linspace(0.0, 1.0, 401)
        for sigma, dw in ((0.477, 0.0), (0.68, 0.1186)):
            integ = np.trapezoid([between_group_regression(p, sigma, 0.68) for p in ps], ps)
            direct = mixed_group_mean_fitness(1.0, sigma, 0.68) - mixed_group_mean_fitness(0.0, sigma, 0.68)
            assert integ == pytest.approx(direct, abs=2e-3)
            assert direct == pytest.approx(dw, abs=2e-3)
        hump = max(mixed_group_mean_fitness(p, 0.477, 0.68) for p in ps) - mixed_group_mean_fitness(0.0, 0.477, 0.68)
        assert hump == pytest.approx(0.135, abs=3e-3)
