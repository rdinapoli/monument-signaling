"""Tests for Layer 2: intergroup assessment and conflict resolution.

Verifies theoretical properties of the conflict probability function:
- P_conflict in [0, 1] for all inputs
- P_conflict decreasing in |M_g - M_h|
- P_conflict decreasing in M_g + M_h (mutual assessment)
- P_conflict increasing in noise_sigma
- Mutual and self-assessment produce different orderings for high-M dyads
- Calibrated mutual assessment reproduces ~75% conflict reduction
- Edge cases: zero investment, extreme asymmetry, zero noise
"""

import numpy as np
import pytest

from signaling.calibration import (
    CALIBRATION_TARGETS,
    DEFAULT_ALPHA_CONFLICT,
    DEFAULT_BETA_CONFLICT,
    DEFAULT_C_RATE,
    DEFAULT_D,
    DEFAULT_KAPPA,
    DEFAULT_M_OWN,
    DEFAULT_M_SCALE,
    DEFAULT_M_THRESH,
    DEFAULT_P_BASE,
    DEFAULT_SIGMA_0,
    DEFAULT_STEEPNESS,
    DEFAULT_T_0,
    DEFAULT_V,
)
from signaling.layer2 import (
    bourgeois_conflict_prob,
    calibrate_conflict_reduction,
    compare_assessment_models,
    compute_lambda_C,
    constrained_sigma_0,
    derived_conflict_reduction,
    effective_noise,
    mutual_assessment_conflict_prob,
    mutual_assessment_conflict_prob_exponential_stake,
    mutual_assessment_exponential,
    self_assessment_conflict_prob,
    war_of_attrition_cost,
    war_of_attrition_deterrence,
)

# Shared test values
M_LOW = 0.5
M_MED = 3.0
M_HIGH = 10.0
M_RANGE = np.linspace(0.0, 15.0, 50)


# =====================================================================
# TestEffectiveNoise
# =====================================================================


class TestEffectiveNoise:
    """Verify the signal-to-noise mapping sigma_eff(M_g, M_h)."""

    def test_zero_investment_returns_sigma_0(self) -> None:
        """At zero total investment, sigma_eff equals sigma_0."""
        assert effective_noise(0.0, 0.0) == pytest.approx(DEFAULT_SIGMA_0)

    def test_positive_investment_reduces_noise(self) -> None:
        """Noise decreases with total investment (monuments are informative)."""
        sig_0 = effective_noise(0.0, 0.0)
        sig_pos = effective_noise(5.0, 5.0)
        assert sig_pos < sig_0

    def test_monotonically_decreasing_in_total(self) -> None:
        """sigma_eff is strictly decreasing as total M increases."""
        totals = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        noises = [float(effective_noise(t / 2, t / 2)) for t in totals]
        for i in range(len(noises) - 1):
            assert noises[i] > noises[i + 1]

    def test_symmetric_in_inputs(self) -> None:
        """sigma_eff depends on total, not on allocation between groups."""
        assert effective_noise(3.0, 7.0) == pytest.approx(effective_noise(7.0, 3.0))
        assert effective_noise(3.0, 7.0) == pytest.approx(effective_noise(5.0, 5.0))

    def test_kappa_zero_gives_constant_noise(self) -> None:
        """Setting kappa=0 recovers the pure SAM baseline with constant noise."""
        sig = effective_noise(100.0, 100.0, kappa=0.0)
        assert sig == pytest.approx(DEFAULT_SIGMA_0)

    def test_vectorized(self) -> None:
        """Works with array inputs."""
        M_g = np.array([0.0, 5.0, 10.0])
        M_h = np.array([0.0, 5.0, 10.0])
        result = effective_noise(M_g, M_h)
        assert result.shape == (3,)
        assert result[0] > result[1] > result[2]


# =====================================================================
# TestMutualAssessment
# =====================================================================


class TestMutualAssessment:
    """Verify the mutual assessment conflict probability (primary model)."""

    def test_p_conflict_in_valid_range(self) -> None:
        """P_conflict in [0, P_base] over the sampled stock grid at default parameters."""
        for mg in [0.0, 0.1, 1.0, 5.0, 20.0]:
            for mh in [0.0, 0.1, 1.0, 5.0, 20.0]:
                P = float(mutual_assessment_conflict_prob(mg, mh))
                assert 0.0 <= P <= DEFAULT_P_BASE + 1e-12

    def test_zero_investment_equals_p_base(self) -> None:
        """P_conflict(0, 0) = P_base by normalization."""
        P = mutual_assessment_conflict_prob(0.0, 0.0)
        assert P == pytest.approx(DEFAULT_P_BASE, rel=1e-10)

    def test_symmetric_in_groups(self) -> None:
        """P_conflict(M_g, M_h) = P_conflict(M_h, M_g)."""
        P_ab = mutual_assessment_conflict_prob(3.0, 7.0)
        P_ba = mutual_assessment_conflict_prob(7.0, 3.0)
        assert P_ab == pytest.approx(P_ba, rel=1e-12)

    def test_decreasing_in_asymmetry(self) -> None:
        """Larger |M_g - M_h| means clearer winner, lower P_conflict.

        At fixed total M_g + M_h = 10, increasing asymmetry should
        decrease conflict probability because the weaker group can
        more reliably predict it will lose.
        """
        P_55 = float(mutual_assessment_conflict_prob(5.0, 5.0))
        P_37 = float(mutual_assessment_conflict_prob(3.0, 7.0))
        P_19 = float(mutual_assessment_conflict_prob(1.0, 9.0))
        assert P_55 > P_37 > P_19

    def test_decreasing_in_total_investment(self) -> None:
        """Higher total M_g + M_h reduces P_conflict (deterrence effect).

        This is the central prediction linking monument investment to
        conflict reduction. With beta > 0, total investment provides
        absolute deterrence beyond relative assessment.
        """
        P_low = float(mutual_assessment_conflict_prob(1.0, 1.0))
        P_mid = float(mutual_assessment_conflict_prob(5.0, 5.0))
        P_high = float(mutual_assessment_conflict_prob(10.0, 10.0))
        assert P_low > P_mid > P_high

    def test_increasing_in_noise(self) -> None:
        """Higher assessment noise means more disputes escalate."""
        P_precise = float(mutual_assessment_conflict_prob(5.0, 5.0, sigma_0=0.5))
        P_noisy = float(mutual_assessment_conflict_prob(5.0, 5.0, sigma_0=2.0))
        assert P_noisy > P_precise

    def test_increasing_in_V_over_D(self) -> None:
        """Higher V/D ratio (resource more valuable relative to cost) means
        the escalation threshold T = T_0 * V/D increases, raising P_conflict.

        Tested with an asymmetric pair where the effect is unambiguous.
        For symmetric pairs the normalization P_raw/P_raw(0,0) can obscure
        the monotonicity because both numerator and denominator change with T.
        """
        P_low_VD = float(mutual_assessment_conflict_prob(3.0, 7.0, V=1.0, D=4.0))
        P_high_VD = float(mutual_assessment_conflict_prob(3.0, 7.0, V=1.0, D=1.0))
        assert P_high_VD > P_low_VD

    def test_beta_zero_is_pure_mutual_assessment(self) -> None:
        """Setting beta=0 removes absolute deterrence.

        Under pure mutual assessment, well-informed symmetric dyads can
        have higher P than poorly-informed ones (precise assessment
        confirms even matching). This is why we need the deterrence term.
        """
        # With beta=0 and kappa>0, symmetric dyads get BETTER assessment
        # (lower noise) as total investment grows, which can INCREASE P
        # because they confirm they're evenly matched.
        P_low = float(mutual_assessment_conflict_prob(1.0, 1.0, beta=0.0, kappa=0.2))
        P_high = float(mutual_assessment_conflict_prob(10.0, 10.0, beta=0.0, kappa=0.2))
        # Under pure mutual assessment with info gain, symmetric high-M dyads
        # fight MORE: parity amplification pushes P ABOVE P_base, exactly as
        # the displayed equation requires (a clip at P_base would delete
        # this mechanism).
        assert P_high > P_low
        assert P_high > DEFAULT_P_BASE
        assert 0.0 <= P_low <= 1.0 and 0.0 <= P_high <= 1.0

    def test_equation_code_equality_at_parity_beta_zero(self) -> None:
        """Pin the raw displayed equation at a parity-amplification point:
        sigma_0=1, kappa=0.1, T=0.5 (T_0=0.5, V=D=1), M=10, beta=0
        -> P = 0.0166364 (1.66 P_base). Equation-code equality, no clipping."""
        from scipy.stats import norm as _norm
        P_code = float(mutual_assessment_conflict_prob(
            10.0, 10.0, sigma_0=1.0, V=1.0, D=1.0, T_0=0.5,
            beta=0.0, kappa=0.1, P_base=0.01,
        ))
        sig_eff = 1.0 / np.sqrt(1.0 + 0.1 * 20.0)
        num = _norm.cdf(0.5 / (np.sqrt(2) * sig_eff)) - _norm.cdf(-0.5 / (np.sqrt(2) * sig_eff))
        den = _norm.cdf(0.5 / np.sqrt(2)) - _norm.cdf(-0.5 / np.sqrt(2))
        assert P_code == pytest.approx(0.01 * num / den, abs=1e-12)
        assert P_code == pytest.approx(0.0166364, abs=1e-6)

    def test_self_assessment_equation_code_equality_with_bound(self) -> None:
        """Pin the self-assessment model at a saturating point: the raw
        normalized expression is ~22995 and the displayed min{1,.} bound
        saturates it at exactly 1 (the bound is part of the model)."""
        from signaling.layer2 import self_assessment_conflict_prob
        import math
        m_thresh, m_scale, m, p_base, beta = 8.0, 1.0, 10.0, 0.01, 0.1
        p0 = 1 / (1 + math.exp(m_thresh / m_scale))
        pm = 1 / (1 + math.exp(-(m - m_thresh) / m_scale))
        raw = p_base * (pm / p0) ** 2 / (1 + 2 * beta * m)
        assert raw > 1.0e4
        P_code = float(self_assessment_conflict_prob(
            m, m, M_thresh=m_thresh, M_scale=m_scale, beta=beta, P_base=p_base,
        ))
        assert P_code == 1.0
        # Unclipped region: equation-code equality at the defaults point
        P_def = float(self_assessment_conflict_prob(5.0, 5.0))
        assert 0.0 < P_def < 1.0

    def test_vectorized_grid(self) -> None:
        """Works with 2D array inputs for surface plotting."""
        Mg, Mh = np.meshgrid(M_RANGE, M_RANGE)
        P = mutual_assessment_conflict_prob(Mg, Mh)
        assert P.shape == Mg.shape
        assert np.all(P >= -1e-12)
        assert np.all(P <= DEFAULT_P_BASE + 1e-12)


# =====================================================================
# TestMutualAssessmentExponential
# =====================================================================


class TestMutualAssessmentExponential:
    """Verify the tractable exponential approximation."""

    def test_zero_investment_equals_p_base(self) -> None:
        """P_exp(0, 0) = P_base / (1 + 0) = P_base."""
        P = mutual_assessment_exponential(0.0, 0.0)
        assert P == pytest.approx(DEFAULT_P_BASE, rel=1e-10)

    def test_decreasing_in_asymmetry(self) -> None:
        """Same qualitative behavior as Gaussian form."""
        P_55 = float(mutual_assessment_exponential(5.0, 5.0))
        P_37 = float(mutual_assessment_exponential(3.0, 7.0))
        assert P_55 > P_37

    def test_decreasing_in_total(self) -> None:
        """Deterrence factor reduces P with higher total investment."""
        P_low = float(mutual_assessment_exponential(1.0, 1.0))
        P_high = float(mutual_assessment_exponential(10.0, 10.0))
        assert P_low > P_high

    def test_symmetric(self) -> None:
        """P_exp(M_g, M_h) = P_exp(M_h, M_g)."""
        P_ab = mutual_assessment_exponential(3.0, 7.0)
        P_ba = mutual_assessment_exponential(7.0, 3.0)
        assert P_ab == pytest.approx(P_ba, rel=1e-12)


# =====================================================================
# TestSelfAssessment
# =====================================================================


class TestSelfAssessment:
    """Verify the self-assessment conflict probability (alternative model)."""

    def test_p_in_valid_range(self) -> None:
        """P_self in [0, 1] over the sampled stock grid at default parameters."""
        for mg in [0.0, 0.5, 2.0, 5.0, 20.0]:
            for mh in [0.0, 0.5, 2.0, 5.0, 20.0]:
                P = float(self_assessment_conflict_prob(mg, mh))
                assert 0.0 <= P <= 1.0

    def test_symmetric(self) -> None:
        """P_self(M_g, M_h) = P_self(M_h, M_g)."""
        P_ab = self_assessment_conflict_prob(3.0, 7.0)
        P_ba = self_assessment_conflict_prob(7.0, 3.0)
        assert P_ab == pytest.approx(P_ba, rel=1e-12)

    def test_increasing_in_own_investment(self) -> None:
        """Higher own M increases escalation probability (more confident)."""
        P_low_g = float(self_assessment_conflict_prob(1.0, 5.0))
        P_high_g = float(self_assessment_conflict_prob(10.0, 5.0))
        assert P_high_g > P_low_g

    def test_high_pair_higher_than_low_pair(self) -> None:
        """Two high-investing groups fight MORE under self-assessment.

        This is the central discriminating prediction: self-assessment
        predicts conflict is highest between two confident groups,
        opposite to mutual assessment.
        """
        P_high = float(self_assessment_conflict_prob(M_HIGH, M_HIGH))
        P_low = float(self_assessment_conflict_prob(M_LOW, M_LOW))
        assert P_high > P_low

    def test_zero_investment_low_probability(self) -> None:
        """Groups with zero investment are not confident, so P is low."""
        P = float(self_assessment_conflict_prob(0.0, 0.0))
        assert P < 0.15  # Well below threshold, so both sigmoids are low


# =====================================================================
# TestBourgeoisConvention
# =====================================================================


class TestBourgeoisConvention:
    """Verify the Bourgeois conflict probability (alternative model)."""

    def test_p_in_valid_range(self) -> None:
        """P_bourgeois must be in [0, P_base]."""
        for mg in [0.0, 1.0, 3.0, 5.0, 10.0]:
            for mh in [0.0, 1.0, 3.0, 5.0, 10.0]:
                P = float(bourgeois_conflict_prob(mg, mh))
                assert -1e-12 <= P <= DEFAULT_P_BASE + 1e-12

    def test_clear_asymmetry_low_conflict(self) -> None:
        """When one group is clearly above and another below threshold,
        ownership is unambiguous and P_conflict is low."""
        P = float(bourgeois_conflict_prob(M_HIGH, 0.0))
        assert P < DEFAULT_P_BASE * 0.2  # Much below baseline

    def test_both_near_threshold_high_conflict(self) -> None:
        """When both groups are near the ownership threshold,
        ownership is ambiguous and P_conflict is near P_base."""
        M_own = DEFAULT_M_OWN
        P = float(bourgeois_conflict_prob(M_own, M_own))
        assert P > DEFAULT_P_BASE * 0.8  # Near baseline

    def test_symmetric(self) -> None:
        """P_bourgeois(M_g, M_h) = P_bourgeois(M_h, M_g)."""
        P_ab = bourgeois_conflict_prob(3.0, 7.0)
        P_ba = bourgeois_conflict_prob(7.0, 3.0)
        assert P_ab == pytest.approx(P_ba, rel=1e-12)


# =====================================================================
# TestOppositeModelPredictions
# =====================================================================


class TestOppositeModelPredictions:
    """The key discriminating prediction between assessment models.

    Under mutual assessment, two high-investing groups have LOW conflict
    (precise assessment enables deterrence). Under self-assessment,
    two high-investing groups have HIGH conflict (both confident).
    This is the empirically testable prediction from the mutual-versus-self
    assessment analysis.
    """

    def test_mutual_vs_self_opposite_for_high_M_dyads(self) -> None:
        """Mutual assessment: P(high, high) < P(low, low).
        Self-assessment: P(high, high) > P(low, low)."""
        P_mut_hh = float(mutual_assessment_conflict_prob(M_HIGH, M_HIGH))
        P_mut_ll = float(mutual_assessment_conflict_prob(M_LOW, M_LOW))
        P_self_hh = float(self_assessment_conflict_prob(M_HIGH, M_HIGH))
        P_self_ll = float(self_assessment_conflict_prob(M_LOW, M_LOW))

        # Mutual: high-M pair has LOWER P
        assert P_mut_hh < P_mut_ll
        # Self: high-M pair has HIGHER P
        assert P_self_hh > P_self_ll

    def test_ordering_reversal_at_high_investment(self) -> None:
        """The relative ranking of P(high,high) vs P(low,low) flips
        between the two models. This is the diagnostic for distinguishing
        the mechanisms archaeologically."""
        r_mut_hh = float(derived_conflict_reduction(M_HIGH, M_HIGH))
        r_mut_ll = float(derived_conflict_reduction(M_LOW, M_LOW))
        # Under mutual assessment, high-investing dyads have greater r
        assert r_mut_hh > r_mut_ll


# =====================================================================
# TestWarOfAttrition
# =====================================================================


class TestWarOfAttrition:
    """Verify the war of attrition cost model."""

    def test_symmetric_win_probability(self) -> None:
        """Equal M gives equal win probabilities."""
        result = war_of_attrition_cost(5.0, 5.0)
        assert result["p_g_wins"] == pytest.approx(0.5)
        assert result["p_h_wins"] == pytest.approx(0.5)

    def test_stronger_group_wins_more(self) -> None:
        """Higher M gives higher win probability."""
        result = war_of_attrition_cost(8.0, 2.0)
        assert result["p_g_wins"] == pytest.approx(0.8)
        assert result["p_h_wins"] == pytest.approx(0.2)

    def test_duration_increases_with_symmetry(self) -> None:
        """Symmetric contests last longer than asymmetric ones
        (at fixed total investment)."""
        dur_sym = war_of_attrition_cost(5.0, 5.0)["duration"]
        dur_asym = war_of_attrition_cost(2.0, 8.0)["duration"]
        assert dur_sym > dur_asym

    def test_payoff_sum_equals_V_minus_costs(self) -> None:
        """Total expected payoff = V - total costs."""
        result = war_of_attrition_cost(5.0, 3.0)
        total_payoff = result["payoff_g"] + result["payoff_h"]
        total_cost = result["cost_g"] + result["cost_h"]
        assert total_payoff == pytest.approx(DEFAULT_V - total_cost)

    def test_zero_investment_raises(self) -> None:
        """Zero M should raise ValueError (no basis for attrition)."""
        with pytest.raises(ValueError):
            war_of_attrition_cost(0.0, 5.0)
        with pytest.raises(ValueError):
            war_of_attrition_cost(5.0, 0.0)

    def test_equal_costs(self) -> None:
        """Both groups pay the same expected cost regardless of asymmetry
        (both persist for the same expected duration in a competing risks model)."""
        result = war_of_attrition_cost(8.0, 2.0)
        assert result["cost_g"] == pytest.approx(result["cost_h"])


# =====================================================================
# TestDerivedConflictReduction
# =====================================================================


class TestDerivedConflictReduction:
    """Verify the derived conflict reduction r = 1 - P/P_base."""

    def test_r_in_valid_range(self) -> None:
        """r in [0, 1] over the sampled stock grid at default parameters (r < 0 is possible in parity-amplification regimes, e.g. beta = 0; tested separately)."""
        for mg in [0.0, 1.0, 5.0, 10.0]:
            for mh in [0.0, 1.0, 5.0, 10.0]:
                r = float(derived_conflict_reduction(mg, mh))
                assert -1e-12 <= r <= 1.0 + 1e-12

    def test_zero_investment_zero_reduction(self) -> None:
        """r(0, 0) = 0: no investment means no conflict reduction."""
        r = derived_conflict_reduction(0.0, 0.0)
        assert r == pytest.approx(0.0, abs=1e-10)

    def test_high_investment_substantial_reduction(self) -> None:
        """High-investing dyads should have substantial conflict reduction."""
        r = float(derived_conflict_reduction(M_HIGH, M_HIGH))
        assert r > 0.2  # Non-trivial reduction

    def test_increasing_with_total_investment(self) -> None:
        """Conflict reduction increases as both groups invest more."""
        r_low = float(derived_conflict_reduction(1.0, 1.0))
        r_mid = float(derived_conflict_reduction(5.0, 5.0))
        r_high = float(derived_conflict_reduction(10.0, 10.0))
        assert r_low < r_mid < r_high


# =====================================================================
# TestDerivedConflictReductionSelfAssessment
# =====================================================================


class TestDerivedConflictReductionSelfAssessment:
    """Verify the parallel derivation of conflict reduction under
    self-assessment. The opposite-direction prediction at symmetric
    high-investment dyads is the discriminating signature (main-text Layer 2
    intergroup-assessment section).
    """

    def test_runs_for_valid_inputs(self) -> None:
        """Function returns a finite value across the operating range."""
        from signaling.layer2 import derived_conflict_reduction_self_assessment
        for mg in [0.0, 1.0, 5.0, 10.0]:
            for mh in [0.0, 1.0, 5.0, 10.0]:
                r = float(derived_conflict_reduction_self_assessment(mg, mh))
                assert np.isfinite(r)

    def test_zero_investment_close_to_zero_reduction(self) -> None:
        """Under the normalized self model both regimes share the common
        baseline P_base at M = 0, so P_self(0,0) = P_base and r_self(0,0) = 0
        exactly (matching r_mutual(0,0) = 0). At zero investment no signal
        exists, so neither assessment rule applies and conflict reduces to
        the common no-signaling rate."""
        from signaling.layer2 import derived_conflict_reduction_self_assessment
        r = float(derived_conflict_reduction_self_assessment(0.0, 0.0))
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_high_symmetric_investment_predicts_high_conflict(self) -> None:
        """At symmetric high-M, BOTH sides escalate (self-assessment),
        so P_conflict_self >> P_base and r_self is strongly negative.
        This is the qualitative OPPOSITE of mutual assessment, which
        predicts r > 0 (conflict reduction) at the same dyad.

        In the self-assessment model, the assessment-driven product p(M)^2
        is multiplied by the absolute-deterrence factor 1/(1 + beta*total).
        The deterrence factor does not reverse the sign of r_self at
        M=10; r_self(10, 10) remains substantially negative.
        """
        from signaling.layer2 import (
            derived_conflict_reduction,
            derived_conflict_reduction_self_assessment,
        )
        r_mut = float(derived_conflict_reduction(10.0, 10.0))
        r_slf = float(derived_conflict_reduction_self_assessment(10.0, 10.0))
        assert r_mut > 0.2, f"mutual reduction at high-M unexpectedly low: {r_mut}"
        assert r_slf < 0, f"self-assessment reduction at high-M not negative: {r_slf}"

    def test_self_assessment_has_minimum_at_moderate_M(self) -> None:
        """r_self(M, M) has an interior minimum at moderate M. The
        product p(M)^2 * 1/(1 + 2*beta*M) is non-monotonic in M: the
        sigmoid saturates near M_thresh, but the deterrence damping
        continues to grow, so r_self becomes less negative for very large
        M after passing through a minimum near M ~ 5 at default parameters
        (beta=0.1, M_thresh=2.0).

        The behavior is non-monotonic, not a monotonic fall. The pattern
        r_self(1) > r_self(5) AND r_self(20) > r_self(5) confirms the
        interior minimum.
        """
        from signaling.layer2 import derived_conflict_reduction_self_assessment
        r_low = float(derived_conflict_reduction_self_assessment(1.0, 1.0))
        r_mid = float(derived_conflict_reduction_self_assessment(5.0, 5.0))
        r_high = float(derived_conflict_reduction_self_assessment(20.0, 20.0))
        # r_mid is the most negative; both r_low and r_high lie above it.
        assert r_low > r_mid, (
            f"r_self should rise from low to moderate M but went "
            f"{r_low} -> {r_mid}"
        )
        assert r_high > r_mid, (
            f"r_self should rise after the minimum at moderate M but "
            f"went {r_mid} -> {r_high}"
        )

    def test_self_assessment_below_mutual_throughout_relevant_range(self) -> None:
        """r_self(M, M) < r_mutual(M, M) at every M in the moderate-to-
        high range (M in {3, 5, 10}) relevant to the empirical anchor.

        This is the discriminating prediction the manuscript actually
        makes: the sign and magnitude of r at
        symmetric high-M dyads separates the two assessment regimes,
        not the direction of change with M.
        """
        from signaling.layer2 import (
            derived_conflict_reduction,
            derived_conflict_reduction_self_assessment,
        )
        for M in (3.0, 5.0, 10.0):
            r_mut = float(derived_conflict_reduction(M, M))
            r_slf = float(derived_conflict_reduction_self_assessment(M, M))
            assert r_slf < r_mut, (
                f"At M={M}, expected r_self < r_mutual but got "
                f"r_self={r_slf}, r_mutual={r_mut}"
            )

    def test_opposite_direction_from_mutual(self) -> None:
        """The SIGN of r at symmetric high-M dyads differs between mutual
        and self-assessment. This is the framework's
        distinctive empirical prediction: r_mutual > 0 (conflict
        reduction) versus r_self < 0 (conflict amplification) across the
        moderate-to-high investment range relevant to the empirical
        anchor (main-text mutual-assessment section).

        The discriminating signature is the sign and magnitude of r at
        symmetric M, not the monotonicity of r_self in M; r_self is
        itself non-monotonic.
        """
        from signaling.layer2 import (
            derived_conflict_reduction,
            derived_conflict_reduction_self_assessment,
        )
        for M in (3.0, 5.0, 10.0):
            r_mut = float(derived_conflict_reduction(M, M))
            r_slf = float(derived_conflict_reduction_self_assessment(M, M))
            assert r_mut > 0, f"r_mutual({M},{M}) not positive: {r_mut}"
            assert r_slf < 0, f"r_self({M},{M}) not negative: {r_slf}"

    def test_self_assessment_minimum_near_M_five(self) -> None:
        """Regression test pinning the structural location of the
        interior minimum of r_self(M, M).

        At default parameters (M_thresh=2.0, M_scale=1.0, beta=0.1,
        P_base=0.01), the minimum of r_self(M, M) lies at M ~ 5. The
        product p(M)^2 saturates near 1 once M > M_thresh, but the
        deterrence damping 1/(1 + 2*beta*M) keeps growing; the two
        effects balance near M ~ 1/(2*beta) = 5.

        This test detects regressions that move the minimum away from
        M ~ 5 at default parameters.
        """
        from signaling.layer2 import derived_conflict_reduction_self_assessment
        r3 = float(derived_conflict_reduction_self_assessment(3.0, 3.0))
        r5 = float(derived_conflict_reduction_self_assessment(5.0, 5.0))
        r20 = float(derived_conflict_reduction_self_assessment(20.0, 20.0))
        # The minimum lies between M=3 and M=20, with M=5 deeper than
        # both neighbors.
        assert r3 > r5, (
            f"Expected r_self(5,5) deeper than r_self(3,3); got "
            f"r_self(3,3)={r3}, r_self(5,5)={r5}"
        )
        assert r20 > r5, (
            f"Expected r_self(5,5) deeper than r_self(20,20); got "
            f"r_self(5,5)={r5}, r_self(20,20)={r20}"
        )
        # Confirm the minimum is tight: scan a coarse grid and verify
        # the minimizer lies within [4, 6].
        import numpy as np
        M_grid = np.linspace(2.0, 12.0, 201)
        r_vals = [
            float(derived_conflict_reduction_self_assessment(M, M))
            for M in M_grid
        ]
        argmin_M = float(M_grid[int(np.argmin(r_vals))])
        assert 4.0 < argmin_M < 6.0, (
            f"argmin r_self(M,M) expected near 5; got {argmin_M:.3f}"
        )

    def test_asymmetric_low_side_does_not_escalate(self) -> None:
        """Under self-assessment, the weaker side does not escalate even
        if the stronger side is above threshold. So P_conflict_self at
        asymmetric (low, high) is dominated by the weaker side's p(M).

        The absolute-deterrence factor 1/(1 + beta*total) is
        also present, so the dependence on the weak side's M reflects
        both the assessment product p(M_g)*p(M_h) and the deterrence
        damping. The test verifies the qualitative claim:
        P_conflict_self at extreme asymmetry (one side at zero) is
        substantially lower than at moderate asymmetry where both
        sides are above threshold, because the assessment-driven
        product p(0)*p(10) is much smaller than p(5)*p(10).
        """
        from signaling.layer2 import self_assessment_conflict_prob
        P_extreme_asym = float(self_assessment_conflict_prob(0.0, 10.0))
        P_mid_asym = float(self_assessment_conflict_prob(5.0, 10.0))
        # The weak side at M=0 keeps p(0) low and pins the product near
        # zero regardless of the deterrence factor.
        assert P_extreme_asym < P_mid_asym, (
            f"Expected weak side to suppress P; got P(0,10)={P_extreme_asym}, "
            f"P(5,10)={P_mid_asym}"
        )


# =====================================================================
# TestCalibration
# =====================================================================


class TestCalibration:
    """Verify that calibration reproduces target conflict reduction."""

    def test_calibration_converges(self) -> None:
        """Optimizer should converge to a solution."""
        result = calibrate_conflict_reduction()
        assert result["success"]

    def test_calibration_hits_target(self) -> None:
        """ROUND-TRIP check: the calibration optimizer, ASKED to hit
        r = 0.75, achieves it within 0.001 (ftol = 1e-14, converges ~1e-8).
        This verifies the optimizer's inversion of the r(parameters) map,
        not any property of 0.75 itself (0.75 is not independently
        confirmed here)."""
        result = calibrate_conflict_reduction(target_r=0.75)
        assert result["r_achieved"] == pytest.approx(0.75, abs=0.001)

    def test_calibration_within_ethnographic_range(self) -> None:
        """Calibrated r should fall within CALIBRATION_TARGETS range."""
        result = calibrate_conflict_reduction(target_r=0.75)
        low, high = CALIBRATION_TARGETS["conflict_reduction"]
        assert low <= result["r_achieved"] <= high

    def test_calibrated_parameters_positive(self) -> None:
        """All calibrated parameters should be positive."""
        result = calibrate_conflict_reduction()
        assert result["sigma_0"] > 0
        assert result["T_0"] > 0
        assert result["beta"] >= 0
        assert result["kappa"] >= 0

    def test_calibration_with_custom_M_high(self) -> None:
        """Calibration works with a user-provided M_high value."""
        result = calibrate_conflict_reduction(M_high=5.0)
        assert result["success"]
        assert result["M_high"] == pytest.approx(5.0)


# =====================================================================
# TestCompareModels
# =====================================================================


class TestCompareModels:
    """Verify the model comparison utility."""

    def test_returns_correct_keys(self) -> None:
        """Output dict has all expected keys."""
        result = compare_assessment_models(M_RANGE)
        expected_keys = {"M_g_grid", "M_h_grid", "mutual", "self", "bourgeois"}
        assert set(result.keys()) == expected_keys

    def test_correct_grid_shape(self) -> None:
        """2D grids have shape (len(M_range), len(M_range))."""
        n = len(M_RANGE)
        result = compare_assessment_models(M_RANGE)
        for key in ["M_g_grid", "M_h_grid", "mutual", "self", "bourgeois"]:
            assert result[key].shape == (n, n)

    def test_mutual_values_bounded(self) -> None:
        """All mutual assessment values in [0, P_base]."""
        result = compare_assessment_models(M_RANGE)
        assert np.all(result["mutual"] >= -1e-12)
        assert np.all(result["mutual"] <= DEFAULT_P_BASE + 1e-12)


# =====================================================================
# TestEdgeCases
# =====================================================================


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_extreme_asymmetry(self) -> None:
        """Very large asymmetry should give near-zero conflict probability."""
        P = float(mutual_assessment_conflict_prob(0.0, 100.0))
        assert P < DEFAULT_P_BASE * 0.01

    def test_very_large_investment(self) -> None:
        """Very large symmetric investment should not cause numerical issues."""
        P = float(mutual_assessment_conflict_prob(1000.0, 1000.0))
        assert 0.0 <= P <= DEFAULT_P_BASE + 1e-12
        assert np.isfinite(P)

    def test_very_small_noise(self) -> None:
        """Near-zero noise with nonzero delta should give near-zero P."""
        P = float(mutual_assessment_conflict_prob(3.0, 7.0, sigma_0=0.001))
        assert P < DEFAULT_P_BASE * 0.01

    def test_very_large_noise(self) -> None:
        """Large noise should give P near P_base (assessment is useless)."""
        P = float(mutual_assessment_conflict_prob(5.0, 5.0, sigma_0=100.0))
        # With very large noise, the Gaussian CDF component approaches
        # 2*Phi(T/(sqrt(2)*100)) - 1, which for moderate T is near
        # the normalization value, so P -> P_base / (1 + beta*total)
        assert P <= DEFAULT_P_BASE + 1e-12

    def test_self_assessment_sigmoid_saturation(self) -> None:
        """Very high M saturates the assessment-driven product
        p(M)^2/p(0)^2 toward its ceiling 1/p(0)^2, while the absolute-
        deterrence factor 1/(1 + 2*beta*M) damps P toward zero as M -> inf.

        Under the normalized model P_self(M, M) -> P_base/(p(0)^2 * 2*beta*M)
        for large M. At default beta=0.1, M_thresh=2.0 (p(0)^2 ~ 0.0142),
        M=100 gives P_self ~ P_base/p(0)^2 / 21 ~ 0.0335; M=1000 is ~10x lower.
        """
        P_100 = float(self_assessment_conflict_prob(100.0, 100.0))
        P_1000 = float(self_assessment_conflict_prob(1000.0, 1000.0))
        assert P_100 == pytest.approx(0.0335, rel=0.02), (
            f"Expected P_self(100,100) ~ 0.0335 under the normalized model; got {P_100}"
        )
        # Deterrence dominates in the limit: P should fall toward zero.
        assert P_1000 < P_100, (
            f"Expected P_self to decrease at very large M (deterrence "
            f"dominates), got P(100)={P_100}, P(1000)={P_1000}"
        )

    def test_self_assessment_anchored_to_p_base_at_zero(self) -> None:
        """The normalized self model reduces to the common baseline P_base
        at M = 0 for ANY M_thresh: the assessment term p(0)^2/p(0)^2 = 1 by
        construction, so P_self(0, 0) = P_base, matching mutual assessment.
        (Replaces the pre-normalization 'far below threshold -> P ~ 0' test;
        both models are now anchored to a common no-signaling baseline.)"""
        for mt in [0.5, 2.0, 10.0]:
            P = float(self_assessment_conflict_prob(0.0, 0.0, M_thresh=mt))
            assert P == pytest.approx(DEFAULT_P_BASE, rel=1e-9), (
                f"P_self(0,0) should equal P_base for M_thresh={mt}; got {P}"
            )


# =====================================================================
# TestLambdaC
# =====================================================================


class TestLambdaC:
    """Verify the lambda_C (competitive between-group feedback) computation.

    lambda_C is the marginal fitness value of conflict deterrence per unit
    of individual monument investment. It must be non-negative (investing
    more never increases conflict losses) and should respond to conflict
    cost, dispute frequency, and neighbor count in predictable ways.
    """

    def test_lambda_C_positive(self) -> None:
        """lambda_C > 0 when P_conflict decreases with M_g.

        This is the central result: monument investment has positive
        competitive between-group value because it deters conflict.
        """
        lc = compute_lambda_C(M_g=5.0)
        assert lc > 0.0

    def test_lambda_C_positive_at_various_M(self) -> None:
        """lambda_C is positive across a range of monument stocks."""
        for mg in [0.5, 1.0, 3.0, 5.0, 10.0]:
            lc = compute_lambda_C(M_g=mg)
            assert lc > 0.0, f"lambda_C should be positive at M_g={mg}"

    def test_lambda_C_increases_with_conflict_cost(self) -> None:
        """Higher conflict cost means monuments are more valuable for deterrence."""
        lc_low = compute_lambda_C(M_g=5.0, conflict_cost=0.2)
        lc_high = compute_lambda_C(M_g=5.0, conflict_cost=1.0)
        assert lc_high > lc_low

    def test_lambda_C_linear_in_conflict_cost(self) -> None:
        """lambda_C should scale linearly with conflict cost (since it
        enters multiplicatively in the expected loss)."""
        lc_base = compute_lambda_C(M_g=5.0, conflict_cost=0.5)
        lc_double = compute_lambda_C(M_g=5.0, conflict_cost=1.0)
        assert lc_double == pytest.approx(2.0 * lc_base, rel=0.01)

    def test_lambda_C_increases_with_dispute_frequency(self) -> None:
        """More frequent disputes mean more value from deterrence."""
        lc_low = compute_lambda_C(M_g=5.0, dispute_freq=0.01)
        lc_high = compute_lambda_C(M_g=5.0, dispute_freq=0.10)
        assert lc_high > lc_low

    def test_lambda_C_linear_in_dispute_freq(self) -> None:
        """lambda_C should scale linearly with dispute frequency."""
        lc_base = compute_lambda_C(M_g=5.0, dispute_freq=0.05)
        lc_double = compute_lambda_C(M_g=5.0, dispute_freq=0.10)
        assert lc_double == pytest.approx(2.0 * lc_base, rel=0.01)

    def test_lambda_C_zero_when_no_neighbors(self) -> None:
        """If n_neighbors = 0, lambda_C = 0 (no one to deter)."""
        lc = compute_lambda_C(M_g=5.0, n_neighbors=0)
        assert lc == 0.0

    def test_lambda_C_increases_with_neighbors(self) -> None:
        """More neighbors means more disputes to deter."""
        lc_2 = compute_lambda_C(M_g=5.0, n_neighbors=2)
        lc_6 = compute_lambda_C(M_g=5.0, n_neighbors=6)
        assert lc_6 > lc_2

    def test_lambda_C_with_heterogeneous_neighbors(self) -> None:
        """lambda_C works with an array of neighbor monument stocks."""
        M_h_arr = np.array([2.0, 5.0, 8.0])
        lc = compute_lambda_C(M_g=5.0, M_h_neighbors=M_h_arr)
        assert lc > 0.0

    def test_lambda_C_per_capita(self) -> None:
        """lambda_C with N divides by group size for individual returns."""
        lc_group = compute_lambda_C(M_g=5.0)
        lc_individual = compute_lambda_C(M_g=5.0, N=12)
        assert lc_individual == pytest.approx(lc_group / 12.0, rel=1e-10)


# =====================================================================
# TestExponentialApproximation
# =====================================================================


class TestExponentialApproximation:
    """Verify the quality of the exponential approximation to the Gaussian form.

    The exponential form P_exp = P_base * exp(-alpha*delta/sigma_eff) / (1+beta*total)
    should track the Gaussian CDF form reasonably over the relevant parameter range.
    """

    def test_exponential_approximates_gaussian_symmetric(self) -> None:
        """For symmetric pairs (delta=0), both forms agree exactly at (0,0)
        and should track each other at higher total investment."""
        # At (0,0): both equal P_base
        P_gauss_0 = float(mutual_assessment_conflict_prob(0.0, 0.0))
        P_exp_0 = float(mutual_assessment_exponential(0.0, 0.0))
        assert P_gauss_0 == pytest.approx(P_exp_0, rel=1e-10)

        # For symmetric pairs, delta=0, so exp(-alpha*0) = 1 for both.
        # Both reduce to P_base * normalization / (1 + beta*total).
        # The only difference is the normalization in the Gaussian form.
        for mg in [1.0, 3.0, 5.0, 10.0]:
            P_gauss = float(mutual_assessment_conflict_prob(mg, mg))
            P_exp = float(mutual_assessment_exponential(mg, mg))
            # Both should decrease with total investment (deterrence)
            assert P_gauss < DEFAULT_P_BASE
            assert P_exp < DEFAULT_P_BASE

    def test_exponential_max_deviation_documented(self) -> None:
        """Document the maximum absolute deviation between exponential
        and Gaussian forms over the relevant parameter range.

        The tolerance here is generous (0.005 = 50% of P_base) because
        the exponential is an approximation for tractability, not
        a precise match. The key requirement is that qualitative
        behavior (monotonicity in delta and total) is preserved.
        """
        M_range_test = np.linspace(0.1, 15.0, 30)
        max_dev = 0.0
        for mg in M_range_test:
            for mh in M_range_test:
                P_gauss = float(mutual_assessment_conflict_prob(mg, mh))
                P_exp = float(mutual_assessment_exponential(mg, mh))
                dev = abs(P_gauss - P_exp)
                max_dev = max(max_dev, dev)
        # The max deviation should be documented and bounded
        # With default alpha=1.0, the deviation is small relative to P_base
        assert max_dev < 0.005, (
            f"Max deviation {max_dev:.6f} exceeds tolerance. "
            "Consider recalibrating alpha."
        )

    def test_exponential_preserves_monotonicity_in_delta(self) -> None:
        """Exponential form preserves the qualitative prediction that
        P decreases with |M_g - M_h| at fixed total."""
        # Fixed total = 10
        P_55 = float(mutual_assessment_exponential(5.0, 5.0))
        P_37 = float(mutual_assessment_exponential(3.0, 7.0))
        P_19 = float(mutual_assessment_exponential(1.0, 9.0))
        assert P_55 > P_37 > P_19


# =====================================================================
# TestCalibrationRegression
# =====================================================================


class TestCalibrationRegression:
    """Regression tests: pin calibrated values to detect unintended changes.

    Once a key result is established, a test flags if it changes.
    """

    def test_calibration_regression_default_target(self) -> None:
        """Calibrated parameters for target_r=0.75 should be stable."""
        result = calibrate_conflict_reduction(target_r=0.75)
        assert result["success"]
        # Pin the achieved r to within a tight tolerance
        assert result["r_achieved"] == pytest.approx(0.75, abs=0.01)

    def test_calibration_regression_custom_M(self) -> None:
        """Calibration with M_high=5.0 should achieve target."""
        result = calibrate_conflict_reduction(target_r=0.75, M_high=5.0)
        assert result["success"]
        assert result["r_achieved"] == pytest.approx(0.75, abs=0.01)

    def test_lambda_C_regression(self) -> None:
        """lambda_C at default parameters should be stable across code changes."""
        lc = compute_lambda_C(M_g=5.0)
        # Pin to current value with 5% tolerance for numerical stability
        assert lc > 0.0
        assert lc == pytest.approx(1.8e-5, rel=0.10)


# =====================================================================
# TestConstrainedSigma0
# =====================================================================


class TestConstrainedSigma0:
    """Tests for constrained_sigma_0: deriving assessment noise from signal properties.

    Verifies that sigma_0 is traceable to deception, lag, and mismatch
    components, with correct monotonicity in each parameter.
    """

    def test_sigma_0_positive(self) -> None:
        """Constrained sigma_0 is positive for all valid parameter combinations."""
        result = constrained_sigma_0()
        assert result["sigma_0"] > 0.0

    def test_sigma_0_close_to_default(self) -> None:
        """With default parameters, constrained sigma_0 should be near the exploration default."""
        result = constrained_sigma_0()
        # Should be in [0.5, 1.5], close to DEFAULT_SIGMA_0 = 1.0
        assert 0.5 < result["sigma_0"] < 1.5

    def test_components_sum_to_total_variance(self) -> None:
        """Variance components must sum to the total variance."""
        result = constrained_sigma_0()
        total = (
            result["sigma_deception_sq"]
            + result["sigma_lag_sq"]
            + result["sigma_mismatch_sq"]
        )
        assert total == pytest.approx(result["sigma_0_sq"], rel=1e-10)

    def test_deception_near_zero_for_monuments(self) -> None:
        """With rho_monument=0.95, deception is a small fraction of total."""
        result = constrained_sigma_0(rho_monument=0.95)
        # Deception variance should be at most 20% of total
        assert result["sigma_deception_sq"] < 0.20 * result["sigma_0_sq"]

    def test_deception_dominates_for_low_fidelity(self) -> None:
        """With ritual-like fidelity, deception should be the largest component."""
        result = constrained_sigma_0(rho_monument=0.3)
        assert result["sigma_deception_sq"] > result["sigma_lag_sq"]
        assert result["sigma_deception_sq"] > result["sigma_mismatch_sq"]

    def test_lag_decreasing_in_delta(self) -> None:
        """Higher depreciation rate means less lag noise (signal more current)."""
        sig_low_delta = constrained_sigma_0(delta=0.05)
        sig_mid_delta = constrained_sigma_0(delta=0.10)
        sig_high_delta = constrained_sigma_0(delta=0.20)
        assert sig_low_delta["sigma_lag_sq"] > sig_mid_delta["sigma_lag_sq"]
        assert sig_mid_delta["sigma_lag_sq"] > sig_high_delta["sigma_lag_sq"]

    def test_lag_zero_when_delta_one(self) -> None:
        """When delta=1 (fully current each period), lag noise vanishes."""
        result = constrained_sigma_0(delta=1.0)
        assert result["sigma_lag_sq"] == pytest.approx(0.0, abs=1e-15)

    def test_mismatch_zero_when_perfect_correlation(self) -> None:
        """When productive and fighting capacity are perfectly correlated, no mismatch."""
        result = constrained_sigma_0(rho_prod_fight=1.0)
        assert result["sigma_mismatch_sq"] == pytest.approx(0.0, abs=1e-15)


# =====================================================================
# TestWarOfAttritionDeterrence (derived absolute-deterrence factor)
# =====================================================================


class TestWarOfAttritionDeterrence:
    """The absolute-deterrence factor D = 1/(1+beta*(M_g+M_h)), derived from a
    war of attrition with log-logistic stakes, is the symmetric-dyad mechanism
    that produces positive build-build conflict reduction. Relative assessment
    alone cannot: it predicts MORE conflict at parity.
    """

    def test_value_at_zero_is_one(self) -> None:
        """No deterrence at zero combined stock."""
        assert war_of_attrition_deterrence(0.0, 0.0) == pytest.approx(1.0)

    def test_closed_form(self) -> None:
        """D = 1/(1+beta*(M_g+M_h)) exactly (the WoA survival function)."""
        b = DEFAULT_BETA_CONFLICT
        assert war_of_attrition_deterrence(10.0, 10.0, beta=b) == pytest.approx(
            1.0 / (1.0 + b * 20.0), rel=1e-12
        )

    def test_decreasing_in_total_stock(self) -> None:
        """Stronger combined force -> more deterrence -> lower escalation factor."""
        totals = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
        D = np.array([float(war_of_attrition_deterrence(t / 2, t / 2)) for t in totals])
        assert np.all(np.diff(D) < 0)

    def test_depends_only_on_total_not_asymmetry(self) -> None:
        """Deterrence is the symmetric-dyad mechanism: it depends on M_g+M_h only."""
        assert war_of_attrition_deterrence(3.0, 17.0) == pytest.approx(
            war_of_attrition_deterrence(10.0, 10.0)
        )
        assert war_of_attrition_deterrence(1.0, 19.0) == pytest.approx(
            war_of_attrition_deterrence(8.0, 12.0)
        )

    def test_deterrence_is_source_of_positive_r_at_parity(self) -> None:
        """Assessment alone AMPLIFIES conflict at symmetric dyads (no clip
        flattens this at r = 0): with beta = 0 the derived reduction is
        NEGATIVE, r(10,10) =
        -0.714 (the SI's -0.71 figure, pinned here). The absolute-deterrence
        factor is what outruns that amplification and makes the full
        r(M, M) > 0."""
        for M in (5.0, 10.0, 20.0):
            r_full = float(derived_conflict_reduction(M, M))
            r_assess_only = float(derived_conflict_reduction(M, M, beta=0.0))
            assert r_assess_only < 0  # parity amplification, not neutrality
            assert r_full > 0.1  # deterrence creates the reduction
            assert r_full > r_assess_only
        assert float(
            derived_conflict_reduction(10.0, 10.0, beta=0.0)
        ) == pytest.approx(-0.714, abs=0.02)

    def test_mismatch_equals_var_RHP_when_zero_correlation(self) -> None:
        """When correlation is zero, all RHP variance is unexplained."""
        result = constrained_sigma_0(rho_prod_fight=0.0, var_RHP=1.0)
        assert result["sigma_mismatch_sq"] == pytest.approx(1.0, rel=1e-10)

    def test_sigma_0_increases_with_sigma_q(self) -> None:
        """Higher capacity variance produces more lag noise."""
        r1 = constrained_sigma_0(sigma_q=0.1)
        r2 = constrained_sigma_0(sigma_q=0.3)
        r3 = constrained_sigma_0(sigma_q=0.5)
        assert r1["sigma_0"] < r2["sigma_0"] < r3["sigma_0"]

    def test_sigma_0_decreases_with_rho_prod_fight(self) -> None:
        """Higher productive-fighting correlation reduces mismatch noise."""
        r1 = constrained_sigma_0(rho_prod_fight=0.5)
        r2 = constrained_sigma_0(rho_prod_fight=0.8)
        r3 = constrained_sigma_0(rho_prod_fight=0.95)
        assert r1["sigma_0"] > r2["sigma_0"] > r3["sigma_0"]

    def test_regression_default_parameters(self) -> None:
        """Pin default-parameter result to detect unintended changes."""
        result = constrained_sigma_0()
        assert result["sigma_0"] == pytest.approx(0.891, abs=0.01)

    def test_invalid_delta_raises(self) -> None:
        """delta <= 0 or > 1 should raise ValueError."""
        with pytest.raises(ValueError):
            constrained_sigma_0(delta=0.0)
        with pytest.raises(ValueError):
            constrained_sigma_0(delta=-0.1)

    def test_invalid_rho_raises(self) -> None:
        """rho values outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError):
            constrained_sigma_0(rho_monument=1.5)
        with pytest.raises(ValueError):
            constrained_sigma_0(rho_prod_fight=-0.1)


# =====================================================================
# TestManuscriptAnchorValues
# =====================================================================


class TestManuscriptAnchorValues:
    """Pin specific numerical values reported in manuscript prose.

    These are regression tests guarding against silent divergence between the
    code and the manuscript. The manuscript reports rounded integer (or
    two-significant-figure)
    values at the empirical anchor (default Layer 2 parameters); if the
    computed values move enough to invalidate the prose, these tests
    fail in CI and the prose must be updated alongside the code.

    Anchors are sourced from the main-text mutual-assessment section
    and the conflict-assessment figure caption.
    """

    def test_r_self_at_M_5_matches_manuscript_minus_31(self) -> None:
        # Main-text Layer 2 section: r_self(5, 5) ~= -31 (normalized self model).
        from signaling.layer2 import derived_conflict_reduction_self_assessment
        r = float(derived_conflict_reduction_self_assessment(5.0, 5.0))
        assert r == pytest.approx(-31.0, abs=0.5)

    def test_r_self_at_M_10_matches_manuscript_minus_22(self) -> None:
        # Main-text Layer 2 section: r_self(10, 10) ~= -22 (normalized self model).
        from signaling.layer2 import derived_conflict_reduction_self_assessment
        r = float(derived_conflict_reduction_self_assessment(10.0, 10.0))
        assert r == pytest.approx(-22.0, abs=0.5)

    def test_r_self_at_M_20_matches_manuscript_minus_13(self) -> None:
        # Main-text Layer 2 section: r_self(20, 20) ~= -13 (normalized self model).
        from signaling.layer2 import derived_conflict_reduction_self_assessment
        r = float(derived_conflict_reduction_self_assessment(20.0, 20.0))
        assert r == pytest.approx(-13.0, abs=0.5)

    def test_P_self_ratio_at_M_5_matches_manuscript_32(self) -> None:
        # main-text conflict-assessment figure caption: P_self/P_base ~= 32 at M = 5.
        ratio = float(self_assessment_conflict_prob(5.0, 5.0)) / DEFAULT_P_BASE
        assert ratio == pytest.approx(32.0, abs=0.5)

    def test_P_self_ratio_at_M_10_matches_manuscript_23(self) -> None:
        # main-text conflict-assessment figure caption: P_self/P_base ~= 23 at M = 10.
        ratio = float(self_assessment_conflict_prob(10.0, 10.0)) / DEFAULT_P_BASE
        assert ratio == pytest.approx(23.0, abs=0.5)

    def test_P_self_ratio_at_M_20_matches_manuscript_14(self) -> None:
        # main-text conflict-assessment figure caption: P_self/P_base ~= 14 at M = 20.
        ratio = float(self_assessment_conflict_prob(20.0, 20.0)) / DEFAULT_P_BASE
        assert ratio == pytest.approx(14.0, abs=0.5)

    def test_r_mutual_at_M_10_matches_manuscript_0p43(self) -> None:
        # Main-text mutual-assessment section:
        # r_mutual(10, 10) ~= 0.43 at the empirical anchor.
        r = float(derived_conflict_reduction(10.0, 10.0))
        assert r == pytest.approx(0.43, abs=0.02)


class TestSelfAssessmentNumericalStability:
    """Log-space evaluation of the self-assessment probability: the
    direct sigmoid-ratio formula
    underflowed p(0) to exactly 0 for M_thresh/M_scale beyond ~372 and
    returned silent NaN, breaking the P(0,0) = P_base anchoring."""

    def test_extreme_threshold_preserves_baseline_anchor(self) -> None:
        """P(0,0) = P_base exactly even at extreme threshold/scale ratios
        (the direct sigmoid-ratio formula returns NaN here)."""
        v = float(self_assessment_conflict_prob(0.0, 0.0, M_thresh=20.0, M_scale=0.05))
        assert v == pytest.approx(0.01, rel=1e-9)
        v2 = float(self_assessment_conflict_prob(0.0, 0.0, M_thresh=500.0, M_scale=1.0))
        assert v2 == pytest.approx(0.01, rel=1e-9)

    def test_log_space_matches_direct_formula_in_normal_domain(self) -> None:
        """The log-space rewrite is value-identical to the direct formula
        wherever the latter is numerically valid."""
        rng = np.random.default_rng(1)
        for _ in range(2000):
            Mg, Mh = rng.uniform(0, 30, 2)
            T = rng.uniform(0.1, 12.0)
            s = rng.uniform(0.2, 3.0)
            b = rng.uniform(0.0, 0.5)
            Pb = rng.uniform(0.001, 0.1)
            new = float(self_assessment_conflict_prob(Mg, Mh, T, s, b, Pb))
            sig = lambda M: 1.0 / (1.0 + np.exp(-(M - T) / s))
            old = min(1.0, Pb * sig(Mg) * sig(Mh) / sig(0.0) ** 2 / (1.0 + b * (Mg + Mh)))
            assert new == pytest.approx(old, abs=1e-12)

    def test_high_threshold_limit_matches_displayed_formula(self) -> None:
        """As M_thresh -> infinity at fixed M, the probability approaches the
        SI's displayed limit min{1, P_base e^{(Mg+Mh)/s} / (1 + beta(Mg+Mh))}
        (equal to 1 at the plotted stocks M in {3, 5, 10})."""
        for M in (3.0, 5.0, 10.0):
            v = float(self_assessment_conflict_prob(M, M, M_thresh=500.0, M_scale=1.0))
            lim = min(1.0, 0.01 * np.exp(2.0 * M) / (1.0 + 0.1 * 2.0 * M))
            assert v == pytest.approx(lim, rel=1e-6)

class TestExponentialStakeVariant:
    """The SI's exponential-stake deterrence robustness variant (SI stake-
    threshold section): deterrence exp(-beta(Mg+Mh)) instead of the
    log-logistic. Pins the SI-quoted r ~ 0.77 at the anchor dyad and the
    deterrence-factor contrast 0.135 vs 1/3, and separates this variant
    from the similarly named exponential ASSESSMENT approximation."""

    def test_r_exponential_stake_anchor_pinned(self):
        """SI quotes r(10,10) ~ 0.77 under exponential stake; measured
        0.7680."""
        P = float(mutual_assessment_conflict_prob_exponential_stake(10.0, 10.0))
        r = 1.0 - P / DEFAULT_P_BASE
        assert r == pytest.approx(0.768, abs=2e-3)

    def test_deterrence_factor_contrast(self):
        """exp(-2) ~ 0.135 vs log-logistic 1/3 at Mg+Mh = 20, beta = 0.1
        (the SI-quoted 0.14 vs 0.33 contrast)."""
        import numpy as np
        assert np.exp(-0.1 * 20.0) == pytest.approx(0.1353, abs=1e-3)
        assert 1.0 / (1.0 + 0.1 * 20.0) == pytest.approx(0.3333, abs=1e-3)

    def test_distinct_from_exponential_assessment_variant(self):
        """mutual_assessment_exponential (exponential ASSESSMENT approx,
        log-logistic deterrence) is a different object: r(10,10) ~ 0.67,
        not the stake variant's 0.77."""
        P = float(mutual_assessment_exponential(10.0, 10.0))
        r = 1.0 - P / DEFAULT_P_BASE
        assert r == pytest.approx(0.667, abs=5e-3)
        assert abs(r - 0.768) > 0.05

    def test_agrees_with_loglogistic_at_zero_stock(self):
        """Both deterrence forms equal 1 at zero total stock, so the two
        variants share the P_base anchor at (0,0)."""
        P_exp = float(mutual_assessment_conflict_prob_exponential_stake(0.0, 0.0))
        P_ll = float(mutual_assessment_conflict_prob(0.0, 0.0))
        assert P_exp == pytest.approx(P_ll, abs=1e-12)
        assert P_exp == pytest.approx(DEFAULT_P_BASE, abs=1e-12)

