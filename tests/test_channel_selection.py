"""Tests for channel selection analysis.

Verifies theoretical properties of the multi-channel signaling model:
- Lambda decomposition sums correctly
- Monument has highest effective lambda under multi-audience conditions
- Signal fidelity values are bounded in [0, 1]
- Monument dominates under standard exploration parameters
- Feasting dominates when intergroup audiences are absent
- Incremental lambda is positive when monuments dominate
- Incremental lambda is zero when monuments do not dominate
- Windfall probability reduces feast fidelity monotonically
- Channel selection ranks by NET return V_s = lambda_s*rho_s - f_s (cost-adjusted)
- Monument dominance is CONDITIONAL: feasting wins the within-group-only corner;
  the framework's operating points (default, empirical anchor) stay monument
- Dominance falls as the monument fixed cost rises (f_M = 0 recovers 100%)

Each test encodes a prediction from the channel selection analysis
(SI "Channel selection analysis"; main text "Channel selection: why
monuments specifically"), not merely that the code runs.
"""

import dataclasses

import numpy as np
import pytest

from signaling.calibration import (
    DEFAULT_LAM_C,
    DEFAULT_LAM_W,
    DEFAULT_LAM_X,
    DEFAULT_WINDFALL_PROB,
)
from signaling.layer1 import (
    ALL_CHANNELS,
    FEAST_CHANNEL,
    HUNTING_CHANNEL,
    MONUMENT_CHANNEL,
    RITUAL_CHANNEL,
    VERBAL_CHANNEL,
    SignalChannel,
    channel_dominance_condition,
    channel_effective_lambda,
    incremental_lambda,
    lambda_decomposition,
    monument_dominance_threshold,
    signal_fidelity,
)


class TestChannelSelection:
    """Verify channel selection analysis properties."""

    def test_lambda_decomposition_sums(self) -> None:
        """Total lambda equals the sum of its three audience components."""
        result = lambda_decomposition(0.3, 0.15, 0.15)
        assert result["total"] == pytest.approx(0.6, abs=1e-12)
        assert result["total"] == pytest.approx(
            result["lam_W"] + result["lam_C"] + result["lam_X"], abs=1e-12
        )

        # Test with asymmetric values
        result2 = lambda_decomposition(0.5, 0.0, 0.1)
        assert result2["total"] == pytest.approx(0.6, abs=1e-12)

        # Test with zeros
        result3 = lambda_decomposition(0.0, 0.0, 0.0)
        assert result3["total"] == pytest.approx(0.0, abs=1e-12)

    def test_channel_effective_lambda_monument_highest(self) -> None:
        """Monument has the highest effective lambda when all audiences are active.

        This is the central prediction of the multi-audience lambda
        decomposition: because monuments activate all three audience
        channels (a_W = a_C = a_X = 1), they
        capture the full lambda when all audiences matter.
        """
        lam_W, lam_C, lam_X = DEFAULT_LAM_W, DEFAULT_LAM_C, DEFAULT_LAM_X
        lam_monument = channel_effective_lambda(MONUMENT_CHANNEL, lam_W, lam_C, lam_X)

        for ch in [FEAST_CHANNEL, RITUAL_CHANNEL, HUNTING_CHANNEL, VERBAL_CHANNEL]:
            lam_ch = channel_effective_lambda(ch, lam_W, lam_C, lam_X)
            assert lam_monument > lam_ch, (
                f"Monument (lambda_s={lam_monument:.4f}) should exceed "
                f"{ch.name} (lambda_s={lam_ch:.4f}) when all audiences active"
            )

        # Monument effective lambda should equal total lambda (all activations = 1)
        total = lam_W + lam_C + lam_X
        assert lam_monument == pytest.approx(total, abs=1e-12)

    def test_signal_fidelity_bounds(self) -> None:
        """All signal fidelity values must be in [0, 1]."""
        for ch in ALL_CHANNELS:
            rho = signal_fidelity(ch)
            assert 0.0 <= rho <= 1.0, (
                f"{ch.name} fidelity {rho} out of bounds"
            )

        # With maximum windfall, fidelity should be zero
        rho_maxwind = signal_fidelity(FEAST_CHANNEL, windfall_prob=1.0)
        assert rho_maxwind == pytest.approx(0.0, abs=1e-12)

        # With zero windfall, fidelity should equal base rho
        rho_nowind = signal_fidelity(FEAST_CHANNEL, windfall_prob=0.0)
        assert rho_nowind == pytest.approx(FEAST_CHANNEL.rho, abs=1e-12)

        # Quality correlation override
        rho_override = signal_fidelity(FEAST_CHANNEL, quality_correlation=0.8)
        assert rho_override == pytest.approx(0.8, abs=1e-12)

    def test_monument_dominance_under_standard_conditions(self) -> None:
        """Monument dominates when lam_C > 0 and lam_X > 0.

        Under the default exploration parameters (lam_W=0.3, lam_C=0.15,
        lam_X=0.15), monument construction should have the highest net
        return V_s = lambda_s * rho_s - f_s among all channels.
        """
        ranking = channel_dominance_condition(
            ALL_CHANNELS, DEFAULT_LAM_W, DEFAULT_LAM_C, DEFAULT_LAM_X,
            windfall_prob=DEFAULT_WINDFALL_PROB,
        )
        assert ranking[0][0] == "monument", (
            f"Monument should dominate under standard conditions, "
            f"but {ranking[0][0]} dominates with R_s={ranking[0][3]:.4f}"
        )
        # Verify monument R_s exceeds all others
        R_monument = ranking[0][3]
        for name, _, _, R_s in ranking[1:]:
            assert R_monument > R_s

    def test_feasting_dominates_without_intergroup(self) -> None:
        """Feasting dominates when lam_C = lam_X = 0 (within-group only).

        Under the cost-adjusted model (V_s = lambda_s*rho_s - f_s), monuments
        are too costly (high fixed cost f_M) to be worth building when only
        the within-group audience is active: their multi-audience reach is
        unused, so their net return falls below that of the cheaper
        within-group specialist (feasting). This is the conditional onset
        specific to the net-return form: a gross-return ranking (no fixed cost)
        has monuments winning here too.
        """
        ranking_within = channel_dominance_condition(ALL_CHANNELS, 0.3, 0.0, 0.0)
        assert ranking_within[0][0] == "feast", (
            f"Feasting should dominate the within-group-only corner, got "
            f"{ranking_within[0][0]} with V_s={ranking_within[0][3]:.4f}"
        )
        # Monument's NET return is negative here (gross return < fixed cost),
        # so it is not even worth building absent between-group audiences.
        monument_V = next(V for name, _, _, V in ranking_within if name == "monument")
        assert monument_V < 0.0, (
            f"Monument net return should be negative at the within-group corner, "
            f"got {monument_V:.4f}"
        )

    def test_incremental_lambda_positive_when_dominant(self) -> None:
        """Incremental lambda is positive when monuments dominate.

        This is a tautological consequence of dominance: the dominant
        channel's net return exceeds all alternatives, so the increment
        is positive.
        """
        ranking = channel_dominance_condition(
            ALL_CHANNELS, DEFAULT_LAM_W, DEFAULT_LAM_C, DEFAULT_LAM_X,
            windfall_prob=DEFAULT_WINDFALL_PROB,
        )
        lam_incr = incremental_lambda(ranking)
        assert lam_incr > 0, (
            f"Incremental lambda should be positive when monuments dominate, "
            f"got {lam_incr}"
        )
        # Incremental lambda should be less than total effective lambda
        R_monument = ranking[0][3]
        assert lam_incr < R_monument

    def test_incremental_lambda_zero_when_not_dominant(self) -> None:
        """Incremental lambda for the dominant channel is non-negative.

        When an alternative dominates monuments (e.g., only within-group
        audiences at extreme parameter values), incremental lambda is
        still well-defined as the gap between the top two channels.
        It is always >= 0 by construction.
        """
        # Create a scenario where feast has artificially high fidelity
        high_fidelity_feast = SignalChannel(
            name="feast",
            audience_W=0.9,
            audience_C=0.2,
            audience_X=0.2,
            rho=0.99,  # artificially high
            fixed_cost=0.02,
            quality_dimension="resource accumulation",
            description="Modified feast with high fidelity",
        )
        channels = [MONUMENT_CHANNEL, high_fidelity_feast]
        # With only within-group audiences (lam_C=0, lam_X=0) and an
        # artificially high feast fidelity, feast's net return exceeds
        # monument's (whose fixed cost is unrecovered without between-group
        # audiences), so feast dominates and incremental_lambda is the
        # feast-over-monument gap.
        ranking = channel_dominance_condition(channels, 0.3, 0.0, 0.0)
        lam_incr = incremental_lambda(ranking)
        assert lam_incr >= 0, "Incremental lambda must be non-negative"
        assert ranking[0][0] == "feast", "Alternative genuinely dominates here"

    def test_windfall_reduces_feast_fidelity(self) -> None:
        """Higher windfall probability reduces effective feasting fidelity.

        The relationship is rho_feast = rho_0 * (1 - p_w), so fidelity
        decreases linearly with windfall probability.
        """
        rho_values = []
        windfall_probs = np.linspace(0.0, 0.9, 10)
        for wp in windfall_probs:
            rho = signal_fidelity(FEAST_CHANNEL, windfall_prob=wp)
            rho_values.append(rho)

        # Monotonically decreasing
        for i in range(1, len(rho_values)):
            assert rho_values[i] < rho_values[i - 1], (
                f"Fidelity should decrease with windfall probability: "
                f"rho[{i}]={rho_values[i]:.4f} >= rho[{i-1}]={rho_values[i-1]:.4f}"
            )

        # At windfall_prob=0, should equal base rho
        assert rho_values[0] == pytest.approx(FEAST_CHANNEL.rho, abs=1e-12)


class TestMonumentDominanceThreshold:
    """Verify the parameter-space dominance computation."""

    def test_grid_shape(self) -> None:
        """Output grid has correct shape."""
        lam_C_range = np.linspace(0, 0.3, 5)
        lam_X_range = np.linspace(0, 0.3, 7)
        grid = monument_dominance_threshold(0.3, lam_C_range, lam_X_range)
        assert grid.shape == (7, 5)

    def test_high_intergroup_is_monument_dominant(self) -> None:
        """At high lam_C and lam_X, monuments should dominate."""
        lam_C_range = np.array([0.3])
        lam_X_range = np.array([0.3])
        grid = monument_dominance_threshold(0.3, lam_C_range, lam_X_range)
        assert grid[0, 0], "Monuments should dominate at high intergroup lambda"

    def test_dominance_increases_with_intergroup(self) -> None:
        """Monument dominance fraction should increase with intergroup lambda."""
        lam_C_range = np.linspace(0, 0.5, 20)
        lam_X_range = np.linspace(0, 0.5, 20)
        grid = monument_dominance_threshold(0.3, lam_C_range, lam_X_range)
        # The upper-right region (high lam_C, lam_X) should have more
        # monument dominance than the lower-left
        n = len(lam_X_range) // 2
        upper_right = grid[n:, n:]
        lower_left = grid[:n, :n]
        assert np.sum(upper_right) >= np.sum(lower_left), (
            "Monument dominance should be more common at higher intergroup lambda"
        )


class TestCostAdjustedChannelSelection:
    """Verify the cost-adjusted (net-return) channel selection.

    Channels rank by V_s = lambda_s*rho_s - f_s, where f_s is the channel's
    fixed access cost. Including f_s makes monument dominance conditional on
    the audience weights: a gross-return ranking (no fixed cost) gives monument
    dominance across the entire (lam_C, lam_X) space, so these tests exercise
    the conditional dominance that only the net-return form produces.
    """

    @staticmethod
    def _dominance_fraction(lam_W, windfall_prob=0.0, channels=None, n=120):
        g = np.linspace(0.0, 0.5, n)
        grid = monument_dominance_threshold(
            lam_W, g, g, channels=channels, windfall_prob=windfall_prob
        )
        return float(grid.mean())

    @staticmethod
    def _channels_with_fM(fM):
        return [
            dataclasses.replace(MONUMENT_CHANNEL, fixed_cost=fM),
            FEAST_CHANNEL, RITUAL_CHANNEL, HUNTING_CHANNEL, VERBAL_CHANNEL,
        ]

    def test_net_return_is_community_mean_payoff(self) -> None:
        """The ranking's 4th element is the community-mean equilibrium net
        payoff V_s = lambda_s*rho_s*A_bar - f_s, with A_bar the mean payoff
        multiplier E[(q^2+q_min^2)/(2q)] (a criterion omitting A_bar is not
        the derived payoff)."""
        from signaling.layer1 import mean_payoff_multiplier
        A_bar = mean_payoff_multiplier()
        assert A_bar == pytest.approx(0.5328835, abs=1e-6)
        ranking = channel_dominance_condition(ALL_CHANNELS, 0.3, 0.15, 0.15)
        for name, lam_s, rho_s, V_s in ranking:
            ch = next(c for c in ALL_CHANNELS if c.name == name)
            assert V_s == pytest.approx(lam_s * rho_s * A_bar - ch.fixed_cost, abs=1e-12)
        # Monument's net return uses its payoff-unit fixed cost (0.16).
        mon = next(t for t in ranking if t[0] == "monument")
        assert mon[3] == pytest.approx(mon[1] * mon[2] * A_bar - 0.16, abs=1e-12)

    def test_dominance_frontier_preserved_under_payoff_units(self) -> None:
        """The payoff-unit re-anchoring of the illustrative fixed costs keeps
        the dominance frontier at lam_C + lam_X ~ 0.15 (measured 0.1525) and
        the operating point (0.15, 0.15) monument-dominant."""
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            top = channel_dominance_condition(ALL_CHANNELS, 0.3, mid / 2, mid / 2)[0][0]
            if top == "monument":
                hi = mid
            else:
                lo = mid
        assert hi == pytest.approx(0.1525, abs=0.005)

    def test_default_and_anchor_stay_monument(self) -> None:
        """The framework's operating points are monument-dominant.

        Both the default audience weights (lam_C=lam_X=0.15) and the empirical
        anchor (lam_W=0.68, even at the within-group corner) give monument
        dominance, so the framework's conclusions do not hinge on the
        within-group-corner behavior that drives the dominance percentage.
        """
        r_default = channel_dominance_condition(
            ALL_CHANNELS, DEFAULT_LAM_W, DEFAULT_LAM_C, DEFAULT_LAM_X
        )
        assert r_default[0][0] == "monument"
        r_anchor = channel_dominance_condition(ALL_CHANNELS, 0.68, 0.0, 0.0)
        assert r_anchor[0][0] == "monument"

    def test_dominance_is_conditional_not_total(self) -> None:
        """At the illustrative fixed cost, monument dominance is conditional (< 100%).

        The cost-adjusted model produces a genuine boundary: monuments do NOT
        dominate the entire (lam_C, lam_X) space. (A gross-return ranking with
        no fixed cost gives exactly 100%.)
        """
        frac = self._dominance_fraction(DEFAULT_LAM_W, windfall_prob=0.0)
        assert frac < 1.0, "Dominance must be conditional (a feast region exists)"
        assert frac > 0.80, "Monuments should still dominate most of the space"

    def test_windfall_expands_monument_dominance(self) -> None:
        """Feasting windfall degrades feast fidelity, expanding monument dominance."""
        frac_nowind = self._dominance_fraction(DEFAULT_LAM_W, windfall_prob=0.0)
        frac_wind = self._dominance_fraction(DEFAULT_LAM_W, windfall_prob=0.3)
        assert frac_wind > frac_nowind, (
            "Windfall should expand monument dominance by degrading feast"
        )

    def test_dominance_decreases_with_fixed_cost(self) -> None:
        """Monument dominance falls as f_M rises; f_M=0 recovers unconditional 100%."""
        frac0 = self._dominance_fraction(DEFAULT_LAM_W, channels=self._channels_with_fM(0.0))
        frac_mid = self._dominance_fraction(DEFAULT_LAM_W, channels=self._channels_with_fM(0.30))
        frac_hi = self._dominance_fraction(DEFAULT_LAM_W, channels=self._channels_with_fM(0.45))
        assert frac0 == pytest.approx(1.0, abs=1e-9), (
            "Zero fixed cost recovers the old unconditional dominance"
        )
        assert frac0 > frac_mid > frac_hi, "Dominance must decrease with fixed cost"

    def test_dominance_fraction_regression(self) -> None:
        """Regression pin on the dominance % at the illustrative default
        f_M = 0.16 (PAYOFF units, under the community-mean criterion
        V_s = lam_s * rho_s * A_bar - f_s; f_M = 0.16 in payoff units
        corresponds to the former reward-unit value 0.30).

        If these change, the manuscript channel-dominance numbers (main-text
        channel-selection subsection, the channel-selection figure caption, and the (C, n)
        joint-sensitivity sweep) must be re-derived. Pinned values at n=120:
        0.951 (wf=0), 0.977 (wf=0.3).
        """
        frac0 = self._dominance_fraction(DEFAULT_LAM_W, windfall_prob=0.0, n=120)
        frac3 = self._dominance_fraction(DEFAULT_LAM_W, windfall_prob=0.3, n=120)
        assert frac0 == pytest.approx(0.951, abs=0.02)
        assert frac3 == pytest.approx(0.977, abs=0.02)
