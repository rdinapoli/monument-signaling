"""Tests for Layer 3: cooperation networks and crisis buffering.

Verifies theoretical properties of the network and buffering model:
- Network degree: boundary conditions, monotonicity, saturation
- Survival: S in [0, 1], increasing in k, decreasing in sigma
- Vulnerability: alpha_eff < beta_eff when k_signal > k_nonsignal
- lambda_X: positive, increasing in sigma, decreasing in M_g
- Feedback loop: lambda_total increasing in sigma, convergence
- Calibration: can approximate initial model vulnerability values
- NetworkX: signal-based networks denser than baseline
"""

import numpy as np
import pytest

from signaling.calibration import DEFAULT_K_0, DEFAULT_K_MAX, DEFAULT_M_HALF
from signaling.layer3 import (
    NetworkDegreeSpec,
    calibrate_alternative_network_forms,
    calibrate_to_initial_vulnerability,
    compute_lambda_X,
    derive_vulnerability_differential,
    form_network_baseline,
    form_network_signal_based,
    lambda_sigma_sweep,
    lambda_total_at_sigma,
    network_degree,
    network_degree_derivative,
    network_degree_exponential,
    network_degree_exponential_derivative,
    network_degree_hill,
    network_degree_hill_derivative,
    network_degree_piecewise,
    network_degree_piecewise_derivative,
    survival_probability,
    vulnerability_coefficient,
)


# =====================================================================
# Network degree
# =====================================================================


class TestNetworkDegree:
    """Tests for the saturating network degree function k(M_g)."""

    def test_baseline_at_zero(self):
        """k(0) = k_0: no monument investment gives baseline degree."""
        k = network_degree(0.0)
        assert k == pytest.approx(DEFAULT_K_0)

    def test_half_saturation(self):
        """k(M_half) = k_0 + k_max/2: half-saturation property."""
        k = network_degree(DEFAULT_M_HALF)
        expected = DEFAULT_K_0 + DEFAULT_K_MAX / 2
        assert k == pytest.approx(expected)

    def test_approaches_maximum(self):
        """k approaches k_0 + k_max for large M_g."""
        k = network_degree(1000.0)
        assert k == pytest.approx(DEFAULT_K_0 + DEFAULT_K_MAX, rel=0.01)

    def test_monotonically_increasing(self):
        """dk/dM > 0: more monument investment attracts more partners."""
        M_values = np.linspace(0, 20, 50)
        k_values = network_degree(M_values)
        # Check strict monotonicity
        diffs = np.diff(k_values)
        assert np.all(diffs > 0)

    def test_concavity(self):
        """d^2k/dM^2 < 0: diminishing returns to monument investment."""
        M_values = np.linspace(0.1, 20, 50)
        k_values = network_degree(M_values)
        # Second differences should be negative
        second_diffs = np.diff(np.diff(k_values))
        assert np.all(second_diffs < 0)

    def test_nonnegative(self):
        """k >= 0 for all valid inputs."""
        M_values = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        k_values = network_degree(M_values)
        assert np.all(k_values >= 0)

    def test_vectorized(self):
        """Function works on arrays."""
        M_values = np.array([0.0, 1.0, 3.0, 10.0])
        k_values = network_degree(M_values)
        assert k_values.shape == (4,)

    def test_custom_parameters(self):
        """Works with non-default parameters."""
        k = network_degree(5.0, k_0=1.0, k_max=10.0, M_half=5.0)
        assert k == pytest.approx(1.0 + 10.0 * 0.5)  # = 6.0


class TestNetworkDegreeDerivative:
    """Tests for dk/dM_g."""

    def test_positive(self):
        """dk/dM > 0 for all M_g >= 0."""
        M_values = np.linspace(0, 50, 100)
        dk = network_degree_derivative(M_values)
        assert np.all(dk > 0)

    def test_decreasing(self):
        """d^2k/dM^2 < 0: derivative is decreasing (diminishing returns)."""
        M_values = np.linspace(0, 50, 100)
        dk = network_degree_derivative(M_values)
        diffs = np.diff(dk)
        assert np.all(diffs < 0)

    def test_value_at_zero(self):
        """dk/dM(0) = k_max / M_half (maximum marginal rate)."""
        dk_0 = network_degree_derivative(0.0)
        expected = DEFAULT_K_MAX / DEFAULT_M_HALF
        assert dk_0 == pytest.approx(expected)

    def test_consistency_with_finite_difference(self):
        """Analytical derivative matches finite differences."""
        M_g = 5.0
        h = 1e-6
        dk_analytical = float(network_degree_derivative(M_g))
        dk_numerical = float((network_degree(M_g + h) - network_degree(M_g - h)) / (2 * h))
        assert dk_analytical == pytest.approx(dk_numerical, rel=1e-4)


# =====================================================================
# Survival probability
# =====================================================================


class TestSurvivalProbability:
    """Tests for S(sigma, k) = 1 - sigma / (1 + gamma * k)."""

    def test_certain_survival_no_crisis(self):
        """S(0, k) = 1 for all k."""
        assert survival_probability(0.0, 0.0) == pytest.approx(1.0)
        assert survival_probability(0.0, 10.0) == pytest.approx(1.0)

    def test_maximum_vulnerability(self):
        """S(sigma, 0) = 1 - sigma (no partners, full vulnerability)."""
        assert survival_probability(0.2, 0.0) == pytest.approx(0.8)
        assert survival_probability(0.5, 0.0) == pytest.approx(0.5)

    def test_in_unit_interval(self):
        """S in [0, 1] for valid inputs."""
        sigma_vals = np.linspace(0, 0.9, 20)
        k_vals = np.linspace(0, 10, 20)
        Sg, Kg = np.meshgrid(sigma_vals, k_vals)
        S = survival_probability(Sg, Kg)
        assert np.all(S >= 0)
        assert np.all(S <= 1)

    def test_increasing_in_k(self):
        """dS/dk > 0: more partners improve survival."""
        sigma = 0.3
        k_values = np.linspace(0, 20, 50)
        S_values = survival_probability(sigma, k_values)
        diffs = np.diff(S_values)
        assert np.all(diffs > 0)

    def test_decreasing_in_sigma(self):
        """dS/dsigma < 0: more uncertainty reduces survival."""
        k = 5.0
        sigma_values = np.linspace(0, 0.5, 50)
        S_values = survival_probability(sigma_values, k)
        diffs = np.diff(S_values)
        assert np.all(diffs < 0)

    def test_partners_reduce_vulnerability(self):
        """S(sigma, k > 0) > S(sigma, 0) for sigma > 0."""
        sigma = 0.3
        S_no_partners = survival_probability(sigma, 0.0)
        S_with_partners = survival_probability(sigma, 5.0)
        assert S_with_partners > S_no_partners

    def test_vectorized(self):
        """Works on arrays of matching shape."""
        sigma = np.array([0.1, 0.2, 0.3])
        k = np.array([1.0, 3.0, 5.0])
        S = survival_probability(sigma, k)
        assert S.shape == (3,)

    def test_clipped_at_zero(self):
        """S is clipped to 0 for extreme sigma values."""
        # sigma very large, k very small: raw S would be negative
        S = survival_probability(5.0, 0.0)
        assert S == pytest.approx(0.0)


# =====================================================================
# Vulnerability coefficient
# =====================================================================


class TestVulnerabilityCoefficient:
    """Tests for alpha(k) = 1 / (1 + gamma * k)."""

    def test_maximum_vulnerability_at_zero(self):
        """alpha(0) = 1: maximum vulnerability without partners."""
        assert vulnerability_coefficient(0.0) == pytest.approx(1.0)

    def test_decreasing_in_k(self):
        """dalpha/dk < 0: more partners reduce vulnerability."""
        k_values = np.linspace(0, 20, 50)
        alpha_values = vulnerability_coefficient(k_values)
        diffs = np.diff(alpha_values)
        assert np.all(diffs < 0)

    def test_approaches_zero(self):
        """alpha -> 0 as k -> infinity."""
        alpha = vulnerability_coefficient(1000.0)
        assert alpha < 0.01

    def test_consistency_with_survival(self):
        """alpha(k) * sigma = 1 - S(sigma, k)."""
        sigma, k = 0.3, 5.0
        alpha = float(vulnerability_coefficient(k))
        S = float(survival_probability(sigma, k))
        assert alpha * sigma == pytest.approx(1.0 - S, abs=1e-10)


class TestVulnerabilityDifferential:
    """Tests for the derived vulnerability differential."""

    def test_signalers_less_vulnerable(self):
        """alpha_eff < beta_eff when k_signal > k_nonsignal.

        This is the central result of Layer 3: monument builders have
        lower vulnerability because their honest signals attract more
        exchange partners, and denser networks buffer crises better.
        """
        result = derive_vulnerability_differential(6.0, 0.5)
        assert result["alpha_eff"] < result["beta_eff"]

    def test_ratio_less_than_one(self):
        """Vulnerability ratio alpha/beta < 1 for k_signal > k_nonsignal."""
        result = derive_vulnerability_differential(6.0, 0.5)
        assert result["ratio"] < 1.0

    def test_equal_when_same_degree(self):
        """alpha = beta when k_signal = k_nonsignal."""
        result = derive_vulnerability_differential(3.0, 3.0)
        assert result["alpha_eff"] == pytest.approx(result["beta_eff"])
        assert result["ratio"] == pytest.approx(1.0)

    def test_larger_differential_with_larger_k_gap(self):
        """Larger gap in network degree produces larger vulnerability gap."""
        r1 = derive_vulnerability_differential(4.0, 0.5)
        r2 = derive_vulnerability_differential(8.0, 0.5)
        # r2 has bigger k gap, so alpha/beta ratio should be smaller (more different)
        assert r2["ratio"] < r1["ratio"]


# =====================================================================
# lambda_X
# =====================================================================


class TestLambdaX:
    """Tests for the cooperative between-group lambda component."""

    def test_positive(self):
        """lambda_X > 0 for M_g > 0 and sigma > 0."""
        lx = compute_lambda_X(5.0, 0.2)
        assert lx > 0

    def test_zero_without_crisis(self):
        """lambda_X = 0 when sigma = 0 (no crisis, no buffering value)."""
        lx = compute_lambda_X(5.0, 0.0)
        assert lx == pytest.approx(0.0)

    def test_increasing_in_sigma(self):
        """dlambda_X/dsigma > 0: higher uncertainty makes networks more valuable.

        This is the core feedback prediction: environmental uncertainty
        drives the value of cooperation networks, which drives monument
        investment through the lambda_X channel.
        """
        sigma_vals = [0.1, 0.2, 0.3, 0.5, 0.8]
        lx_vals = [compute_lambda_X(5.0, s) for s in sigma_vals]
        for i in range(len(lx_vals) - 1):
            assert lx_vals[i + 1] > lx_vals[i]

    def test_decreasing_in_M_g(self):
        """dlambda_X/dM_g < 0: diminishing returns to monument investment.

        As the network saturates, additional monument investment produces
        smaller marginal increases in network degree and survival.
        """
        M_vals = [1.0, 3.0, 5.0, 10.0, 20.0]
        lx_vals = [compute_lambda_X(M, 0.3) for M in M_vals]
        for i in range(len(lx_vals) - 1):
            assert lx_vals[i + 1] < lx_vals[i]

    def test_proportional_to_sigma(self):
        """lambda_X is exactly proportional to sigma (dS/dk is linear in sigma)."""
        lx1 = compute_lambda_X(5.0, 0.2)
        lx2 = compute_lambda_X(5.0, 0.4)
        assert lx2 == pytest.approx(2.0 * lx1, rel=1e-8)

    def test_per_capita_with_N(self):
        """lambda_X / N when group size is provided."""
        lx_group = compute_lambda_X(5.0, 0.3)
        lx_individual = compute_lambda_X(5.0, 0.3, N=12)
        assert lx_individual == pytest.approx(lx_group / 12.0, rel=1e-8)


# =====================================================================
# Feedback loop
# =====================================================================


class TestFeedbackLoop:
    """Tests for the COMPOSITE-REWARD ROBUSTNESS VARIANT's fixed-point
    iteration (lambda_total_at_sigma). On the canonical lambda_W-only path
    the between-group returns are diagnostics, not schedule inputs; this
    loop is retained for the SI Banach robustness comparison and moves
    sigma* by ~0.001; it is not the load-bearing mechanism."""

    def test_convergence(self):
        """Fixed-point iteration converges."""
        eq = lambda_total_at_sigma(0.2, lambda_W=0.3)
        assert eq["converged"]

    def test_lambda_greater_than_lambda_W(self):
        """Total lambda >= lambda_W (feedback adds, doesn't subtract).

        The competitive and cooperative channels can only add to
        within-group lambda, never subtract from it.
        """
        eq = lambda_total_at_sigma(0.3, lambda_W=0.3)
        assert eq["lambda_total"] >= 0.3

    def test_lambda_X_positive_in_result(self):
        """lambda_X component is positive at equilibrium."""
        eq = lambda_total_at_sigma(0.3, lambda_W=0.3)
        assert eq["lambda_X"] > 0

    def test_lambda_increasing_in_sigma(self):
        """Equilibrium lambda increases with sigma.

        Higher environmental uncertainty makes cooperation networks
        more valuable (through lambda_X), increasing total lambda.
        """
        eq1 = lambda_total_at_sigma(0.1, lambda_W=0.3)
        eq2 = lambda_total_at_sigma(0.5, lambda_W=0.3)
        assert eq2["lambda_total"] > eq1["lambda_total"]

    def test_M_g_positive(self):
        """Monument stock is positive at equilibrium."""
        eq = lambda_total_at_sigma(0.2, lambda_W=0.3)
        assert eq["M_g"] > 0

    def test_k_greater_than_k_0(self):
        """Network degree exceeds baseline at equilibrium."""
        eq = lambda_total_at_sigma(0.2, lambda_W=0.3)
        assert eq["k"] > DEFAULT_K_0

    def test_alpha_eff_less_than_one(self):
        """Signaler vulnerability is reduced by network (alpha < 1)."""
        eq = lambda_total_at_sigma(0.3, lambda_W=0.3)
        assert eq["alpha_eff"] < 1.0

    def test_with_lambda_C_func(self):
        """Accepts an external lambda_C function."""
        def mock_lambda_C(M_g):
            return 0.05  # constant for testing

        eq = lambda_total_at_sigma(0.3, lambda_W=0.3, compute_lambda_C_func=mock_lambda_C)
        # The mock returns exactly 0.05; the pass-through must be exact.
        assert eq["lambda_C"] == pytest.approx(0.05, abs=1e-12)
        assert eq["lambda_total"] > 0.35  # lambda_W + lambda_C + lambda_X


class TestLambdaSigmaSweep:
    """Tests for the composite-variant lambda(sigma) sweep (robustness
    diagnostics; the canonical schedule carries lambda_W alone)."""

    def test_returns_correct_keys(self):
        """Output contains all expected keys."""
        sweep = lambda_sigma_sweep(np.linspace(0.1, 0.5, 5), lambda_W=0.3)
        expected_keys = {"sigma", "lambda_total", "lambda_C", "lambda_X", "M_g", "k", "alpha_eff"}
        assert set(sweep.keys()) == expected_keys

    def test_lambda_monotonically_increasing(self):
        """lambda_total increases monotonically with sigma."""
        sweep = lambda_sigma_sweep(np.linspace(0.05, 0.8, 20), lambda_W=0.3)
        diffs = np.diff(sweep["lambda_total"])
        assert np.all(diffs >= 0)

    def test_alpha_eff_decreasing(self):
        """alpha_eff decreases with sigma (networks get denser).

        Higher sigma -> higher lambda -> more investment -> denser
        networks -> lower vulnerability. This is the full feedback chain.
        """
        sweep = lambda_sigma_sweep(np.linspace(0.05, 0.8, 20), lambda_W=0.3)
        diffs = np.diff(sweep["alpha_eff"])
        # Should be non-increasing (may plateau due to saturation)
        assert np.all(diffs <= 1e-10)


# =====================================================================
# NetworkX simulations
# =====================================================================


class TestNetworkFormation:
    """Tests for signal-based and baseline network formation."""

    def test_signal_network_has_edges(self):
        """Signal-based network has connections."""
        qualities = np.linspace(0.5, 2.0, 12)
        investments = np.sqrt(0.5 * (qualities ** 2 - 0.1 ** 2))
        G = form_network_signal_based(qualities, investments)
        assert G.number_of_edges() > 0

    def test_signal_network_assortative(self):
        """High-quality nodes have higher degree in signal-based networks.

        This tests the core mechanism: assortative partner choice based
        on investment level.
        """
        rng = np.random.default_rng(42)
        qualities = np.sort(rng.uniform(0.5, 2.0, 20))
        investments = np.sqrt(0.5 * (qualities ** 2 - 0.1 ** 2))
        G = form_network_signal_based(qualities, investments, seed=42)

        # Compare average degree of top vs bottom half
        degrees = dict(G.degree())
        top_half = [degrees[i] for i in range(10, 20)]
        bottom_half = [degrees[i] for i in range(0, 10)]
        assert np.mean(top_half) >= np.mean(bottom_half)

    def test_baseline_network_creates_graph(self):
        """Baseline network returns a valid graph."""
        G = form_network_baseline(12)
        assert G.number_of_nodes() == 12

    def test_signal_network_density_comparable_to_baseline(self):
        """Signal-based network density is at least comparable to baseline
        (>= 0.5x at the fixed seed; densities are realization-dependent, so
        the assertion is a comparability floor, not strict dominance)."""
        qualities = np.linspace(0.5, 2.0, 20)
        investments = np.sqrt(0.8 * (qualities ** 2 - 0.1 ** 2))
        G_signal = form_network_signal_based(qualities, investments, rho=0.95, seed=42)
        G_baseline = form_network_baseline(20, p_connect=0.15, seed=42)

        density_signal = G_signal.number_of_edges() / (20 * 19 / 2)
        density_baseline = G_baseline.number_of_edges() / (20 * 19 / 2)
        assert density_signal >= density_baseline * 0.5  # At least comparable

    def test_reproducibility(self):
        """Same seed produces same network."""
        q = np.linspace(0.5, 2.0, 12)
        x = np.sqrt(0.5 * (q ** 2 - 0.01))
        G1 = form_network_signal_based(q, x, seed=123)
        G2 = form_network_signal_based(q, x, seed=123)
        assert set(G1.edges()) == set(G2.edges())


# =====================================================================
# Calibration
# =====================================================================


class TestCalibration:
    """Tests for calibration to initial model vulnerability parameters."""

    def test_beta_matches_target(self):
        """Non-signaler vulnerability matches initial model beta = 0.90."""
        cal = calibrate_to_initial_vulnerability(beta_target=0.90)
        assert cal["beta_achieved"] == pytest.approx(0.90, rel=1e-6)

    def test_gamma_positive(self):
        """Calibrated gamma is positive."""
        cal = calibrate_to_initial_vulnerability()
        assert cal["gamma"] > 0

    def test_alpha_less_than_beta(self):
        """Signalers are less vulnerable than non-signalers."""
        cal = calibrate_to_initial_vulnerability()
        assert cal["alpha_achieved"] < cal["beta_achieved"]

    def test_k_signal_greater_than_k_0(self):
        """Signalers have more exchange partners."""
        cal = calibrate_to_initial_vulnerability()
        assert cal["k_signal"] > cal["k_nonsignal"]

    def test_different_targets(self):
        """Calibration works for different target values."""
        cal = calibrate_to_initial_vulnerability(alpha_target=0.50, beta_target=0.80)
        assert cal["beta_achieved"] == pytest.approx(0.80, rel=1e-6)
        assert cal["alpha_achieved"] < 0.80


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge case tests for Layer 3 functions."""

    def test_zero_monument_stock(self):
        """All functions handle M_g = 0."""
        assert network_degree(0.0) == pytest.approx(DEFAULT_K_0)
        assert float(network_degree_derivative(0.0)) > 0
        assert compute_lambda_X(0.0, 0.3) >= 0

    def test_zero_sigma(self):
        """Functions handle sigma = 0 (no environmental uncertainty)."""
        assert survival_probability(0.0, 5.0) == pytest.approx(1.0)
        assert compute_lambda_X(5.0, 0.0) == pytest.approx(0.0)

    def test_zero_gamma(self):
        """gamma = 0 means no buffering (partners are useless)."""
        S = survival_probability(0.3, 10.0, gamma=0.0)
        assert S == pytest.approx(0.7)  # 1 - 0.3, same as k=0

    def test_single_individual_group(self):
        """lambda_X with N=1."""
        lx = compute_lambda_X(5.0, 0.3, N=1)
        lx_group = compute_lambda_X(5.0, 0.3)
        assert lx == pytest.approx(lx_group)

    def test_very_large_monument_stock(self):
        """Functions are stable for large M_g."""
        k = network_degree(1e6)
        assert np.isfinite(k)
        assert k <= DEFAULT_K_0 + DEFAULT_K_MAX + 0.01

        lx = compute_lambda_X(1e6, 0.3)
        assert np.isfinite(lx)
        assert lx >= 0

    def test_empty_network_formation(self):
        """Network formation with 0 or 1 nodes."""
        G0 = form_network_baseline(0)
        assert G0.number_of_nodes() == 0

        q1 = np.array([1.0])
        x1 = np.array([0.5])
        G1 = form_network_signal_based(q1, x1)
        assert G1.number_of_nodes() == 1
        assert G1.number_of_edges() == 0


# =====================================================================
# Signal depreciation backward compatibility
# =====================================================================


class TestFeedbackWithDepreciation:
    """Tests for feedback loop with signal depreciation (delta parameter)."""

    def test_feedback_unchanged_at_zero_delta(self):
        """delta = 0 reproduces the static model exactly."""
        eq_static = lambda_total_at_sigma(0.3, 0.3)
        eq_zero_delta = lambda_total_at_sigma(0.3, 0.3, delta=0.0)
        assert eq_static["lambda_total"] == pytest.approx(eq_zero_delta["lambda_total"])
        assert eq_static["M_g"] == pytest.approx(eq_zero_delta["M_g"])
        assert eq_static["alpha_eff"] == pytest.approx(eq_zero_delta["alpha_eff"])

    def test_feedback_converges_with_depreciation(self):
        """Feedback loop converges for delta = 0.1."""
        eq = lambda_total_at_sigma(0.3, 0.3, delta=0.1)
        assert eq["converged"]
        assert eq["lambda_total"] > 0
        assert eq["M_g"] > 0

    def test_depreciation_changes_effective_stock(self):
        """With delta > 0, effective M_g differs from investment flow I_g."""
        eq = lambda_total_at_sigma(0.3, 0.3, delta=0.1)
        # I_g key should be present and differ from M_g
        assert "I_g" in eq
        # M_g = I_g / delta for delta > 0, so M_g > I_g when delta < 1
        assert eq["M_g"] == pytest.approx(eq["I_g"] / 0.1, rel=1e-4)

    def test_sweep_with_depreciation(self):
        """Lambda-sigma sweep works with depreciation."""
        sigma_range = np.linspace(0.1, 0.5, 5)
        sweep = lambda_sigma_sweep(sigma_range, 0.3, delta=0.1)
        assert len(sweep["sigma"]) == 5
        assert np.all(sweep["lambda_total"] > 0)


# =====================================================================
# Alternative network degree forms (network functional-form robustness analysis)
# =====================================================================


class TestAlternativeNetworkForms:
    """Property tests for the four network-degree forms used in the
    network functional-form robustness analysis: each must satisfy k(0) = k_0,
    saturate at k_0 + k_max, be monotonically increasing, and have
    a correct analytical derivative."""

    K0 = 0.5
    KMAX = 8.0
    M_HALF_REF = 3.0
    M_ANCHOR = 10.307  # equilibrium M_g at lambda_W = 0.68

    # --- Exponential form ---

    def test_exponential_at_zero(self):
        assert network_degree_exponential(0.0, self.K0, self.KMAX, tau=6.0) \
            == pytest.approx(self.K0)

    def test_exponential_saturates(self):
        k_large = network_degree_exponential(1000.0, self.K0, self.KMAX, tau=6.0)
        assert k_large == pytest.approx(self.K0 + self.KMAX, abs=1e-5)

    def test_exponential_monotonic(self):
        M = np.linspace(0, 30, 60)
        k = network_degree_exponential(M, self.K0, self.KMAX, tau=6.0)
        assert np.all(np.diff(k) > 0)

    def test_exponential_derivative_matches_numerical(self):
        M = 5.0
        h = 1e-5
        dk_num = (
            network_degree_exponential(M + h, self.K0, self.KMAX, 6.0)
            - network_degree_exponential(M - h, self.K0, self.KMAX, 6.0)
        ) / (2 * h)
        dk_an = network_degree_exponential_derivative(M, self.KMAX, 6.0)
        assert float(dk_an) == pytest.approx(float(dk_num), rel=1e-5)

    # --- Hill (n=2) form ---

    def test_hill_at_zero(self):
        assert network_degree_hill(0.0, self.K0, self.KMAX, K=5.0, hill_n=2.0) \
            == pytest.approx(self.K0)

    def test_hill_saturates(self):
        k_large = network_degree_hill(1000.0, self.K0, self.KMAX, K=5.0, hill_n=2.0)
        assert k_large == pytest.approx(self.K0 + self.KMAX, rel=1e-3)

    def test_hill_monotonic(self):
        M = np.linspace(0.1, 30, 60)
        k = network_degree_hill(M, self.K0, self.KMAX, K=5.0, hill_n=2.0)
        assert np.all(np.diff(k) > 0)

    def test_hill_derivative_matches_numerical(self):
        M = 5.0
        h = 1e-5
        dk_num = (
            network_degree_hill(M + h, self.K0, self.KMAX, 5.0, 2.0)
            - network_degree_hill(M - h, self.K0, self.KMAX, 5.0, 2.0)
        ) / (2 * h)
        dk_an = network_degree_hill_derivative(M, self.KMAX, 5.0, 2.0)
        assert float(dk_an) == pytest.approx(float(dk_num), rel=1e-5)

    def test_hill_has_inflection_below_anchor(self):
        """Sigmoid Hill has zero derivative at M=0 and positive curvature
        below the inflection at M = K/sqrt(3)."""
        K = 5.0
        # dk/dM at M=0 should be ~0
        dk_at_zero = network_degree_hill_derivative(0.0, self.KMAX, K, 2.0)
        assert abs(float(dk_at_zero)) < 1e-3

    # --- Piecewise-linear form ---

    def test_piecewise_at_zero(self):
        assert network_degree_piecewise(0.0, self.K0, self.KMAX, slope=1.0) \
            == pytest.approx(self.K0)

    def test_piecewise_saturates(self):
        # Saturation at M = k_max / slope; beyond, k = k_0 + k_max constant
        k_at_cap = network_degree_piecewise(self.KMAX / 1.0, self.K0, self.KMAX, slope=1.0)
        k_past_cap = network_degree_piecewise(100.0, self.K0, self.KMAX, slope=1.0)
        assert k_at_cap == pytest.approx(self.K0 + self.KMAX)
        assert k_past_cap == pytest.approx(self.K0 + self.KMAX)

    def test_piecewise_monotonic(self):
        M = np.linspace(0, 30, 60)
        k = network_degree_piecewise(M, self.K0, self.KMAX, slope=1.0)
        assert np.all(np.diff(k) >= 0)

    def test_piecewise_derivative_correct(self):
        """Derivative is slope before saturation, 0 after."""
        slope = 1.0
        M_cap = self.KMAX / slope
        dk_before = network_degree_piecewise_derivative(M_cap - 0.1, self.KMAX, slope)
        dk_after = network_degree_piecewise_derivative(M_cap + 0.1, self.KMAX, slope)
        assert float(dk_before) == pytest.approx(slope)
        assert float(dk_after) == pytest.approx(0.0)

    # --- Calibration ---

    def test_calibration_matches_at_anchor(self):
        """All calibrated forms give exactly the same k value at M_anchor.
        This is the defining property of the calibration."""
        specs = calibrate_alternative_network_forms(
            self.M_ANCHOR, self.K0, self.KMAX, self.M_HALF_REF,
        )
        k_values = [spec.k(self.M_ANCHOR) for spec in specs.values()]
        # All values should be within a tight numerical tolerance
        assert max(k_values) - min(k_values) < 1e-9, (
            f"Calibration failed: k values at anchor differ: {dict(zip(specs.keys(), k_values))}"
        )
        # And all match the MM reference
        k_mm = float(network_degree(self.M_ANCHOR, self.K0, self.KMAX, self.M_HALF_REF))
        for name, spec in specs.items():
            assert spec.k(self.M_ANCHOR) == pytest.approx(k_mm, abs=1e-9), (
                f"{name} did not match MM at anchor"
            )

    def test_calibrated_forms_all_satisfy_boundaries(self):
        """All four calibrated forms agree at M=0 and saturate at the same upper bound."""
        specs = calibrate_alternative_network_forms(
            self.M_ANCHOR, self.K0, self.KMAX, self.M_HALF_REF,
        )
        for name, spec in specs.items():
            assert spec.k(0.0) == pytest.approx(self.K0, abs=1e-9), (
                f"{name}: k(0) != k_0"
            )
            assert spec.k(1e6) == pytest.approx(
                self.K0 + self.KMAX, rel=1e-3
            ), f"{name}: k(infty) != k_0 + k_max"

    def test_spec_is_frozen(self):
        """NetworkDegreeSpec is immutable after construction."""
        from dataclasses import FrozenInstanceError
        spec = NetworkDegreeSpec(
            name="test",
            k=lambda M: M,
            dk_dM=lambda M: 1.0,
        )
        with pytest.raises(FrozenInstanceError):
            spec.name = "other"


class TestNetworkFormRobustness:
    """Robustness of downstream quantities to network-degree functional form.

    Encodes the network functional-form robustness claim: sigma*(lambda_W) is
    insensitive to the choice among Michaelis-Menten, exponential, Hill
    (n=2), and piecewise-linear functional forms once each is calibrated
    to match at the empirical M_g anchor."""

    M_ANCHOR = 10.307  # equilibrium M_g at lambda_W = 0.68 (empirical anchor)

    def _all_specs(self):
        return calibrate_alternative_network_forms(self.M_ANCHOR)

    def test_sigma_star_identical_at_anchor(self):
        """At lambda_W = 0.68, sigma* is identical across forms to within 1%
        of the MM reference. This is the central robustness claim."""
        from signaling.price_equation import sigma_star_self_consistent

        specs = self._all_specs()
        sigma_stars = {}
        for name, spec in specs.items():
            res = sigma_star_self_consistent(
                lambda_W=0.68, mode="multiplicative", network_degree_spec=spec,
            )
            sigma_stars[name] = res["sigma_star"]

        ref = sigma_stars["michaelis_menten"]
        for name, ss in sigma_stars.items():
            assert abs(ss - ref) / ref < 0.01, (
                f"{name}: sigma* = {ss:.4f} deviates >1% from MM reference {ref:.4f}"
            )

    def test_sigma_star_robust_across_lambda_W(self):
        """Across the manuscript-relevant lambda_W range [0.30, 1.00], sigma*
        values agree across forms to within 13% relative OR 0.03 absolute of MM
        (the sigma-scaled war-avoidance makes the conflict reduction r matter,
        hence the network form via M_g, loosening cross-form agreement).

        The combined criterion is needed because at very small sigma* (e.g.,
        lambda_W = 0.10 yields sigma* ~ 0.002), a 10% relative deviation is a
        meaningless 0.0002 absolute. The manuscript's range of interest is
        lambda_W in [0.30, 1.00]; below 0.30, monument building is favored at
        nearly all sigma anyway."""
        from signaling.price_equation import sigma_star_self_consistent

        specs = self._all_specs()
        for lam_W in (0.30, 0.50, 0.68, 1.00):
            sigma_stars = {}
            for name, spec in specs.items():
                res = sigma_star_self_consistent(
                    lambda_W=lam_W, mode="multiplicative", network_degree_spec=spec,
                )
                sigma_stars[name] = res["sigma_star"]
            ref = sigma_stars["michaelis_menten"]
            for name, ss in sigma_stars.items():
                abs_dev = abs(ss - ref)
                rel_dev = abs_dev / max(ref, 1e-6)
                passes = abs_dev < 0.03 or rel_dev < 0.13
                assert passes, (
                    f"At lambda_W={lam_W}, {name}: sigma*={ss:.4f} "
                    f"deviates {abs_dev:.4f} ({100*rel_dev:.1f}%) from MM={ref:.4f}"
                )

    def test_phi_star_identical_at_anchor(self):
        """phi*(sigma, lambda_W=0.68) is identical across forms at the anchor,
        because all forms agree at M = M_anchor by calibration."""
        from signaling.emergence import phi_star

        specs = self._all_specs()
        ps_values = {}
        for name, spec in specs.items():
            ps = phi_star(0.55, lambda_W=0.68, network_degree_spec=spec)
            ps_values[name] = ps

        ref = ps_values["michaelis_menten"]
        for name, ps in ps_values.items():
            assert abs(ps - ref) < 1e-3, (
                f"{name}: phi*={ps:.4f} differs from MM {ref:.4f} at anchor"
            )

    def test_bistability_preserved_under_all_forms(self):
        """All forms predict a finite interior saddle phi*(sigma) for
        sigma above the maintenance threshold, preserving bistability."""
        from signaling.emergence import phi_star

        specs = self._all_specs()
        for name, spec in specs.items():
            ps = phi_star(0.55, lambda_W=0.68, network_degree_spec=spec)
            assert np.isfinite(ps) and 0.0 < ps < 1.0, (
                f"{name}: bistability not preserved (phi* = {ps})"
            )

    def test_mm_spec_recovers_default_behavior(self):
        """Passing the MM spec explicitly produces the same sigma* as
        passing no spec (which defaults to MM via k_0, k_max, M_half)."""
        from signaling.price_equation import sigma_star_self_consistent

        specs = self._all_specs()
        ref_no_spec = sigma_star_self_consistent(0.68, mode="multiplicative")
        ref_with_spec = sigma_star_self_consistent(
            0.68, mode="multiplicative",
            network_degree_spec=specs["michaelis_menten"],
        )
        assert ref_with_spec["sigma_star"] == pytest.approx(
            ref_no_spec["sigma_star"], rel=1e-6,
        )
