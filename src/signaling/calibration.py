"""Parameter calibration and case study values.

Stores empirically grounded parameter values for archaeological case
studies and ethnographic calibration targets. Provides functions to
calibrate derived model parameters against initial model values
and independent ethnographic data.

All parameter values are sourced from published literature; sources are
documented inline. Assumed values are marked explicitly.

See the SI model-parameter section for the assembled
parameter table.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Case study parameters for the assumed-coefficient baseline (the initial
# Price-equation model that the assembly section of the manuscript compares
# against)
# ---------------------------------------------------------------------------

RAPA_NUI: dict[str, float] = {
    # Environmental and demographic parameters
    # ASSUMED: the 60% shortfall magnitude is an illustrative value for
    # the assumed-coefficient baseline, not an empirically verified
    # estimate. The manuscript does not cite it, and production code
    # consumes RAPA_NUI only for "C".
    "shortfall_magnitude": 0.6,       # 60% reduction (ASSUMED, see above)
    "shortfall_frequency": 6,          # ENSO-driven drought clustering
    "shortfall_duration": 2.5,         # Derived from magnitude formula
    "base_productivity": 0.8,          # Marginal conditions
    "carrying_capacity_per_cell": 8,
    # Reproductive cost of monument investment
    # CALIBRATED FROM: Erasmus 1965 + Abrams 1994 architectural energetics.
    # C is the per-builder peak labor fraction during active construction
    # (i = individual person, not per-household-year). Erasmus 1965 p. 283
    # reports ~25 person-days per household per year at Uxmal chiefdom, ~10%
    # of a 250-day household-year budget; each contributing person within that
    # household devotes a higher fraction during active construction episodes.
    # ASSUMED: the 150-250-working-day annual denominator is a project
    # assumption (consistent with the 9-10 months of post-subsistence
    # labor Erasmus 1965 p. 279 attributes to Maya farmers, citing Morley
    # and Brainerd 1956); neither Erasmus nor Abrams states a working-day
    # budget. A builder who contributes 30-60 of their ~150-250 working days per
    # construction episode operates in the 0.20-0.50 peak range. Abrams 1994
    # Table 12 (p. 103): Copan range 24-160 p-d per construction project under
    # familial reciprocity. Abrams 1994 pp. 105-106: state-level Maya corvée
    # ~60 days per construction event every 15-20 years, ~0.30-0.40 of a
    # builder's working year. C = 0.35 sits at the upper end of this per-
    # builder peak range, consistent with active monument-building episodes;
    # C ∈ [0.20, 0.50] is the in-scope per-builder bracket swept in the main text.
    "C": 0.35,
    # Environmental uncertainty (LEGACY, UNUSED): no current code path reads
    # this entry. The illustrative recipe once quoted here,
    # (magnitude * duration) / frequency, evaluates to 0.6 * 2.5 / 6 = 0.25
    # with the values above, NOT the stored 0.20, which is the initial
    # model's historical value kept only as a record. The manuscript does
    # not classify Rapa Nui against
    # sigma*, and the model's sigma is defined as the per-period rate of
    # shortfalls a group cannot weather alone unbuffered (main text,
    # crisis-survival section), for which no validated proxy transfer
    # function exists yet (SI empirical-evaluation section).
    "sigma_initial": 0.20,
    # Initial model assumed parameters (to be derived from signaling
    # dynamics in Layers 2 and 3; stored here for comparison only)
    "r_initial": 0.75,                 # Conflict reduction (assumed)
    "alpha_initial": 0.30,             # Signaler vulnerability (assumed)
    "beta_initial": 0.90,              # Non-signaler vulnerability (assumed)
}

RAPA_ITI: dict[str, float] = {
    # Control case: low environmental uncertainty, no monumentality
    "shortfall_magnitude": 0.3,        # Modern climate analogy
    "shortfall_frequency": 18,         # Attenuated ENSO
    "shortfall_duration": 1.0,
    "base_productivity": 1.2,          # 2500mm rainfall, favorable conditions
    "carrying_capacity_per_cell": 15,
    "C": 0.35,
    "sigma_initial": 0.017,            # Well below any plausible threshold
}

# ---------------------------------------------------------------------------
# Initial model assumed values (for threshold comparison)
# ---------------------------------------------------------------------------

INITIAL_MODEL_PARAMS: dict[str, float] = {
    "C": 0.35,
    "alpha": 0.30,
    "beta": 0.90,
    "r": 0.75,
    # sigma* = C / [beta - (1-C)*alpha] = 0.35 / [0.90 - 0.65*0.30]
    #        = 0.35 / 0.705 ~ 0.496 (simplified)
    # Full formula gives ~0.39 (see initial model supplement S1.7)
    "sigma_star_full": 0.39,
    "sigma_star_simplified": 0.496,
}

# ---------------------------------------------------------------------------
# Ethnographic calibration targets
# ---------------------------------------------------------------------------

CALIBRATION_TARGETS: dict[str, tuple[float, float]] = {
    # (lower_bound, upper_bound) ranges from published sources
    # CALIBRATED FROM: Erasmus 1965 p. 283 (~10% chiefdom-level Uxmal,
    # 25/250 working days) and Abrams 1994 pp. 105-106 (state-level Maya
    # corvée ~60 days/event under the ASSUMED 150-250 working-day
    # denominator (a project assumption; see RAPA_NUI "C" note) gives
    # ~25-40% peak). Range 0.25-0.40 brackets peak labor diversion during
    # active monument construction; sensitivity sweep in main text uses
    # the wider bracket [0.20, 0.50].
    "labor_diversion": (0.25, 0.40),
    # ASSUMED: Wiessner 2006 provides qualitative evidence that Enga tee
    # exchange cycles created economic disincentives for warfare and
    # facilitated peacemaking, but no quantitative conflict reduction
    # estimate. Range 0.60-0.80 is inferred from the qualitative account,
    # not directly reported. PDF now in repo (jar.0521004.0062.203.pdf).
    "conflict_reduction": (0.60, 0.80),
    # INFERRED FROM: Wiessner 1982 (!Kung hxaro) documents network
    # structure and crisis use (69% of possessions obtained via hxaro,
    # p. 70; 48% of the /Xai/xai population dispersed to partners in the
    # 1974 crisis, p. 77) but reports no buffering-effectiveness
    # coefficient. The 0.60-0.70 magnitude is set by variance-reduction
    # analyses: Winterhalder 1986 (~58% SD reduction at six sharers,
    # p. 380; six or fewer capture ~60% of potential reduction, Fig. 5
    # p. 383) and Cashdan 1985 (stored/harvested variance ratio ~0.2,
    # p. 469). PDFs in repo: Wiessner_1982.pdf, Winterhalder_1986.pdf,
    # Cashdan_1985.pdf.
    "network_buffering": (0.60, 0.70),
    # DERIVED FROM: Bowles 2009 reports per-generation decisive conflict
    # probability kappa = 2*delta where delta ~ 0.14 (Table 1, fraction of
    # total mortality from warfare). Annualizing: 0.28/25 ~ 0.011/year.
    # Range 0.005-0.02 brackets this estimate. Bowles does not perform
    # the annualization himself. PDF now in repo.
    "baseline_conflict": (0.005, 0.02),
}

# ---------------------------------------------------------------------------
# Default exploration parameters for Layer 1
# ---------------------------------------------------------------------------

# ASSUMED: mid-range value for initial exploration before Layer 3 feedback
# is formalized. Lambda should ultimately be endogenous via lambda(sigma).
DEFAULT_LAMBDA: float = 0.5

# ASSUMED: quality range for numerical exploration. Qualitative results
# do not depend on the specific range as long as q_min > 0.
DEFAULT_Q_MIN: float = 0.1
DEFAULT_Q_MAX: float = 2.0

# ILLUSTRATION DEFAULT: group size for code that hard-codes a single n. The
# framework treats n as a sweep across the face-to-face audience range
# n ∈ [10, 250]; the lower bound is set by where the framework's signals are
# operative (M_g large enough for the network function to matter), the upper
# bound by the face-to-face CRED audience constraint (beyond which witnessing
# breaks down). Production code should pass n explicitly; n = 12 is retained
# here only as a single-point illustration anchored within the
# subsistence-group sizes Roscoe 2009 reports for New Guinea ("about
# eight to 20 people", p. 78; Table I group forms extend to ~25). Note
# that 12 lies within Roscoe's reported spans but is not a midpoint of
# any range Roscoe states. Qualitative
# results (β₀ > 0, polarity finding, σ* exists) are invariant in n; the σ*
# value itself depends on n through Layer 3 saturation and Layer 2 deterrence.
DEFAULT_N_COMPUTE: int = 12

# Alias kept so existing import sites (src, tests, figure scripts)
# can continue to import DEFAULT_N without simultaneous mass migration. New
# code should import DEFAULT_N_COMPUTE.
DEFAULT_N: int = DEFAULT_N_COMPUTE

# ---------------------------------------------------------------------------
# Signal depreciation
# ---------------------------------------------------------------------------

# ASSUMED: signal depreciation rate per period. Monuments persist physically
# but their informational value as signals of *current* group capacity
# depreciates over time: a structure built decades ago does not credibly
# indicate that the present group can mobilize comparable labor.
# Reinvestment (renovation, expansion) is required to maintain signal
# currency. At delta = 0.10, the signal half-life is approximately 6.6
# periods, and steady-state monument stock is M_g* = I_g / delta = 10 * I_g.
# Alternative: 0.05 (durable stone construction in stable environments)
# or 0.20 (perishable materials or rapid social change).
# Setting delta = 0 recovers the static model (no depreciation).
DEFAULT_DELTA: float = 0.10

# Sensitivity sweep range for delta: (start, stop, step)
DELTA_SENSITIVITY_RANGE: tuple[float, float, float] = (0.01, 0.30, 0.05)

# ---------------------------------------------------------------------------
# Between-group conflict: scarcity-driven war cost (war-avoidance coordination)
# ---------------------------------------------------------------------------

# DEFINITION (load-bearing): W(sigma) = DEFAULT_WAR_COST * sigma is the EXPECTED
# PER-PERIOD fitness cost of warfare exposure for an unassessed group. The
# coefficient w0 folds together the frequency of hostile encounters in the
# unassessed baseline and the fitness lost per conflict; it is NOT the cost of a
# single war (a conditional per-war coefficient would be w0 divided by the
# per-period conflict frequency and plays no role in the model). War cost rises
# with environmental uncertainty sigma (scarcity sharpens competition over
# resources). Build-build dyads settle their contest by mutual assessment
# (reduction r_bb from Layer 2) and avoid the fraction r of this expected cost;
# unassessed dyads pay it in full. P_base enters the derivation of r only as the
# normalization of the probability ratio r = 1 - P_conflict/P_base, from which
# it cancels; it is not an additional multiplier on W(sigma). This is the
# competitive between-group coordination channel, realized as mutual
# war-avoidance -- a positive-sum, absolute benefit (both build-build parties
# save), not a positional penalty.
# ASSUMED (soft calibration): w0 sets the expected-cost scale; the war-avoidance
# benefit at full adoption is ~W(sigma)*r_bb. Default chosen so war-avoidance is
# a meaningful, non-dominant benefit.
DEFAULT_WAR_COST: float = 0.30

# ---------------------------------------------------------------------------
# Channel selection analysis: audience-specific lambda decomposition
# ---------------------------------------------------------------------------

# ASSUMED: exploration defaults for the audience-level returns in the
# channel-selection comparison. Manuscript notation: tilde-lambda_C and
# tilde-lambda_X, the between-group returns available at the EMERGENCE STAGE
# (low stock), distinct from the derived marginal feedback values lambda_C,
# lambda_X, which are evaluated at the saturated anchor stock and sit below
# 1% of lambda_W there. Verified reconciliation: the derived cooperative
# return compute_lambda_X passes through 0.15 at emergence-stage stocks
# (0.10-0.18 for M_g in [0.5, 1] at sigma = 0.68); the competitive anchor
# is an exploration weight whose realized fitness counterpart is the
# war-avoidance advantage W(sigma)*phi*r. See the SI sections
# "Channel-selection audience-weight decomposition" and "Channel-selection
# sensitivity".
#
# The relative magnitudes reflect the expectation that within-group rewards
# are the largest component, with competitive and cooperative between-group
# returns roughly equal and each about half the within-group component.
# These are exploration values subject to empirical calibration; qualitative
# results (monument dominance when all three audiences are active) hold for
# any values satisfying lam_W > lam_C and lam_W > lam_X.
DEFAULT_LAM_W: float = 0.3     # Within-group social rewards
DEFAULT_LAM_C: float = 0.15    # Competitive between-group (conflict deterrence)
DEFAULT_LAM_X: float = 0.15    # Cooperative between-group (exchange access)

# Empirical anchor for within-group reward parameter lambda_W in σ* reporting.
# CALIBRATED FROM: lambda_W = 0.68 is the central anchor used in
# the main text (results, sensitivity analysis, and conclusion) to report σ* at the empirical anchor, consistent
# with Erasmus 1965 per-builder peak C = 0.35 and the in-scope (C, n)
# rectangle. The value derives from the per-builder C ≈ 0.35 calibration via
# lambda_W ≈ 1.93 · C (see the main-text C--lambda_W relation).
# Centralized here to eliminate duplication across the docstrings, the standalone figure scripts, and the tests. The framework sweeps lambda_W via the
# (C, n) rectangle; 0.68 remains the central anchor for reporting.
DEFAULT_LAMBDA_W_ANCHOR: float = 0.68

# ---------------------------------------------------------------------------
# Channel selection analysis: signal fidelity defaults
# ---------------------------------------------------------------------------

# Status: Exploration anchors, NOT calibrated from primary sources.
# ---------------------------------------------------------------------------
# The rho_s values below are illustrative defaults consistent with the
# *qualitative* ordering established in the costly-signaling literature on
# small-scale forager and horticulturalist societies: turtle hunting and
# big-game hunting return broad social benefits but only moderate productive-
# capacity information (Hawkes & Bliege Bird 2002; Bliege Bird & Smith 2005;
# Smith, Bliege Bird & Bird 2003), feasting is degraded by transient windfall
# substitution (Bliege Bird & Smith 2005), ritual ordeal and ascetic display
# convey commitment more reliably than capacity (Sosis & Bressler 2003;
# Norenzayan et al. 2016; cf. Henrich 2009 on credibility-enhancing displays
# / CREDs for the ritual-as-commitment channel), and verbal claims approximate
# cheap talk. None of these papers report numerical rho_s values directly;
# the magnitudes below are exploration anchors chosen to span the qualitative
# ordering, and the framework's headline results are tested for sensitivity
# to this ordering rather than to specific values. See the main-text
# channel-selection analysis and the supplementary channel-selection
# sensitivity figure (rho_s exploration-anchor disclosure) for the manuscript-side treatment.
# ---------------------------------------------------------------------------
#
# Monument: near-perfect fidelity. No windfall substitution pathway;
#   labor is physical, cumulative, publicly witnessed. Slightly below 1.0
#   to allow for minor noise (e.g., favorable stone geology).
DEFAULT_RHO_MONUMENT: float = 0.95
#
# Feast: moderate fidelity. Vulnerable to windfall substitution: a lucky
#   harvest or trade windfall allows low-quality households to produce
#   high-signal feasts. The value 0.5 reflects a windfall probability of
#   ~0.3 applied to a baseline fidelity of ~0.7.
DEFAULT_RHO_FEAST: float = 0.5
#
# Ritual: low fidelity for productive capacity. Ritual endurance signals
#   commitment and pain tolerance, which have weak correlation with the
#   productive/coordinative capacity that audiences care about.
DEFAULT_RHO_RITUAL: float = 0.3
#
# Hunting: moderate fidelity. Individual hunting skill has some correlation
#   with productive capacity but signals individual prowess rather than
#   collective capacity. The mismatch reduces effective fidelity.
DEFAULT_RHO_HUNTING: float = 0.4
#
# Verbal: near-zero fidelity. Cheap talk carries minimal information
#   about productive capacity.
DEFAULT_RHO_VERBAL: float = 0.05

# ASSUMED: exploration default for feasting windfall probability.
# Represents the fraction of high-signal feasts funded by transient
# windfalls rather than sustained productive capacity. At 0.3, roughly
# one in three impressive feasts is "fake" in the sense of not reflecting
# underlying quality. Used in the fidelity sensitivity analysis.
DEFAULT_WINDFALL_PROB: float = 0.3

# ---------------------------------------------------------------------------
# Channel selection analysis: per-channel fixed (access) costs
# ---------------------------------------------------------------------------
#
# Status: SWEPT exploration parameters, NOT calibrated. Channel selection
# ranks channels by the community-mean NET return
#   V_s = lambda_s * rho_s * A_bar - f_s,
# where A_bar = E[(q^2+q_min^2)/(2q)] ~ 0.533 is the mean payoff multiplier
# (layer1.mean_payoff_multiplier) and f_s is the channel's scale-independent
# FIXED (access) cost in payoff units. The per-unit (marginal) cost cancels
# at the Spence separating equilibrium -- a signaler facing a higher marginal
# cost simply invests less, leaving the equilibrium net payoff
# lambda_s*rho_s*(q^2+q_min^2)/(2q) unchanged -- so only a fixed cost can
# make channel choice conditional. The payoff-unit defaults below equal the
# former reward-unit values (0.30, 0.02) times A_bar, preserving the
# dominance frontier under the payoff-unit criterion. Monuments carry a substantial
# fixed cost (a minimum recognizable scale plus collective-labor overhead);
# feasting/ritual/hunting scale down cheaply; verbal is ~free. Derivation: the
# main-text subsection "Channel selection: why monuments specifically" and the
# SI section "Channel-selection sensitivity: monument fixed cost and audience weights".
#
# f_M is uncalibrated (abstract payoff units, no empirical anchor), so the
# manuscript reports monument dominance as a RANGE swept over f_M in
# [0, 0.24] (fixed-cost sensitivity), NOT at a single value. The monument
# value below is an ILLUSTRATIVE default for the (lambda_C, lambda_X)
# boundary panel and worked example, not a calibration.
DEFAULT_FIXED_COST_MONUMENT: float = 0.16   # illustrative; swept over [0, 0.24]
DEFAULT_FIXED_COST_FEAST: float = 0.011
DEFAULT_FIXED_COST_RITUAL: float = 0.011
DEFAULT_FIXED_COST_HUNTING: float = 0.011
DEFAULT_FIXED_COST_VERBAL: float = 0.0

# ---------------------------------------------------------------------------
# Layer 2: Assessment noise decomposition parameters
# ---------------------------------------------------------------------------

# The assessment noise sigma_0 has three independent variance components:
#   sigma_0^2 = sigma_deception^2 + sigma_lag^2 + sigma_mismatch^2
# The following parameters allow sigma_0 to be derived from physical
# quantities rather than set as a free parameter. See constrained_sigma_0()
# in layer2.py.

# ASSUMED: standard deviation of group capacity fluctuations over time.
# A group's productive capacity varies from period to period due to
# demographic changes, environmental shocks, and leadership transitions.
# At sigma_q = 0.3, capacity fluctuates by roughly 30% of the mean per
# period. This controls the lag noise component: more variable capacity
# means the monument stock (an exponentially weighted moving average of
# past investment) diverges more from current capacity.
# Alternative: 0.15 (stable groups) or 0.5 (volatile groups).
DEFAULT_SIGMA_Q: float = 0.3

# ASSUMED: correlation between productive capacity (what monuments reveal)
# and fighting capacity (what matters for conflict). In small-scale
# societies, both depend on the same underlying resources: labor pool,
# food stores, coordination ability, and leadership quality. At 0.8,
# productive capacity explains 64% of fighting capacity variance.
# Alternative: 0.6 (weak coupling, e.g., pastoral societies where herding
# skill differs from raiding skill) or 0.9 (strong coupling, e.g.,
# societies where the same labor force farms and fights).
DEFAULT_RHO_PROD_FIGHT: float = 0.8

# ASSUMED: variance of fighting capacity (RHP) across groups, normalized
# to the contest value scale V^2. Following the convention of normalizing
# contest parameters relative to V = 1.0, this defaults to 1.0.
DEFAULT_VAR_RHP: float = 1.0

# ---------------------------------------------------------------------------
# Layer 2: Intergroup assessment default parameters
# ---------------------------------------------------------------------------

# ASSUMED: baseline assessment noise for the Enquist-Leimar mutual assessment
# model. Represents the standard deviation of the perceived RHP difference
# when total monument investment is zero. Benchmarked to sigma_0 = 1.0 in
# Enquist & Leimar (1983, Table 1) as the natural scale for assessment
# uncertainty relative to normalized contest value V = 1.
# The constrained_sigma_0() function in layer2.py derives sigma_0 from
# physical quantities (signal fidelity, depreciation rate, capacity variance,
# productive-fighting correlation), yielding approximately 0.89 at default
# parameters. The value 1.0 is retained here as a round-number default for
# backward compatibility; the constrained derivation serves as validation.
DEFAULT_SIGMA_0: float = 1.0

# ASSUMED: information gain rate from monument investment. Controls how
# rapidly assessment noise decreases with total monument stock:
# sigma_eff = sigma_0 / sqrt(1 + kappa * (M_g + M_h)).
# At kappa = 0.1, doubling total monument stock from 10 to 20 reduces noise
# by ~30%. Setting kappa = 0 recovers constant noise (pure SAM baseline).
# Alternative: kappa proportional to monument visibility or archaeological
# preservation. Higher kappa means monuments are more informative signals.
DEFAULT_KAPPA: float = 0.1

# ASSUMED: base escalation threshold scale. The actual threshold is
# T = T_0 * V / D, so T_0 controls how narrowly groups discriminate.
# Refined by calibration to hit r ~ 0.75. Starting value 0.5 means groups
# tolerate a half-unit perceived difference before deferring.
# Alternative: T_0 could be endogenous (groups adjust thresholds via
# learning), but we treat it as fixed per the static game formulation.
DEFAULT_T_0: float = 0.5

# ASSUMED: absolute deterrence coefficient. Captures the idea that high total
# investment signals destructive potential regardless of relative difference.
# Pure mutual assessment (beta = 0) predicts symmetric high-investing dyads
# fight MORE because precise assessment confirms even matching. The beta
# term corrects this by adding absolute deterrence.
# Alternative: beta could depend on monument type (defensive vs. ceremonial).
# Refined by calibration; starting at 0.1 gives moderate deterrence.
DEFAULT_BETA_CONFLICT: float = 0.1

# ASSUMED: normalized contest value following Enquist & Leimar (1983)
# convention. All other parameters are scaled relative to V = 1.
# In absolute terms, V represents the reproductive fitness value of the
# contested resource (territory, freshwater, foraging area).
DEFAULT_V: float = 1.0

# ASSUMED: fighting cost parameter. D > V means fighting costs exceed
# resource value for intergroup contests, consistent with the observation
# that intergroup violence is typically destructive enough to deter
# escalation when assessment is available.
# Alternative: D = 1.5 (moderate), D = 3.0 (highly destructive).
# Higher D lowers the escalation threshold T = T_0 * V / D.
DEFAULT_D: float = 2.0

# ASSUMED: annual intergroup conflict probability per group pair.
# Bowles (2009, Table 1) reports fraction of total mortality from warfare
# delta ~ 0.14, yielding per-generation decisive conflict probability
# kappa = 2*delta ~ 0.28. Annualized over ~25-year generation: ~0.011.
# P_base = 0.01 is consistent with this derivation. Note: Bowles's
# "decisive conflict" (group elimination) differs from the less severe
# contests modeled here, so P_base may undercount minor conflicts.
# Alternative: 0.005 (low end) or 0.02 (high end).
DEFAULT_P_BASE: float = 0.01

# ASSUMED: exponential approximation decay rate for the tractable form
# P_exp = P_base * exp(-alpha * delta / sigma_eff) / (1 + beta*(M_g+M_h)).
# Calibrated so that the exponential form closely tracks the Gaussian CDF
# form over the relevant parameter range. Starting value alpha = 1.0
# gives roughly comparable behavior to the Gaussian model near T ~ 0.5.
DEFAULT_ALPHA_CONFLICT: float = 1.0

# ASSUMED: self-assessment model threshold. The inflection point of the
# sigmoid p(M) = sigmoid((M - M_thresh) / M_scale). Groups with M above
# this threshold are more likely to escalate. Set to the expected monument
# stock for a mid-quality group; refined by numerical exploration.
DEFAULT_M_THRESH: float = 2.0

# ASSUMED: self-assessment model scale. Controls the steepness of the
# sigmoid: smaller values mean a sharper transition from "defer" to
# "escalate." At M_scale = 1.0, the transition spans roughly 4 units
# of monument stock centered on M_thresh.
DEFAULT_M_SCALE: float = 1.0

# ASSUMED: Bourgeois model ownership threshold. Groups with M above this
# are recognized as "owners." Set equal to M_thresh for consistency.
DEFAULT_M_OWN: float = 2.0

# ASSUMED: Bourgeois model steepness. Controls how sharply the ownership
# clarity function transitions. Higher values mean a narrower ambiguity
# zone around M_own.
DEFAULT_STEEPNESS: float = 2.0

# ASSUMED: war of attrition cost rate. The rate at which fitness costs
# accumulate during conflict. Higher c_rate means conflicts are costlier,
# leading to shorter expected durations.
DEFAULT_C_RATE: float = 0.1

# ---------------------------------------------------------------------------
# Layer 3: Network formation and crisis buffering parameters
# ---------------------------------------------------------------------------

# CALIBRATED FROM: Winterhalder 1986, Wiessner 1982 (hxaro network
# buffering in !Kung San). With gamma = 0.3 and 6 exchange partners,
# the fraction of crisis impact absorbed by the network is
# 1 - 1/(1 + 0.3*6) = 0.64, consistent with the CALIBRATION_TARGETS
# range of 0.60-0.70 for network buffering efficiency.
# Alternative: 0.15 (weak buffering) or 0.5 (strong buffering).
# Sensitivity: alpha_eff = 1/(1 + gamma*k), so doubling gamma has
# the same effect as doubling k on vulnerability reduction.
DEFAULT_GAMMA: float = 0.3

# ASSUMED: baseline intergroup network degree without monument signaling.
# Represents exchange partnerships maintained through kinship, proximity,
# and verbal agreements alone. Set to 0.5 to reflect that reliable
# crisis-buffering partnerships (not just social contacts) require more
# commitment than kinship alone provides. At k_0 = 0.5, non-signalers
# have vulnerability beta_eff = 1/(1 + 0.3*0.5) = 0.87, close to the
# initial model's beta = 0.90.
# Alternative: 1.0 (more kinship-based partnerships) or 0.2 (very sparse).
DEFAULT_K_0: float = 0.5

# ASSUMED: maximum additional network degree achievable through monument
# signaling. With k_max = 8, a fully signal-saturated group can have up
# to k_0 + k_max = 8.5 exchange partners. This is consistent with
# ethnographic data on exchange network sizes in small-scale societies
# (Wiessner 1982 reports hxaro networks of 2-30 partners, median ~12,
# but not all are reliable crisis-buffering partners).
# Alternative: 5 (conservative) or 12 (generous).
DEFAULT_K_MAX: float = 8.0

# ASSUMED: half-saturation monument stock for network formation (Michaelis-
# Menten parameter). At M_g = M_half, half the signal-based connections are
# active. Set to 3.0, meaning moderate monument investment already yields
# substantial network growth.
# CALIBRATED UNDER n=12 ILLUSTRATION: with default Layer 1 parameters (n=12,
# lambda=0.5, q_min=0.1, q_max=2.0), expected M_g is roughly 7-8, so
# M_half = 3.0 means the saturating region is reached. The framework treats
# n as a sweep across n ∈ [10, 250] (face-to-face audience range), so the
# expected M_g at fixed lambda_W also varies with n; the M_half = 3.0 anchor
# is preserved here as an illustrative reference for the n = 12 cross-
# section, and tests / figure scripts that depend on the specific saturation
# behavior should note their n assumption explicitly.
# Alternative: 1.0 (fast saturation) or 6.0 (slow, requires heavy investment).
DEFAULT_M_HALF: float = 3.0

# ASSUMED: baseline connection probability for Erdos-Renyi random network
# (kinship/proximity model).
# CALIBRATED UNDER n=12 ILLUSTRATION: with n = 12 and p = 0.15, expected
# degree is ~1.7, somewhat higher than k_0 because the ER model includes
# unreliable connections as well. The framework sweeps n; the expected ER
# degree at fixed p scales linearly with n, so tests / figure scripts that
# depend on a specific connectivity at non-default n should adjust p
# accordingly or note the implicit n assumption.
DEFAULT_P_CONNECT_BASE: float = 0.15

# ---------------------------------------------------------------------------
# Price equation: assembled fitness parameters
# ---------------------------------------------------------------------------

# ASSUMED: group-level mortality per conflict event. When intergroup
# conflict occurs, this fraction of the group is killed or suffers
# fitness loss equivalent to death. At 0.1, conflict is damaging but
# not catastrophic per event (consistent with raiding rather than
# annihilation). Combined with P_base = 0.01, the annual conflict
# mortality rate is P_base * m = 0.001. Bowles (2009, Table 1) reports
# lifetime war mortality delta ~ 0.14; annualized over ~50 adult years
# this is ~0.003/year, bracketing our 0.001 estimate given that most
# conflicts in this model are not decisive.
# Alternative: 0.05 (minor skirmishes) or 0.3 (intense warfare).
DEFAULT_CONFLICT_MORTALITY: float = 0.1

# ---------------------------------------------------------------------------
# Layer 2: lambda_C computation parameters
# ---------------------------------------------------------------------------

# ASSUMED: fitness cost incurred by each group when intergroup conflict
# occurs. Expressed in the same fitness units as V. A conflict that occurs
# costs each group this amount in destroyed resources, casualties, and
# disrupted production. At 0.5 (half the contest value V=1.0), conflict is
# costly but not catastrophic, consistent with ethnographic accounts of
# raiding and territorial skirmishes rather than total warfare.
# Alternative: 0.2 (minor raiding) or 1.0 (devastating conflict where
# costs equal the full resource value). Higher values increase lambda_C,
# making monument investment more valuable for deterrence.
# Sensitivity: lambda_C is approximately linear in conflict_cost, so
# doubling this parameter roughly doubles lambda_C.
DEFAULT_CONFLICT_COST: float = 0.5

# ASSUMED: annual probability of territorial dispute per neighbor pair.
# Bowles (2009) data implies ~1% annual decisive conflict rate (see
# P_BASE note above). Not all disputes escalate to decisive conflict,
# so the dispute rate should be somewhat higher. We use 0.05 (5% per
# neighbor pair per year) as a moderate estimate, implying roughly one
# in twenty neighbor-pair-years involves a territorial dispute that
# could escalate.
# Alternative: 0.02 (low-conflict environments) or 0.10 (high-competition
# contexts like Rapa Nui). Higher values increase lambda_C.
DEFAULT_DISPUTE_FREQ: float = 0.05

# ASSUMED: number of neighboring groups in the territory graph. For
# small-scale societies with roughly contiguous territories, 4-6 neighbors
# is typical for irregular tessellations. We use 4 as a conservative
# default; Rapa Nui had approximately 11 territorial clans, giving each
# interior territory roughly 3-5 neighbors.
# Alternative: 3 (linear coastal arrangement) or 6 (hexagonal packing).
# lambda_C scales linearly with n_neighbors.
DEFAULT_N_NEIGHBORS: int = 4

# =====================================================================
# Placement prediction (SI section "Further derived predictions: placement,
# competitive investment, and size inequality").
# Downstream module: consumes the audience weights and a per-unit return;
# feeds back into nothing. Exposure coordinate z is normalized to [0, 1].
# =====================================================================
PLACEMENT: dict[str, float] = {
    # ASSUMED: exposure normalized; z=0 core, z=1 most peripheral. Scale is
    # immaterial because the qualitative results are scale-free.
    "z_max": 1.0,
    # ASSUMED: half-saturation exposure for between-group visibility. Sets
    # where eta_C reaches 1/2. Alternative: any z_half in (0, z_max); only
    # the magnitude of z*, not its interiority or the comparative-static
    # signs, depends on it.
    "z_half": 0.3,
    # ASSUMED: within-group visibility decay rate (eta_W = exp(-b z)).
    # Larger b concentrates peers more tightly at the core. Sensitivity:
    # shifts z* toward the core, does not change result signs.
    "b": 3.0,
    # ASSUMED: resource-cost gradient (scarcity). Unit default makes the cost
    # commensurate with the unit visibility benefit at R = 1. Alternative:
    # gamma in (0.1, 10) rescales the cost curve. Sensitivity: larger gamma
    # pulls z* toward the core, smaller gamma allows z* further out; the
    # comparative-static signs are robust for all gamma > 0. Swept in the
    # test dz*/d gamma < 0.
    "gamma": 1.0,
    # ASSUMED: illustrative unit default. R is location-independent and scales
    # the visibility benefit uniformly, so for any R > 0 it changes neither z*
    # nor the signs of the comparative statics (results are R-independent).
    # Alternative: set R = DEFAULT_LAMBDA * DEFAULT_RHO_MONUMENT (the monument
    # channel net return). Sensitivity: sign-robust for all R > 0.
    "R": 1.0,
}

# =====================================================================
# Competitive-investment prediction (SI section "Further derived predictions:
# placement, competitive investment, and size inequality").
# Decoupled module: monument labor as Tullock-contest effort; output is a
# standalone result, never wired into lambda_C / lambda / x* / sigma*.
# =====================================================================
COMPETITION: dict[str, float] = {
    # ASSUMED: common contest prize (deterrence/standing value). Illustrative
    # unit default; the hump, the directed peak, and the regional-inequality
    # result are all V-independent (V scales every effort uniformly).
    # Alternative: a prize that scales with capacity (richer groups value the
    # contest more) would tilt efforts toward the stronger group; that is a
    # flagged robustness check, not the default. Sensitivity: V > 0 rescales
    # magnitudes only, not any sign or peak location.
    "V": 1.0,
}
