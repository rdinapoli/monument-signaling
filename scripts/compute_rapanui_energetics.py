"""In-scope Rapa Nui ahu labor-diversion estimate for the C_emp anchor (SI).

Produces the numbers reported in the supplementary "Rapa Nui energetics" passage.
This is an order-of-magnitude, in-scope cross-check on the architectural-energetics
bracket C_emp in [0.20, 0.50], whose only directly measured anchors are Mesoamerican
(Erasmus 1965 Uxmal chiefdom; Abrams 1994 Copan, a state-corvee context outside the
framework's scope). The point is to show that small-scale, non-state Polynesian
construction can reproduce the bracket from WITHIN scope.

Every input is labeled DATA (measured, sourced) or ASSUMED (reasoned, with rationale).
The honest summary: the rate and volumes are well grounded; the community size N and
the construction-burst duration T are NOT measured and are carried as ranges. The
qualitative conclusion (in-scope construction reaches bracket-level diversion during
active episodes, and is negligible when averaged over the centuries-long use-life) is
robust across the plausible N, T ranges; the exact fraction is not.

Run: python3 scripts/compute_rapanui_energetics.py
"""

# --- Labor rate -------------------------------------------------------------
# CALIBRATED FROM Kolb 1994 (Current Anthropology 35:521-547), Table 1: eight
# excavated Maui heiau with paired Volume (m3) and Labor-Days. Effective rate =
# total labor-days / total volume. Kolb's labor-days already embed rock density x
# experimental per-task rates (procurement/transport/construction, kg/laborer/day;
# Erasmus 1965; Kolb 1991), so labor-days/m3 transfers directly to DiNapoli 2020's
# SfM ahu volumes (also total stone volume) without reapplying density.
# Validation: Pi'ilanihale 21,938 m3 / 128,155 LD = 5.84 LD/m3, and its 128,155 LD
# matches the figure cited via DiNapoli & Morrison 2017.
KOLB_TABLE1 = {  # heiau: (volume_m3, labor_days)
    "Halekii": (9513, 39368), "Lanikele": (785, 2775), "Loaloa": (5834, 21462),
    "Kaimupeelua": (92, 237), "Molohai": (126, 1609), "Pihana": (6686, 24011),
    "Piilanihale": (21938, 128155), "Popoiwi": (11099, 53537),
}
RATE = sum(l for _, l in KOLB_TABLE1.values()) / sum(v for v, _ in KOLB_TABLE1.values())
# ASSUMPTION (rate transfer): Hawaiian heiau and Rapa Nui ahu are both dry-stone
# basalt platform/terrace construction, so the per-m3 rate is taken as transferable.
# This is the single biggest cross-tradition assumption; flag it in the prose.

# --- Ahu volumes ------------------------------------------------------------
# DATA: DiNapoli 2020 dissertation, SfM photogrammetry of 90 platform ahu.
AHU_MEDIAN_M3 = 297     # DATA: median volume (range 12.5-4629; right-skewed)
AHU_UPPER_M3 = 1000     # DATA-ish: representative of the upper tercile (17 ahu > 1000 m3)
AHU_MAX_M3 = 4629       # DATA: largest, Heki'i 1

# --- Labor budget -----------------------------------------------------------
# DATA: Erasmus 1965 (p. 279) attributes ~9-10 months of post-subsistence labor to
# Maya farmers; we take ~285 person-days/household/year as the denominator (same
# basis as the 0.10 Uxmal figure: 25 man-days / ~250-285 ~ 0.10).
BUDGET_DAYS = 285       # ASSUMED denominator (in-scope post-subsistence labor budget)

# --- Community size and burst duration (NOT measured; carried as ranges) -----
# ASSUMED N: from the convergent ~5000 maximum island population (multiple lines of
# evidence) divided among ~150 platform ahu accumulated over ~500 yr. If a third to
# a half were active contemporaneously (~50-75 communities), that is ~13-20 households
# of ~5 people each. Small, dispersed kin groups. Carried as a range.
N_HOUSEHOLDS = (10, 15, 20)
# ASSUMED T: a median ahu platform (~1400 labor-days) is small enough that a 10-20
# household community could raise it in roughly one to a few years of seasonal effort.
# Direct construction-duration data do not exist; carried as a range.
T_BURST_YEARS = (1, 2, 3)
# ASSUMED lifetime: ahu were used and added to over centuries; total lifetime labor is
# the initial platform plus episodic renovation. We illustrate with ~3x the initial
# labor over a ~300-year use-life (initial + ~2x in cumulative renovations). This is
# illustrative, not measured; its only role is to show the lifetime average is tiny.
LIFETIME_RENOVATION_FACTOR = 3
LIFETIME_YEARS = 300


def labor_days(volume_m3):
    return volume_m3 * RATE


def per_episode_per_household(volume_m3, n):
    return labor_days(volume_m3) / n


def annual_fraction(volume_m3, n, t_years):
    return labor_days(volume_m3) / (n * t_years * BUDGET_DAYS)


if __name__ == "__main__":
    print(f"Kolb-derived labor rate: {RATE:.2f} labor-days/m3 "
          f"(per-temple range {min(l/v for v,l in KOLB_TABLE1.values()):.2f}"
          f"-{max(l/v for v,l in KOLB_TABLE1.values()):.2f})")
    print(f"\nTotal construction labor (rate {RATE:.1f} LD/m3):")
    for label, v in [("median ahu (297 m3)", AHU_MEDIAN_M3),
                     ("upper-tercile (1000 m3)", AHU_UPPER_M3),
                     ("largest Heki'i 1 (4629 m3)", AHU_MAX_M3)]:
        print(f"  {label:28} {labor_days(v):8.0f} labor-days")

    print(f"\nPer construction EPISODE, person-days/household "
          f"(Abrams 1994 in-scope familial range: 24-160 pd/hh):")
    for n in N_HOUSEHOLDS:
        print(f"  N={n:>2} hh:  median ahu {per_episode_per_household(AHU_MEDIAN_M3,n):4.0f}"
              f" | upper-tercile {per_episode_per_household(AHU_UPPER_M3,n):4.0f} pd/hh")

    print(f"\nAnnual diversion FRACTION during a construction burst "
          f"(budget {BUDGET_DAYS} pd/hh/yr; Erasmus ~0.10, Abrams peak ~0.40, bracket [0.20,0.50]):")
    print("  median ahu:")
    for n in N_HOUSEHOLDS:
        row = "  ".join(f"T={t}:{annual_fraction(AHU_MEDIAN_M3,n,t):.2f}" for t in T_BURST_YEARS)
        print(f"    N={n:<3} {row}")
    print("  upper-tercile ahu (major episode):")
    for n in N_HOUSEHOLDS:
        row = "  ".join(f"T={t}:{annual_fraction(AHU_UPPER_M3,n,t):.2f}" for t in T_BURST_YEARS)
        print(f"    N={n:<3} {row}")

    life = labor_days(AHU_MEDIAN_M3) * LIFETIME_RENOVATION_FACTOR / (15 * LIFETIME_YEARS * BUDGET_DAYS)
    print(f"\nLifetime-averaged fraction (median ahu, N=15, ~3x initial labor over "
          f"{LIFETIME_YEARS} yr): {life:.4f}  (negligible)")
