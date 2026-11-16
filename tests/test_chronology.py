"""
Tests for crater-count dating (simulation/chronology.py, simulation/surface_dating.py).

The production and chronology functions are checked against published
values; the age inference against synthetic surfaces of known age; and the
real-data conclusions (Chang'e-6 recovered, naive circles biased old) are
pinned to bands so a change that moves them gets noticed.
"""

import math

import numpy as np
import pytest

from simulation.chronology import (
    NEUKUM_2001,
    YUE_2022,
    bombard,
    cumulative_density,
    infer_age,
    production_shape,
    sample_diameters,
    sample_formation_ages,
    visible_at,
)
from simulation.surface_dating import (
    LANDING_SITES,
    change6_analysis,
    great_circle_km,
    landing_site_ages,
    landing_site_craters,
    spherical_cap_area_km2,
    synthetic_history,
)


# ---------------------------------------------------------------------------
# Production and chronology functions
# ---------------------------------------------------------------------------

def test_chronology_n1_at_1_gyr_matches_the_pf_a0():
    # Craterstats: the published a0 = -3.0876 is a typo for -3.0768 = log10 N1(1 Gyr)
    assert math.log10(NEUKUM_2001.n1(1.0)) == pytest.approx(-3.0768, abs=1e-4)


def test_chronology_is_nearly_linear_after_3_gyr_ago():
    # The last 3 Gyr: almost constant flux, so N1 grows ~linearly
    assert NEUKUM_2001.n1(2.0) / NEUKUM_2001.n1(1.0) == pytest.approx(2.0, rel=0.01)


def test_chronology_explodes_before_3_5_gyr():
    # Heavy early bombardment: 3.5 → 4.0 Gyr adds more than the 3.5 Gyr before
    assert NEUKUM_2001.n1(4.0) - NEUKUM_2001.n1(3.5) > 10 * NEUKUM_2001.n1(3.5)


@pytest.mark.parametrize("chronology", [NEUKUM_2001, YUE_2022])
@pytest.mark.parametrize("age", [0.1, 1.0, 2.5, 3.6, 4.2])
def test_chronology_inverse(chronology, age):
    assert chronology.age(float(chronology.n1(age))) == pytest.approx(age, abs=1e-6)


def test_chronology_rate_is_the_derivative():
    t, h = 3.7, 1e-6
    numeric = (NEUKUM_2001.n1(t + h) - NEUKUM_2001.n1(t - h)) / (2 * h)
    assert float(NEUKUM_2001.rate(t)) == pytest.approx(float(numeric), rel=1e-5)


def test_chronology_rejects_impossible_densities():
    with pytest.raises(ValueError):
        NEUKUM_2001.age(-1.0)
    with pytest.raises(ValueError):
        NEUKUM_2001.age(1e6)


def test_production_shape_is_one_at_1_km_and_falls_with_size():
    assert float(production_shape(1.0)) == pytest.approx(1.0)
    s = production_shape([0.1, 1.0, 10.0, 100.0])
    assert all(a > b for a, b in zip(s, s[1:]))


def test_production_function_slope_near_1_km():
    # The lunar PF is steep near 1 km (cumulative slope about -2 to -3)
    slope = math.log10(production_shape(2.0) / production_shape(0.5)) / math.log10(4.0)
    assert -3.5 < slope < -1.5


def test_isochrons_scale_with_n1():
    assert float(cumulative_density(1.0, 3.0)) == pytest.approx(float(NEUKUM_2001.n1(3.0)))
    ratio = cumulative_density(5.0, 3.5) / cumulative_density(5.0, 1.0)
    assert float(ratio) == pytest.approx(float(NEUKUM_2001.n1(3.5) / NEUKUM_2001.n1(1.0)))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sampled_diameters_follow_the_production_function():
    rng = np.random.default_rng(0)
    d = sample_diameters(200_000, 0.5, 50.0, rng)
    assert d.min() >= 0.5 and d.max() <= 50.0
    expected = (production_shape(2.0) - production_shape(50.0)) / (production_shape(0.5) - production_shape(50.0))
    assert np.mean(d >= 2.0) == pytest.approx(float(expected), rel=0.03)


def test_formation_ages_follow_the_chronology():
    rng = np.random.default_rng(0)
    ages = sample_formation_ages(100_000, 4.0, NEUKUM_2001, rng)
    # Fraction formed in the last 3 Gyr = N1(3) / N1(4)
    expected = NEUKUM_2001.n1(3.0) / NEUKUM_2001.n1(4.0)
    assert np.mean(ages <= 3.0) == pytest.approx(float(expected), rel=0.05)


# ---------------------------------------------------------------------------
# Age inference
# ---------------------------------------------------------------------------

def test_expected_count_recovers_the_age():
    area, age = 10_000.0, 3.4
    n = area * cumulative_density(1.0, age)
    # A count exactly at the expectation: the posterior mode sits at the age
    fit = infer_age(np.full(int(round(float(n))), 2.0), area, 1.0)
    assert fit["median"] == pytest.approx(age, abs=0.05)


def test_more_craters_mean_an_older_surface():
    few = infer_age(np.full(10, 2.0), 1000.0, 1.0)["median"]
    many = infer_age(np.full(100, 2.0), 1000.0, 1.0)["median"]
    assert many > few


def test_no_craters_means_a_young_surface():
    fit = infer_age([], 10_000.0, 1.0)
    assert fit["count"] == 0
    assert fit["p95"] < 0.5


def test_weights_count_partial_craters():
    full = infer_age(np.full(40, 2.0), 1000.0, 1.0)
    halves = infer_age(np.full(80, 2.0), 1000.0, 1.0, weights=np.full(80, 0.5))
    assert halves["count"] == pytest.approx(40.0)
    assert halves["median"] == pytest.approx(full["median"])


def test_fit_range_is_validated():
    with pytest.raises(ValueError):
        infer_age([1.0], 100.0, 0.001)
    with pytest.raises(ValueError):
        infer_age([1.0], 100.0, 5.0, 2.0)
    with pytest.raises(ValueError):
        infer_age([1.0], 0.0, 1.0)


@pytest.mark.parametrize("age", [0.8, 2.0, 3.3, 3.8])
def test_synthetic_surfaces_are_dated_correctly(age):
    # No erasure: the counts follow the model exactly, so the 90% interval
    # must contain the true age (pinned seeds keep this deterministic)
    surface = bombard(age, 100.0, 0.5, seed=11, erase=False)
    fit = infer_age(surface["diameter"], 100.0 ** 2, 0.5, surface["d_max"])
    assert fit["p5"] <= age <= fit["p95"]


def test_age_intervals_are_calibrated():
    # Over many synthetic surfaces, the 90% interval holds the truth ~90% of the time
    rng = np.random.default_rng(5)
    trials, hits = 200, 0
    for trial in range(trials):
        age = float(rng.uniform(0.5, 4.0))
        surface = bombard(age, 30.0, 0.5, seed=trial, erase=False)
        fit = infer_age(surface["diameter"], 30.0 ** 2, 0.5, surface["d_max"])
        hits += fit["p5"] <= age <= fit["p95"]
    assert 0.84 <= hits / trials <= 0.96


# ---------------------------------------------------------------------------
# Bombardment and saturation
# ---------------------------------------------------------------------------

def test_bombardment_is_reproducible():
    a = bombard(3.5, 30.0, 0.5, seed=4)
    b = bombard(3.5, 30.0, 0.5, seed=4)
    assert np.array_equal(a["diameter"], b["diameter"])
    assert np.array_equal(np.isnan(a["erased"]), np.isnan(b["erased"]))


def test_craters_are_erased_only_by_later_larger_craters():
    s = bombard(4.1, 30.0, 0.4, seed=3)
    erased = ~np.isnan(s["erased"])
    assert erased.any()
    # Erased after they formed (time runs towards 0)
    assert np.all(s["erased"][erased] <= s["formed"][erased])
    # Each erased crater sits inside some later, larger crater
    for i in np.flatnonzero(erased)[:50]:
        later = (s["formed"] <= s["formed"][i]) & (s["diameter"] > s["diameter"][i])
        dist = np.hypot(s["x"][later] - s["x"][i], s["y"][later] - s["y"][i])
        assert np.any(dist < s["diameter"][later] / 2)


def test_bucketed_erasure_matches_brute_force():
    s = bombard(4.0, 20.0, 0.4, seed=8)
    n = len(s["x"])
    expected = np.full(n, np.nan)
    for j in range(n):
        older = np.arange(j)
        hit = (np.isnan(expected[older]) & (s["diameter"][older] < s["diameter"][j])
               & (np.hypot(s["x"][older] - s["x"][j], s["y"][older] - s["y"][j]) < s["diameter"][j] / 2))
        expected[older[hit]] = s["formed"][j]
    assert np.array_equal(np.isnan(expected), np.isnan(s["erased"]))
    assert np.allclose(expected[~np.isnan(expected)], s["erased"][~np.isnan(s["erased"])])


def test_visible_at_tracks_history():
    s = bombard(3.9, 30.0, 0.4, seed=2)
    assert visible_at(s, s["age"] + 0.01).sum() == 0
    assert visible_at(s, 0.0).sum() == np.isnan(s["erased"]).sum()


def test_saturation_makes_small_craters_underestimate_old_ages():
    s = bombard(4.3, 40.0, 0.4, seed=1)
    today = visible_at(s, 0.0)
    small = infer_age(s["diameter"][today], 1600.0, 0.4)["median"]
    large = infer_age(s["diameter"][today], 1600.0, 1.5)["median"]
    assert small < large <= 4.3 + 0.1
    assert 4.3 - small > 0.1


def test_young_surfaces_do_not_saturate():
    s = bombard(2.0, 40.0, 0.4, seed=1)
    assert np.isnan(s["erased"]).mean() > 0.98


def test_bombardment_refuses_huge_surfaces():
    with pytest.raises(ValueError):
        bombard(4.4, 200.0, 0.1)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def test_great_circle_and_cap_area():
    # A quarter of the circumference along the equator
    assert great_circle_km(0, 0, 0, 90) == pytest.approx(math.pi * 1737.4 / 2)
    # Small caps are flat circles
    assert spherical_cap_area_km2(10.0) == pytest.approx(math.pi * 100.0, rel=1e-4)


# Function-scoped on purpose: mutmut 3 runs, for each mutant, only the tests
# that called the mutated function.  A module-scoped fixture is computed during
# the first test alone, so the others would never see the mutant.
@pytest.fixture
def change6():
    return change6_analysis()


def test_change6_count_loaded(change6):
    assert change6["area_km2"] == pytest.approx(52.49, abs=0.01)
    assert len(change6["craters"]["diameter"]) == 799


@pytest.mark.parametrize("chronology", ["neukum2001", "yue2022"])
def test_change6_sample_age_inside_the_90_percent_interval(change6, chronology):
    # Fitting craters ≥ 250 m, past the saturated small sizes
    fit = next(f for f in change6["fits"][chronology] if f["d_min"] == 0.25)
    assert fit["p5"] <= 2.807 <= fit["p95"]


def test_change6_small_craters_underestimate(change6):
    # Below ~200 m the craters are in equilibrium: the age rises with d_min
    medians = [f["median"] for f in change6["fits"]["neukum2001"]]
    assert medians[0] < 1.0 < medians[3]
    assert all(a < b for a, b in zip(medians[:5], medians[1:5]))


def test_yue_chronology_is_closer_for_change6(change6):
    # Yue et al. 2022 recalibrated with Chang'e-5's 2.0 Ga; it reads ~0.1 Gyr older here
    neukum = next(f for f in change6["fits"]["neukum2001"] if f["d_min"] == 0.25)["median"]
    yue = next(f for f in change6["fits"]["yue2022"] if f["d_min"] == 0.25)["median"]
    assert abs(yue - 2.807) < abs(neukum - 2.807)


@pytest.fixture
def circles():
    return landing_site_ages(100.0, 1.0, 20.0)


def test_all_landing_sites_present(circles):
    assert [s["id"] for s in circles["sites"]] == [s.id for s in LANDING_SITES]


def test_naive_circles_date_every_site_too_old(circles):
    for site in circles["sites"]:
        age = site["ages"]["neukum2001"]["median"]
        assert age > min(site["sample_ages"]) - 0.05, site["id"]


def test_naive_circles_are_worst_for_young_basalts(circles):
    by_id = {s["id"]: s for s in circles["sites"]}
    for site_id, sample in [("change5", 2.03), ("change6", 2.807)]:
        assert by_id[site_id]["ages"]["neukum2001"]["median"] - sample > 0.7


def test_landing_site_ages_validate_inputs():
    with pytest.raises(ValueError):
        landing_site_ages(500.0)
    with pytest.raises(ValueError):
        landing_site_ages(100.0, 0.5)


def test_landing_site_craters_within_radius():
    site = landing_site_craters("apollo11", 50.0)
    assert len(site["craters"]["diameter"]) > 50
    assert all(great_circle_km(0.6741, 23.4730, la, lo) <= 50.01
               for la, lo in zip(site["craters"]["latitude"][:100], site["craters"]["longitude"][:100]))


# ---------------------------------------------------------------------------
# Synthetic history payload
# ---------------------------------------------------------------------------

def test_synthetic_history_payload():
    h = synthetic_history(3.8, 40.0, 0.4, seed=1)
    n = len(h["craters"]["x"])
    assert n == len(h["craters"]["formed"]) == len(h["craters"]["erased"])
    assert h["history"]["formed"][-1] == n
    assert h["history"]["visible"][-1] <= n
    assert h["history"]["formed"][0] == 0
    # ~8% of the smallest craters are already erased at 3.8 Gyr, enough to put
    # the truth just outside the very narrow interval (the chronology is steep
    # here), but the median is still within a few tens of Myr
    assert h["fits"][0]["median"] == pytest.approx(3.8, abs=0.05)


# ---------------------------------------------------------------------------
# Mapped geological units
# ---------------------------------------------------------------------------

def test_point_in_polygon_handles_holes():
    from scripts.build_crater_data import points_in_polygon
    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)
    hole = np.array([[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]], float)
    inside = points_in_polygon([1, 5, 11, 5], [1, 5, 5, 9], [outer, hole])
    assert inside.tolist() == [True, False, False, True]


def test_unit_area_integral_matches_the_map():
    # The Chang'e-6 unit lies entirely within 150 km, so the grid integral
    # must reproduce the map's own geodesic area
    from simulation.surface_dating import load_landing_site_craters
    unit = load_landing_site_craters()["units"]["change6"]
    assert unit["area_by_radius_km2"]["150"] == pytest.approx(unit["unit_area_km2"], rel=0.005)


def test_unit_area_grows_with_radius_and_fits_in_the_circle():
    from simulation.surface_dating import load_landing_site_craters
    for site_id, unit in load_landing_site_craters()["units"].items():
        areas = [unit["area_by_radius_km2"][str(r)] for r in range(25, 155, 5)]
        assert all(a <= b for a, b in zip(areas, areas[1:])), site_id
        assert all(a <= spherical_cap_area_km2(r) * 1.01 for a, r in zip(areas, range(25, 155, 5))), site_id


@pytest.fixture
def units_150():
    return landing_site_ages(150.0, 1.0, 20.0, count_area="unit")


def test_unit_counts_are_a_subset_of_the_circle(units_150):
    circles = landing_site_ages(150.0, 1.0, 20.0, count_area="circle")
    for unit_site, circle_site in zip(units_150["sites"], circles["sites"]):
        assert unit_site["ages"]["neukum2001"]["count"] <= circle_site["ages"]["neukum2001"]["count"]
        assert unit_site["area_km2"] <= circle_site["area_km2"]


def test_counting_on_the_unit_helps_the_young_basalts(units_150):
    circles = {s["id"]: s for s in landing_site_ages(150.0, 1.0, 20.0)["sites"]}
    for site in units_150["sites"]:
        if site["id"] in ("change5", "change6"):
            sample = site["sample_ages"][0]
            assert abs(site["ages"]["neukum2001"]["median"] - sample) < abs(circles[site["id"]]["ages"]["neukum2001"]["median"] - sample)


def test_young_basalts_still_read_too_old_from_kilometre_craters(units_150):
    by_id = {s["id"]: s for s in units_150["sites"]}
    for site_id in ("change5", "change6"):
        assert by_id[site_id]["ages"]["neukum2001"]["median"] - by_id[site_id]["sample_ages"][0] > 0.5


def test_resurfacing_signature_on_the_young_basalts(units_150):
    # Ghost craters of the older floor: the age climbs with the smallest crater counted
    by_id = {s["id"]: s for s in units_150["sites"]}
    for site_id in ("change5", "change6"):
        medians = [f["median"] for f in by_id[site_id]["age_by_d_min"]]
        assert all(a < b for a, b in zip(medians, medians[1:])), site_id


def test_unknown_count_area_and_radius_rejected():
    with pytest.raises(ValueError):
        landing_site_ages(100.0, count_area="square")
    with pytest.raises(ValueError):
        landing_site_ages(42.0, count_area="unit")


def test_site_craters_carry_unit_flags_and_outline():
    site = landing_site_craters("change5", 150.0)
    flags = site["craters"]["on_unit"]
    assert set(flags) == {0, 1}
    assert site["unit"]["code"] == "Em"
    assert len(site["unit"]["outline"]) >= 1


def test_synthetic_history_caps_craters_for_the_browser():
    # The page's own settings stay well inside the cap...
    synthetic_history(4.3, 40.0, 0.4)
    # ...but the API's extreme corner would send ~130k craters
    with pytest.raises(ValueError, match="limit 50,000"):
        synthetic_history(4.3, 60.0, 0.3)


# ---------------------------------------------------------------------------
# Gaps found by mutation testing
# ---------------------------------------------------------------------------

from simulation.surface_dating import _weighted_cumulative  # noqa: E402


def test_fresh_surface_has_no_craters():
    assert float(NEUKUM_2001.n1(0.0)) == 0.0
    assert NEUKUM_2001.age(0.0) == pytest.approx(0.0, abs=1e-9)


def test_chronology_errors_explain_themselves():
    with pytest.raises(ValueError, match="≥ 0"):
        NEUKUM_2001.age(-1.0)
    with pytest.raises(ValueError, match="5 Gyr"):
        NEUKUM_2001.age(1e6)
    with pytest.raises(ValueError, match="fit range"):
        infer_age([1.0], 100.0, 0.001)
    with pytest.raises(ValueError, match="age and size"):
        bombard(0.0, 10.0, 0.5)


def test_finite_fit_window_recovers_the_age():
    # Craters only between 1 and 2 km: the expected count uses S(1) − S(2)
    area, age = 50_000.0, 3.6
    n = area * NEUKUM_2001.n1(age) * (production_shape(1.0) - production_shape(2.0))
    fit = infer_age(np.full(int(round(float(n))), 1.5), area, 1.0, 2.0)
    assert fit["median"] == pytest.approx(age, abs=0.02)
    # ...and reports the N(1) density that count implies
    assert fit["n1"] == pytest.approx(float(NEUKUM_2001.n1(age)), rel=0.01)


def test_small_areas_and_the_smallest_pf_diameter_are_allowed():
    assert infer_age([0.02], 0.5, 0.01)["count"] == 1
    infer_age([0.02], 100.0, 0.01, 0.05)


def test_age_zero_only_when_nothing_was_counted():
    none = infer_age([], 1000.0, 1.0)
    one = infer_age([2.0], 1000.0, 1.0)
    # No craters: the most likely age is zero; one crater rules zero out
    assert none["posterior"][0] == none["posterior"].max()
    assert one["posterior"][0] == 0.0


def test_bombardment_rejects_empty_surfaces():
    with pytest.raises(ValueError):
        bombard(1.0, 0.0, 0.5)


def test_bombardment_defaults_and_coverage():
    s = bombard(3.9, 30.0, 0.4, seed=6, erase=False)
    assert s["d_max"] == 15.0                               # half the surface
    assert s["x"].min() < 0.5 and s["y"].min() < 0.5          # craters reach every edge
    assert s["x"].max() > 29.5 and s["y"].max() > 29.5
    assert np.isnan(s["erased"]).all()                       # nothing erased without erasure


def test_bombardment_crater_count_matches_expectation():
    # Poisson mean = area · N1(T) · (S(d_min) − S(d_max)); average over seeds
    counts = [len(bombard(3.0, 40.0, 0.5, seed=k, erase=False)["x"]) for k in range(30)]
    expected = 1600.0 * NEUKUM_2001.n1(3.0) * (production_shape(0.5) - production_shape(20.0))
    assert np.mean(counts) == pytest.approx(float(expected), rel=0.05)


def test_weighted_cumulative_density_and_errors():
    sfd = _weighted_cumulative([1.0, 3.0, 2.0], [1.0, 0.5, 1.0], 10.0)
    assert sfd["diameter"] == [3.0, 2.0, 1.0]
    assert sfd["density"] == pytest.approx([0.05, 0.15, 0.25])
    assert sfd["error"] == pytest.approx([math.sqrt(0.5) / 10, math.sqrt(1.5) / 10, math.sqrt(2.5) / 10])


def test_great_circle_along_a_meridian_and_off_the_equator():
    assert great_circle_km(0, 0, 90, 0) == pytest.approx(math.pi * 1737.4 / 2)
    # 45°N, 90° apart in longitude: a 60° central angle
    assert great_circle_km(45, 0, 45, 90) == pytest.approx(math.pi * 1737.4 / 3)


def test_landing_site_craters_report_diameters():
    from simulation.surface_dating import _site_rows
    rows = _site_rows("change5")
    site = landing_site_craters("change5", 150.0)
    assert site["craters"]["diameter"] == rows[rows[:, 3] <= 150.0][:, 2].tolist()
    assert site["craters"]["on_unit"] == rows[rows[:, 3] <= 150.0][:, 4].astype(int).tolist()


def test_count_area_errors_explain_themselves():
    with pytest.raises(ValueError, match="count_area"):
        landing_site_ages(100.0, count_area="square")
    with pytest.raises(ValueError, match="radius"):
        landing_site_ages(42.0)
    with pytest.raises(ValueError, match="d_min"):
        landing_site_ages(100.0, 0.5)


def test_isochron_scales_with_the_production_function():
    assert float(cumulative_density(2.0, 3.0)) == pytest.approx(float(NEUKUM_2001.n1(3.0) * production_shape(2.0)))


def test_crater_at_the_fit_minimum_is_counted():
    assert infer_age([1.0], 1000.0, 1.0)["count"] == 1


def test_no_craters_posterior_is_flat_near_zero():
    # λ(1 Myr) is tiny, so with nothing counted 0 and 1 Myr are equally likely
    post = infer_age([], 100.0, 1.0)["posterior"]
    assert post[0] == pytest.approx(post[1], rel=1e-3)


def test_bombardment_count_in_a_narrow_size_window():
    # 0.5-0.7 km: S(d_max) is most of S(d_min), so a sign slip in the window shows
    counts = [len(bombard(3.0, 40.0, 0.5, d_max=0.7, seed=k, erase=False)["x"]) for k in range(30)]
    expected = 1600.0 * NEUKUM_2001.n1(3.0) * (production_shape(0.5) - production_shape(0.7))
    assert np.mean(counts) == pytest.approx(float(expected), rel=0.08)


def test_synthetic_history_marks_erased_craters():
    h = synthetic_history(4.2, 30.0, 0.4)
    erased = h["craters"]["erased"]
    assert any(e is not None for e in erased) and any(e is None for e in erased)
    # Fits use today's surface: the smallest-crater fit counts every survivor
    assert h["fits"][0]["count"] == h["history"]["visible"][-1]


def test_change6_counts_partial_craters_and_has_coordinates(change6):
    fit = change6["fits"]["neukum2001"][0]          # d_min = 50 m
    d = np.array(change6["craters"]["diameter"]); f = np.array(change6["craters"]["fraction"])
    assert fit["count"] == pytest.approx(f[d >= 0.05].sum())
    assert fit["count"] != pytest.approx(float((d >= 0.05).sum()))
    lon, lat = change6["craters"]["longitude"], change6["craters"]["latitude"]
    assert -155 < min(lon) and max(lon) < -153 and -42.5 < min(lat) and max(lat) < -41


def test_landing_site_ages_defaults_and_chronologies():
    result = landing_site_ages()
    assert (result["radius_km"], result["d_min"], result["d_max"], result["count_area"]) == (100.0, 1.0, 20.0, "circle")
    # The two chronologies give different ages from the same count
    site = result["sites"][0]
    assert site["ages"]["neukum2001"]["median"] != site["ages"]["yue2022"]["median"]
