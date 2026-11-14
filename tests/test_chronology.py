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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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
