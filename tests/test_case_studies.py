"""
Tests for the 2013-03-17 case study (simulation/case_studies.py).

The flash→crater chain is checked against the paper's own intermediate
numbers, so a wrong η formula or unit slip shows up before any conclusion
is drawn from it.  The conclusions themselves (where LRO's crater falls,
which η fits) are pinned to bands like tests/test_real_impacts.py, so a model
change that moves them shows up here and the case-study page gets reviewed.
"""

import math
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from simulation.case_studies import (
    FLASH_2013,
    analyze,
    implied_luminous_efficiency,
    monte_carlo,
    nominal_chain,
    observing_geometry,
    predict_crater,
    suggs_luminous_efficiency,
)


# ---------------------------------------------------------------------------
# The chain reproduces the paper's numbers
# ---------------------------------------------------------------------------

def test_luminous_efficiency_reproduces_the_papers_kinetic_energy():
    # Moser et al. 2014: 7.1e6 J of light → 5.4e9 J of kinetic energy at 25.6 km/s
    eta = suggs_luminous_efficiency(25_600.0)
    assert 7.1e6 / eta == pytest.approx(5.4e9, rel=0.01)


def test_luminous_efficiency_rises_with_velocity():
    assert suggs_luminous_efficiency(15_000.0) < suggs_luminous_efficiency(25_000.0) < 1.5e-3


def test_nominal_chain_recovers_the_papers_16_kg():
    chain = nominal_chain(FLASH_2013, "regolith")
    assert chain["impactor_mass"] == pytest.approx(16.0, rel=0.05)


def test_nominal_chain_matches_a_direct_simulation():
    from simulation.crater_model import ImpactParameters, run_simulation

    chain = nominal_chain(FLASH_2013, "regolith")
    direct = run_simulation(ImpactParameters(
        impactor_diameter=chain["impactor_diameter"],
        impact_velocity=FLASH_2013.velocity,
        impact_angle=FLASH_2013.impact_angle,
        impactor_density=FLASH_2013.nominal_impactor_density,
        target_density=FLASH_2013.nominal_target_density,
        target_material="regolith",
    ))
    assert chain["crater_diameter"] == pytest.approx(direct.crater_diameter)


def test_more_light_means_a_bigger_crater():
    args = (FLASH_2013.nominal_eta, 25_600.0, 56.0, 2400.0, 1500.0, "regolith")
    assert predict_crater(1e6, *args) < predict_crater(1e7, *args)


def test_higher_efficiency_means_a_smaller_crater():
    def crater(eta):
        return predict_crater(7.1e6, eta, 25_600.0, 56.0, 2400.0, 1500.0, "regolith")
    assert crater(1e-2) < crater(1e-3) < crater(1e-4)


# ---------------------------------------------------------------------------
# Observing geometry — the ephemeris must agree that the flash was visible
# ---------------------------------------------------------------------------

def test_flash_site_was_in_lunar_night():
    geometry = observing_geometry(FLASH_2013)
    assert geometry["site_in_darkness"]
    assert geometry["sun_elevation"] < -30.0


def test_flash_site_faced_earth():
    geometry = observing_geometry(FLASH_2013)
    assert geometry["earth_above_horizon"]
    # Near the centre of the near side: Earth high in the sky
    assert geometry["earth_elevation"] > 45.0


def test_moon_was_a_waxing_crescent():
    # 17 March 2013 was ~5 days after new Moon; flashes are monitored then,
    # when the night side is large and the Moon is up after sunset
    geometry = observing_geometry(FLASH_2013)
    assert geometry["phase_name"] == "Waxing Crescent"
    assert 0.15 < geometry["illuminated_fraction"] < 0.4


def test_geometry_detects_a_daylit_site():
    # Same instant, the opposite longitude: sunlit and facing away from Earth
    far_side = replace(FLASH_2013, longitude=FLASH_2013.longitude + 180.0)
    geometry = observing_geometry(far_side)
    assert not geometry["site_in_darkness"]
    assert not geometry["earth_above_horizon"]


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def test_monte_carlo_is_reproducible_with_a_seed():
    a = monte_carlo(FLASH_2013, "regolith", samples=200, seed=7)
    b = monte_carlo(FLASH_2013, "regolith", samples=200, seed=7)
    assert (a == b).all()


def test_monte_carlo_spreads_around_the_nominal_prediction():
    samples = monte_carlo(FLASH_2013, "regolith", samples=1000)
    nominal = nominal_chain(FLASH_2013, "regolith")["crater_diameter"]
    assert samples.min() < nominal < samples.max()
    assert float(sorted(samples)[500]) == pytest.approx(nominal, rel=0.1)


def test_no_scatter_collapses_the_distribution():
    exact = replace(
        FLASH_2013, velocity_sigma=0.0, impact_angle_sigma=0.0, eta_scatter=1.0,
        impactor_density_range=(2400.0, 2400.0), target_density_range=(1550.0, 1550.0),
    )
    samples = monte_carlo(exact, "regolith", samples=50)
    assert samples.min() == pytest.approx(samples.max())


def test_monte_carlo_rejects_unknown_material():
    with pytest.raises(ValueError):
        monte_carlo(FLASH_2013, "cheese", samples=10)


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("material", ["regolith", "rock"])
def test_implied_eta_reproduces_the_observed_crater(material):
    eta = implied_luminous_efficiency(FLASH_2013, material)
    crater = predict_crater(
        FLASH_2013.luminous_energy, eta, FLASH_2013.velocity, FLASH_2013.impact_angle,
        FLASH_2013.nominal_impactor_density, FLASH_2013.nominal_target_density, material,
    )
    assert crater == pytest.approx(FLASH_2013.observed_diameter, rel=1e-3)


def test_implied_eta_rejects_an_unreachable_crater():
    with pytest.raises(ValueError):
        implied_luminous_efficiency(replace(FLASH_2013, observed_diameter=1e6), "regolith")


# ---------------------------------------------------------------------------
# The findings the case-study page reports (pinned to bands)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result():
    return analyze(FLASH_2013)


def test_regolith_scaling_underpredicts_the_crater(result):
    m = result["models"]["regolith"]
    # LRO's 18.8 m sits in the upper tail: the median is about half of it
    assert 0.4 < m["percentiles"]["p50"] / FLASH_2013.observed_diameter < 0.7
    assert m["observed_percentile"] > 0.9


def test_rock_scaling_overpredicts_the_crater(result):
    m = result["models"]["rock"]
    assert 1.5 < m["percentiles"]["p50"] / FLASH_2013.observed_diameter < 3.0
    assert m["observed_percentile"] < 0.1


def test_observed_crater_lies_between_the_two_models(result):
    regolith = result["models"]["regolith"]["percentiles"]["p50"]
    rock = result["models"]["rock"]["percentiles"]["p50"]
    assert regolith < FLASH_2013.observed_diameter < rock


def test_regolith_fit_needs_an_eta_about_ten_times_lower(result):
    assert 0.03 < result["models"]["regolith"]["implied_eta_ratio"] < 0.3


def test_rock_fit_needs_an_implausibly_high_eta(result):
    # Several percent of the kinetic energy as visible light: far above any
    # efficiency proposed for lunar flashes
    assert result["models"]["rock"]["implied_eta"] > 0.01


def test_eta_sweep_brackets_both_implied_values(result):
    for m in result["models"].values():
        etas = m["eta_sweep"]["eta"]
        assert etas[0] < m["implied_eta"] < etas[-1]
        diameters = m["eta_sweep"]["crater_diameter"]
        assert all(a >= b for a, b in zip(diameters, diameters[1:]))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

client = TestClient(app)


def test_api_lists_the_case_study():
    res = client.get("/api/case-studies")
    assert res.status_code == 200
    assert [c["id"] for c in res.json()] == ["2013-03-17"]


def test_api_returns_the_analysis_without_login():
    res = client.get("/api/case-studies/2013-03-17")
    assert res.status_code == 200
    body = res.json()
    assert body["event"]["observed_diameter"] == 18.8
    assert set(body["models"]) == {"regolith", "rock"}
    assert len(body["models"]["regolith"]["samples"]) == body["assumptions"]["samples"]
    assert math.isfinite(body["models"]["regolith"]["implied_eta"])


def test_api_unknown_case_study_is_404():
    assert client.get("/api/case-studies/1999-01-01").status_code == 404


def test_case_study_page_is_served():
    res = client.get("/case-study")
    assert res.status_code == 200
    assert "17 March 2013" in res.text
