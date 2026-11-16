"""
The shape of every science payload the frontend reads.

The pages (case-study.html, inference.html, surface-age.html) index these
dicts by key, so a renamed or dropped key breaks a chart without any Python
test failing.  Mutation testing showed exactly that: renaming a key survived.
These tests pin the key structure; a deliberate change updates both the page
and the expected tree here.
"""

import pytest

from simulation.case_studies import FLASH_2013, analyze
from simulation.inference import Observation, infer
from simulation.surface_dating import (
    change6_analysis,
    landing_site_ages,
    landing_site_craters,
    synthetic_history,
)


def keys(obj):
    """Nested key structure: dicts become {key: structure}, a list of dicts its first element."""
    if isinstance(obj, dict):
        return {k: keys(v) for k, v in obj.items()}
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return [keys(obj[0])]
    return None


PERCENTILES = {"p5": None, "p50": None, "p95": None}
AGE_FIT = {k: None for k in ("count", "median", "p5", "p95", "n1", "d_min", "d_max")}
ISOCHRONS = {"diameter": None, "curves": {k: None for k in ("1", "2", "3", "3.5", "3.8", "4")}}
SFD = {"diameter": None, "density": None, "error": None}


def test_case_study_payload():
    model = {
        "nominal": {k: None for k in ("luminous_energy", "luminous_efficiency", "kinetic_energy",
                                      "impactor_mass", "impactor_diameter", "crater_diameter")},
        "samples": None,
        "percentiles": PERCENTILES,
        "observed_percentile": None,
        "observed_within_90": None,
        "implied_eta": None,
        "implied_eta_ratio": None,
        "eta_sweep": {"eta": None, "crater_diameter": None},
    }
    assert keys(analyze(FLASH_2013, samples=50)) == {
        "event": {k: None for k in ("id", "title", "time", "latitude", "longitude", "luminous_energy",
                                    "velocity", "impact_angle", "observed_diameter", "references")},
        "assumptions": {k: None for k in ("velocity_sigma", "impact_angle_sigma", "impactor_density_range",
                                          "target_density_range", "eta_scatter", "samples", "seed")},
        "geometry": {k: None for k in ("sun_elevation", "earth_elevation", "phase_name",
                                       "illuminated_fraction", "site_in_darkness", "earth_above_horizon")},
        "models": {"rock": model, "regolith": model},
    }


def test_inference_payload():
    names = ("impactor_diameter", "angle", "impactor_density", "target_density", "velocity",
             "luminous_efficiency", "mass", "kinetic_energy")
    result = infer(Observation(18.8, luminous_energy=7.1e6, velocity=25_600.0), samples=20_000, posterior_samples=10)
    assert keys(result) == {
        "observation": {k: None for k in ("crater_diameter", "diameter_error", "material", "luminous_energy", "velocity")},
        "effective_sample_size": None,
        "prior_samples": None,
        "parameters": {n: {"prior": PERCENTILES, "posterior": PERCENTILES, "narrowing": None} for n in names},
        "posterior_samples": {n: None for n in names},
        "prior_draws": {n: None for n in names},
    }


def test_synthetic_surface_payload():
    result = synthetic_history(2.0, 20.0, 0.5)
    assert keys(result) == {
        "age": None, "size_km": None, "d_min": None, "d_max": None, "seed": None,
        "craters": {k: None for k in ("x", "y", "diameter", "formed", "erased")},
        "history": {"time_ago": None, "formed": None, "visible": None},
        "sfd": SFD,
        "fits": [AGE_FIT],
        "isochrons": ISOCHRONS,
    }
    # The fits use the surface's own size range
    assert all(f["d_max"] == result["d_max"] for f in result["fits"])
    assert result["fits"][0]["d_min"] == result["d_min"]


def test_change6_payload():
    result = change6_analysis()
    assert keys(result) == {
        "site": {k: None for k in ("name", "latitude", "longitude", "sample_age", "reference")},
        "source": None,
        "area_km2": None,
        "boundary": None,
        "craters": {k: None for k in ("diameter", "fraction", "longitude", "latitude")},
        "sfd": SFD,
        "fits": {"neukum2001": [AGE_FIT], "yue2022": [AGE_FIT]},
        "chronologies": {"neukum2001": None, "yue2022": None},
        "isochrons": ISOCHRONS,
    }
    # The page reads the d_min = 250 m fit, and the isochrons span the count's sizes
    assert [f["d_min"] for f in result["fits"]["neukum2001"]] == [0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]
    assert result["isochrons"]["diameter"][0] == pytest.approx(0.03)


def test_landing_site_ages_payload():
    assert keys(landing_site_ages(100.0, count_area="unit")) == {
        "source": None, "unit_source": None, "count_area": None, "radius_km": None, "d_min": None, "d_max": None,
        "sites": [{
            "id": None, "name": None, "latitude": None, "longitude": None, "sample_ages": None, "reference": None,
            "unit": {k: None for k in ("code", "period", "description", "unit_area_km2")},
            "area_km2": None,
            "ages": {"neukum2001": AGE_FIT, "yue2022": AGE_FIT},
            "age_by_d_min": [AGE_FIT],
        }],
    }


def test_landing_site_craters_payload():
    result = landing_site_craters("apollo11", 50.0)
    assert keys(result) == {
        "id": None, "name": None, "latitude": None, "longitude": None, "radius_km": None,
        "unit": {
            "code": None, "period": None, "description": None, "unit_area_km2": None,
            "area_by_radius_km2": {str(r): None for r in range(25, 155, 5)},
            "outline": None,
        },
        "craters": {k: None for k in ("latitude", "longitude", "diameter", "on_unit")},
    }
