"""
Tests for the Bayesian crater inversion (simulation/inference.py).

Three layers:
1. The vectorised forward model must agree with crater_model.run_simulation,
   or the inversion would be inverting a different model.
2. Calibration: over synthetic craters made by impactors drawn from the
   prior, the 90% credible interval must contain the true impactor about
   90% of the time (simulation-based calibration, Talts et al. 2018).  A
   wrong likelihood, prior or weighting breaks this, whatever the output
   looks like.
3. The physical conclusions the page reports, pinned to bands.
"""

import math

import numpy as np
import pytest

from simulation.crater_model import ImpactParameters, run_simulation
from simulation.inference import (
    Observation,
    Priors,
    crater_diameter,
    default_material,
    infer,
    sample_prior,
)


# ---------------------------------------------------------------------------
# 1. Vectorised forward model == run_simulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("material", ["regolith", "rock"])
def test_vectorised_model_matches_run_simulation(material):
    rng = np.random.default_rng(1)
    n = 200
    d = np.exp(rng.uniform(math.log(0.05), math.log(50_000.0), n))
    v = rng.uniform(2_500.0, 60_000.0, n)
    angle = rng.uniform(15.0, 90.0, n)
    rho_i = rng.uniform(1000.0, 8000.0, n)
    rho_t = rng.uniform(1300.0, 3000.0, n)

    vectorised = crater_diameter(d, v, angle, rho_i, rho_t, material)
    for i in range(n):
        expected = run_simulation(ImpactParameters(
            impactor_diameter=d[i], impact_velocity=v[i], impact_angle=angle[i],
            impactor_density=rho_i[i], target_density=rho_t[i], target_material=material,
        )).crater_diameter
        # run_simulation rounds to 0.1 mm
        assert vectorised[i] == pytest.approx(expected, rel=1e-6, abs=1e-4)


def test_vectorised_model_covers_complex_craters():
    # A 10 km impactor makes a complex crater: the collapse branch must be taken
    d = crater_diameter(np.array([10_000.0]), np.array([20_000.0]), np.array([45.0]),
                        np.array([3000.0]), np.array([2550.0]), "rock")
    assert d[0] > 15_000.0


def test_vectorised_model_rejects_unknown_material():
    with pytest.raises(ValueError):
        crater_diameter(np.ones(1), np.ones(1), np.ones(1), np.ones(1), np.ones(1), "cheese")


def test_default_material_switches_at_the_regolith_limit():
    assert default_material(18.8) == "regolith"
    assert default_material(125.0) == "regolith"
    assert default_material(126.0) == "rock"


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prior():
    return sample_prior(Observation(1000.0), Priors(), 200_000, np.random.default_rng(0))


def test_angle_prior_follows_sin_2_theta(prior):
    # p(θ) ∝ sin 2θ has CDF sin²θ; truncated at 15°, it is
    # (sin²θ - sin²15°) / (1 - sin²15°).  Untruncated, the median is 45°.
    floor = math.sin(math.radians(15.0)) ** 2
    cdf = lambda deg: (math.sin(math.radians(deg)) ** 2 - floor) / (1 - floor)
    angle = prior["angle"]
    assert angle.min() >= 15.0 and angle.max() <= 90.0
    assert np.median(angle) == pytest.approx(math.degrees(math.asin(math.sqrt((1 + floor) / 2))), abs=0.3)
    assert np.mean((angle > 30) & (angle < 60)) == pytest.approx(cdf(60) - cdf(30), abs=0.005)


def test_velocity_prior_mean_is_about_20_km_s(prior):
    v = prior["velocity"]
    assert v.min() >= 2_380.0 and v.max() <= 50_000.0
    assert v.mean() == pytest.approx(20_000.0, rel=0.03)


def test_diameter_prior_is_log_uniform(prior):
    log_d = np.log(prior["impactor_diameter"])
    lo, hi = math.log(1000.0 / 2000), math.log(1000.0 / 2)
    assert log_d.min() >= lo and log_d.max() <= hi
    # Uniform in log: each half of the log range holds half the draws
    assert np.mean(log_d < (lo + hi) / 2) == pytest.approx(0.5, abs=0.01)


def test_known_velocity_replaces_the_prior():
    s = sample_prior(Observation(18.8, velocity=25_600.0), Priors(), 10_000, np.random.default_rng(0))
    assert s["velocity"].mean() == pytest.approx(25_600.0, rel=0.01)


# ---------------------------------------------------------------------------
# 2. Calibration (simulation-based calibration)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("material,crater_scale", [("regolith", 20.0), ("rock", 5_000.0)])
def test_credible_intervals_are_calibrated(material, crater_scale):
    rng = np.random.default_rng(42)
    priors = Priors()
    trials, counted, hits_d, hits_e = 120, 0, 0, 0
    # A large measurement error, so the likelihood width matters: at 5% the
    # size/velocity degeneracy dominates the posterior, and a likelihood 5x
    # too narrow or too wide still passes.  At 30% both fail this test.
    error = 0.3

    for trial in range(trials):
        # A true impactor from the prior, and the crater it makes, measured with noise
        reference = Observation(crater_scale, material=material)
        truth = {k: v[0] for k, v in sample_prior(reference, priors, 1, rng).items()}
        true_crater = crater_diameter(*(np.array([truth[k]]) for k in (
            "impactor_diameter", "velocity", "angle", "impactor_density", "target_density")), material)[0]
        measured = true_crater * math.exp(rng.normal(0.0, math.log1p(error)))
        true_energy = 0.5 * math.pi / 6 * truth["impactor_diameter"] ** 3 * truth["impactor_density"] * truth["velocity"] ** 2

        # The diameter prior scales with the observed crater, so the truth must
        # lie inside it for the test to be fair: it was drawn relative to
        # crater_scale, so skip the rare draws outside the new range
        lo, hi = (r * measured for r in priors.diameter_ratio_range)
        if not lo <= truth["impactor_diameter"] <= hi:
            continue

        counted += 1
        result = infer(Observation(measured, error, material), priors, samples=40_000, seed=trial)
        d = result["parameters"]["impactor_diameter"]["posterior"]
        e = result["parameters"]["kinetic_energy"]["posterior"]
        hits_d += d["p5"] <= truth["impactor_diameter"] <= d["p95"]
        hits_e += e["p5"] <= true_energy <= e["p95"]

    # 90% intervals: with ~120 trials, 0.8-0.98 is a generous binomial band
    assert counted > 100
    assert 0.8 <= hits_d / counted <= 0.98
    assert 0.8 <= hits_e / counted <= 0.98


# ---------------------------------------------------------------------------
# 3. What the inversion shows
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_crater():
    return infer(Observation(18.8))


def test_crater_constrains_energy_much_more_than_velocity(small_crater):
    p = small_crater["parameters"]
    # Energy and mass: the posterior is ~10% as wide as the prior
    assert p["kinetic_energy"]["narrowing"] < 0.2
    assert p["impactor_diameter"]["narrowing"] < 0.2
    # Velocity and angle: the crater alone barely narrows them (degeneracy)
    assert p["velocity"]["narrowing"] > 0.8
    assert p["angle"]["narrowing"] > 0.8


def test_size_and_velocity_trade_off(small_crater):
    # Along the degeneracy, faster impactors must be smaller
    s = small_crater["posterior_samples"]
    r = np.corrcoef(np.log(s["impactor_diameter"]), np.log(s["velocity"]))[0, 1]
    assert r < -0.5


def test_bigger_crater_needs_a_bigger_impactor():
    small = infer(Observation(2_000.0))["parameters"]["impactor_diameter"]["posterior"]["p50"]
    large = infer(Observation(20_000.0))["parameters"]["impactor_diameter"]["posterior"]["p50"]
    assert large > 5 * small


def test_tighter_measurement_gives_a_tighter_posterior():
    loose = infer(Observation(2_000.0, diameter_error=0.3))["parameters"]["kinetic_energy"]["narrowing"]
    tight = infer(Observation(2_000.0, diameter_error=0.02))["parameters"]["kinetic_energy"]["narrowing"]
    assert tight < loose


def test_effective_sample_size_is_enough(small_crater):
    assert small_crater["effective_sample_size"] > 2000


def test_inference_is_reproducible():
    a = infer(Observation(500.0), samples=20_000, seed=3)
    b = infer(Observation(500.0), samples=20_000, seed=3)
    assert a["parameters"] == b["parameters"]


def test_invalid_observations_rejected():
    with pytest.raises(ValueError):
        infer(Observation(0.0))
    with pytest.raises(ValueError):
        infer(Observation(10.0, material="cheese"))


# ---------------------------------------------------------------------------
# The 2013-03-17 event: crater + flash → luminous efficiency
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flash_2013():
    return infer(Observation(18.8, luminous_energy=7.1e6, velocity=25_600.0))


def test_flash_event_infers_luminous_efficiency(flash_2013):
    eta = flash_2013["parameters"]["luminous_efficiency"]
    # Agrees with the case study's point inversion (~1.4e-4)...
    assert 1e-4 < eta["posterior"]["p50"] < 2e-4
    # ...and the 90% interval excludes the published 1.31e-3
    assert eta["posterior"]["p95"] < 1.31e-3 / 3


def test_flash_tightens_the_mass_estimate(flash_2013, small_crater):
    with_flash = flash_2013["parameters"]["mass"]["posterior"]
    crater_only = small_crater["parameters"]["mass"]["posterior"]
    assert (with_flash["p95"] / with_flash["p5"]) < (crater_only["p95"] / crater_only["p5"])


def test_flash_event_has_usable_sample_size(flash_2013):
    assert flash_2013["effective_sample_size"] > 300
