"""
Bayesian inversion — from an observed crater back to the impactor.

The forward model (crater_model.run_simulation) maps an impactor to a crater.
This module runs it backwards: given a measured crater diameter, what
impactors could have made it, and how likely is each?

    posterior(impactor | crater)  ∝  likelihood(crater | impactor) × prior(impactor)

A single diameter cannot pin down four unknowns (size, velocity, angle,
density): a small fast impactor and a large slow one dig the same crater.
The posterior shows that degeneracy instead of hiding it, and shows which
combination the crater *does* constrain: the kinetic energy, far more tightly
than size or velocity alone.

Method: importance sampling.  Draw many impactors from the prior, run the
(vectorised) forward model on all of them, and weight each by the likelihood
of the observed crater.  With 4-6 parameters this is exact up to sampling
noise and has no convergence to diagnose, unlike MCMC; the effective sample
size (ESS) reports how many prior draws actually informed the answer.
tests/test_inference.py checks the calibration: over synthetic craters with
a known impactor, the 90% credible interval must contain the truth ~90% of
the time.

Priors (assumptions, chosen to be broad and documented):
    impactor diameter   log-uniform over [D/2000, D/2]: scale-free ignorance
    velocity            normal, mean 20 km/s, σ 6 km/s, truncated to
                        [2.38, 50] km/s.  Mean impact speeds on the Moon:
                        19.7 km/s (Le Feuvre & Wieczorek 2011), 20 km/s
                        (Gallant et al. 2009); the width is an assumption.
    angle               p(θ) ∝ sin 2θ (Gilbert 1893, Shoemaker 1962): the
                        distribution of isotropic impactors, most likely
                        45°.  Truncated to ≥ 15°, the model's range.
    impactor density    uniform 1500-3500 kg/m³ (carbonaceous to stony)
    target density      regolith 1300-1800 kg/m³, rock 2400-2700 kg/m³
    luminous efficiency (flash events only) log-normal around the Suggs et
                        al. (2014) value, within a factor of 3

Likelihoods are log-normal: a 5% diameter error means ln D_obs has σ ≈ 0.05.

References:
    - Le Feuvre, M. & Wieczorek, M.A. (2011). Icarus 214, 1-20.
    - Shoemaker, E.M. (1962). "Interpretation of lunar craters." In Physics
      and Astronomy of the Moon, 283-359.
    - Talts, S. et al. (2018). "Validating Bayesian inference algorithms
      with simulation-based calibration." arXiv:1804.06788.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from simulation.crater_model import (
    LUNAR_ESCAPE_VELOCITY,
    LUNAR_GRAVITY,
    REGOLITH_K1, REGOLITH_K2, REGOLITH_KR, REGOLITH_MU, REGOLITH_NU, REGOLITH_STRENGTH,
    REGOLITH_MAX_TRANSIENT_DIAMETER,
    TARGET_MATERIALS,
    transition_diameter,
)
from simulation.case_studies import FLASH_2013, suggs_luminous_efficiency


# ---------------------------------------------------------------------------
# Vectorised forward model
# ---------------------------------------------------------------------------

def crater_diameter(
    impactor_diameter: np.ndarray,
    velocity: np.ndarray,
    angle: np.ndarray,
    impactor_density: np.ndarray,
    target_density: np.ndarray,
    material: str,
    gravity: float = LUNAR_GRAVITY,
) -> np.ndarray:
    """
    Final rim-to-rim diameter (m) for arrays of impactors.

    The same equations as crater_model.run_simulation, on numpy arrays, so
    hundreds of thousands of impactors take milliseconds.
    tests/test_inference.py checks that the two agree.
    """
    if material == "regolith":
        mu, nu = REGOLITH_MU, REGOLITH_NU
        density_ratio = target_density / impactor_density
        vertical_velocity = velocity * np.sin(np.radians(angle))
        pi2 = gravity * (impactor_diameter / 2.0) / vertical_velocity ** 2
        pi3 = REGOLITH_STRENGTH / (target_density * vertical_velocity ** 2)
        gravity_term = pi2 * density_ratio ** ((6 * nu - 2 - mu) / (3 * mu))
        strength_term = (REGOLITH_K2 * pi3 * density_ratio ** ((6 * nu - 2) / (3 * mu))) ** ((2 + mu) / 2)
        pi_v = REGOLITH_K1 * (gravity_term + strength_term) ** (-3 * mu / (2 + mu))
        mass = math.pi / 6.0 * impactor_diameter ** 3 * impactor_density
        transient = 2.0 * REGOLITH_KR * (mass * pi_v / target_density) ** (1.0 / 3.0) / 1.25
    elif material == "rock":
        transient = (
            1.161
            * (impactor_density / target_density) ** (1.0 / 3.0)
            * impactor_diameter ** 0.78
            * velocity ** 0.44
            * gravity ** -0.22
            * np.sin(np.radians(angle)) ** (1.0 / 3.0)
        )
    else:
        raise ValueError(f"material must be one of {TARGET_MATERIALS}")

    d_c = transition_diameter(gravity)
    return np.maximum(1.25 * transient, 1.17 * transient ** 1.13 / d_c ** 0.13)


def default_material(observed_diameter: float) -> str:
    """Regolith scaling for craters that stay inside the regolith, rock beyond."""
    return "regolith" if observed_diameter / 1.25 <= REGOLITH_MAX_TRANSIENT_DIAMETER else "rock"


# ---------------------------------------------------------------------------
# Observation and priors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    crater_diameter: float                  # m, rim to rim
    diameter_error: float = 0.05            # relative (1σ)
    material: str | None = None             # None: chosen from the diameter
    # Optional impact flash: the light constrains η × kinetic energy
    luminous_energy: float | None = None    # J
    luminous_energy_error: float = 0.2      # relative (1σ)
    # Optional known velocity (e.g. a meteor shower); otherwise the prior
    velocity: float | None = None           # m/s
    velocity_error: float = 1000.0          # m/s (1σ)

    @property
    def target_material(self) -> str:
        return self.material or default_material(self.crater_diameter)


@dataclass(frozen=True)
class Priors:
    diameter_ratio_range: tuple[float, float] = (1 / 2000, 1 / 2)   # impactor / crater
    velocity_mean: float = 20_000.0
    velocity_sigma: float = 6_000.0
    velocity_range: tuple[float, float] = (LUNAR_ESCAPE_VELOCITY, 50_000.0)
    min_angle: float = 15.0
    impactor_density_range: tuple[float, float] = (1500.0, 3500.0)
    target_density_range: dict = field(default_factory=lambda: {
        "regolith": (1300.0, 1800.0),
        "rock": (2400.0, 2700.0),
    })
    eta_scatter: float = 3.0


PARAMETERS = ("impactor_diameter", "velocity", "angle", "impactor_density", "target_density")


def _truncated_normal(rng, mean, sigma, low, high, n):
    """Normal draws restricted to [low, high], by redrawing the ones outside."""
    values = rng.normal(mean, sigma, n)
    outside = (values < low) | (values > high)
    while outside.any():
        values[outside] = rng.normal(mean, sigma, outside.sum())
        outside = (values < low) | (values > high)
    return values


def sample_prior(obs: Observation, priors: Priors, n: int, rng: np.random.Generator) -> dict:
    """n impactors drawn from the prior, as arrays keyed by parameter name."""
    lo, hi = (r * obs.crater_diameter for r in priors.diameter_ratio_range)
    sample = {
        "impactor_diameter": np.exp(rng.uniform(math.log(lo), math.log(hi), n)),
        # θ with p ∝ sin 2θ has CDF sin²θ: invert a uniform draw
        "angle": np.degrees(np.arcsin(np.sqrt(rng.uniform(math.sin(math.radians(priors.min_angle)) ** 2, 1.0, n)))),
        "impactor_density": rng.uniform(*priors.impactor_density_range, n),
        "target_density": rng.uniform(*priors.target_density_range[obs.target_material], n),
    }
    if obs.velocity is None:
        sample["velocity"] = _truncated_normal(
            rng, priors.velocity_mean, priors.velocity_sigma, *priors.velocity_range, n)
    else:
        sample["velocity"] = rng.normal(obs.velocity, obs.velocity_error, n).clip(LUNAR_ESCAPE_VELOCITY)
    if obs.luminous_energy is not None:
        nominal = suggs_luminous_efficiency(obs.velocity or priors.velocity_mean)
        sample["luminous_efficiency"] = nominal * np.exp(rng.normal(0.0, math.log(priors.eta_scatter), n))
    return sample


def derived_quantities(sample: dict) -> dict:
    mass = math.pi / 6.0 * sample["impactor_diameter"] ** 3 * sample["impactor_density"]
    return {"mass": mass, "kinetic_energy": 0.5 * mass * sample["velocity"] ** 2}


def log_likelihood(obs: Observation, sample: dict) -> np.ndarray:
    predicted = crater_diameter(
        sample["impactor_diameter"], sample["velocity"], sample["angle"],
        sample["impactor_density"], sample["target_density"], obs.target_material,
    )
    sigma = math.log1p(obs.diameter_error)
    log_l = -0.5 * ((np.log(predicted) - math.log(obs.crater_diameter)) / sigma) ** 2
    if obs.luminous_energy is not None:
        energy = sample["kinetic_energy"] if "kinetic_energy" in sample else derived_quantities(sample)["kinetic_energy"]
        light = sample["luminous_efficiency"] * energy
        sigma = math.log1p(obs.luminous_energy_error)
        log_l += -0.5 * ((np.log(light) - math.log(obs.luminous_energy)) / sigma) ** 2
    return log_l


# Real craters to try the inversion on.  Diameters are approximate, except
# the 2013 crater, which LRO measured to the metre.
PRESETS = [
    {
        "id": "2013-crater-only",
        "name": "17 March 2013 crater only",
        "description": "Measured by LRO. From the crater alone, size and speed trade off.",
        "crater_diameter": FLASH_2013.observed_diameter,
        "diameter_error": 0.05,
    },
    {
        "id": "2013-flash",
        "name": "17 March 2013 crater + flash",
        "description": "Add the flash: a second constraint pins the speed and infers η too.",
        "crater_diameter": FLASH_2013.observed_diameter,
        "diameter_error": 0.05,
        "luminous_energy": FLASH_2013.luminous_energy,
        "velocity": FLASH_2013.velocity,
    },
    {
        "id": "linne",
        "name": "Linné (~2.2 km)",
        "description": "A young, simple bowl-shaped crater in Mare Serenitatis.",
        "crater_diameter": 2_200.0,
        "diameter_error": 0.05,
    },
    {
        "id": "kepler",
        "name": "Kepler (~31 km)",
        "description": "A complex crater with terraced walls and a central peak.",
        "crater_diameter": 31_000.0,
        "diameter_error": 0.05,
    },
    {
        "id": "copernicus",
        "name": "Copernicus (~93 km)",
        "description": "A large complex crater, about 800 million years old.",
        "crater_diameter": 93_000.0,
        "diameter_error": 0.05,
    },
]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

PRIOR_SUMMARY_SAMPLES = 100_000
# Scale parameters, whose spread is compared in log space
LOG_SCALE = {"impactor_diameter", "mass", "kinetic_energy", "luminous_efficiency"}


def _summary_and_width(values: np.ndarray, name: str) -> tuple[dict, float]:
    """5/50/95th percentiles, and the 90% width (in log space for scale parameters)."""
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    width = math.log(p95 / p5) if name in LOG_SCALE else p95 - p5
    return {"p5": float(p5), "p50": float(p50), "p95": float(p95)}, width


def infer(
    obs: Observation,
    priors: Priors = Priors(),
    samples: int | None = None,
    posterior_samples: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Posterior over the impactor for an observed crater.

    Returns prior and posterior percentiles for every parameter, an
    equal-weight posterior sample (resampled by weight) for plotting, and
    the effective sample size.
    """
    if obs.crater_diameter <= 0:
        raise ValueError("crater_diameter must be > 0")
    if obs.target_material not in TARGET_MATERIALS:
        raise ValueError(f"material must be one of {TARGET_MATERIALS}")
    if samples is None:
        # A flash adds a second narrow likelihood, so fewer prior draws survive
        samples = 1_000_000 if obs.luminous_energy is not None else 300_000
    rng = np.random.default_rng(seed)

    prior = sample_prior(obs, priors, samples, rng)
    prior.update(derived_quantities(prior))
    log_w = log_likelihood(obs, prior)
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()
    ess = float(1.0 / np.sum(weights ** 2))

    picked = rng.choice(samples, size=posterior_samples, p=weights)
    posterior = {name: values[picked] for name, values in prior.items()}

    names = list(prior)
    # Prior draws are independent, so the first 100k summarise the prior as
    # well as all of them, at a tenth of the sorting cost
    prior_view = {name: values[:PRIOR_SUMMARY_SAMPLES] for name, values in prior.items()}
    parameters = {}
    for name in names:
        prior_stats, prior_width = _summary_and_width(prior_view[name], name)
        post_stats, post_width = _summary_and_width(posterior[name], name)
        # How much the crater narrowed this parameter: posterior 90% width
        # over prior 90% width
        parameters[name] = {"prior": prior_stats, "posterior": post_stats, "narrowing": float(post_width / prior_width)}

    return {
        "observation": {
            "crater_diameter": obs.crater_diameter,
            "diameter_error": obs.diameter_error,
            "material": obs.target_material,
            "luminous_energy": obs.luminous_energy,
            "velocity": obs.velocity,
        },
        "effective_sample_size": ess,
        "prior_samples": samples,
        "parameters": parameters,
        "posterior_samples": {name: posterior[name].tolist() for name in names},
        # A matching prior subsample, to draw prior vs posterior side by side
        "prior_draws": {name: prior[name][:posterior_samples].tolist() for name in names},
    }
