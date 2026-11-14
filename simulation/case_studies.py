"""
Case studies — the crater model tested against an observed lunar impact.

On 2013-03-17 NASA's lunar impact monitoring cameras recorded the brightest
flash of their eight-year campaign, and the Lunar Reconnaissance Orbiter later
found the fresh crater at the same spot.  That makes it one of the few impacts
where both the input (the flash) and the output (the crater) were measured.

The chain from flash to crater:

    luminous energy  --η-->  kinetic energy  --v-->  mass  --scaling-->  crater

Every link but the last is an observation with an uncertainty, and the
largest is the luminous efficiency η: the fraction of kinetic energy that
becomes visible light, which nobody has measured at lunar impact speeds.
This module:

1. Checks the observing geometry with our own ephemeris: the flash can only
   have been seen if the site was in darkness and Earth was above its horizon.
2. Propagates the input uncertainties through the crater model by Monte Carlo
   and asks where the observed crater falls in the predicted distribution.
3. Inverts the model: which η would make it predict the observed crater?
   (The crater diameter grows monotonically with the impactor mass, and the
   mass falls monotonically with η, so a bisection on log η finds it.)

The input distributions (Event.*_sigma, *_range) are documented assumptions,
not published error bars, except where a reference is given.

References:
    - Moser, D.E., Suggs, R.M. & Suggs, R.J. (2014). "Large meteoroid impact
      on the Moon on 17 March 2013." Asteroids, Comets, Meteors, Helsinki.
    - Suggs, R.M. et al. (2014). "The flux of kilogram-sized meteoroids from
      lunar impact monitoring." Icarus 238, 23-36.
    - Robinson, M.S. et al. (2015). "New crater on the Moon and a swarm of
      secondaries." Icarus 252, 229-235.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from simulation.crater_model import ImpactParameters, run_simulation, TARGET_MATERIALS
from simulation.lunar_ephemeris import moon_illumination


def suggs_luminous_efficiency(velocity: float) -> float:
    """Velocity-dependent luminous efficiency η (Suggs et al. 2014); velocity in m/s."""
    v_km = velocity / 1000.0
    return 1.5e-3 * math.exp(-(9.3 ** 2) / v_km ** 2)


@dataclass(frozen=True)
class ImpactEvent:
    id: str
    title: str
    time: datetime
    latitude: float                 # selenographic, degrees north
    longitude: float                # selenographic, degrees east
    luminous_energy: float          # J, from the flash photometry
    velocity: float                 # m/s
    velocity_sigma: float           # m/s
    impact_angle: float             # degrees from horizontal
    impact_angle_sigma: float       # degrees
    impactor_density_range: tuple[float, float]  # kg/m³
    target_density_range: tuple[float, float]    # kg/m³
    eta_scatter: float              # η is log-normal; σ of ln η, as a factor
    observed_diameter: float        # m, rim to rim
    references: tuple[str, ...]

    @property
    def nominal_eta(self) -> float:
        return suggs_luminous_efficiency(self.velocity)

    @property
    def nominal_impactor_density(self) -> float:
        return sum(self.impactor_density_range) / 2.0

    @property
    def nominal_target_density(self) -> float:
        return sum(self.target_density_range) / 2.0


FLASH_2013 = ImpactEvent(
    id="2013-03-17",
    title="17 March 2013 impact flash",
    time=datetime(2013, 3, 17, 3, 50, 54, 312000, tzinfo=timezone.utc),
    # Moser et al. 2014: 20.60 ± 0.17° N, 23.92 ± 0.30° W
    latitude=20.60,
    longitude=-23.92,
    luminous_energy=7.1e6,          # Moser et al. 2014
    velocity=25_600.0,              # Virginid fireball velocity, Moser et al. 2014
    velocity_sigma=1_000.0,         # assumption
    impact_angle=56.0,              # Moser et al. 2014
    impact_angle_sigma=10.0,        # assumption
    impactor_density_range=(1800.0, 3000.0),  # Moser et al. 2014
    target_density_range=(1300.0, 1800.0),    # assumption: lunar regolith bulk density
    eta_scatter=3.0,                # assumption: η known to within a factor of ~3
    observed_diameter=18.8,         # Robinson et al. 2015
    references=(
        "Moser, Suggs & Suggs (2014), Asteroids, Comets, Meteors, Helsinki",
        "Suggs et al. (2014), Icarus 238, 23-36",
        "Robinson et al. (2015), Icarus 252, 229-235",
    ),
)

CASE_STUDIES = {FLASH_2013.id: FLASH_2013}


def get_case_study(event_id: str) -> ImpactEvent | None:
    return CASE_STUDIES.get(event_id)


# ---------------------------------------------------------------------------
# Observing geometry
# ---------------------------------------------------------------------------

def _elevation_above_horizon(lat: float, lon: float, sub_lat: float, sub_lon: float) -> float:
    """Elevation (degrees) of a body whose sub-point is (sub_lat, sub_lon), seen from (lat, lon)."""
    la, sla = math.radians(lat), math.radians(sub_lat)
    cos_zenith = (math.sin(la) * math.sin(sla)
                  + math.cos(la) * math.cos(sla) * math.cos(math.radians(lon - sub_lon)))
    return math.degrees(math.asin(max(-1.0, min(1.0, cos_zenith))))


def observing_geometry(event: ImpactEvent) -> dict:
    """Sun and Earth elevation at the impact site when the flash was seen."""
    m = moon_illumination(event.time)
    sun = _elevation_above_horizon(event.latitude, event.longitude, m.sub_solar_lat, m.sub_solar_lon)
    earth = _elevation_above_horizon(event.latitude, event.longitude, m.sub_earth_lat, m.sub_earth_lon)
    return {
        "sun_elevation": sun,
        "earth_elevation": earth,
        "phase_name": m.phase_name,
        "illuminated_fraction": m.illuminated_fraction,
        # Flashes are only detectable against the unlit Moon, from an Earth above the horizon
        "site_in_darkness": sun < 0.0,
        "earth_above_horizon": earth > 0.0,
    }


# ---------------------------------------------------------------------------
# Flash → crater
# ---------------------------------------------------------------------------

def predict_crater(
    luminous_energy: float,
    eta: float,
    velocity: float,
    impact_angle: float,
    impactor_density: float,
    target_density: float,
    material: str,
) -> float:
    """Rim-to-rim crater diameter (m) for a flash, through the whole chain."""
    kinetic_energy = luminous_energy / eta
    mass = 2.0 * kinetic_energy / velocity ** 2
    diameter = (6.0 * mass / (math.pi * impactor_density)) ** (1.0 / 3.0)
    return run_simulation(ImpactParameters(
        impactor_diameter=diameter,
        impact_velocity=velocity,
        impact_angle=impact_angle,
        impactor_density=impactor_density,
        target_density=target_density,
        target_material=material,
    )).crater_diameter


def nominal_chain(event: ImpactEvent, material: str) -> dict:
    """Each step of the chain at the published (nominal) inputs."""
    eta = event.nominal_eta
    kinetic_energy = event.luminous_energy / eta
    mass = 2.0 * kinetic_energy / event.velocity ** 2
    return {
        "luminous_energy": event.luminous_energy,
        "luminous_efficiency": eta,
        "kinetic_energy": kinetic_energy,
        "impactor_mass": mass,
        "impactor_diameter": (6.0 * mass / (math.pi * event.nominal_impactor_density)) ** (1.0 / 3.0),
        "crater_diameter": predict_crater(
            event.luminous_energy, eta, event.velocity, event.impact_angle,
            event.nominal_impactor_density, event.nominal_target_density, material,
        ),
    }


def monte_carlo(event: ImpactEvent, material: str, samples: int = 2000, seed: int = 0) -> np.ndarray:
    """Predicted crater diameters (m) with the inputs drawn from their uncertainties."""
    if material not in TARGET_MATERIALS:
        raise ValueError(f"material must be one of {TARGET_MATERIALS}")
    rng = np.random.default_rng(seed)
    eta = event.nominal_eta * np.exp(rng.normal(0.0, math.log(event.eta_scatter), samples))
    velocity = rng.normal(event.velocity, event.velocity_sigma, samples)
    # Grazing (< 15°) and beyond-vertical angles are outside the model's range
    angle = np.clip(rng.normal(event.impact_angle, event.impact_angle_sigma, samples), 15.0, 90.0)
    rho_i = rng.uniform(*event.impactor_density_range, samples)
    rho_t = rng.uniform(*event.target_density_range, samples)
    return np.array([
        predict_crater(event.luminous_energy, *inputs, material)
        for inputs in zip(eta, velocity, angle, rho_i, rho_t)
    ])


def implied_luminous_efficiency(event: ImpactEvent, material: str) -> float:
    """η at which the nominal chain predicts exactly the observed crater."""
    def crater(log_eta):
        return predict_crater(
            event.luminous_energy, math.exp(log_eta), event.velocity, event.impact_angle,
            event.nominal_impactor_density, event.nominal_target_density, material,
        )

    # Diameter falls as η rises (less mass for the same light)
    lo, hi = math.log(1e-7), math.log(1.0)
    if not crater(hi) <= event.observed_diameter <= crater(lo):
        raise ValueError("observed crater is outside the range any η can produce")
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if crater(mid) > event.observed_diameter:
            lo = mid
        else:
            hi = mid
    return math.exp((lo + hi) / 2.0)


def eta_sweep(event: ImpactEvent, material: str, points: int = 49) -> dict:
    """Predicted crater across η from 1e-5 to 1e-1, for the η-vs-diameter chart."""
    etas = np.logspace(-5, -1, points)
    return {
        "eta": etas.tolist(),
        "crater_diameter": [
            predict_crater(event.luminous_energy, eta, event.velocity, event.impact_angle,
                           event.nominal_impactor_density, event.nominal_target_density, material)
            for eta in etas
        ],
    }


def analyze(event: ImpactEvent, samples: int = 2000, seed: int = 0) -> dict:
    """Everything the case-study page shows, for both target materials."""
    models = {}
    for material in TARGET_MATERIALS:
        diameters = monte_carlo(event, material, samples, seed)
        p5, p50, p95 = np.percentile(diameters, [5, 50, 95])
        implied = implied_luminous_efficiency(event, material)
        models[material] = {
            "nominal": nominal_chain(event, material),
            "samples": diameters.round(3).tolist(),
            "percentiles": {"p5": float(p5), "p50": float(p50), "p95": float(p95)},
            # Share of predictions at or below what LRO measured: ~0.5 means a good fit
            "observed_percentile": float(np.mean(diameters <= event.observed_diameter)),
            "observed_within_90": bool(p5 <= event.observed_diameter <= p95),
            "implied_eta": implied,
            "implied_eta_ratio": implied / event.nominal_eta,
            "eta_sweep": eta_sweep(event, material),
        }
    return {
        "event": {
            "id": event.id,
            "title": event.title,
            "time": event.time.isoformat(),
            "latitude": event.latitude,
            "longitude": event.longitude,
            "luminous_energy": event.luminous_energy,
            "velocity": event.velocity,
            "impact_angle": event.impact_angle,
            "observed_diameter": event.observed_diameter,
            "references": list(event.references),
        },
        "assumptions": {
            "velocity_sigma": event.velocity_sigma,
            "impact_angle_sigma": event.impact_angle_sigma,
            "impactor_density_range": list(event.impactor_density_range),
            "target_density_range": list(event.target_density_range),
            "eta_scatter": event.eta_scatter,
            "samples": samples,
            "seed": seed,
        },
        "geometry": observing_geometry(event),
        "models": models,
    }
