"""
Surface dating — crater-count ages for synthetic and real lunar surfaces.

Builds on simulation/chronology.py.  Three analyses, each a test of the
method against something whose answer is known:

1. Synthetic bombardment: a surface of known age, built crater by crater
   from the production and chronology functions.  Fitting its craters must
   recover that age, except where small craters have saturated.

2. Chang'e-6: an expert crater count of the landing site (Craterstats sample
   data) against the 2.807 Ga radiometric age of the returned basalts.
   Those samples came back in 2024, so unlike the Apollo sites they played
   no part in calibrating the Neukum (2001) chronology: an independent test.

3. Landing sites with the Robbins (2018) catalogue, counted in circles: a
   deliberately naive count area, to show why crater counting needs a
   mapped geological unit.  Circles mix in neighbouring terrain: at 100 km
   no site comes out younger than its samples, and the young basalts
   (Chang'e-5, Chang'e-6) come out over a billion years too old.  Smaller
   circles let in less foreign terrain but leave fewer craters to count.

Data files are built by scripts/build_crater_data.py.
"""

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from simulation.chronology import (
    CHRONOLOGIES,
    NEUKUM_2001,
    bombard,
    cumulative_density,
    infer_age,
    production_shape,
    visible_at,
)

MOON_RADIUS_KM = 1737.4
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance (km) along the lunar surface (haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2.0 * MOON_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def spherical_cap_area_km2(radius_km: float) -> float:
    """Area of a circle of `radius_km` (measured along the surface) on the Moon."""
    return 2.0 * math.pi * MOON_RADIUS_KM ** 2 * (1.0 - math.cos(radius_km / MOON_RADIUS_KM))


@dataclass(frozen=True)
class LandingSite:
    id: str
    name: str
    latitude: float
    longitude: float
    sample_ages: tuple[float, ...]    # Gyr, radiometric ages of the mare basalts
    reference: str


LANDING_SITES = (
    LandingSite("apollo11", "Apollo 11", 0.6741, 23.4730, (3.58, 3.80),
                "Stöffler & Ryder (2001), Space Sci. Rev. 96, 9-54"),
    LandingSite("apollo12", "Apollo 12", -3.0124, -23.4216, (3.15,),
                "Stöffler & Ryder (2001), Space Sci. Rev. 96, 9-54"),
    LandingSite("apollo15", "Apollo 15", 26.1322, 3.6339, (3.30,),
                "Stöffler & Ryder (2001), Space Sci. Rev. 96, 9-54"),
    LandingSite("apollo17", "Apollo 17", 20.1908, 30.7717, (3.75,),
                "Stöffler & Ryder (2001), Space Sci. Rev. 96, 9-54"),
    LandingSite("luna24", "Luna 24", 12.7145, 62.2129, (3.22,),
                "Stöffler & Ryder (2001), Space Sci. Rev. 96, 9-54"),
    LandingSite("change5", "Chang'e-5", 43.0576, -51.9161, (2.030,),
                "Li, Q.-L. et al. (2021), Nature 600, 54-58"),
    LandingSite("change6", "Chang'e-6", -41.625, -153.978, (2.807,),
                "Zhang, Q.W.L. et al. (2025), Nature 643, 356-360"),
)
SITES_BY_ID = {site.id: site for site in LANDING_SITES}

ISOCHRON_AGES = (1.0, 2.0, 3.0, 3.5, 3.8, 4.0)


def isochrons(d_min: float, d_max: float, chronology=NEUKUM_2001, points: int = 60) -> dict:
    """Cumulative-density curves for reference ages, for the SFD plots."""
    d = np.logspace(math.log10(d_min), math.log10(d_max), points)
    return {
        "diameter": d.tolist(),
        "curves": {f"{age:g}": cumulative_density(d, age, chronology).tolist() for age in ISOCHRON_AGES},
    }


def _age_summary(fit: dict) -> dict:
    return {k: fit[k] for k in ("count", "median", "p5", "p95", "n1", "d_min", "d_max")}


def _weighted_cumulative(diameters, weights, area_km2: float) -> dict:
    order = np.argsort(diameters)[::-1]
    d = np.asarray(diameters)[order]
    counts = np.cumsum(np.asarray(weights)[order])
    return {"diameter": d.tolist(), "density": (counts / area_km2).tolist(),
            "error": (np.sqrt(counts) / area_km2).tolist()}


# ---------------------------------------------------------------------------
# 1. Synthetic bombardment
# ---------------------------------------------------------------------------

def synthetic_history(age: float, size_km: float, d_min: float, seed: int = 0) -> dict:
    """A bombarded surface, its crater history, and the ages recovered from it."""
    surface = bombard(age, size_km, d_min, seed=seed)
    area = size_km ** 2
    today = visible_at(surface, 0.0)

    # Fit with ever larger smallest craters: small ones saturate on old surfaces
    fit_minimums = np.logspace(math.log10(d_min), math.log10(max(d_min * 1.01, size_km / 10)), 8)
    fits = []
    for fit_min in fit_minimums:
        fit = infer_age(surface["diameter"][today], area, float(fit_min), surface["d_max"])
        fits.append(_age_summary(fit))

    # Crater density through time: craters formed so far vs still visible
    times = np.linspace(age, 0.0, 90)
    history = {
        "time_ago": times.tolist(),
        "formed": [int(np.sum(surface["formed"] >= t)) for t in times],
        "visible": [int(np.sum(visible_at(surface, t))) for t in times],
    }

    def rounded(key, digits):
        return [None if math.isnan(v) else round(float(v), digits) for v in surface[key]]

    return {
        "age": age,
        "size_km": size_km,
        "d_min": d_min,
        "d_max": surface["d_max"],
        "seed": seed,
        "craters": {
            "x": rounded("x", 2), "y": rounded("y", 2), "diameter": rounded("diameter", 3),
            "formed": rounded("formed", 3), "erased": rounded("erased", 3),
        },
        "history": history,
        "sfd": _weighted_cumulative(surface["diameter"][today], np.ones(int(today.sum())), area),
        "fits": fits,
        "isochrons": isochrons(d_min, surface["d_max"]),
    }


# ---------------------------------------------------------------------------
# 2. Chang'e-6 expert count
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_change6() -> dict:
    with open(os.path.join(DATA_DIR, "change6_count.json")) as f:
        return json.load(f)


def change6_analysis() -> dict:
    data = load_change6()
    craters = np.array(data["craters"])
    d, fraction = craters[:, 0], craters[:, 1]
    area = data["area_km2"]
    site = SITES_BY_ID["change6"]

    fit_minimums = [0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]
    by_minimum = {
        key: [_age_summary(infer_age(d, area, m, chronology=chron, weights=fraction)) for m in fit_minimums]
        for key, chron in CHRONOLOGIES.items()
    }
    return {
        "site": {"name": site.name, "latitude": site.latitude, "longitude": site.longitude,
                 "sample_age": site.sample_ages[0], "reference": site.reference},
        "source": data["source"],
        "area_km2": area,
        "boundary": data["boundary"],
        "craters": {"diameter": d.tolist(), "fraction": fraction.tolist(),
                    "longitude": craters[:, 2].tolist(), "latitude": craters[:, 3].tolist()},
        "sfd": _weighted_cumulative(d, fraction, area),
        "fits": by_minimum,
        "chronologies": {key: chron.name for key, chron in CHRONOLOGIES.items()},
        "isochrons": isochrons(0.03, 0.6),
    }


# ---------------------------------------------------------------------------
# 3. Landing sites, Robbins catalogue, circular count areas
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_landing_site_craters() -> dict:
    with open(os.path.join(DATA_DIR, "landing_site_craters.json")) as f:
        return json.load(f)


def landing_site_ages(radius_km: float = 100.0, d_min: float = 1.0, d_max: float = 20.0) -> dict:
    data = load_landing_site_craters()
    if not 10.0 <= radius_km <= data["max_distance_km"]:
        raise ValueError(f"radius must be between 10 and {data['max_distance_km']:g} km")
    if not 1.0 <= d_min < d_max:
        raise ValueError("need 1 km ≤ d_min < d_max (the catalogue starts at 1 km)")
    area = spherical_cap_area_km2(radius_km)

    sites = []
    for site in LANDING_SITES:
        rows = np.array(data["sites"][site.id]).reshape(-1, 4)
        inside = rows[rows[:, 3] <= radius_km]
        sites.append({
            "id": site.id,
            "name": site.name,
            "latitude": site.latitude,
            "longitude": site.longitude,
            "sample_ages": list(site.sample_ages),
            "reference": site.reference,
            "ages": {key: _age_summary(infer_age(inside[:, 2], area, d_min, d_max, chron))
                     for key, chron in CHRONOLOGIES.items()},
        })
    return {
        "source": data["source"],
        "radius_km": radius_km,
        "area_km2": area,
        "d_min": d_min,
        "d_max": d_max,
        "sites": sites,
    }


def landing_site_craters(site_id: str, radius_km: float = 100.0) -> dict:
    """The catalogue craters around one site, for its map and SFD."""
    site = SITES_BY_ID[site_id]
    rows = np.array(load_landing_site_craters()["sites"][site_id]).reshape(-1, 4)
    inside = rows[rows[:, 3] <= radius_km]
    area = spherical_cap_area_km2(radius_km)
    return {
        "id": site.id,
        "name": site.name,
        "latitude": site.latitude,
        "longitude": site.longitude,
        "radius_km": radius_km,
        "craters": {"latitude": inside[:, 0].tolist(), "longitude": inside[:, 1].tolist(),
                    "diameter": inside[:, 2].tolist()},
        "sfd": _weighted_cumulative(inside[:, 2], np.ones(len(inside)), area),
        "isochrons": isochrons(1.0, 50.0),
    }


__all__ = [
    "LANDING_SITES", "SITES_BY_ID", "great_circle_km", "spherical_cap_area_km2",
    "synthetic_history", "change6_analysis", "landing_site_ages", "landing_site_craters",
    "production_shape",
]
