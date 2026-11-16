"""
Crater chronology — dating a lunar surface by counting its craters.

A fresh surface (a lava flow, say) starts with no craters and collects them
at a known, slowly declining rate.  Count the craters on it, and the count
tells you how long it has been exposed.  This is how ages are assigned to
almost every surface in the Solar System that no sample has come from.

Two calibrated ingredients:

1. Production function (PF): the *shape* of the size distribution of new
   craters, the same on every surface.  Neukum et al. (2001), an
   11th-degree polynomial in log D, valid for 10 m < D < 300 km:

       log10 N(≥D) = a0 + Σ_{k=1..11} a_k (log10 D)^k      (D in km, per km²)

2. Chronology function (CF): how many craters ≥ 1 km a surface collects in
   T billion years, calibrated against radiometric ages of Apollo and Luna
   samples:

       N1(T) = a (exp(b T) − 1) + c T                       (per km²)

   The exponential term is the heavy early bombardment; the linear term
   the near-constant flux of the last ~3 Gyr.

Together:  N(≥D, T) = N1(T) · S(D),  with S(D) = N(≥D) / N(≥1 km) from the PF.

Age inference (Michael et al. 2016): craters form as a Poisson process, so
the number n counted in [D_min, D_max] over area A has

    p(n | T) = Poisson(n; λ(T)),   λ(T) = A · N1(T) · (S(D_min) − S(D_max))

and, with the PF shape fixed, n is a sufficient statistic for T.  The
posterior on a fine grid of T is therefore exact, with no curve fitting.

Coefficients as in Craterstats (Michael 2013; github.com/ggmichael/craterstats),
including its correction of the published Neukum (2001) a0 = −3.0876, a typo
for −3.0768 (which matches N1(1 Gyr)).

References:
    - Neukum, G., Ivanov, B.A. & Hartmann, W.K. (2001). "Cratering records in
      the inner solar system in relation to the lunar reference system."
      Space Sci. Rev. 96, 55-86.
    - Yue, Z. et al. (2022). "Updated lunar cratering chronology model with
      the radiometric age of Chang'e-5 samples." Nature Astronomy 6, 541-545.
    - Michael, G.G. et al. (2016). "Planetary surface dating from crater
      size-frequency distribution measurements: Poisson timing analysis."
      Icarus 277, 279-285.
"""

import math
from dataclasses import dataclass

import numpy as np


# Neukum et al. (2001) production function, a1..a11 (a0 cancels out of S(D))
NEUKUM_PF = (
    -3.557528, 0.781027, 1.021521, -0.156012, -0.444058, 0.019977,
    0.086850, -0.005874, -0.006809, 8.25e-04, 5.54e-05,
)
PF_RANGE_KM = (0.01, 300.0)


@dataclass(frozen=True)
class Chronology:
    name: str
    a: float      # km⁻²
    b: float      # Gyr⁻¹
    c: float      # km⁻² Gyr⁻¹
    reference: str

    def n1(self, age):
        """Cumulative craters ≥ 1 km per km² after `age` Gyr (scalar or array)."""
        age = np.asarray(age, dtype=float)
        return self.a * (np.exp(self.b * age) - 1.0) + self.c * age

    def rate(self, age):
        """dN1/dT: craters ≥ 1 km per km² per Gyr, `age` Gyr before present."""
        return self.a * self.b * np.exp(self.b * np.asarray(age, dtype=float)) + self.c

    def age(self, n1: float) -> float:
        """Inverse of n1(): the age (Gyr) at which a surface reaches n1 craters ≥ 1 km per km²."""
        if n1 < 0:
            raise ValueError("n1 must be ≥ 0")
        lo, hi = 0.0, 5.0
        if n1 > self.n1(hi):
            raise ValueError("crater density exceeds a 5 Gyr surface")
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if self.n1(mid) < n1:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0


NEUKUM_2001 = Chronology(
    "Neukum et al. (2001)", 5.44e-14, 6.93, 8.38e-4,
    "Neukum, Ivanov & Hartmann (2001), Space Sci. Rev. 96, 55-86",
)
YUE_2022 = Chronology(
    "Yue et al. (2022)", 1.089e-13, 6.757, 7.660e-4,
    "Yue et al. (2022), Nature Astronomy 6, 541-545 (recalibrated with Chang'e-5)",
)
CHRONOLOGIES = {"neukum2001": NEUKUM_2001, "yue2022": YUE_2022}


def production_shape(diameter_km):
    """S(D) = N(≥D) / N(≥1 km), the Neukum (2001) production function shape."""
    log_d = np.log10(np.asarray(diameter_km, dtype=float))
    exponent = np.zeros_like(log_d)
    for k, a in enumerate(NEUKUM_PF, start=1):
        exponent += a * log_d ** k
    return 10.0 ** exponent


def cumulative_density(diameter_km, age: float, chronology: Chronology = NEUKUM_2001):
    """Isochron: craters ≥ D per km² on a surface `age` Gyr old."""
    return chronology.n1(age) * production_shape(diameter_km)


# ---------------------------------------------------------------------------
# Age inference
# ---------------------------------------------------------------------------

AGE_GRID = np.linspace(0.0, 4.5, 4501)   # Gyr, 1 Myr steps; uniform prior


def infer_age(
    diameters_km,
    area_km2: float,
    d_min: float,
    d_max: float = PF_RANGE_KM[1],
    chronology: Chronology = NEUKUM_2001,
    weights=None,
) -> dict:
    """
    Posterior over the surface age from the craters counted in [d_min, d_max).

    `weights` is the fraction of each crater counted (buffered crater
    counting counts a crater straddling the area boundary partially); the
    Poisson likelihood then takes the summed, non-integer count.

    Returns the median, the 90% credible interval, the count used and the
    posterior on AGE_GRID (for plotting).
    """
    if area_km2 <= 0:
        raise ValueError("area must be > 0")
    if not PF_RANGE_KM[0] <= d_min < d_max <= PF_RANGE_KM[1]:
        raise ValueError(f"fit range must lie within {PF_RANGE_KM} km")

    d = np.asarray(diameters_km, dtype=float)
    w = np.ones_like(d) if weights is None else np.asarray(weights, dtype=float)
    n = float(np.sum(w[(d >= d_min) & (d < d_max)]))
    window = float(production_shape(d_min) - production_shape(d_max))
    expected = area_km2 * chronology.n1(AGE_GRID) * window

    # log Poisson(n; λ), dropping log n! (constant in T); λ = 0 at T = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = n * np.log(expected) - expected
    log_p[0] = 0.0 if n == 0 else -np.inf
    posterior = np.exp(log_p - log_p.max())
    posterior /= posterior.sum()
    cdf = np.cumsum(posterior)

    def quantile(q):
        return float(np.interp(q, cdf, AGE_GRID))

    return {
        "count": n,
        "area_km2": area_km2,
        "d_min": d_min,
        "d_max": d_max,
        "chronology": chronology.name,
        # Crater density ≥ 1 km implied by the count (the classic "N(1)")
        "n1": n / (area_km2 * window),
        "median": quantile(0.5),
        "p5": quantile(0.05),
        "p95": quantile(0.95),
        "posterior": posterior,
    }


# ---------------------------------------------------------------------------
# Synthetic bombardment
# ---------------------------------------------------------------------------

def sample_diameters(n: int, d_min: float, d_max: float, rng: np.random.Generator) -> np.ndarray:
    """n crater diameters (km) following the production function in [d_min, d_max]."""
    # Inverse CDF on a fine log grid: P(D ≥ x) ∝ S(x) − S(d_max)
    grid = np.logspace(math.log10(d_min), math.log10(d_max), 4000)
    survival = production_shape(grid) - production_shape(d_max)
    survival /= survival[0]
    # survival falls from 1 to 0 as D grows; interp needs increasing x
    return np.interp(rng.uniform(0.0, 1.0, n), survival[::-1], grid[::-1])


def sample_formation_ages(n: int, age: float, chronology: Chronology, rng: np.random.Generator) -> np.ndarray:
    """When each crater formed (Gyr ago), following the declining impact rate."""
    grid = np.linspace(0.0, age, 2000)
    accumulated = chronology.n1(grid) / chronology.n1(age)   # 0 today → 1 at `age`
    return np.interp(rng.uniform(0.0, 1.0, n), accumulated, grid)


MAX_SYNTHETIC_CRATERS = 200_000


def _erase_covered(x, y, diameter, formed, cell: float) -> np.ndarray:
    """
    When each crater was erased (Gyr ago, NaN if never), craters in time order.

    A crater is erased by the first later crater that is larger and covers
    its centre.  Craters are bucketed on a grid of `cell` km, so each new
    crater only checks the buckets under it: small craters, the vast
    majority, look at a handful of neighbours.
    """
    erased = np.full(len(x), np.nan)
    buckets: dict[tuple[int, int], list[int]] = {}
    for j in range(len(x)):
        r = diameter[j] / 2.0
        for cx in range(int((x[j] - r) // cell), int((x[j] + r) // cell) + 1):
            for cy in range(int((y[j] - r) // cell), int((y[j] + r) // cell) + 1):
                bucket = buckets.get((cx, cy))
                if not bucket:
                    continue
                keep = []
                for i in bucket:
                    if diameter[i] < diameter[j] and (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < r * r:
                        erased[i] = formed[j]
                    else:
                        keep.append(i)
                buckets[(cx, cy)] = keep
        buckets.setdefault((int(x[j] // cell), int(y[j] // cell)), []).append(j)
    return erased


def bombard(
    age: float,
    size_km: float,
    d_min: float,
    d_max: float | None = None,
    chronology: Chronology = NEUKUM_2001,
    seed: int = 0,
    erase: bool = True,
    max_craters: int = MAX_SYNTHETIC_CRATERS,
) -> dict:
    """
    A square surface `size_km` across, exposed for `age` Gyr.

    Every crater in [d_min, d_max] that the production and chronology
    functions predict is placed at random.  With `erase`, a crater whose
    centre a later, larger crater lands on is destroyed: on old surfaces the
    small craters saturate and stop recording the age.

    Returns craters oldest first, each with x, y (km), diameter (km),
    formation age (Gyr ago) and the age at which it was erased (None if it
    survives).
    """
    if age <= 0 or size_km <= 0:
        raise ValueError("age and size must be > 0")
    d_max = d_max or size_km / 2.0
    rng = np.random.default_rng(seed)
    area = size_km ** 2
    expected = area * chronology.n1(age) * float(production_shape(d_min) - production_shape(d_max))
    if expected > max_craters:
        raise ValueError(
            f"about {expected:,.0f} craters: use a smaller area, a younger age or a larger d_min "
            f"(limit {max_craters:,})")
    n = int(rng.poisson(expected))

    formed = np.sort(sample_formation_ages(n, age, chronology, rng))[::-1]   # oldest first
    diameter = sample_diameters(n, d_min, d_max, rng)
    x = rng.uniform(0.0, size_km, n)
    y = rng.uniform(0.0, size_km, n)
    erased = np.full(n, np.nan)

    if erase:
        erased = _erase_covered(x, y, diameter, formed, cell=2.0 * d_min)

    return {
        "age": age,
        "size_km": size_km,
        "d_min": d_min,
        "d_max": d_max,
        "chronology": chronology.name,
        "x": x, "y": y, "diameter": diameter, "formed": formed, "erased": erased,
    }


def visible_at(surface: dict, time_ago: float) -> np.ndarray:
    """Mask of craters on the surface `time_ago` Gyr before present."""
    formed = surface["formed"] >= time_ago
    not_yet_erased = np.isnan(surface["erased"]) | (surface["erased"] < time_ago)
    return formed & not_yet_erased
