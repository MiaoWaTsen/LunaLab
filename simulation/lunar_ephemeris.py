"""
Lunar Ephemeris — where the Sun and the Earth are, as seen from the Moon.

Computes, for a given instant:

    * the sub-solar point (selenographic latitude/longitude of the Sun),
      which places the day/night terminator on the 3D Moon;
    * the sub-Earth point (optical libration), i.e. the point at the
      centre of the Moon's disc as seen from Earth;
    * phase angle, illuminated fraction and phase name.

Method (Meeus, *Astronomical Algorithms*, 2nd ed., 1998):

    ch. 25  Sun's geometric longitude and distance (low accuracy, ~0.01°)
    ch. 47  Moon's longitude, latitude and distance — the largest terms
            of the ELP-2000/82 series (truncated; ~0.01° in longitude)
    ch. 48  Phase angle and illuminated fraction
    ch. 53  Optical libration and the selenographic position of the Sun

Physical libration (< 0.04°) and nutation are left out.  Nutation cancels
because geometric longitudes are used throughout (Meeus subtracts Δψ from
apparent longitudes in ch. 53).  The result is accurate to about 0.1°,
far below what the 3D view can show.

All angles are in degrees; selenographic longitudes are east-positive in
[-180, 180).
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

# TT − UTC.  ~69.2 s since 2017; the Moon moves 0.5′ per minute, so a
# second-level error here is irrelevant.
DELTA_T_SECONDS = 69.2

AU_KM = 149_597_870.7
LUNAR_EQUATOR_INCLINATION = 1.54242   # I, to the ecliptic (Cassini's laws)


def _sin(deg: float) -> float:
    return math.sin(math.radians(deg))


def _cos(deg: float) -> float:
    return math.cos(math.radians(deg))


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def julian_ephemeris_day(when: datetime) -> float:
    """Julian Ephemeris Day (TT) for a UTC datetime."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    unix = when.timestamp() + DELTA_T_SECONDS
    return unix / 86400.0 + 2440587.5


# ---------------------------------------------------------------------------
# Moon position (Meeus ch. 47, Tables 47.A and 47.B, largest terms)
# ---------------------------------------------------------------------------

# D, M, M', F, Σl coefficient (1e-6 °), Σr coefficient (1e-3 km)
_LON_DIST_TERMS = [
    (0, 0, 1, 0, 6288774, -20905355), (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968), (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888), (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158), (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733), (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620), (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755), (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0), (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782), (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636), (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824), (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675), (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445), (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403), (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0), (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322), (2, -2, 0, 0, 2236, -9884),
]

# D, M, M', F, Σb coefficient (1e-6 °)
_LAT_TERMS = [
    (0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237), (2, 0, -1, 1, 55413), (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573), (0, 0, 2, 1, 17198), (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822), (2, -1, 0, -1, 8216), (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200), (2, 1, 0, -1, -3359), (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211), (2, -1, -1, -1, 2065), (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828), (0, 1, 0, 1, -1794), (0, 0, 0, 3, -1749),
    (0, 1, -1, 1, -1565), (1, 0, 0, 1, -1491), (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410), (0, 1, 0, -1, -1344), (1, 0, 0, -1, -1335),
    (0, 0, 3, 1, 1107), (4, 0, 0, -1, 1021), (4, 0, -1, 1, 833),
]


def _fundamental_arguments(T: float) -> dict:
    return {
        "Lp": 218.3164477 + 481267.88123421 * T - 0.0015786 * T**2 + T**3 / 538841 - T**4 / 65194000,
        "D": 297.8501921 + 445267.1114034 * T - 0.0018819 * T**2 + T**3 / 545868 - T**4 / 113065000,
        "M": 357.5291092 + 35999.0502909 * T - 0.0001536 * T**2 + T**3 / 24490000,
        "Mp": 134.9633964 + 477198.8675055 * T + 0.0087414 * T**2 + T**3 / 69699 - T**4 / 14712000,
        "F": 93.2720950 + 483202.0175233 * T - 0.0036539 * T**2 - T**3 / 3526000 + T**4 / 863310000,
        "E": 1 - 0.002516 * T - 0.0000074 * T**2,
    }


def moon_position(jde: float) -> tuple[float, float, float]:
    """Geometric ecliptic longitude, latitude (°) and distance (km) of the Moon."""
    T = (jde - 2451545.0) / 36525.0
    a = _fundamental_arguments(T)
    Lp, D, M, Mp, F, E = a["Lp"], a["D"], a["M"], a["Mp"], a["F"], a["E"]

    sum_l = sum_r = sum_b = 0.0
    for d, m, mp, f, cl, cr in _LON_DIST_TERMS:
        arg = d * D + m * M + mp * Mp + f * F
        e = E ** abs(m)
        sum_l += cl * e * _sin(arg)
        sum_r += cr * e * _cos(arg)
    for d, m, mp, f, cb in _LAT_TERMS:
        sum_b += cb * E ** abs(m) * _sin(d * D + m * M + mp * Mp + f * F)

    A1 = 119.75 + 131.849 * T
    A2 = 53.09 + 479264.290 * T
    A3 = 313.45 + 481266.484 * T
    sum_l += 3958 * _sin(A1) + 1962 * _sin(Lp - F) + 318 * _sin(A2)
    sum_b += (-2235 * _sin(Lp) + 382 * _sin(A3) + 175 * _sin(A1 - F)
              + 175 * _sin(A1 + F) + 127 * _sin(Lp - Mp) - 115 * _sin(Lp + Mp))

    lon = (Lp + sum_l / 1e6) % 360.0
    lat = sum_b / 1e6
    dist = 385000.56 + sum_r / 1000.0
    return lon, lat, dist


# ---------------------------------------------------------------------------
# Sun position (Meeus ch. 25, low accuracy)
# ---------------------------------------------------------------------------

def sun_position(jde: float) -> tuple[float, float]:
    """Geometric ecliptic longitude (°) and distance (km) of the Sun."""
    T = (jde - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
    C = ((1.914602 - 0.004817 * T - 0.000014 * T**2) * _sin(M)
         + (0.019993 - 0.000101 * T) * _sin(2 * M) + 0.000289 * _sin(3 * M))
    true_lon = (L0 + C) % 360.0
    nu = M + C
    R_au = 1.000001018 * (1 - e**2) / (1 + e * _cos(nu))
    return true_lon, R_au * AU_KM


# ---------------------------------------------------------------------------
# Selenographic coordinates (Meeus ch. 53)
# ---------------------------------------------------------------------------

def _selenographic(lon: float, lat: float, node: float, F: float) -> tuple[float, float]:
    """Selenographic (lat, lon) of a direction given in ecliptic coordinates."""
    I = LUNAR_EQUATOR_INCLINATION
    W = lon - node
    A = math.degrees(math.atan2(
        _sin(W) * _cos(lat) * _cos(I) - _sin(lat) * _sin(I),
        _cos(W) * _cos(lat),
    ))
    sel_lon = _wrap180(A - F)
    sel_lat = math.degrees(math.asin(-_sin(W) * _cos(lat) * _sin(I) - _sin(lat) * _cos(I)))
    return sel_lat, sel_lon


@dataclass
class MoonIllumination:
    sub_solar_lat: float
    sub_solar_lon: float
    sub_earth_lat: float      # optical libration in latitude, b
    sub_earth_lon: float      # optical libration in longitude, l
    phase_angle: float
    illuminated_fraction: float
    phase_name: str
    waxing: bool


_PHASE_NAMES = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
                "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]


def illumination_at_jde(jde: float) -> MoonIllumination:
    T = (jde - 2451545.0) / 36525.0
    lam, beta, delta = moon_position(jde)
    sun_lon, R = sun_position(jde)

    F = _fundamental_arguments(T)["F"]
    node = 125.0445479 - 1934.1362891 * T + 0.0020754 * T**2 + T**3 / 467441 - T**4 / 60616000

    # Sub-Earth point = optical libration
    b, l = _selenographic(lam, beta, node, F)

    # Heliocentric direction of the Moon ≈ direction of the Sun seen from it
    lam_h = sun_lon + 180.0 + (delta / R) * 57.296 * _cos(beta) * _sin(sun_lon - lam)
    beta_h = (delta / R) * beta
    b0, l0 = _selenographic(lam_h, beta_h, node, F)

    # Phase (Meeus ch. 48)
    psi = math.degrees(math.acos(_cos(beta) * _cos(lam - sun_lon)))
    phase_angle = math.degrees(math.atan2(R * _sin(psi), delta - R * _cos(psi)))
    k = (1 + _cos(phase_angle)) / 2
    elongation = (lam - sun_lon) % 360.0          # 0 new, 90 first quarter, 180 full
    name = _PHASE_NAMES[int(((elongation + 22.5) % 360) // 45)]

    return MoonIllumination(
        sub_solar_lat=b0, sub_solar_lon=l0,
        sub_earth_lat=b, sub_earth_lon=l,
        phase_angle=phase_angle, illuminated_fraction=k,
        phase_name=name, waxing=elongation < 180.0,
    )


def moon_illumination(when: datetime) -> MoonIllumination:
    """Sun and Earth as seen from the Moon at a UTC instant."""
    return illumination_at_jde(julian_ephemeris_day(when))
