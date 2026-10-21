"""
Asymmetric Crater Terrain Generator — terrain/asymmetric_crater.py

SCIENTIFIC DISCLAIMER
======================
This module generates a *synthetic geometric approximation* of
oblique-impact crater morphology for **visualization purposes only**.
It is NOT the output of a hydrocode or validated geophysical model.

The angle-dependent features follow the qualitative regimes observed in
laboratory and planetary craters (Gault & Wedekind 1978; Bottke et al.
2000; Herrick & Forsberg-Taylor 2003):

    θ ≥ 45°   Circular crater, symmetric ejecta blanket
    θ < 45°   Ejecta increasingly concentrated downrange
    θ < 25°   Uprange "forbidden zone" with no ejecta
    θ < 15°   Crater footprint elongates along the impact direction
    θ < 10°   "Butterfly" ejecta: forbidden zones uprange AND downrange

The radial crater shape is shared with ``terrain/crater_terrain.py``
(``radial_elevation``); this module only adds the directional terms.

Dependencies
-------------
    numpy — vectorised grid computation
"""

import time
import math
import numpy as np

from terrain.crater_terrain import (
    TERRAIN_WIDTH,
    TERRAIN_HEIGHT,
    DOMAIN_HALF_FACTOR,
    radial_elevation,
)


# ---------------------------------------------------------------------------
# Oblique-impact regime thresholds (degrees from horizontal)
# ---------------------------------------------------------------------------

ASYMMETRIC_EJECTA_ANGLE = 45.0
FORBIDDEN_ZONE_ANGLE    = 25.0
ELLIPTICAL_CRATER_ANGLE = 15.0
BUTTERFLY_EJECTA_ANGLE  = 10.0

_MIN_AXIS_RATIO = 0.5     # minor / major axis at grazing incidence
_RIM_BLEND_WIDTH = 0.15   # (r/R) over which the rim blends into the ejecta pattern


def ejecta_weight(phi: np.ndarray, impact_angle: float) -> np.ndarray:
    """
    Relative ejecta thickness as a function of azimuth from downrange.

    phi = 0 is downrange, phi = π uprange.  The weight averages to 1 over
    all azimuths, so total ejecta volume is independent of impact angle.
    """
    # Downrange bias: A = 0 at 45°, 1 at 25° (forbidden zone appears
    # exactly uprange), 2 at 5° (forbidden zone spans ±60°).
    a = np.clip((ASYMMETRIC_EJECTA_ANGLE - impact_angle) / 20.0, 0.0, 2.0)
    w = np.maximum(0.0, 1.0 + a * np.cos(phi))

    # Butterfly pattern blends in below 10°
    b = np.clip((BUTTERFLY_EJECTA_ANGLE - impact_angle) / BUTTERFLY_EJECTA_ANGLE, 0.0, 1.0)
    w = (1.0 - b) * w + b * 2.0 * np.sin(phi) ** 2

    # Normalise to unit mean
    sample = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    w_sample = (1.0 - b) * np.maximum(0.0, 1.0 + a * np.cos(sample)) + b * 2.0 * np.sin(sample) ** 2
    return w / w_sample.mean()


def axis_ratio(impact_angle: float) -> float:
    """Minor/major axis ratio of the crater footprint (1 = circular)."""
    if impact_angle >= ELLIPTICAL_CRATER_ANGLE:
        return 1.0
    return _MIN_AXIS_RATIO + (1.0 - _MIN_AXIS_RATIO) * impact_angle / ELLIPTICAL_CRATER_ANGLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_asymmetric_elevation_field(
    crater_diameter: float,
    crater_depth: float,
    rim_height: float,
    impact_angle: float = 90.0,
    impact_azimuth: float = 0.0,
    width: int = TERRAIN_WIDTH,
    height: int = TERRAIN_HEIGHT,
    crater_type: str = "simple",
) -> dict:
    """
    Generate a synthetic 2-D elevation field with oblique-impact asymmetry.

    For impact_angle >= 45° the result is identical to the symmetric
    generator in ``crater_terrain.py``.

    Parameters
    ----------
    crater_diameter, crater_depth, rim_height : float
        Crater geometry in metres (from the scaling model).
    impact_angle : float
        Degrees from horizontal (0–90].  90° = vertical.
    impact_azimuth : float
        Direction of travel in degrees, 0° = north (top of grid), clockwise.
    width, height : int
        Grid dimensions.
    crater_type : str
        'simple', 'complex' or 'basin'.

    Returns
    -------
    dict
        Same format as ``crater_terrain.generate_elevation_field()`` plus
        ``is_asymmetric``, ``impact_angle``, ``impact_azimuth``.
    """
    # --- Input guard --------------------------------------------------------
    if crater_diameter <= 0:
        raise ValueError(f"crater_diameter must be > 0, got {crater_diameter}")
    if crater_depth <= 0:
        raise ValueError(f"crater_depth must be > 0, got {crater_depth}")
    if rim_height <= 0:
        raise ValueError(f"rim_height must be > 0, got {rim_height}")
    if impact_angle <= 0 or impact_angle > 90:
        raise ValueError(
            f"impact_angle must be in (0, 90], got {impact_angle}"
        )
    if width < 2 or height < 2:
        raise ValueError(
            f"width and height must be >= 2, got {width}×{height}"
        )

    t_start = time.perf_counter()

    radius = crater_diameter / 2.0
    ratio = axis_ratio(impact_angle)
    # Keep the footprint area equal to the circular crater's
    major = radius / math.sqrt(ratio)
    minor = radius * math.sqrt(ratio)
    half_domain = major * DOMAIN_HALF_FACTOR

    x_axis = np.linspace(-half_domain, half_domain, width)
    y_axis = np.linspace(-half_domain, half_domain, height)
    xg, yg = np.meshgrid(x_axis, y_axis)

    # Rotate into the impact frame: yr along the direction of travel
    azimuth_rad = math.radians(impact_azimuth)
    sin_az, cos_az = math.sin(azimuth_rad), math.cos(azimuth_rad)
    xr = xg * cos_az - yg * sin_az
    yr = xg * sin_az + yg * cos_az

    s = np.sqrt((xr / minor) ** 2 + (yr / major) ** 2)   # normalised elliptical radius
    phi = np.arctan2(xr, yr)                             # 0 = downrange

    z = radial_elevation(s, crater_depth, rim_height, crater_type)

    if impact_angle < ASYMMETRIC_EJECTA_ANGLE:
        outside = s > 1.0
        w = ejecta_weight(phi[outside], impact_angle)
        # Blend from the full rim crest at s = 1 into the ejecta pattern
        blend = np.exp(-((s[outside] - 1.0) / _RIM_BLEND_WIDTH) ** 2)
        z[outside] *= blend + (1.0 - blend) * w

    z = z.astype(np.float32)
    z_min = float(z.min())
    z_max = float(z.max())
    gen_time_ms = round((time.perf_counter() - t_start) * 1000, 3)

    return {
        "x":              x_axis.round(3).tolist(),
        "y":              y_axis.round(3).tolist(),
        "z":              z.round(4).tolist(),
        "z_min":          round(z_min, 4),
        "z_max":          round(z_max, 4),
        "width":          width,
        "height":         height,
        "gen_time_ms":    gen_time_ms,
        "is_asymmetric":  impact_angle < ASYMMETRIC_EJECTA_ANGLE,
        "impact_angle":   impact_angle,
        "impact_azimuth": impact_azimuth,
    }
