"""
Asymmetric Crater Terrain Generator — terrain/asymmetric_crater.py

SCIENTIFIC DISCLAIMER
======================
This module generates a *synthetic geometric approximation* of
asymmetric (oblique-impact) crater morphology for **visualization
purposes only**.

The directional deformations applied here are NOT the output of a
hydrocode, SPH simulation, or validated geophysical model.  They are
simplified mathematical transformations that produce visually plausible
directional asymmetry based on well-known qualitative observations:

    1. Oblique impacts produce elongated crater footprints
       (Gault & Wedekind, 1978)
    2. Rim height is enhanced uprange of the impact direction
    3. Ejecta distribution is biased downrange
    4. Crater depth may vary with azimuthal direction

These effects are approximated using:
    - Elliptical footprint scaling
    - Cosine-weighted directional modulation
    - Anisotropic ejecta decay

The module takes crater dimensions from ``simulation/crater_model.py``
(which implements the Pi-scaling framework) and generates a 2-D
elevation grid.  It does NOT modify the scaling model itself.

Relationship to Other Modules
------------------------------
    simulation/crater_model.py  → crater dimensions (science)
    terrain/crater_terrain.py   → symmetric elevation grid (existing)
    terrain/asymmetric_crater.py → THIS: asymmetric elevation grid (new)

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
    _RIM_WIDTH_FACTOR,
    _EJECTA_RANGE_FACTOR,
    _EJECTA_PEAK_RATIO,
    _DOMAIN_HALF_FACTOR,
)


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
) -> dict:
    """
    Generate a synthetic 2-D elevation field with directional asymmetry.

    When ``impact_angle == 90`` (vertical), this produces results
    identical to the symmetric generator in ``crater_terrain.py``.

    When ``impact_angle < 90``, directional deformations are applied:
        - Elliptical crater footprint (major axis along impact direction)
        - Rim height enhanced uprange
        - Crater depth shifted downrange
        - Ejecta preferentially distributed downrange

    IMPORTANT: This is a synthetic geometric approximation for
    visualization, NOT a physical simulation of oblique impact
    cratering processes.

    Parameters
    ----------
    crater_diameter : float
        Final crater diameter in metres (from the scaling model).
    crater_depth : float
        Maximum crater depth in metres (positive value).
    rim_height : float
        Height of the raised rim above the surrounding terrain.
    impact_angle : float
        Impact angle in degrees from vertical (0–90].
        90° = vertical (symmetric), lower = more oblique.
    impact_azimuth : float
        Direction of impact in degrees.  0° = from the north (top of grid),
        measured clockwise.  90° = from the east.
    width : int
        Number of columns in the output grid.
    height : int
        Number of rows in the output grid.

    Returns
    -------
    dict
        Same format as ``crater_terrain.generate_elevation_field()``:
        ``x``, ``y``, ``z``, ``z_min``, ``z_max``, ``width``, ``height``,
        ``gen_time_ms``, plus additional metadata:
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
    rim_width = radius * _RIM_WIDTH_FACTOR
    half_domain = radius * _DOMAIN_HALF_FACTOR
    ejecta_outer = radius + rim_width * _EJECTA_RANGE_FACTOR

    # ── Asymmetry parameters ──────────────────────────────────────────
    angle_rad = math.radians(impact_angle)
    azimuth_rad = math.radians(impact_azimuth)

    # Obliquity factor: 0 at 90° (vertical), approaches 1 at very oblique
    obliquity = 1.0 - math.sin(angle_rad)

    # Ellipticity: ratio of minor/major axis
    # At 90°: 1.0 (circle).  At 45°: ~0.85.  At 15°: ~0.6
    ellipticity = math.sin(angle_rad) ** 0.5
    ellipticity = max(ellipticity, 0.4)  # clamp to prevent degenerate shapes

    # Directional strength factors
    depth_asym_strength = 0.3 * obliquity     # depth shift downrange
    rim_asym_strength = 0.4 * obliquity       # rim height uprange bias
    ejecta_asym_strength = 0.5 * obliquity    # ejecta downrange bias

    # ── Build coordinate axes ─────────────────────────────────────────
    x_axis = np.linspace(-half_domain, half_domain, width)
    y_axis = np.linspace(-half_domain, half_domain, height)
    xg, yg = np.meshgrid(x_axis, y_axis)

    # ── Azimuthal angle of each grid point relative to centre ─────────
    # theta[i,j] = angle from impact direction for point (xg, yg)
    grid_angle = np.arctan2(xg, yg)  # angle from +Y axis (north), CW
    relative_angle = grid_angle - azimuth_rad

    # ── Elliptical radial distance ────────────────────────────────────
    # Rotate grid so that impact direction aligns with y-axis,
    # then apply elliptical scaling
    cos_az = math.cos(azimuth_rad)
    sin_az = math.sin(azimuth_rad)

    # Rotated coordinates (impact direction → +Y)
    xr = xg * cos_az - yg * sin_az
    yr = xg * sin_az + yg * cos_az

    # Elliptical distance: stretch perpendicular to impact direction
    r_ellip = np.sqrt((xr / ellipticity) ** 2 + yr ** 2)

    # Original radial distance (for fallback / blending)
    r = np.sqrt(xg ** 2 + yg ** 2)

    # ── Directional weighting ─────────────────────────────────────────
    # cos_dir > 0 = downrange (in impact direction)
    # cos_dir < 0 = uprange  (opposite to impact direction)
    cos_dir = np.cos(relative_angle)

    # ── Allocate output array ─────────────────────────────────────────
    z = np.zeros((height, width), dtype=np.float32)

    # ── Zone 1: Parabolic crater bowl (with asymmetry) ────────────────
    bowl_mask = r_ellip <= radius

    # Base parabolic profile
    bowl_depth = -crater_depth * (1.0 - (r_ellip[bowl_mask] / radius) ** 2)

    # Depth asymmetry: deeper downrange
    depth_mod = 1.0 + depth_asym_strength * cos_dir[bowl_mask]
    z[bowl_mask] = bowl_depth * depth_mod

    # ── Zone 2: Gaussian rim (with asymmetry) ─────────────────────────
    rim_outer = radius + rim_width
    rim_mask = (r_ellip > radius) & (r_ellip <= rim_outer)
    t_rim = (r_ellip[rim_mask] - radius) / rim_width

    # Rim height asymmetry: higher uprange (opposite to impact direction)
    rim_mod = 1.0 - rim_asym_strength * cos_dir[rim_mask]
    z[rim_mask] = rim_height * rim_mod * np.exp(-3.0 * t_rim ** 2)

    # ── Zone 3: Power-law ejecta blanket (with asymmetry) ─────────────
    ejecta_mask = (r_ellip > rim_outer) & (r_ellip <= ejecta_outer)
    t_ej = (r_ellip[ejecta_mask] - rim_outer) / (ejecta_outer - rim_outer)
    ejecta_peak = rim_height * _EJECTA_PEAK_RATIO

    # Ejecta asymmetry: more ejecta downrange
    ejecta_mod = 1.0 + ejecta_asym_strength * cos_dir[ejecta_mask]
    z[ejecta_mask] = ejecta_peak * ejecta_mod * (1.0 - t_ej) ** 2

    # Zone 4: Undisturbed terrain — already zero

    z_min = float(z.min())
    z_max = float(z.max())

    t_end = time.perf_counter()
    gen_time_ms = round((t_end - t_start) * 1000, 3)

    return {
        "x":              x_axis.round(3).tolist(),
        "y":              y_axis.round(3).tolist(),
        "z":              z.round(4).tolist(),
        "z_min":          round(z_min, 4),
        "z_max":          round(z_max, 4),
        "width":          width,
        "height":         height,
        "gen_time_ms":    gen_time_ms,
        "is_asymmetric":  impact_angle < 89.5,
        "impact_angle":   impact_angle,
        "impact_azimuth": impact_azimuth,
    }
