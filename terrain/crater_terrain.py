"""
Synthetic Crater Terrain Generator — terrain/crater_terrain.py

IMPORTANT SCIENTIFIC DISCLAIMER
================================
This module generates a *synthetic* elevation field for **visualization
purposes only**.  The output is NOT a real lunar Digital Elevation Model
(DEM) and does NOT represent actual lunar topography.

The terrain is constructed analytically from the crater scaling results:

    1. Interior — parabolic bowl (simple craters), or flat floor with a
       central peak (complex craters) / peak ring (basins)
    2. Rim crest at the crater radius
    3. Ejecta blanket — thickness ∝ (r/R)^-3 (McGetchin et al. 1973),
       tapered to zero at the edge of the domain

These shapes are geometric approximations intended to give a plausible
visual impression of a crater morphology.  They are *not* the output of
a hydrodynamic code, SPH simulation, or validated geophysical model.

Relationship to crater_model.py
--------------------------------
``simulation/crater_model.py`` is responsible for the *Crater Scaling
Model*: it takes impactor parameters and returns physical crater
characteristics (diameter, depth, rim height, energy …) using the
pi-scaling framework (Schmidt & Housen 1987; Holsapple 1993).

This module (``terrain/crater_terrain.py``) is responsible for the
*Synthetic Terrain Model*: it takes those physical crater characteristics
and converts them into a 2-D elevation grid suitable for Plotly heatmap,
3-D surface, or contour visualizations.

Separation rationale
---------------------
Terrain generation is NOT crater science.  Keeping it in a separate
module means:

* ``crater_model.py`` stays focused on physics — it can be replaced with
  a different scaling law without touching visualization code.
* ``crater_terrain.py`` can be updated (e.g. to add noise, asymmetry, or
  multiple craters) without touching the science module.
* Both modules are independently testable without a web server or database.

Dependencies
------------
* numpy — required for vectorized grid computation (see rationale below).

NumPy rationale
---------------
A 1024 × 512 = 524,288-cell grid computed with pure-Python nested loops
takes roughly 3–5 seconds on a modern laptop.  NumPy vectorization
reduces this to < 50 ms by operating on entire arrays at once using
compiled C kernels, with no GPU required.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Public constants — canonical grid dimensions
# ---------------------------------------------------------------------------

TERRAIN_WIDTH  = 1024   # number of columns (x-axis samples)
TERRAIN_HEIGHT = 512    # number of rows    (y-axis samples)
# Resulting z array shape: (TERRAIN_HEIGHT, TERRAIN_WIDTH) = (512, 1024)


# ---------------------------------------------------------------------------
# Terrain geometry parameters (relative to crater radius R)
# ---------------------------------------------------------------------------

DOMAIN_HALF_FACTOR = 2.50   # domain spans ±2.5 R

# Complex-crater interior (illustrative values for fresh lunar craters)
_COMPLEX_FLOOR_FRACTION = 0.40   # flat floor radius / R
_CENTRAL_PEAK_HEIGHT    = 0.30   # central peak height / depth
_CENTRAL_PEAK_WIDTH     = 0.12   # central peak half-width / R

# Basin interior
_BASIN_FLOOR_FRACTION = 0.60
_PEAK_RING_RADIUS     = 0.45     # peak ring radius / R
_PEAK_RING_HEIGHT     = 0.20     # peak ring height / depth
_PEAK_RING_WIDTH      = 0.06


def radial_elevation(s, crater_depth: float, rim_height: float, crater_type: str = "simple"):
    """
    Axisymmetric crater elevation as a function of normalised radius.

    Parameters
    ----------
    s : array-like
        Radial distance divided by the crater radius (r / R), >= 0.
    crater_depth, rim_height : float
        Floor depth below and rim height above the pre-impact surface (m).
    crater_type : str
        'simple', 'complex' or 'basin'.

    Returns
    -------
    numpy.ndarray of elevations (m), same shape as *s*.  The profile is
    continuous: -depth at the floor, +rim_height at s = 1, and 0 at
    s = DOMAIN_HALF_FACTOR.
    """
    s = np.asarray(s, dtype=np.float64)
    d, h = crater_depth, rim_height
    z = np.zeros_like(s)

    inside = s <= 1.0
    si = s[inside]
    if crater_type == "simple":
        # Paraboloid bowl rising from the floor to the rim crest
        zi = -d + (d + h) * si ** 2
    else:
        floor = _BASIN_FLOOR_FRACTION if crater_type == "basin" else _COMPLEX_FLOOR_FRACTION
        t = np.clip((si - floor) / (1.0 - floor), 0.0, 1.0)
        zi = -d + (d + h) * t ** 2          # flat floor, terraced-looking wall
        if crater_type == "basin":
            zi += _PEAK_RING_HEIGHT * d * np.exp(
                -((si - _PEAK_RING_RADIUS) / _PEAK_RING_WIDTH) ** 2
            )
        else:
            zi += _CENTRAL_PEAK_HEIGHT * d * np.exp(-(si / _CENTRAL_PEAK_WIDTH) ** 2)
    z[inside] = zi

    # Ejecta blanket: (r/R)^-3 decay, tapered to zero at the domain edge
    outside = ~inside
    so = s[outside]
    taper = np.clip(1.0 - (so - 1.0) / (DOMAIN_HALF_FACTOR - 1.0), 0.0, 1.0) ** 2
    z[outside] = h * so ** -3.0 * taper
    return z


# ---------------------------------------------------------------------------
# Core terrain generator
# ---------------------------------------------------------------------------

def generate_elevation_field(
    crater_diameter: float,
    crater_depth: float,
    rim_height: float,
    width: int = TERRAIN_WIDTH,
    height: int = TERRAIN_HEIGHT,
    crater_type: str = "simple",
) -> dict:
    """
    Generate a synthetic 2-D elevation field from crater geometry.

    This function converts the output of the crater scaling model
    (diameter, depth, rim height) into a rectangular elevation grid
    suitable for interactive visualization.  The terrain is NOT a
    real DEM — see module docstring for the full scientific disclaimer.

    The grid is rectangular (width × height) rather than square so the
    x-axis can carry more detail at the same data volume.

    Parameters
    ----------
    crater_diameter : float
        Final crater diameter in metres (from the scaling model).
    crater_depth : float
        Maximum crater depth in metres (positive value).
    rim_height : float
        Height of the raised rim above the surrounding terrain, in metres.
    width : int
        Number of columns in the output grid (default: 1024).
    height : int
        Number of rows in the output grid (default: 512).
    crater_type : str
        'simple', 'complex' or 'basin' — selects the interior morphology.

    Returns
    -------
    dict
        ``x``       — list[float] of length *width*.  X-axis coordinates (m).
        ``y``       — list[float] of length *height*. Y-axis coordinates (m).
        ``z``       — list[list[float]] of shape (height, width).
                      Row-major: ``z[row][col]`` gives elevation at
                      ``(x[col], y[row])``.
        ``z_min``   — float. Minimum elevation (crater floor).
        ``z_max``   — float. Maximum elevation (rim peak).
        ``width``   — int.  Number of columns.
        ``height``  — int.  Number of rows.
        ``gen_time_ms`` — float. Wall-clock generation time in milliseconds.

    Raises
    ------
    ValueError
        If any of the geometric parameters are non-positive, or if
        width / height are less than 2.
    """
    # --- Input guard --------------------------------------------------------
    if crater_diameter <= 0:
        raise ValueError(f"crater_diameter must be > 0, got {crater_diameter}")
    if crater_depth <= 0:
        raise ValueError(f"crater_depth must be > 0, got {crater_depth}")
    if rim_height <= 0:
        raise ValueError(f"rim_height must be > 0, got {rim_height}")
    if width < 2 or height < 2:
        raise ValueError(f"width and height must be >= 2, got {width}×{height}")

    t_start = time.perf_counter()

    radius      = crater_diameter / 2.0
    half_domain = radius * DOMAIN_HALF_FACTOR

    # ── Build coordinate axes (NumPy) ──────────────────────────────────────
    x_axis = np.linspace(-half_domain, half_domain, width)    # (width,)
    y_axis = np.linspace(-half_domain, half_domain, height)   # (height,)

    # ── Broadcast to 2-D radial distance grid ─────────────────────────────
    # xg[i, j] = x_axis[j],  yg[i, j] = y_axis[i]
    xg, yg = np.meshgrid(x_axis, y_axis)   # both shape: (height, width)
    r = np.sqrt(xg ** 2 + yg ** 2)         # radial distance from centre

    z = radial_elevation(r / radius, crater_depth, rim_height, crater_type).astype(np.float32)

    z_min = float(z.min())
    z_max = float(z.max())

    t_end = time.perf_counter()
    gen_time_ms = round((t_end - t_start) * 1000, 3)

    return {
        "x":          x_axis.round(3).tolist(),
        "y":          y_axis.round(3).tolist(),
        "z":          z.round(4).tolist(),    # list[list[float]], shape (height, width)
        "z_min":      round(z_min, 4),
        "z_max":      round(z_max, 4),
        "width":      width,
        "height":     height,
        "gen_time_ms": gen_time_ms,
    }
