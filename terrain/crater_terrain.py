"""
Synthetic Crater Terrain Generator — terrain/crater_terrain.py

IMPORTANT SCIENTIFIC DISCLAIMER
================================
This module generates a *synthetic* elevation field for **visualization
purposes only**.  The output is NOT a real lunar Digital Elevation Model
(DEM) and does NOT represent actual lunar topography.

The terrain is constructed analytically from three simplified geometric
zones derived from the crater scaling results:

    1. Parabolic bowl  — interior of the crater cavity
    2. Gaussian rim    — raised lip around the crater edge
    3. Power-law ejecta blanket — material deposited beyond the rim

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
# Terrain geometry parameters (relative to crater radius)
# ---------------------------------------------------------------------------

_RIM_WIDTH_FACTOR    = 0.30   # rim extends 30% of radius beyond the edge
_EJECTA_RANGE_FACTOR = 4.00   # ejecta blanket fades at 4× rim_width
_EJECTA_PEAK_RATIO   = 0.25   # ejecta peak height = 25% of rim height
_DOMAIN_HALF_FACTOR  = 2.50   # domain spans ±2.5 × crater radius


# ---------------------------------------------------------------------------
# Core terrain generator
# ---------------------------------------------------------------------------

def generate_elevation_field(
    crater_diameter: float,
    crater_depth: float,
    rim_height: float,
    width: int = TERRAIN_WIDTH,
    height: int = TERRAIN_HEIGHT,
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

    radius    = crater_diameter / 2.0
    rim_width = radius * _RIM_WIDTH_FACTOR
    half_domain = radius * _DOMAIN_HALF_FACTOR

    # Ejecta blanket outer boundary
    ejecta_outer = radius + rim_width * _EJECTA_RANGE_FACTOR

    # ── Build coordinate axes (NumPy) ──────────────────────────────────────
    x_axis = np.linspace(-half_domain, half_domain, width)    # (width,)
    y_axis = np.linspace(-half_domain, half_domain, height)   # (height,)

    # ── Broadcast to 2-D radial distance grid ─────────────────────────────
    # xg[i, j] = x_axis[j],  yg[i, j] = y_axis[i]
    xg, yg = np.meshgrid(x_axis, y_axis)   # both shape: (height, width)
    r = np.sqrt(xg ** 2 + yg ** 2)         # radial distance from centre

    # ── Allocate output array ──────────────────────────────────────────────
    z = np.zeros((height, width), dtype=np.float32)

    # ── Zone 1: Parabolic crater bowl ─────────────────────────────────────
    bowl_mask = r <= radius
    z[bowl_mask] = -crater_depth * (1.0 - (r[bowl_mask] / radius) ** 2)

    # ── Zone 2: Gaussian rim ──────────────────────────────────────────────
    rim_outer = radius + rim_width
    rim_mask  = (r > radius) & (r <= rim_outer)
    t_rim     = (r[rim_mask] - radius) / rim_width
    z[rim_mask] = rim_height * np.exp(-3.0 * t_rim ** 2)

    # ── Zone 3: Power-law ejecta blanket ─────────────────────────────────
    ejecta_mask = (r > rim_outer) & (r <= ejecta_outer)
    t_ej = (r[ejecta_mask] - rim_outer) / (ejecta_outer - rim_outer)
    ejecta_peak = rim_height * _EJECTA_PEAK_RATIO
    z[ejecta_mask] = ejecta_peak * (1.0 - t_ej) ** 2

    # Zone 4: Undisturbed terrain — already zero from np.zeros

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
