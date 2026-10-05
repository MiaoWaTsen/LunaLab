"""
Crater Scaling Model — Simulation Module

This module implements a simplified crater scaling law for educational
and prototype purposes. It is decoupled from the web API and database
so it can be tested, replaced, or extended independently.

SCIENTIFIC DISCLAIMER:
    This is a simplified model based on the pi-scaling framework
    (Schmidt & Housen, 1987; Holsapple, 1993). The implementation
    uses approximate coefficients suitable for demonstration.
    It is NOT a validated physical simulation of lunar impact processes.

References (simplified from):
    - Schmidt, R.M. & Housen, K.R. (1987). "Some recent advances in
      the scaling of impact and explosion cratering."
    - Holsapple, K.A. (1993). "The scaling of impact processes in
      planetary sciences."
"""

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Constants & defaults for a lunar-like target
# ---------------------------------------------------------------------------

LUNAR_GRAVITY = 1.62           # m/s² — surface gravity of the Moon
LUNAR_TARGET_DENSITY = 2500.0  # kg/m³ — approximate regolith bulk density
DEFAULT_IMPACTOR_DENSITY = 3000.0  # kg/m³ — stony asteroid default


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ImpactParameters:
    """Input parameters for a single impact simulation."""
    impactor_diameter: float        # metres
    impact_velocity: float          # m/s
    impact_angle: float             # degrees from vertical (0–90)
    impactor_density: float = DEFAULT_IMPACTOR_DENSITY   # kg/m³
    target_density: float = LUNAR_TARGET_DENSITY         # kg/m³
    surface_gravity: float = LUNAR_GRAVITY               # m/s²


@dataclass
class SimulationResult:
    """Output of a crater‑scaling calculation."""
    crater_diameter: float          # metres
    crater_depth: float             # metres
    rim_height: float               # metres
    ejecta_volume: float            # m³
    transient_crater_diameter: float  # metres
    energy_joules: float            # impact kinetic energy
    model_name: str                 # identifier for the scaling model
    warnings: list                  # any warnings about input ranges


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when input parameters are physically invalid."""
    pass


class ParameterWarning:
    """Lightweight container for non-fatal warnings."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message

    def to_dict(self):
        return {"field": self.field, "message": self.message}


def validate_parameters(params: ImpactParameters) -> list[ParameterWarning]:
    """
    Validate impact parameters.

    Raises ValidationError for physically impossible values.
    Returns a list of ParameterWarning for values outside the
    comfortable range of the simplified model.
    """
    # --- Hard errors ---
    if params.impactor_diameter <= 0:
        raise ValidationError("impactor_diameter must be > 0.")
    if params.impact_velocity <= 0:
        raise ValidationError("impact_velocity must be > 0.")
    if params.impact_angle <= 0 or params.impact_angle > 90:
        raise ValidationError("impact_angle must be in the range (0, 90].")
    if params.impactor_density <= 0:
        raise ValidationError("impactor_density must be > 0.")
    if params.target_density <= 0:
        raise ValidationError("target_density must be > 0.")
    if params.surface_gravity <= 0:
        raise ValidationError("surface_gravity must be > 0.")

    # --- Soft warnings ---
    warnings: list[ParameterWarning] = []

    if params.impactor_diameter > 100_000:
        warnings.append(ParameterWarning(
            "impactor_diameter",
            "Impactor diameter > 100 km — this may be outside the "
            "validated range of the current model."
        ))
    if params.impactor_diameter < 0.01:
        warnings.append(ParameterWarning(
            "impactor_diameter",
            "Impactor diameter < 0.01 m — micro-impacts may involve "
            "different physics (e.g. strength-dominated regime)."
        ))
    if params.impact_velocity > 100_000:
        warnings.append(ParameterWarning(
            "impact_velocity",
            "Impact velocity > 100 km/s — relativistic or extreme "
            "velocity effects are not modelled."
        ))
    if params.impact_velocity < 1_000:
        warnings.append(ParameterWarning(
            "impact_velocity",
            "Impact velocity < 1 km/s — low-velocity impacts may be "
            "in the strength-dominated regime."
        ))
    if params.impact_angle < 10:
        warnings.append(ParameterWarning(
            "impact_angle",
            "Very oblique impacts (< 10°) involve complex asymmetric "
            "cratering not captured by this simplified model."
        ))

    return warnings


# ---------------------------------------------------------------------------
# Core crater scaling calculation
# ---------------------------------------------------------------------------

MODEL_NAME = "Pi-Scaling Prototype Model v0.1"


def run_simulation(params: ImpactParameters) -> SimulationResult:
    """
    Run a simplified crater-scaling calculation.

    The model uses a pi-scaling approach (gravity regime) with
    approximate exponents.  The angle dependence uses the widely
    adopted sin(θ) scaling for the vertical component of velocity.

    Steps:
        1. Compute impactor kinetic energy.
        2. Compute the transient crater diameter using pi-scaling.
        3. Apply angle correction.
        4. Estimate final crater diameter (≈ 1.2 × transient).
        5. Estimate depth and rim height from geometric ratios.

    Returns a SimulationResult dataclass.
    """
    # 1. Validate
    warnings = validate_parameters(params)

    # 2. Derived quantities
    impactor_radius = params.impactor_diameter / 2.0
    impactor_volume = (4.0 / 3.0) * math.pi * impactor_radius ** 3  # m³
    impactor_mass = impactor_volume * params.impactor_density         # kg

    # Kinetic energy  (½mv²)
    energy = 0.5 * impactor_mass * params.impact_velocity ** 2       # Joules

    # 3. Pi-scaling — transient crater diameter (gravity regime)
    #
    #    D_tc = K₁ · a · (ρ_i / ρ_t)^(1/3) · (v² / (g · a))^μ
    #
    #    where a = impactor radius, K₁ ≈ 1.03, μ ≈ 0.22
    #    (approximate values for competent rock / regolith target)

    K1 = 1.03
    mu = 0.22

    density_ratio = params.impactor_density / params.target_density
    gravity_term = (params.impact_velocity ** 2) / (
        params.surface_gravity * impactor_radius
    )

    transient_diameter = (
        K1
        * params.impactor_diameter
        * (density_ratio ** (1.0 / 3.0))
        * (gravity_term ** mu)
    )

    # 4. Angle correction — sin(θ) scaling on velocity
    angle_rad = math.radians(params.impact_angle)
    angle_factor = math.sin(angle_rad) ** (2.0 * mu)
    transient_diameter *= angle_factor

    # 5. Final (simple) crater diameter ≈ 1.2 × transient
    final_diameter = 1.2 * transient_diameter

    # 6. Depth and rim from typical ratios
    #    Simple craters: depth ≈ D / 5,  rim_height ≈ D / 20
    crater_depth = final_diameter / 5.0
    rim_height = final_diameter / 20.0

    # 7. Ejecta volume (rough hemisphere bowl approximation)
    crater_radius = final_diameter / 2.0
    ejecta_volume = (2.0 / 3.0) * math.pi * crater_radius ** 2 * crater_depth

    return SimulationResult(
        crater_diameter=round(final_diameter, 4),
        crater_depth=round(crater_depth, 4),
        rim_height=round(rim_height, 4),
        ejecta_volume=round(ejecta_volume, 2),
        transient_crater_diameter=round(transient_diameter, 4),
        energy_joules=round(energy, 2),
        model_name=MODEL_NAME,
        warnings=[w.to_dict() for w in warnings],
    )


# ---------------------------------------------------------------------------
# Crater profile generation (for visualization)
# ---------------------------------------------------------------------------

def generate_crater_profile(
    crater_diameter: float,
    crater_depth: float,
    rim_height: float,
    num_points: int = 200,
) -> dict:
    """
    Generate (x, y) arrays for a simplified crater cross-section.

    The profile is symmetric: a raised rim, parabolic bowl,
    and flat surrounding terrain at y = 0.

    Returns {"x": [...], "y": [...]}.
    """
    radius = crater_diameter / 2.0
    rim_width = radius * 0.3          # rim extends ~30% beyond the radius
    total_half_width = radius + rim_width * 2.5

    xs: list[float] = []
    ys: list[float] = []

    for i in range(num_points):
        x = -total_half_width + (2 * total_half_width) * i / (num_points - 1)
        abs_x = abs(x)

        if abs_x <= radius:
            # Parabolic bowl: y = -depth · (1 - (x/R)²)
            y = -crater_depth * (1.0 - (abs_x / radius) ** 2)
        elif abs_x <= radius + rim_width:
            # Rim bump — Gaussian-ish raised lip
            t = (abs_x - radius) / rim_width
            y = rim_height * math.exp(-3.0 * t ** 2)
        else:
            # Flat terrain
            y = 0.0

        xs.append(round(x, 4))
        ys.append(round(y, 4))

    return {"x": xs, "y": ys}


# ---------------------------------------------------------------------------
# Synthetic 2-D elevation field (for terrain map visualization)
# ---------------------------------------------------------------------------

def generate_elevation_field(
    crater_diameter: float,
    crater_depth: float,
    rim_height: float,
    grid_size: int = 80,
) -> dict:
    """
    Build a square 2-D elevation grid centred on the crater.

    The grid covers ±(2.5 × crater_radius) in both X and Y.
    Elevation is computed radially using the same geometry as the
    cross-section profile, giving a rotationally symmetric synthetic
    terrain surface suitable for a Plotly heatmap / surface plot.

    Parameters
    ----------
    crater_diameter : float
        Final crater diameter in metres.
    crater_depth : float
        Maximum crater depth in metres (positive value).
    rim_height : float
        Height of the raised rim above the surrounding terrain.
    grid_size : int
        Number of grid cells per axis (grid_size × grid_size matrix).
        Default 80 gives good visual resolution without heavy data transfer.

    Returns
    -------
    dict with keys:
        "x"  – 1-D list of x-axis coordinate values (length = grid_size)
        "y"  – 1-D list of y-axis coordinate values (length = grid_size)
        "z"  – 2-D list (grid_size rows × grid_size cols) of elevation values
        "z_min", "z_max"  – scalar range for colour-scale normalisation
    """
    radius = crater_diameter / 2.0
    rim_width = radius * 0.3
    # Domain: 2.5× crater radius on each side
    half_domain = radius * 2.5

    # Build 1-D axis arrays
    axis: list[float] = []
    for i in range(grid_size):
        v = -half_domain + (2.0 * half_domain) * i / (grid_size - 1)
        axis.append(round(v, 3))

    # Ejecta blanket: elevation decays smoothly from rim outward
    ejecta_outer = radius + rim_width * 4.0   # where blanket fades to zero

    z_grid: list[list[float]] = []
    z_min = 0.0
    z_max = 0.0

    for yi in axis:
        row: list[float] = []
        for xi in axis:
            r = math.sqrt(xi * xi + yi * yi)   # radial distance from centre

            if r <= radius:
                # Parabolic bowl interior
                z = -crater_depth * (1.0 - (r / radius) ** 2)
            elif r <= radius + rim_width:
                # Gaussian rim
                t = (r - radius) / rim_width
                z = rim_height * math.exp(-3.0 * t * t)
            elif r <= ejecta_outer:
                # Ejecta blanket — power-law decay
                t = (r - (radius + rim_width)) / (ejecta_outer - radius - rim_width)
                ejecta_height = rim_height * 0.25   # blanket peak ≈ 25% rim height
                z = ejecta_height * (1.0 - t) ** 2
            else:
                # Undisturbed flat terrain
                z = 0.0

            z = round(z, 4)
            row.append(z)
            if z < z_min:
                z_min = z
            if z > z_max:
                z_max = z

        z_grid.append(row)

    return {
        "x": axis,
        "y": axis,
        "z": z_grid,
        "z_min": round(z_min, 4),
        "z_max": round(z_max, 4),
    }
