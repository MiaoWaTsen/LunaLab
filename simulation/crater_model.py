"""
Crater Scaling Model — Simulation Module

This module estimates lunar impact crater dimensions from impactor
parameters.  It is decoupled from the web API and database so it can be
tested, replaced, or extended independently.

Model summary
-------------
1. Transient crater (gravity regime, pi-scaling), Collins, Melosh &
   Marcus (2005), eq. 21:

       D_tc = 1.161 (ρ_i/ρ_t)^(1/3) L^0.78 v^0.44 g^-0.22 sin^(1/3)θ

   Its exponents correspond to μ ≈ 0.55: a competent, non-porous target
   (target_material="rock", the default).

   target_material="regolith" uses Holsapple (1993) point-source scaling
   with porous-regolith parameters instead (μ = 0.4, ν = 0.333,
   K1 = 0.2, K2 = 0.75, Kr = 1.1, Y = 1 kPa, as applied to lunar impacts
   by Madiedo et al. 2019):

       π_V = K1 [ π2 (ρ_t/ρ_i)^((6ν-2-μ)/3μ)
                  + (K2 π3 (ρ_t/ρ_i)^((6ν-2)/3μ))^((2+μ)/2) ]^(-3μ/(2+μ))
       π2 = g a / U²,  π3 = Y / (ρ_t U²),  U = v sinθ,  a = impactor radius
       D  = 2 Kr (m π_V / ρ_t)^(1/3)          (rim-to-rim)

   Energy lost to compacting the pores makes these craters about half
   the size of the "rock" ones.  Validated against two observed lunar
   impacts in tests/test_real_impacts.py.

2. Simple → complex transition at D_c ≈ 15 km on the Moon (Pike 1977),
   scaled inversely with surface gravity for other bodies.

3. Final rim-to-rim diameter (Collins et al. 2005, eqs. 22 & 27):
       simple:  D = 1.25 D_tc
       complex: D = 1.17 D_tc^1.13 / D_c^0.13
   The larger of the two is used, which keeps D monotonic in D_tc
   across the transition.

4. Depth and rim height from fresh lunar crater morphometry (Pike 1977):
       simple:  d ≈ 0.2 D,  h ≈ 0.036 D
       complex: d ∝ D^0.301, h ∝ D^0.399  (anchored at D_c for continuity)

5. Excavated (transient cavity) volume, Collins et al. (2005):
       V_tc ≈ π D_tc³ / (16 √2)

SCIENTIFIC DISCLAIMER:
    This is a simplified, educational model.  The default "rock" scaling
    ignores the porosity of the lunar regolith (which overestimates small
    craters about 2x, see tests/test_real_impacts.py) and target strength,
    and neither option models the detailed physics of very oblique
    impacts.  It is NOT a
    validated hydrocode simulation.

References:
    - Collins, G.S., Melosh, H.J. & Marcus, R.A. (2005). "Earth Impact
      Effects Program." Meteoritics & Planetary Science 40(6), 817–840.
    - Pike, R.J. (1977). "Size-dependence in the shape of fresh impact
      craters on the Moon." In Impact and Explosion Cratering, 489–509.
    - Holsapple, K.A. (1993). "The scaling of impact processes in
      planetary sciences." Annu. Rev. Earth Planet. Sci. 21, 333–373.
    - Madiedo, J.M. et al. (2019). "Multiwavelength observations of a
      bright impact flash during the January 2019 total lunar eclipse."
      MNRAS 486, 3380–3387.
    - Wieczorek, M.A. et al. (2013). "The crust of the Moon as seen by
      GRAIL." Science 339, 671–675.
"""

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Constants & defaults for a lunar-like target
# ---------------------------------------------------------------------------

LUNAR_GRAVITY = 1.62           # m/s² — surface gravity of the Moon
LUNAR_TARGET_DENSITY = 2550.0  # kg/m³ — bulk crustal density from GRAIL (Wieczorek et al. 2013)
DEFAULT_IMPACTOR_DENSITY = 3000.0  # kg/m³ — stony asteroid default

LUNAR_ESCAPE_VELOCITY = 2380.0     # m/s — minimum natural impact speed on the Moon
MAX_SOLAR_SYSTEM_IMPACT_VELOCITY = 72_000.0  # m/s — head-on retrograde comet limit
LUNAR_DIAMETER = 3_474_800.0       # m

# Simple-to-complex transition diameter on the Moon (Pike 1977)
LUNAR_TRANSITION_DIAMETER = 15_000.0  # m
# Above this, lunar craters become peak-ring / multi-ring basins
LUNAR_BASIN_DIAMETER = 300_000.0      # m

TARGET_MATERIALS = ("rock", "regolith")

# Holsapple (1993) scaling parameters for porous lunar regolith, as used by
# Madiedo et al. (2019) for lunar impact flashes
REGOLITH_K1 = 0.2
REGOLITH_K2 = 0.75
REGOLITH_KR = 1.1
REGOLITH_MU = 0.4
REGOLITH_NU = 0.333
REGOLITH_STRENGTH = 1000.0  # Pa — cohesion of lunar regolith
# The regolith is typically 4-15 m thick (McKay et al. 1991, Lunar
# Sourcebook), and a crater excavates to about a tenth of its transient
# diameter, so transient craters wider than ~100 m dig into the bedrock below
# and the regolith scaling no longer applies.
REGOLITH_MAX_TRANSIENT_DIAMETER = 100.0  # m


def transition_diameter(surface_gravity: float = LUNAR_GRAVITY) -> float:
    """Simple-to-complex transition diameter (m), scaled as 1/g from the Moon."""
    # Ratio first: 15000 * 1.62 / 1.62 is 14999.999999999998 in floating point,
    # which would misclassify a crater exactly at the transition.
    return LUNAR_TRANSITION_DIAMETER * (LUNAR_GRAVITY / surface_gravity)


def classify_crater(crater_diameter: float, surface_gravity: float = LUNAR_GRAVITY) -> str:
    """Return 'simple', 'complex' or 'basin' for a final crater diameter (m)."""
    scale = LUNAR_GRAVITY / surface_gravity
    if crater_diameter >= LUNAR_BASIN_DIAMETER * scale:
        return "basin"
    if crater_diameter > transition_diameter(surface_gravity):
        return "complex"
    return "simple"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ImpactParameters:
    """Input parameters for a single impact simulation."""
    impactor_diameter: float        # metres
    impact_velocity: float          # m/s
    impact_angle: float             # degrees from horizontal (0–90], 90 = vertical
    impactor_density: float = DEFAULT_IMPACTOR_DENSITY   # kg/m³
    target_density: float = LUNAR_TARGET_DENSITY         # kg/m³
    surface_gravity: float = LUNAR_GRAVITY               # m/s²
    target_material: str = "rock"                         # 'rock' or 'regolith'


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
    crater_type: str = "simple"     # 'simple', 'complex' or 'basin'


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
    if params.target_material not in TARGET_MATERIALS:
        raise ValidationError(f"target_material must be one of {TARGET_MATERIALS}.")

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
    if params.impact_velocity > MAX_SOLAR_SYSTEM_IMPACT_VELOCITY:
        warnings.append(ParameterWarning(
            "impact_velocity",
            "Impact velocity > 72 km/s — faster than any natural Solar "
            "System impactor (head-on retrograde comet limit)."
        ))
    if params.impact_velocity < LUNAR_ESCAPE_VELOCITY:
        warnings.append(ParameterWarning(
            "impact_velocity",
            "Impact velocity < 2.38 km/s (lunar escape velocity) — natural "
            "impactors cannot hit the Moon this slowly; the scaling law "
            "assumes hypervelocity impacts."
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

MODEL_NAME = "Collins-Pike Lunar Scaling Model v0.2"
REGOLITH_MODEL_NAME = "Holsapple Regolith-Pike Lunar Scaling Model v0.2"


def regolith_rim_diameter(params: ImpactParameters, impactor_mass: float) -> float:
    """Rim-to-rim diameter (m) in porous regolith, Holsapple (1993) point-source scaling."""
    mu, nu = REGOLITH_MU, REGOLITH_NU
    rho_t = params.target_density
    density_ratio = rho_t / params.impactor_density
    vertical_velocity = params.impact_velocity * math.sin(math.radians(params.impact_angle))

    pi2 = params.surface_gravity * (params.impactor_diameter / 2.0) / vertical_velocity ** 2
    pi3 = REGOLITH_STRENGTH / (rho_t * vertical_velocity ** 2)
    gravity_term = pi2 * density_ratio ** ((6 * nu - 2 - mu) / (3 * mu))
    strength_term = (REGOLITH_K2 * pi3 * density_ratio ** ((6 * nu - 2) / (3 * mu))) ** ((2 + mu) / 2)
    pi_v = REGOLITH_K1 * (gravity_term + strength_term) ** (-3 * mu / (2 + mu))

    crater_volume = impactor_mass * pi_v / rho_t
    return 2.0 * REGOLITH_KR * crater_volume ** (1.0 / 3.0)


def run_simulation(params: ImpactParameters) -> SimulationResult:
    """
    Estimate crater dimensions for a single impact.

    Steps:
        1. Validate inputs and collect warnings.
        2. Compute impactor kinetic energy.
        3. Transient crater diameter (Collins et al. 2005).
        4. Final crater diameter with simple/complex collapse.
        5. Depth and rim height from lunar morphometry (Pike 1977).
        6. Excavated volume of the transient cavity.

    Returns a SimulationResult dataclass.
    """
    # 1. Validate
    warnings = validate_parameters(params)

    # 2. Kinetic energy (½mv²)
    impactor_radius = params.impactor_diameter / 2.0
    impactor_volume = (4.0 / 3.0) * math.pi * impactor_radius ** 3  # m³
    impactor_mass = impactor_volume * params.impactor_density         # kg
    energy = 0.5 * impactor_mass * params.impact_velocity ** 2       # J

    # 3. Transient crater diameter — gravity-regime pi-scaling.
    if params.target_material == "regolith":
        # Holsapple gives the rim-to-rim diameter; step 4 turns the
        # transient diameter back into it for simple craters.
        transient_diameter = regolith_rim_diameter(params, impactor_mass) / 1.25
        if transient_diameter > REGOLITH_MAX_TRANSIENT_DIAMETER:
            warnings.append(ParameterWarning(
                "target_material",
                "Crater > ~125 m digs through the regolith (typically 4-15 m "
                "thick) into bedrock; use target_material='rock' instead."
            ))
    else:
        # sin^(1/3)θ: only the vertical momentum component drives excavation.
        density_ratio = params.impactor_density / params.target_density
        angle_factor = math.sin(math.radians(params.impact_angle)) ** (1.0 / 3.0)
        transient_diameter = (
            1.161
            * density_ratio ** (1.0 / 3.0)
            * params.impactor_diameter ** 0.78
            * params.impact_velocity ** 0.44
            * params.surface_gravity ** -0.22
            * angle_factor
        )

    # 4. Final diameter — wall collapse widens the cavity.
    d_c = transition_diameter(params.surface_gravity)
    simple_diameter = 1.25 * transient_diameter
    complex_diameter = 1.17 * transient_diameter ** 1.13 / d_c ** 0.13
    final_diameter = max(simple_diameter, complex_diameter)
    crater_type = classify_crater(final_diameter, params.surface_gravity)

    # 5. Depth and rim height (fresh lunar craters, Pike 1977).
    #    Complex-crater power laws are anchored to the simple values at D_c
    #    so the curves join without a jump.
    if final_diameter <= d_c:
        crater_depth = 0.2 * final_diameter
        rim_height = 0.036 * final_diameter
    else:
        size_ratio = final_diameter / d_c
        crater_depth = 0.2 * d_c * size_ratio ** 0.301
        rim_height = 0.036 * d_c * size_ratio ** 0.399

    # 6. Excavated volume ≈ transient cavity volume (Collins et al. 2005)
    ejecta_volume = math.pi * transient_diameter ** 3 / (16.0 * math.sqrt(2.0))

    if crater_type == "basin":
        warnings.append(ParameterWarning(
            "crater_diameter",
            "Final crater > 300 km — this is a multi-ring impact basin "
            "(like Orientale); simple scaling laws are only indicative here."
        ))
    if final_diameter > LUNAR_DIAMETER:
        warnings.append(ParameterWarning(
            "crater_diameter",
            "Predicted crater is larger than the Moon itself — an impact "
            "this energetic would disrupt the Moon rather than form a crater."
        ))

    return SimulationResult(
        crater_diameter=round(final_diameter, 4),
        crater_depth=round(crater_depth, 4),
        rim_height=round(rim_height, 4),
        ejecta_volume=round(ejecta_volume, 2),
        transient_crater_diameter=round(transient_diameter, 4),
        energy_joules=round(energy, 2),
        model_name=REGOLITH_MODEL_NAME if params.target_material == "regolith" else MODEL_NAME,
        warnings=[w.to_dict() for w in warnings],
        crater_type=crater_type,
    )


# ---------------------------------------------------------------------------
# Crater profile generation (for visualization)
# ---------------------------------------------------------------------------

def generate_crater_profile(
    crater_diameter: float,
    crater_depth: float,
    rim_height: float,
    num_points: int = 200,
    crater_type: str = "simple",
) -> dict:
    """
    Generate (x, y) arrays for a crater cross-section.

    The shape comes from terrain.crater_terrain.radial_elevation, so the
    1-D profile and the 2-D terrain map always agree.

    Returns {"x": [...], "y": [...]}.
    """
    import numpy as np
    from terrain.crater_terrain import radial_elevation, DOMAIN_HALF_FACTOR

    radius = crater_diameter / 2.0
    half_width = radius * DOMAIN_HALF_FACTOR
    xs = np.linspace(-half_width, half_width, num_points)
    ys = radial_elevation(np.abs(xs) / radius, crater_depth, rim_height, crater_type)
    return {"x": xs.round(4).tolist(), "y": ys.round(4).tolist()}
