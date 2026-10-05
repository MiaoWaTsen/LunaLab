"""
Pydantic schemas for request/response validation.

These schemas define the API contract and are used by FastAPI
for automatic request validation and OpenAPI documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Request body for POST /api/simulations."""
    impactor_diameter: float = Field(
        ..., gt=0, description="Impactor diameter in metres"
    )
    impact_velocity: float = Field(
        ..., gt=0, description="Impact velocity in m/s"
    )
    impact_angle: float = Field(
        ..., gt=0, le=90, description="Impact angle in degrees from vertical (0–90]"
    )
    impactor_density: Optional[float] = Field(
        None, gt=0, description="Impactor density in kg/m³ (default: 3000)"
    )
    target_density: Optional[float] = Field(
        None, gt=0, description="Target surface density in kg/m³ (default: 2500)"
    )
    surface_gravity: Optional[float] = Field(
        None, gt=0, description="Surface gravity in m/s² (default: 1.62 — Moon)"
    )


class CompareRequest(BaseModel):
    """Request body for POST /api/experiments/compare."""
    experiment_ids: list[int] = Field(
        ..., min_length=2, description="List of experiment IDs to compare"
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class WarningItem(BaseModel):
    field: str
    message: str


class SimulationResponse(BaseModel):
    """Response for POST /api/simulations."""
    experiment_id: int
    crater_diameter: float
    crater_depth: float
    rim_height: float
    ejecta_volume: float
    transient_crater_diameter: float
    energy_joules: float
    model: str
    warnings: list[WarningItem] = []


class ExperimentDetail(BaseModel):
    """Full experiment record."""
    model_config = {"protected_namespaces": ()}

    id: int
    created_at: str
    impactor_diameter: float
    impact_velocity: float
    impact_angle: float
    impactor_density: float
    target_density: float
    surface_gravity: float
    crater_diameter: float
    crater_depth: float
    rim_height: float
    ejecta_volume: float
    transient_crater_diameter: float
    energy_joules: float
    model_name: str
    warnings_json: str


class CraterProfile(BaseModel):
    """Crater cross-section data for visualization."""
    x: list[float]
    y: list[float]


class ElevationField(BaseModel):
    """
    2-D synthetic elevation grid for terrain-map visualization.

    x / y are 1-D coordinate arrays; z is a 2-D matrix (rows = y, cols = x).
    z_min / z_max provide the colour-scale range.
    """
    x: list[float]
    y: list[float]
    z: list[list[float]]
    z_min: float
    z_max: float
