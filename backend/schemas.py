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

    Response design rationale
    -------------------------
    The grid is 1024 × 512 = 524,288 cells.  Returning those as a flat
    list of {x, y, z} objects would create ~524,288 JSON objects with
    redundant coordinate repetition — roughly 20–30 MB of JSON per
    request, which is unacceptable for a prototype.

    Instead we use the Plotly-native row-major format:
        x     — 1-D coordinate array (length = width)
        y     — 1-D coordinate array (length = height)
        z     — 2-D array of shape (height, width), matching Plotly's
                heatmap / surface expectation

    This reduces the JSON payload to approximately 2–4 MB, which is
    acceptable for a local prototype over loopback.  For production the
    data would be compressed (gzip) or served as a binary format (e.g.
    MessagePack, Arrow).

    Fields
    ------
    x, y    — Coordinate axes.  z[row][col] is the elevation at
              (x[col], y[row]).
    z       — Row-major elevation matrix: shape (height, width).
    z_min, z_max — Scalar range for Plotly colour-scale.
    width, height — Grid dimensions for client-side validation.
    gen_time_ms   — Server-side terrain generation time in milliseconds.
                    Useful for performance monitoring.
    """
    x:            list[float]
    y:            list[float]
    z:            list[list[float]]
    z_min:        float
    z_max:        float
    width:        int
    height:       int
    gen_time_ms:  float
