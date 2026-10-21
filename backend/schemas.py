"""
Pydantic schemas for request/response validation.

These schemas define the API contract and are used by FastAPI
for automatic request validation and OpenAPI documentation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Request body for POST /api/simulations."""
    impactor_diameter: float = Field(
        ..., gt=0, le=1_000_000, description="Impactor diameter in metres (max 1000 km)"
    )
    impact_velocity: float = Field(
        ..., gt=0, le=100_000, description="Impact velocity in m/s (max 100 km/s)"
    )
    impact_angle: float = Field(
        ..., gt=0, le=90, description="Impact angle in degrees from horizontal (0–90], 90 = vertical"
    )
    impactor_density: Optional[float] = Field(
        None, gt=0, le=30_000, description="Impactor density in kg/m³ (default: 3000)"
    )
    target_density: Optional[float] = Field(
        None, gt=0, le=30_000, description="Target surface density in kg/m³ (default: 2550, lunar crust)"
    )
    surface_gravity: Optional[float] = Field(
        None, ge=0.01, le=100, description="Surface gravity in m/s² (default: 1.62 — Moon)"
    )
    impact_azimuth: Optional[float] = Field(
        0.0, description="Impact azimuth in degrees (0=North, clockwise)"
    )
    latitude: Optional[float] = Field(
        None, ge=-90, le=90, description="Latitude of impact"
    )
    longitude: Optional[float] = Field(
        None, ge=-180, le=180, description="Longitude of impact"
    )

    @field_validator("impact_azimuth")
    @classmethod
    def _normalise_azimuth(cls, v: Optional[float]) -> float:
        """Any heading is accepted and folded into [0, 360)."""
        return 0.0 if v is None else v % 360.0

    @model_validator(mode="after")
    def _location_pair(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be given together.")
        return self


class CompareRequest(BaseModel):
    """Request body for POST /api/experiments/compare."""
    experiment_ids: list[int] = Field(
        ..., min_length=2, description="List of experiment IDs to compare"
    )


class RegisterRequest(BaseModel):
    """Request body for POST /api/auth/register."""
    username: str = Field(..., min_length=2, max_length=30, description="Commander username")
    password: str = Field(..., min_length=4, max_length=100, description="Account password")


class LoginRequest(BaseModel):
    """Request body for POST /api/auth/login."""
    username: str = Field(..., description="Commander username")
    password: str = Field(..., description="Account password")


class UserProfile(BaseModel):
    """User profile data."""
    id: int
    username: str
    created_at: str
    xp: int
    level: int
    rank_title: str


class AuthResponse(BaseModel):
    """Response containing auth token and user profile."""
    token: str
    user: UserProfile


class ChallengeSubmitRequest(BaseModel):
    """Request body for POST /api/challenges/{challenge_id}/run."""
    simulation: SimulationRequest


class ChallengeRecordItem(BaseModel):
    id: int
    challenge_id: str
    score: float
    grade: str
    diameter_error_pct: float
    depth_error_pct: float
    experiment_id: Optional[int]
    created_at: str


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class WarningItem(BaseModel):
    field: str
    message: str


class ImpactLocation(BaseModel):
    latitude: float
    longitude: float
    elevation: Optional[float] = None


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
    impact_azimuth: float = 0.0
    impact_location: Optional[ImpactLocation] = None
    crater_type: str = "simple"


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
    impact_azimuth: float
    latitude: Optional[float]
    longitude: Optional[float]
    surface_elevation: Optional[float]


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
    is_asymmetric:  Optional[bool] = None
    impact_angle:   Optional[float] = None
    impact_azimuth: Optional[float] = None
