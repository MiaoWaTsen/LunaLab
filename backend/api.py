"""
FastAPI application — API routes for LunaLab.

This module is the thin orchestration layer:
  - Receives HTTP requests
  - Validates input (via Pydantic schemas)
  - Delegates to the simulation module
  - Persists results via the database module
  - Returns structured responses

It does NOT contain any simulation math or direct SQL.
"""

import json
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    SimulationRequest,
    SimulationResponse,
    CompareRequest,
    ExperimentDetail,
    CraterProfile,
    ElevationField,
    RegisterRequest,
    LoginRequest,
    UserProfile,
    AuthResponse,
    ChallengeSubmitRequest,
    ChallengeRecordItem,
)
from simulation.crater_model import (
    ImpactParameters,
    run_simulation,
    generate_crater_profile,
    classify_crater,
    ValidationError,
    DEFAULT_IMPACTOR_DENSITY,
    LUNAR_TARGET_DENSITY,
    LUNAR_GRAVITY,
)
from simulation.challenges import (
    CHALLENGES,
    get_challenge_by_id,
    evaluate_challenge,
)
from terrain.lunar_surface import get_surface_provider
from terrain.asymmetric_crater import generate_asymmetric_elevation_field
from database.models import (
    init_db,
    save_experiment,
    get_experiment,
    get_all_experiments,
    get_experiments_by_ids,
    delete_experiment,
    create_user,
    authenticate_user,
    create_session,
    get_user_by_session,
    get_user_by_id,
    delete_session,
    add_user_xp,
    save_challenge_record,
    get_user_challenge_records,
)


# ---------------------------------------------------------------------------
# Lifespan — initialise database on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LunaLab API",
    description=(
        "Lunar Impact Simulation & Experiment Analysis Platform — "
        "Prototype API. Uses a simplified crater scaling model for "
        "educational and experimental purposes."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow frontend (served on same origin or dev server) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth Helper
# ---------------------------------------------------------------------------

def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Extract user from Bearer token if provided."""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    return get_user_by_session(token)


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """Register a new user commander."""
    try:
        user = create_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_session(user["id"])
    return {"token": token, "user": user}


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Authenticate commander and return session token."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_session(user["id"])
    return {"token": token, "user": user}


@app.get("/api/auth/me", response_model=UserProfile)
def get_me(authorization: Optional[str] = Header(None)):
    """Retrieve logged in user profile."""
    user = get_current_user_optional(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    """Invalidate current session token."""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        delete_session(token)
    return {"detail": "Logged out successfully."}


# ---------------------------------------------------------------------------
# Gamification / Challenge routes
# ---------------------------------------------------------------------------

@app.get("/api/challenges")
def list_challenges():
    """List all available scientific crater challenge missions."""
    return {"challenges": CHALLENGES}


@app.post("/api/challenges/{challenge_id}/run")
def run_challenge(challenge_id: str, req: ChallengeSubmitRequest, authorization: Optional[str] = Header(None)):
    """
    Run simulation within a challenge mission, evaluate accuracy, and award XP.
    """
    ch = get_challenge_by_id(challenge_id)
    if not ch:
        raise HTTPException(status_code=404, detail="Challenge mission not found.")

    sim_res = create_simulation(req.simulation, authorization=authorization)
    eval_res = evaluate_challenge(
        challenge_id,
        {
            "crater_diameter": sim_res.crater_diameter,
            "crater_depth": sim_res.crater_depth,
            "energy_joules": sim_res.energy_joules,
        },
        req.simulation.model_dump(),
    )

    user = get_current_user_optional(authorization)
    user_id = user["id"] if user else None
    updated_user = None

    if user:
        add_user_xp(user["id"], eval_res["xp_gain"])
        updated_user = get_user_by_id(user["id"])

    rec_id = save_challenge_record({
        "user_id": user_id,
        "challenge_id": challenge_id,
        "score": eval_res["score"],
        "grade": eval_res["grade"],
        "diameter_error_pct": eval_res["diameter_error_pct"],
        "depth_error_pct": eval_res["depth_error_pct"],
        "experiment_id": sim_res.experiment_id,
    })
    eval_res["record_id"] = rec_id

    return {
        "evaluation": eval_res,
        "simulation": sim_res,
        "user": updated_user,
    }


@app.get("/api/challenges/history")
def get_challenge_history(authorization: Optional[str] = Header(None)):
    """Retrieve recent challenge mission attempts from all users (login required)."""
    user = get_current_user_optional(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    records = get_user_challenge_records(limit=50)
    return {"records": records}


# ---------------------------------------------------------------------------
# Simulation API routes
# ---------------------------------------------------------------------------

@app.post("/api/simulations", response_model=SimulationResponse)
def create_simulation(req: SimulationRequest, authorization: Optional[str] = Header(None)):
    """
    Run a crater-scaling simulation and persist the result.
    """
    user = get_current_user_optional(authorization)

    # Build domain object
    params = ImpactParameters(
        impactor_diameter=req.impactor_diameter,
        impact_velocity=req.impact_velocity,
        impact_angle=req.impact_angle,
        impactor_density=req.impactor_density or DEFAULT_IMPACTOR_DENSITY,
        target_density=req.target_density or LUNAR_TARGET_DENSITY,
        surface_gravity=req.surface_gravity or LUNAR_GRAVITY,
    )

    # Fetch elevation if location is provided
    elevation = None
    if req.latitude is not None and req.longitude is not None:
        provider = get_surface_provider()
        try:
            elevation = provider.get_elevation(req.latitude, req.longitude)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    # Run simulation (may raise ValidationError)
    try:
        result = run_simulation(params)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Persist to database
    experiment_data = {
        "impactor_diameter": params.impactor_diameter,
        "impact_velocity": params.impact_velocity,
        "impact_angle": params.impact_angle,
        "impactor_density": params.impactor_density,
        "target_density": params.target_density,
        "surface_gravity": params.surface_gravity,
        "crater_diameter": result.crater_diameter,
        "crater_depth": result.crater_depth,
        "rim_height": result.rim_height,
        "ejecta_volume": result.ejecta_volume,
        "transient_crater_diameter": result.transient_crater_diameter,
        "energy_joules": result.energy_joules,
        "model_name": result.model_name,
        "warnings_json": json.dumps(result.warnings),
        "impact_azimuth": req.impact_azimuth,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "surface_elevation": elevation,
        "user_id": user["id"] if user else None,
    }
    experiment_id = save_experiment(experiment_data)

    # Award +15 Research XP if logged in
    if user:
        add_user_xp(user["id"], 15)

    impact_loc = None
    if req.latitude is not None and req.longitude is not None:
        impact_loc = {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "elevation": elevation
        }

    return SimulationResponse(
        experiment_id=experiment_id,
        crater_diameter=result.crater_diameter,
        crater_depth=result.crater_depth,
        rim_height=result.rim_height,
        ejecta_volume=result.ejecta_volume,
        transient_crater_diameter=result.transient_crater_diameter,
        energy_joules=result.energy_joules,
        model=result.model_name,
        warnings=[{"field": w["field"], "message": w["message"]} for w in result.warnings],
        impact_azimuth=req.impact_azimuth,
        impact_location=impact_loc,
        crater_type=result.crater_type,
    )


def _with_crater_type(exp: dict) -> dict:
    """Derive crater morphology class (not stored in the database)."""
    return {**exp, "crater_type": classify_crater(exp["crater_diameter"], exp["surface_gravity"])}


@app.get("/api/experiments")
def list_experiments():
    """Return all experiments, newest first."""
    experiments = [_with_crater_type(e) for e in get_all_experiments()]
    return {"experiments": experiments}


@app.get("/api/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: int):
    """Return a single experiment by ID."""
    exp = get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return _with_crater_type(exp)


@app.get("/api/experiments/{experiment_id}/profile")
def get_experiment_profile(experiment_id: int):
    """Generate crater profile data for visualization."""
    exp = get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    exp = _with_crater_type(exp)

    profile = generate_crater_profile(
        crater_diameter=exp["crater_diameter"],
        crater_depth=exp["crater_depth"],
        rim_height=exp["rim_height"],
        crater_type=exp["crater_type"],
    )
    return profile


@app.get("/api/experiments/{experiment_id}/elevation", response_model=ElevationField)
def get_experiment_elevation(experiment_id: int):
    """
    Generate a synthetic 2-D elevation field (terrain map) for an experiment.

    Orchestration:
        1. Fetch stored crater geometry from the database.
        2. Delegate to the oblique-impact generator, which reduces to the
           symmetric crater for impact angles >= 45°.
        3. Return the field as a JSON response.

    The terrain generation is intentionally separated from the crater
    scaling model (simulation/crater_model.py).  This route does NOT
    perform any scientific calculations — it delegates entirely.

    Returns a 1024 × 512 grid (width × height) plus 1-D x/y coordinate
    arrays and scalar z_min/z_max for Plotly colour-scale normalisation.
    """
    exp = get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    exp = _with_crater_type(exp)

    return generate_asymmetric_elevation_field(
        crater_diameter=exp["crater_diameter"],
        crater_depth=exp["crater_depth"],
        rim_height=exp["rim_height"],
        impact_angle=exp.get("impact_angle", 90.0),
        impact_azimuth=exp.get("impact_azimuth", 0.0) or 0.0,
        crater_type=exp["crater_type"],
    )


@app.post("/api/experiments/compare")
def compare_experiments(req: CompareRequest):
    """Return data for multiple experiments for side-by-side comparison."""
    experiments = get_experiments_by_ids(req.experiment_ids)
    if len(experiments) < 2:
        raise HTTPException(
            status_code=404,
            detail="Could not find enough experiments with the provided IDs.",
        )

    # Also generate profiles for each
    results = []
    for exp in experiments:
        exp = _with_crater_type(exp)
        profile = generate_crater_profile(
            crater_diameter=exp["crater_diameter"],
            crater_depth=exp["crater_depth"],
            rim_height=exp["rim_height"],
            crater_type=exp["crater_type"],
        )
        results.append({**exp, "profile": profile})

    return {"experiments": results}


@app.delete("/api/experiments/{experiment_id}")
def remove_experiment(experiment_id: int):
    """Delete an experiment."""
    deleted = delete_experiment(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {"detail": "Experiment deleted."}


@app.get("/api/moon/elevation")
def get_moon_elevation(lat: float, lon: float):
    """Surface elevation (m, relative to the 1737.4 km sphere) at a coordinate."""
    provider = get_surface_provider()
    try:
        elevation = provider.get_elevation(lat, lon)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "latitude": lat,
        "longitude": lon,
        "elevation": elevation,
        "source": provider.get_provider_name(),
        "is_real_data": provider.is_real_data(),
    }


@app.get("/api/moon/terrain-texture")
def get_moon_terrain_texture():
    """Return an 8-bit raw displacement map for 3D sphere visualization."""
    provider = get_surface_provider()
    # Fixed resolution 1024x512 for Three.js
    tex = provider.get_terrain_texture(1024, 512)
    return Response(content=tex.tobytes(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------

import os

_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend",
)

# Serve static assets (CSS, JS, images)
app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(_FRONTEND_DIR, "moon.html"))


@app.get("/experiment")
def serve_experiment():
    return FileResponse(os.path.join(_FRONTEND_DIR, "experiment.html"))


@app.get("/history")
def serve_history():
    return FileResponse(os.path.join(_FRONTEND_DIR, "history.html"))


@app.get("/compare")
def serve_compare():
    return FileResponse(os.path.join(_FRONTEND_DIR, "compare.html"))


@app.get("/moon")
def serve_moon():
    return FileResponse(os.path.join(_FRONTEND_DIR, "moon.html"))


@app.get("/challenge")
def serve_challenge():
    return FileResponse(os.path.join(_FRONTEND_DIR, "challenge.html"))


@app.get("/login")
def serve_login():
    return FileResponse(os.path.join(_FRONTEND_DIR, "login.html"))


