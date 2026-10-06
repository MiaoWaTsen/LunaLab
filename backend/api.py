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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
)
from simulation.crater_model import (
    ImpactParameters,
    run_simulation,
    generate_crater_profile,
    ValidationError,
)
from terrain.crater_terrain import generate_elevation_field
from database.models import (
    init_db,
    save_experiment,
    get_experiment,
    get_all_experiments,
    get_experiments_by_ids,
    delete_experiment,
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
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/simulations", response_model=SimulationResponse)
def create_simulation(req: SimulationRequest):
    """
    Run a crater-scaling simulation and persist the result.
    """
    # Build domain object
    params = ImpactParameters(
        impactor_diameter=req.impactor_diameter,
        impact_velocity=req.impact_velocity,
        impact_angle=req.impact_angle,
        impactor_density=req.impactor_density or 3000.0,
        target_density=req.target_density or 2500.0,
        surface_gravity=req.surface_gravity or 1.62,
    )

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
    }
    experiment_id = save_experiment(experiment_data)

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
    )


@app.get("/api/experiments")
def list_experiments():
    """Return all experiments, newest first."""
    experiments = get_all_experiments()
    return {"experiments": experiments}


@app.get("/api/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: int):
    """Return a single experiment by ID."""
    exp = get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return exp


@app.get("/api/experiments/{experiment_id}/profile")
def get_experiment_profile(experiment_id: int):
    """Generate crater profile data for visualization."""
    exp = get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    profile = generate_crater_profile(
        crater_diameter=exp["crater_diameter"],
        crater_depth=exp["crater_depth"],
        rim_height=exp["rim_height"],
    )
    return profile


@app.get("/api/experiments/{experiment_id}/elevation", response_model=ElevationField)
def get_experiment_elevation(experiment_id: int):
    """
    Generate a synthetic 2-D elevation field (terrain map) for an experiment.

    Orchestration:
        1. Fetch stored crater geometry from the database.
        2. Delegate to terrain.crater_terrain.generate_elevation_field().
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

    field = generate_elevation_field(
        crater_diameter=exp["crater_diameter"],
        crater_depth=exp["crater_depth"],
        rim_height=exp["rim_height"],
    )
    return field


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
        profile = generate_crater_profile(
            crater_diameter=exp["crater_diameter"],
            crater_depth=exp["crater_depth"],
            rim_height=exp["rim_height"],
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
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))


@app.get("/experiment")
def serve_experiment():
    return FileResponse(os.path.join(_FRONTEND_DIR, "experiment.html"))


@app.get("/history")
def serve_history():
    return FileResponse(os.path.join(_FRONTEND_DIR, "history.html"))


@app.get("/compare")
def serve_compare():
    return FileResponse(os.path.join(_FRONTEND_DIR, "compare.html"))
