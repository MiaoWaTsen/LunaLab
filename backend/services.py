"""
Use cases shared by more than one router.

Routers translate HTTP to and from these functions; the functions
orchestrate the simulation, terrain and database modules.
"""

import json
from datetime import datetime, timezone

from fastapi import HTTPException

from backend.schemas import SimulationRequest, SimulationResponse
from database.models import (
    add_user_xp,
    count_user_experiments_since,
    get_experiment,
    save_experiment,
)
from simulation.crater_model import (
    DEFAULT_IMPACTOR_DENSITY,
    LUNAR_GRAVITY,
    LUNAR_TARGET_DENSITY,
    ImpactParameters,
    ValidationError,
    classify_crater,
    run_simulation,
)
from terrain.lunar_surface import get_surface_provider

# Posterior draws per inference.  The reported percentiles are computed from
# these draws, so every caller must use the same number to get the same answer.
INFERENCE_POSTERIOR_SAMPLES = 1000

# Research XP for plain simulations: +15 each, for the first 10 per UTC day.
SIMULATION_XP = 15
DAILY_SIMULATION_XP_LIMIT = 10


def can_access(exp: dict, user: dict) -> bool:
    """Owners see their experiments; legacy rows with no owner are shared."""
    return exp.get("user_id") is None or exp["user_id"] == user["id"]


def get_owned_experiment(experiment_id: int, user: dict) -> dict:
    exp = get_experiment(experiment_id)
    if exp is None or not can_access(exp, user):
        # 404 rather than 403 so other users' experiment ids are not revealed
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return exp


def with_crater_type(exp: dict) -> dict:
    """Derive crater morphology class (not stored in the database)."""
    return {**exp, "crater_type": classify_crater(exp["crater_diameter"], exp["surface_gravity"])}


def simulate_and_save(req: SimulationRequest, user: dict) -> SimulationResponse:
    """Run a crater-scaling simulation, persist it and award research XP."""

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
        "user_id": user["id"],
    }
    experiment_id = save_experiment(experiment_data)

    # Research XP, capped per UTC day (this experiment is already counted)
    day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    runs_today = count_user_experiments_since(user["id"], day_start.isoformat())
    xp_awarded = SIMULATION_XP if runs_today <= DAILY_SIMULATION_XP_LIMIT else 0
    if xp_awarded:
        add_user_xp(user["id"], xp_awarded)

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
        xp_awarded=xp_awarded,
    )
