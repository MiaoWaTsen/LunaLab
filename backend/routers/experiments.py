"""Simulations and the experiments they are saved as."""

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import require_user
from backend.schemas import CompareRequest, ElevationField, SimulationRequest, SimulationResponse
from backend.services import can_access, get_owned_experiment, simulate_and_save, with_crater_type
from database.models import delete_experiment, get_all_experiments, get_experiments_by_ids
from simulation.crater_model import generate_crater_profile
from terrain.asymmetric_crater import generate_asymmetric_elevation_field

router = APIRouter(prefix="/api", tags=["experiments"])


@router.post("/simulations", response_model=SimulationResponse)
def create_simulation(req: SimulationRequest, user: dict = Depends(require_user)):
    """
    Run a crater-scaling simulation and persist the result.
    """
    return simulate_and_save(req, user)


@router.get("/experiments")
def list_experiments(user: dict = Depends(require_user)):
    """Return the caller's experiments (plus legacy unowned ones), newest first."""
    experiments = [with_crater_type(e) for e in get_all_experiments() if can_access(e, user)]
    return {"experiments": experiments}


@router.get("/experiments/{experiment_id}")
def get_experiment_detail(experiment_id: int, user: dict = Depends(require_user)):
    """Return a single experiment by ID."""
    exp = get_owned_experiment(experiment_id, user)
    return with_crater_type(exp)


@router.get("/experiments/{experiment_id}/profile")
def get_experiment_profile(experiment_id: int, user: dict = Depends(require_user)):
    """Generate crater profile data for visualization."""
    exp = get_owned_experiment(experiment_id, user)
    exp = with_crater_type(exp)

    profile = generate_crater_profile(
        crater_diameter=exp["crater_diameter"],
        crater_depth=exp["crater_depth"],
        rim_height=exp["rim_height"],
        crater_type=exp["crater_type"],
    )
    return profile


@router.get("/experiments/{experiment_id}/elevation", response_model=ElevationField)
def get_experiment_elevation(experiment_id: int, user: dict = Depends(require_user)):
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
    exp = get_owned_experiment(experiment_id, user)
    exp = with_crater_type(exp)

    return generate_asymmetric_elevation_field(
        crater_diameter=exp["crater_diameter"],
        crater_depth=exp["crater_depth"],
        rim_height=exp["rim_height"],
        impact_angle=exp.get("impact_angle", 90.0),
        impact_azimuth=exp.get("impact_azimuth", 0.0) or 0.0,
        crater_type=exp["crater_type"],
    )


@router.post("/experiments/compare")
def compare_experiments(req: CompareRequest, user: dict = Depends(require_user)):
    """Return data for multiple experiments for side-by-side comparison."""
    experiments = [e for e in get_experiments_by_ids(req.experiment_ids) if can_access(e, user)]
    if len(experiments) < 2:
        raise HTTPException(
            status_code=404,
            detail="Could not find enough experiments with the provided IDs.",
        )

    # Also generate profiles for each
    results = []
    for exp in experiments:
        exp = with_crater_type(exp)
        profile = generate_crater_profile(
            crater_diameter=exp["crater_diameter"],
            crater_depth=exp["crater_depth"],
            rim_height=exp["rim_height"],
            crater_type=exp["crater_type"],
        )
        results.append({**exp, "profile": profile})

    return {"experiments": results}


@router.delete("/experiments/{experiment_id}")
def remove_experiment(experiment_id: int, user: dict = Depends(require_user)):
    """Delete one of the caller's experiments."""
    get_owned_experiment(experiment_id, user)
    deleted = delete_experiment(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {"detail": "Experiment deleted."}
