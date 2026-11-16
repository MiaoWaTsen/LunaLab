"""
The analyses built on the crater model: case studies, impactor inference
and crater-count surface dating.
"""

from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.deps import require_user
from backend.schemas import InferenceRequest
from simulation.case_studies import CASE_STUDIES, analyze, get_case_study
from simulation.inference import PRESETS as INFERENCE_PRESETS, Observation, infer
from simulation.surface_dating import (
    SITES_BY_ID,
    change6_analysis,
    landing_site_ages,
    landing_site_craters,
    synthetic_history,
)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Case studies — the model against observed impacts (public, read-only)
# ---------------------------------------------------------------------------

@router.get("/case-studies", tags=["case-studies"])
def list_case_studies():
    return [
        {"id": e.id, "title": e.title, "time": e.time.isoformat(),
         "latitude": e.latitude, "longitude": e.longitude}
        for e in CASE_STUDIES.values()
    ]


# The analysis is deterministic (fixed seed), so it is computed once per event
@lru_cache(maxsize=None)
def _case_study_analysis(event_id: str) -> dict:
    return analyze(CASE_STUDIES[event_id])


@router.get("/case-studies/{event_id}", tags=["case-studies"])
def get_case_study_analysis(event_id: str):
    if get_case_study(event_id) is None:
        raise HTTPException(status_code=404, detail="Case study not found.")
    return _case_study_analysis(event_id)


# ---------------------------------------------------------------------------
# Inference — from an observed crater back to the impactor
# ---------------------------------------------------------------------------

@router.get("/inference/presets", tags=["inference"])
def list_inference_presets():
    return INFERENCE_PRESETS


# Signed-in only: each call runs up to a million forward models
@router.post("/inference", tags=["inference"])
def run_inference(req: InferenceRequest, user: dict = Depends(require_user)):
    obs = Observation(
        crater_diameter=req.crater_diameter,
        diameter_error=req.diameter_error,
        material=req.material,
        luminous_energy=req.luminous_energy,
        velocity=req.velocity,
    )
    return infer(obs, posterior_samples=1000)


# ---------------------------------------------------------------------------
# Surface age — crater-count dating
# ---------------------------------------------------------------------------

# Signed-in only: an old surface means tens of thousands of craters
@router.get("/surface-age/synthetic", tags=["surface-age"])
def get_synthetic_surface(
    age: float = Query(3.8, ge=0.1, le=4.3, description="Surface age in Gyr"),
    size_km: float = Query(40.0, ge=10.0, le=60.0),
    d_min: float = Query(0.4, ge=0.3, le=5.0, description="Smallest crater simulated, km"),
    seed: int = Query(0, ge=0, le=1_000_000),
    user: dict = Depends(require_user),
):
    try:
        return synthetic_history(age, size_km, d_min, seed)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@lru_cache(maxsize=None)
def _change6_analysis() -> dict:
    return change6_analysis()


@router.get("/surface-age/change6", tags=["surface-age"])
def get_change6_analysis():
    return _change6_analysis()


@router.get("/surface-age/sites", tags=["surface-age"])
def get_landing_site_ages(
    radius_km: float = Query(100.0, ge=25.0, le=150.0, description="Multiple of 5 km"),
    d_min: float = Query(1.0, ge=1.0, le=50.0),
    d_max: float = Query(20.0, gt=1.0, le=300.0),
    count_area: Literal["circle", "unit"] = Query("circle"),
):
    try:
        return landing_site_ages(radius_km, d_min, d_max, count_area)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/surface-age/sites/{site_id}", tags=["surface-age"])
def get_landing_site_craters(site_id: str, radius_km: float = Query(100.0, ge=25.0, le=150.0)):
    if site_id not in SITES_BY_ID:
        raise HTTPException(status_code=404, detail="Landing site not found.")
    return landing_site_craters(site_id, radius_km)
