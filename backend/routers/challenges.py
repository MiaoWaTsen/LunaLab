"""Gamified challenge missions."""

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import require_user
from backend.schemas import ChallengeSubmitRequest
from backend.services import simulate_and_save
from database.models import (
    add_user_xp,
    get_best_challenge_grade,
    get_user_by_id,
    get_user_challenge_records,
    save_challenge_record,
)
from simulation.challenges import (
    CHALLENGES,
    award_challenge_xp,
    evaluate_challenge,
    get_challenge_by_id,
)

router = APIRouter(prefix="/api/challenges", tags=["challenges"])


@router.get("")
def list_challenges():
    """List all available scientific crater challenge missions."""
    return {"challenges": CHALLENGES}


@router.post("/{challenge_id}/run")
def run_challenge(challenge_id: str, req: ChallengeSubmitRequest, user: dict = Depends(require_user)):
    """
    Run simulation within a challenge mission, evaluate accuracy, and award XP.
    """
    ch = get_challenge_by_id(challenge_id)
    if not ch:
        raise HTTPException(status_code=404, detail="Challenge mission not found.")

    sim_res = simulate_and_save(req.simulation, user)
    eval_res = evaluate_challenge(
        challenge_id,
        {
            "crater_diameter": sim_res.crater_diameter,
            "crater_depth": sim_res.crater_depth,
            "energy_joules": sim_res.energy_joules,
        },
        req.simulation.model_dump(),
    )

    previous_best = get_best_challenge_grade(user["id"], challenge_id)
    xp = award_challenge_xp(eval_res["grade"], previous_best)
    eval_res["xp_gain"] = xp
    eval_res["previous_best_grade"] = previous_best
    if xp == 0:
        eval_res["xp_note"] = f"No XP: your best grade on this mission is already {previous_best}. Beat it to earn more."
    elif previous_best:
        eval_res["xp_note"] = f"New best (previous {previous_best}): you earn the difference."
    add_user_xp(user["id"], xp)
    updated_user = get_user_by_id(user["id"])

    rec_id = save_challenge_record({
        "user_id": user["id"],
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


@router.get("/history")
def get_challenge_history(user: dict = Depends(require_user)):
    """Retrieve recent challenge mission attempts from all users (login required)."""
    records = get_user_challenge_records(limit=50)
    return {"records": records}
