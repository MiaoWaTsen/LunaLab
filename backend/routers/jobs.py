"""Background jobs: submit long-running work, follow it, cancel it."""

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from backend.deps import require_user
from backend.job_types import JOB_TYPES
from database import jobs as queue

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Jobs run one at a time, so one user must not be able to fill the queue
MAX_ACTIVE_JOBS_PER_USER = 3

_PUBLIC_FIELDS = (
    "id", "type", "status", "progress", "params", "result", "error",
    "cancel_requested", "created_at", "started_at", "finished_at", "provenance",
)


class JobRequest(BaseModel):
    type: str = Field(..., description="One of GET /api/jobs/types")
    params: dict = Field(default_factory=dict)


def _public(job: dict) -> dict:
    public = {field: job[field] for field in _PUBLIC_FIELDS if field in job}
    # A result is outdated when today's code would not reproduce it: the model
    # has changed since, or the job type no longer exists
    job_type = JOB_TYPES.get(job["type"])
    public["outdated"] = job["status"] == queue.SUCCEEDED and (
        job_type is None or not job_type.is_current(job["provenance"])
    )
    return public


def _enqueue(job_type_name: str, raw_params: dict, request: Request, user: dict) -> dict:
    job_type = JOB_TYPES.get(job_type_name)
    if job_type is None:
        raise HTTPException(status_code=422, detail=f"Unknown job type '{job_type_name}'.")
    try:
        params = job_type.params_model.model_validate(raw_params)
    except ValidationError as error:
        # Same shape as FastAPI's own validation errors
        detail = json.loads(error.json(include_url=False, include_input=False))
        raise HTTPException(status_code=422, detail=detail)
    if queue.count_active_jobs(user["id"]) >= MAX_ACTIVE_JOBS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"You already have {MAX_ACTIVE_JOBS_PER_USER} jobs queued or running. "
                   "Wait for one to finish, or cancel one.",
        )

    job = queue.create_job(user["id"], job_type_name, params.model_dump())
    worker = getattr(request.app.state, "job_worker", None)
    if worker is not None:
        worker.notify()
    return _public(job)


def _get_owned_job(job_id: int, user: dict) -> dict:
    job = queue.get_job(job_id)
    if job is None or job["user_id"] != user["id"]:
        # 404 rather than 403 so other users' job ids are not revealed
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("/types")
def list_job_types():
    """The kinds of job that can be submitted, with the JSON schema of their params."""
    return [
        {"type": name, "description": job_type.description,
         "params_schema": job_type.params_model.model_json_schema()}
        for name, job_type in JOB_TYPES.items()
    ]


@router.post("", status_code=202)
def submit_job(req: JobRequest, request: Request, user: dict = Depends(require_user)):
    """Queue a job.  Poll GET /api/jobs/{id} until its status is no longer queued or running."""
    return _enqueue(req.type, req.params, request, user)


@router.get("")
def list_jobs(user: dict = Depends(require_user)):
    """The caller's jobs, newest first, without their results."""
    return {"jobs": [_public(job) for job in queue.list_user_jobs(user["id"])]}


@router.get("/{job_id}")
def get_job(job_id: int, user: dict = Depends(require_user)):
    """A job's status and progress, and its result once it has succeeded."""
    return _public(_get_owned_job(job_id, user))


@router.post("/{job_id}/cancel")
def cancel_job(job_id: int, user: dict = Depends(require_user)):
    """
    Cancel a job.  A queued job is cancelled at once; a running one stops at
    its next progress report, so it may still read 'running' in the reply.
    """
    _get_owned_job(job_id, user)
    return _public(queue.request_cancel(job_id))


@router.post("/{job_id}/rerun", status_code=202)
def rerun_job(job_id: int, request: Request, user: dict = Depends(require_user)):
    """
    Queue a new job with the same type and params as an existing one, to
    refresh an outdated result or check that a result reproduces.  The
    original job is kept.
    """
    job = _get_owned_job(job_id, user)
    return _enqueue(job["type"], job["params"], request, user)
