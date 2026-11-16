"""The frontend's HTML pages (its assets are mounted at /static by backend/api.py)."""

import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend",
)

router = APIRouter()


def _page(filename: str) -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_DIR, filename))


@router.get("/")
def serve_index():
    return _page("moon.html")


@router.get("/experiment")
def serve_experiment():
    return _page("experiment.html")


@router.get("/history")
def serve_history():
    return _page("history.html")


@router.get("/compare")
def serve_compare():
    return _page("compare.html")


@router.get("/moon")
def serve_moon():
    return _page("moon.html")


@router.get("/challenge")
def serve_challenge():
    return _page("challenge.html")


@router.get("/case-study")
def serve_case_study():
    return _page("case-study.html")


@router.get("/inference")
def serve_inference():
    return _page("inference.html")


@router.get("/batch")
def serve_batch():
    return _page("batch.html")


@router.get("/surface-age")
def serve_surface_age():
    return _page("surface-age.html")


@router.get("/login")
def serve_login():
    return _page("login.html")
