"""
FastAPI application — assembles LunaLab's API.

The routes live in backend/routers/, one module per area:

    auth         accounts and sessions
    challenges   gamified missions
    experiments  simulations and saved experiments
    jobs         long-running work, run in the background
    moon         elevation, illumination, relief texture
    science      case studies, impactor inference, surface dating
    pages        the frontend's HTML pages

Routers translate HTTP to and from the simulation, terrain and database
modules; logic shared between routers is in backend/services.py, and the
authentication dependencies in backend/deps.py.  None of them contain
simulation math or direct SQL.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from backend.job_types import JOB_TYPES
from backend.jobs import Worker
from backend.routers import auth, challenges, experiments, jobs, moon, pages, science
from database.models import init_db


# ---------------------------------------------------------------------------
# Lifespan — migrate the database, run the job worker
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    # The worker shares this process (see backend/jobs.py).  LUNALAB_JOB_WORKER=0
    # turns it off: jobs are then queued but run only by another process.
    worker = None
    if os.environ.get("LUNALAB_JOB_WORKER", "1") != "0":
        worker = Worker(JOB_TYPES)
        worker.start()
    application.state.job_worker = worker
    yield
    if worker is not None:
        worker.stop()


# ---------------------------------------------------------------------------
# App
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

# The science endpoints return long arrays of numbers, which compress ~3x.
# Level 1: on the 1 MB synthetic surface, level 9 (the default) takes ~0.9 s
# to save another 20%; level 1 takes ~15 ms.
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=1)

# Allow frontend (served on same origin or dev server) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(challenges.router)
app.include_router(experiments.router)
app.include_router(jobs.router)
app.include_router(moon.router)
app.include_router(science.router)

# Serve static assets (CSS, JS, images)
app.mount("/static", StaticFiles(directory=pages.FRONTEND_DIR), name="static")

app.include_router(pages.router)
