"""
FastAPI application — assembles LunaLab's API.

The routes live in backend/routers/, one module per area:

    auth         accounts and sessions
    challenges   gamified missions
    experiments  simulations and saved experiments
    moon         elevation, illumination, relief texture
    science      case studies, impactor inference, surface dating
    pages        the frontend's HTML pages

Routers translate HTTP to and from the simulation, terrain and database
modules; logic shared between routers is in backend/services.py, and the
authentication dependencies in backend/deps.py.  None of them contain
simulation math or direct SQL.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import auth, challenges, experiments, moon, pages, science
from database.models import init_db


# ---------------------------------------------------------------------------
# Lifespan — initialise database on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


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
app.include_router(moon.router)
app.include_router(science.router)

# Serve static assets (CSS, JS, images)
app.mount("/static", StaticFiles(directory=pages.FRONTEND_DIR), name="static")

app.include_router(pages.router)
