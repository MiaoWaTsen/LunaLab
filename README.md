# LunaLab

**Lunar Impact Simulation & Experiment Analysis Platform**

> ⚠️ This prototype uses a simplified crater scaling model for educational and experimental purposes. It is not a full physical simulation of impact processes.

---

## Overview

LunaLab is a modular, web-based platform for simulating lunar impact craters using a **lunar crater scaling model** (Collins et al. 2005 pi-scaling + Pike 1977 lunar morphometry). Users can:

1. **Run Simulations** — Input impactor parameters and compute crater dimensions
2. **Visualise Results** — Interactive Plotly charts showing crater profiles and parameter comparisons
3. **Save Experiments** — Every simulation is persisted to a database (SQLite locally, PostgreSQL in production)
4. **Browse History** — View, inspect, and manage past experiments
5. **Compare Experiments** — Select multiple experiments for side-by-side visual comparison
6. **Case Study** — Test the model against the 17 March 2013 impact, where both the flash and the LRO crater were measured: observing geometry from the ephemeris, Monte Carlo uncertainty propagation, and inversion for the luminous efficiency η
7. **Impactor Inference** — Bayesian inversion from an observed crater back to the impactor: posterior over size, velocity, angle and density, the size–speed degeneracy, and (for impact flashes) the luminous efficiency; validated by simulation-based calibration
8. **Surface Age** — Date a surface by counting its craters (Neukum 2001 production and chronology functions, Poisson likelihood): watch a synthetic plain collect craters over 4 billion years, see small craters saturate, then test the method on the Chang'e-6 landing site against its 2.807 Ga sample age, and on seven sampled landing sites counted in circles vs on their mapped geological units
9. **3D Moon** — Pick the impact site on a globe built from LRO imagery and LOLA elevation, lit by the real Sun for the current date
10. **Batch Inference** — Infer the impactor behind every crater in a list, or around a landing site, as a background job with progress, cancellation and CSV download
11. **Six interface languages** — English, Traditional and Simplified Chinese, Japanese, German and French, switchable from the navbar
12. **Accounts & Missions** — Sign in to keep your own experiments, and earn research XP on challenge missions graded against known craters

---

## Architecture

LunaLab follows a **modular, separation-of-concerns** architecture:

```
Frontend (HTML / CSS / Vanilla JS)
       ↓
   REST API (FastAPI)
       ↓
  Simulation Module (Pure Python)
       ↓
  Database Module (SQLite / PostgreSQL)
```

### Project Structure

```
LunaLab/
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── README.md
│
├── backend/                 # API layer
│   ├── __init__.py
│   ├── api.py               # App: middleware, routers, static files
│   ├── routers/             # Routes, one module per area (auth, challenges,
│   │                        #   experiments, jobs, moon, science, pages)
│   ├── services.py          # Use cases shared between routers
│   ├── jobs.py              # Background job runner and worker thread
│   ├── job_types.py         # The kinds of background job
│   ├── provenance.py        # Fingerprint of the code behind a result
│   ├── deps.py              # Authentication dependencies
│   └── schemas.py           # Pydantic request/response models
│
├── simulation/              # Science (independent of the web layer)
│   ├── crater_model.py      # Crater scaling (Collins/Pike, Holsapple regolith)
│   ├── case_studies.py      # Model vs. the observed 2013-03-17 impact
│   ├── inference.py         # Bayesian crater → impactor inversion
│   ├── chronology.py        # Crater production/chronology functions, age inference
│   ├── surface_dating.py    # Synthetic bombardment, Chang'e-6, landing sites
│   ├── lunar_ephemeris.py   # Sun and Earth as seen from the Moon (Meeus)
│   ├── challenges.py        # Challenge missions and their grading
│   └── data/                # Crater counts (built by scripts/build_crater_data.py)
│
├── terrain/                 # Elevation: real and synthetic
│   ├── lunar_surface.py     # LOLA elevation provider (+ procedural fallback)
│   ├── crater_terrain.py    # Synthetic crater shape
│   ├── asymmetric_crater.py # Oblique-impact asymmetry
│   └── data/                # LOLA LDEM_4 grid
│
├── database/                # Data persistence layer
│   ├── __init__.py
│   ├── models.py            # Connections & CRUD (SQLite or PostgreSQL)
│   ├── migrations.py        # Numbered schema migrations
│   └── jobs.py              # Durable job queue
│
├── frontend/                # Static frontend files
│   ├── moon.html, moon.js   # 3D Moon (home page)
│   ├── experiment.html      # New experiment / view experiment
│   ├── history.html         # Experiment history
│   ├── compare.html         # Experiment comparison
│   ├── challenge.html       # Challenge missions
│   ├── case-study.html      # The 2013-03-17 impact
│   ├── inference.html       # Impactor inference
│   ├── batch.html           # Batch inference as a background job
│   ├── surface-age.html     # Crater-count dating
│   ├── login.html           # Sign in / register
│   ├── index.html           # Earlier dashboard (reachable only at /static/index.html)
│   ├── styles.css           # Design system
│   ├── app.js               # Shared JavaScript utilities
│   ├── i18n.js              # Interface translation engine and language button
│   ├── i18n/messages.json   # Translations: one row per English string
│   └── textures/            # Moon colour and normal maps
│
├── scripts/
│   └── build_crater_data.py # Rebuilds simulation/data/ from the source catalogues
│
├── tests/                   # Automated tests (pytest)
│   ├── conftest.py          # Runs database tests on SQLite and PostgreSQL
│   ├── test_*.py            # Unit and API tests, one file per area
│   └── e2e/                 # Browser tests (Playwright)
│
├── .github/workflows/       # CI: tests on PostgreSQL, then E2E
└── lunalab.db               # SQLite database (auto-created, not tracked)
```

### Module Responsibilities

| Module | Responsibility | Dependencies |
|--------|---------------|-------------|
| `simulation/` | Crater scaling, inference, chronology, ephemeris, challenge grading | numpy |
| `terrain/` | Lunar elevation data and synthetic crater terrain | numpy |
| `database/` | Schema migrations, CRUD operations | sqlite3 (stdlib), psycopg (PostgreSQL only) |
| `backend/` | HTTP routing, request validation, orchestration | FastAPI, simulation, terrain, database |
| `frontend/` | UI, user interaction, visualisation | Plotly.js, Three.js (CDN) |
| `tests/` | Automated verification | pytest, Playwright (E2E only) |

**Key design principle:** The simulation module can be imported and tested independently without any web server or database.

---

## Getting Started

### Prerequisites

- Python 3.11 (the version deployed and tested in CI; see `runtime.txt`)
- pip

### Installation

```bash
cd LunaLab
pip install -r requirements.txt
```

### Start the Server

```bash
python3 main.py
```

The server starts at **http://localhost:8765**.

The frontend is served from the same origin — no separate frontend server needed.

### Run Tests

```bash
python3 -m pytest
```

Two groups of tests need something extra and are skipped without it:

- **PostgreSQL** — the database tests run on SQLite, and again on PostgreSQL when
  `TEST_DATABASE_URL` points at a disposable database (the tests drop its tables):

  ```bash
  docker run -d --rm --name lunalab-pg -e POSTGRES_PASSWORD=postgres \
      -e POSTGRES_DB=lunalab_test -p 55432:5432 postgres:16
  TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:55432/lunalab_test python3 -m pytest
  docker stop lunalab-pg
  ```

- **End-to-end** — `tests/e2e/` drives a real browser:
  `pip install playwright && python3 -m playwright install chromium`.

CI runs both. `mutmut run` (configured in `setup.cfg`) mutation-tests the science modules.

### Database Schema Changes

The schema is defined by the numbered migrations in `database/migrations.py`, applied
automatically at startup and recorded in the `schema_migrations` table. To change the
schema, append a migration with the next version number; never edit a released one.

---

## Interface Languages

The language button in the navbar switches the interface between English, 繁體中文, 简体中文,
日本語, Deutsch and Français. The choice is remembered; the first visit follows the browser.

The pages are written in English and are not marked up for translation. `frontend/i18n.js`
translates what is on screen by looking each piece of text up in `frontend/i18n/messages.json`,
keyed by the English source, and keeps watching, so text that scripts insert later is
translated as it appears. A string with no translation stays in English.

To translate a new string, add a row to `messages.json`:

```json
"{0} craters (limit {1})": [null, "{0} 個撞擊坑（上限 {1}）", "{0} 个撞击坑（上限 {1}）", "クレーター {0} 個（上限 {1}）", "{0} Krater (Limit {1})", "{0} cratères (limite {1})"]
```

- The columns follow `languages` at the top of the file; the first (English) is `null`.
- `{0}` stands for a value a script interpolates; `<0>…</0>` for an inline element such as
  `<strong>`, which the translation may move; `<0/>` for one it must not touch.
- Icons and arrows around a phrase are not part of the key.
- Bump `MESSAGES_VERSION` in `i18n.js` so browsers drop their cached copy.

In the browser console, `LunaI18n.missing` lists the strings met on the current page that
have no translation. `tests/test_i18n_messages.py` checks the table; an E2E test walks every
page and fails if any interface text is left untranslated.

Not translated yet: chart labels, the long generated explanations with computed numbers
(case study findings, surface-age commentary), preset descriptions, literature references,
and text that comes from the server (mission descriptions, validation messages).

---

## API Reference

Interactive documentation for every endpoint is served at `/docs`.

| Area | Endpoints | Sign-in |
|------|-----------|---------|
| Accounts | `POST /api/auth/register`, `/login`, `/logout` · `GET /api/auth/me` | — |
| Simulations | `POST /api/simulations` · `GET`/`DELETE /api/experiments/{id}` · `GET /api/experiments`, `/{id}/profile`, `/{id}/elevation` · `POST /api/experiments/compare` | required |
| Challenges | `GET /api/challenges` · `POST /api/challenges/{id}/run` · `GET /api/challenges/history` | required, except the list |
| Moon | `GET /api/moon/elevation`, `/illumination`, `/terrain-texture` | — |
| Case studies | `GET /api/case-studies`, `/{id}` | — |
| Inference | `GET /api/inference/presets` · `POST /api/inference` | required to run |
| Surface age | `GET /api/surface-age/synthetic`, `/change6`, `/sites`, `/sites/{id}` | required for `synthetic` |
| Background jobs | `GET /api/jobs/types` · `POST /api/jobs` · `GET /api/jobs`, `/{id}` · `POST /api/jobs/{id}/cancel`, `/rerun` | required, except the types |

Register or log in to get a token, then send it as `Authorization: Bearer <token>`.
Sessions expire after 30 idle minutes. You only see your own experiments.

### Background jobs

Work too long for one request runs as a job: `POST /api/jobs` returns `202` with the job,
and the client polls `GET /api/jobs/{id}` for `status` (`queued`, `running`, `succeeded`,
`failed`, `cancelled`), `progress` (0–1) and, at the end, `result` or `error`.

```json
{ "type": "inference_batch", "params": { "crater_diameters": [18.8, 43.0, 2000] } }
```

`inference_batch` infers the impactor behind each of up to 500 craters, giving each the same
answer as `POST /api/inference`. A user may have 3 jobs queued or running; finished jobs are
kept for 7 days.

The queue is a database table and the worker a thread in the web process, because the free
hosting tier has no separate worker service and may restart or suspend the process at any
time. Jobs therefore survive it: on a clean shutdown the running job returns to the queue, and
one whose worker dies without a word is retried when its heartbeat has been silent for two
minutes (up to 3 starts). Jobs run one at a time, and only while the service is awake.
`LUNALAB_JOB_WORKER=0` disables the worker. See `database/jobs.py` and `backend/jobs.py`.

### Reproducible results

A stored result says which code produced it. Each successful job records a `provenance`:
a fingerprint of the science modules behind it (found by following imports from the module
the job type names, and hashed without comments, docstrings or formatting), the numpy
version, and the settings used. Nobody maintains a version number by hand.

- **Same input, same output.** `inference_batch` takes a `seed`; the same params give the
  same result bit for bit on one deployment (across machines, to floating-point rounding).
- **Outdated results are flagged.** A job reads `"outdated": true` once the fingerprint of the
  running code differs from the one it recorded, i.e. the model has changed since.
- **Refreshing is one call.** `POST /api/jobs/{id}/rerun` queues the same work again; the
  original is kept, so the two can be compared.

The fingerprint errs towards flagging: any code change in a module the result depends on
marks it outdated, including changes that happen not to move the numbers. It does not cover
data files. See `backend/provenance.py`.

The simulation endpoints in detail:

### `POST /api/simulations`

Run a new simulation and save it as an experiment.

**Request:**
```json
{
  "impactor_diameter": 10,
  "impact_velocity": 20000,
  "impact_angle": 45,
  "impactor_density": 3000,
  "target_density": 2550,
  "surface_gravity": 1.62
}
```

Only `impactor_diameter`, `impact_velocity`, and `impact_angle` are required. Others default to lunar conditions.

**Response:**
```json
{
  "experiment_id": 1,
  "crater_diameter": 577.3744,
  "crater_depth": 115.4749,
  "rim_height": 20.7855,
  "ejecta_volume": 13682243.42,
  "transient_crater_diameter": 461.8995,
  "energy_joules": 3.142e+14,
  "model": "Collins-Pike Lunar Scaling Model v0.2",
  "crater_type": "simple",
  "warnings": []
}
```

### `GET /api/experiments`

List your experiments (newest first).

### `GET /api/experiments/{id}`

Get a single experiment by ID.

### `GET /api/experiments/{id}/profile`

Get crater cross-section profile data (`x` and `y` arrays) for visualisation.

### `POST /api/experiments/compare`

Compare multiple experiments.

**Request:**
```json
{
  "experiment_ids": [1, 2, 3]
}
```

### `DELETE /api/experiments/{id}`

Delete an experiment.

---

## Simulation Model

### Collins-Pike Lunar Scaling Model v0.2

**1. Transient crater** — gravity-regime pi-scaling (Collins, Melosh & Marcus 2005, eq. 21):

```
D_tc = 1.161 · (ρ_i / ρ_t)^(1/3) · L^0.78 · v^0.44 · g^-0.22 · sin^(1/3)θ
```

`L` = impactor diameter, `v` = velocity, `g` = surface gravity, `θ` = angle from horizontal.

**2. Final crater** — the transient cavity collapses. On the Moon craters larger than
`D_c ≈ 15 km` (Pike 1977, scaled as 1/g) become *complex* craters with flat floors and central peaks;
above ~300 km they become multi-ring *basins*.

```
simple:  D = 1.25 · D_tc
complex: D = 1.17 · D_tc^1.13 / D_c^0.13
```

**3. Depth & rim height** — fresh lunar crater morphometry (Pike 1977):
simple `d ≈ 0.2 D`, `h ≈ 0.036 D`; complex `d ∝ D^0.301`, `h ∝ D^0.399` (anchored at `D_c`).

**4. Excavated volume** — `V ≈ π D_tc³ / (16√2)`.

**Defaults:** target density 2550 kg/m³ (lunar crust, GRAIL — Wieczorek et al. 2013),
impactor density 3000 kg/m³, gravity 1.62 m/s².

**Terrain visualisation** follows the oblique-impact regimes of Gault & Wedekind (1978):
asymmetric ejecta below 45°, an uprange forbidden zone below 25°, elongated craters below 15°,
and butterfly ejecta below 10°. Ejecta thickness decays as `(r/R)^-3` (McGetchin et al. 1973).

**Lunar surface data:** the 3D Moon is textured from NASA's [CGI Moon Kit](https://svs.gsfc.nasa.gov/4720):
the LROC colour mosaic, which has no shading baked in, and a normal map computed from LOLA slopes
(16 px/deg), so all relief shading follows the real Sun. The 4K colour map keeps one texel per
screen pixel at the default zoom up to a pixel ratio of 2. Elevations come
from the LOLA DEM (LDEM_4, 4 px/deg, [PDS Geosciences Node](https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/)):
`GET /api/moon/elevation?lat=&lon=` returns real elevation in metres relative to the
1737.4 km reference sphere; experiments store it as `surface_elevation`.

**Sunlight & libration:** `GET /api/moon/illumination?at=<ISO time>` returns the sub-solar point,
the sub-Earth point (optical libration) and the phase, from a Meeus ephemeris (*Astronomical
Algorithms*, ch. 25/47/48/53; checked against the book's worked examples to ~0.03°). The 3D Moon
uses it for the day/night terminator and to start the camera where Earth sees the Moon today.

**Research XP:** each simulation earns +15 XP for the first 10 per UTC day. A challenge earns the XP
for its grade (S 150 · A 100 · B 60 · C 25) only above your best previous grade on that mission.

**Not modelled:** target strength (small craters in cohesive regolith), porosity, melt, and
hydrocode-level oblique-impact physics.

---

## Input Validation

| Parameter | Valid Range | Error |
|-----------|-----------|-------|
| Impactor Diameter | > 0 | `422 Unprocessable Entity` |
| Impact Velocity | > 0 | `422 Unprocessable Entity` |
| Impact Angle | (0, 90] | `422 Unprocessable Entity` |

**Soft warnings** (non-blocking) are returned when parameters exceed the model's comfortable range:
- Diameter > 100 km or < 0.01 m
- Velocity > 72 km/s (no natural impactor) or < 2.38 km/s (lunar escape velocity)
- Final crater > 300 km (multi-ring basin) or larger than the Moon
- Angle < 10°

---

## Risk Management

| Risk | Mitigation |
|------|-----------|
| Scientific Model Risk | Model is clearly labelled as simplified, and tested against observed impacts and sample ages (`tests/test_real_impacts.py`, case study, Chang'e-6). |
| Scope Risk | MVP priority levels (P0–P3). Architecture supports extension without rewrite. |
| Technical Learning Risk | Simple stack: FastAPI + Vanilla JS + plain SQL. No unnecessary frameworks. |
| Input / Numerical Risk | Hard validation + soft warnings + boundary checking. |
| Integration Risk | Simulation, API, and Database are independently testable modules; CI runs the database tests on PostgreSQL and the user flows in a browser. |
| Data Loss Risk | Production data lives in PostgreSQL, not on the host's disk; schema changes are numbered migrations applied in a single transaction. |

---

## Roadmap

Done since the first version: validation against observed impacts, a porous-regolith
scaling model, 3D terrain, PostgreSQL for multi-user deployment, background jobs.

Next:

- Parameter sweeps as a second kind of background job
- Regolith scaling as an option for ordinary simulations (the model has it; the API does not expose it yet)
- CSV export beyond batch results (experiments, history)

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, FastAPI |
| Frontend | HTML, CSS, Vanilla JavaScript |
| 3D | Three.js |
| Visualisation | Plotly.js |
| Database | SQLite (local), PostgreSQL (production) |
| Testing | pytest, Playwright, mutmut |
| CI | GitHub Actions |

---

## License

This is a course project / prototype. All rights reserved.

## Deploying on Render (with Supabase)

Render's free tier wipes the local disk on every restart, redeploy and
spin-down, so a SQLite file there loses all accounts and experiments.
In production, store data in PostgreSQL instead:

1. In Supabase: **Connect** → copy the **Session pooler** URI
   (Render has no IPv6, and the direct connection is IPv6-only).
2. In Render: **Environment** → add `DATABASE_URL` = that URI, with your
   database password filled in.
3. Redeploy. Tables are created automatically on startup.

Without `DATABASE_URL`, LunaLab uses the local SQLite file as before.
Run the test suite with `DATABASE_URL` unset, so it never touches the
production database.

---

## Data Sources

| File | Source | License |
|------|--------|---------|
| `terrain/data/lola_ldem4_m.npy` | NASA LRO LOLA LDEM_4 | Public domain |
| `frontend/textures/moon_color_4k.jpg` | LROC colour map, [CGI Moon Kit](https://svs.gsfc.nasa.gov/4720) (NASA's Scientific Visualization Studio) | Public domain (NASA) |
| `frontend/textures/moon_normal_4k.jpg` | Computed from LOLA `ldem_16.tif`, [CGI Moon Kit](https://svs.gsfc.nasa.gov/4720) (NASA's Scientific Visualization Studio) | Public domain (NASA) |
| `simulation/data/landing_site_craters.json` | Robbins (2018) lunar crater database, via [Zenodo record 21500453](https://zenodo.org/records/21500453) | CC BY 4.0 |
| `simulation/data/landing_site_craters.json` (unit flags, areas, outlines) | [Unified Geologic Map of the Moon, 1:5M](https://astrogeology.usgs.gov/search/map/unified_geologic_map_of_the_moon_1_5m_2020) (Fortezzo, Spudis & Harrel 2020), USGS | Public domain |
| `simulation/data/change6_count.json` | Chang'e-6 landing-site crater count, [Craterstats](https://github.com/ggmichael/craterstats) sample data (G. Michael) | BSD 3-Clause |

The Moon textures are rebuilt with `python scripts/build_moon_textures.py lroc_color_poles_8k.tif ldem_16.tif`
(download links in the script).

The crater files are rebuilt from the originals with
`python scripts/build_crater_data.py ROBBINS_CSV CHANGE6_SCC GEOUNITS_SHP` (see the script for download
links; reading the shapefile needs `pip install pyshp`, used only by this script).
