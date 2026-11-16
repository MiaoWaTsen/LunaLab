# LunaLab

**Lunar Impact Simulation & Experiment Analysis Platform**

> ⚠️ This prototype uses a simplified crater scaling model for educational and experimental purposes. It is not a full physical simulation of impact processes.

---

## Overview

LunaLab is a modular, web-based platform for simulating lunar impact craters using a **lunar crater scaling model** (Collins et al. 2005 pi-scaling + Pike 1977 lunar morphometry). Users can:

1. **Run Simulations** — Input impactor parameters and compute crater dimensions
2. **Visualise Results** — Interactive Plotly charts showing crater profiles and parameter comparisons
3. **Save Experiments** — Every simulation is persisted to a SQLite database
4. **Browse History** — View, inspect, and manage past experiments
5. **Compare Experiments** — Select multiple experiments for side-by-side visual comparison
6. **Case Study** — Test the model against the 17 March 2013 impact, where both the flash and the LRO crater were measured: observing geometry from the ephemeris, Monte Carlo uncertainty propagation, and inversion for the luminous efficiency η
7. **Impactor Inference** — Bayesian inversion from an observed crater back to the impactor: posterior over size, velocity, angle and density, the size–speed degeneracy, and (for impact flashes) the luminous efficiency; validated by simulation-based calibration
8. **Surface Age** — Date a surface by counting its craters (Neukum 2001 production and chronology functions, Poisson likelihood): watch a synthetic plain collect craters over 4 billion years, see small craters saturate, then test the method on the Chang'e-6 landing site against its 2.807 Ga sample age, and on seven sampled landing sites counted in circles vs on their mapped geological units

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
  Database Module (SQLite)
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
│   │                        #   experiments, moon, science, pages)
│   ├── services.py          # Use cases shared between routers
│   ├── deps.py              # Authentication dependencies
│   └── schemas.py           # Pydantic request/response models
│
├── simulation/              # Simulation engine (independent)
│   ├── __init__.py
│   ├── crater_model.py      # Crater scaling (Collins/Pike)
│   ├── case_studies.py      # Model vs. the observed 2013-03-17 impact
│   ├── inference.py         # Bayesian crater → impactor inversion
│   ├── chronology.py        # Crater production/chronology functions, age inference
│   ├── surface_dating.py    # Synthetic bombardment, Chang'e-6, landing sites
│   └── data/                # Crater counts (built by scripts/build_crater_data.py)
│
├── database/                # Data persistence layer
│   ├── __init__.py
│   └── models.py            # SQLite schema & CRUD operations
│
├── frontend/                # Static frontend files
│   ├── index.html           # Dashboard
│   ├── experiment.html      # New experiment / view experiment
│   ├── history.html         # Experiment history
│   ├── compare.html         # Experiment comparison
│   ├── styles.css           # Design system
│   └── app.js               # Shared JavaScript utilities
│
├── tests/                   # Automated tests
│   ├── __init__.py
│   ├── test_simulation.py   # Simulation module tests (30 tests)
│   ├── test_api.py          # API endpoint tests (15 tests)
│   └── test_database.py     # Database CRUD tests (12 tests)
│
└── lunalab.db               # SQLite database (auto-created)
```

### Module Responsibilities

| Module | Responsibility | Dependencies |
|--------|---------------|-------------|
| `simulation/` | Crater scaling calculations, validation, profile generation | None (pure Python + math) |
| `database/` | SQLite schema, CRUD operations | sqlite3 (stdlib) |
| `backend/` | HTTP routing, request validation, orchestration | FastAPI, simulation, database |
| `frontend/` | UI, user interaction, visualisation | Plotly.js (CDN) |
| `tests/` | Automated verification | pytest |

**Key design principle:** The simulation module can be imported and tested independently without any web server or database.

---

## Getting Started

### Prerequisites

- Python 3.10+
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
python3 -m pytest tests/ -v
```

---

## API Reference

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

List all experiments (newest first).

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

**Lunar surface data:** the 3D Moon uses NASA LRO imagery for colour and the LOLA laser-altimeter
DEM (LDEM_4, 4 px/deg, [PDS Geosciences Node](https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/))
for relief. `GET /api/moon/elevation?lat=&lon=` returns real elevation in metres relative to the
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
| Scientific Model Risk | Model is clearly labelled as simplified/prototype. Future: validate against reference cases. |
| Scope Risk | MVP priority levels (P0–P3). Architecture supports extension without rewrite. |
| Technical Learning Risk | Simple stack: FastAPI + Vanilla JS + SQLite. No unnecessary frameworks. |
| Input / Numerical Risk | Hard validation + soft warnings + boundary checking. |
| Integration Risk | Simulation, API, and Database are independently testable modules. |

---

## Future Roadmap

- **P1:** Multiple scaling models (e.g. strength-regime model)
- **P2:** Reference data validation with known craters
- **P2:** CSV / report export
- **P3:** 3D terrain visualisation
- **P3:** PostgreSQL for multi-user deployment

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, FastAPI |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Visualisation | Plotly.js |
| Database | SQLite |
| Testing | pytest |

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
| `simulation/data/landing_site_craters.json` | Robbins (2018) lunar crater database, via [Zenodo record 21500453](https://zenodo.org/records/21500453) | CC BY 4.0 |
| `simulation/data/landing_site_craters.json` (unit flags, areas, outlines) | [Unified Geologic Map of the Moon, 1:5M](https://astrogeology.usgs.gov/search/map/unified_geologic_map_of_the_moon_1_5m_2020) (Fortezzo, Spudis & Harrel 2020), USGS | Public domain |
| `simulation/data/change6_count.json` | Chang'e-6 landing-site crater count, [Craterstats](https://github.com/ggmichael/craterstats) sample data (G. Michael) | BSD 3-Clause |

The crater files are rebuilt from the originals with
`python scripts/build_crater_data.py ROBBINS_CSV CHANGE6_SCC GEOUNITS_SHP` (see the script for download
links; reading the shapefile needs `pip install pyshp`, used only by this script).
