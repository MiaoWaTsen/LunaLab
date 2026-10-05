# LunaLab

**Lunar Impact Simulation & Experiment Analysis Platform**

> ⚠️ This prototype uses a simplified crater scaling model for educational and experimental purposes. It is not a full physical simulation of impact processes.

---

## Overview

LunaLab is a modular, web-based platform for simulating lunar impact craters using a simplified **Pi-Scaling Crater Model**. Users can:

1. **Run Simulations** — Input impactor parameters and compute crater dimensions
2. **Visualise Results** — Interactive Plotly charts showing crater profiles and parameter comparisons
3. **Save Experiments** — Every simulation is persisted to a SQLite database
4. **Browse History** — View, inspect, and manage past experiments
5. **Compare Experiments** — Select multiple experiments for side-by-side visual comparison

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
│   ├── api.py               # FastAPI routes & app factory
│   └── schemas.py           # Pydantic request/response models
│
├── simulation/              # Simulation engine (independent)
│   ├── __init__.py
│   └── crater_model.py      # Pi-Scaling crater calculations
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

The server starts at **http://localhost:8000**.

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
  "target_density": 2500,
  "surface_gravity": 1.62
}
```

Only `impactor_diameter`, `impact_velocity`, and `impact_angle` are required. Others default to lunar conditions.

**Response:**
```json
{
  "experiment_id": 1,
  "crater_diameter": 120.5,
  "crater_depth": 24.1,
  "rim_height": 6.025,
  "ejecta_volume": 50000.0,
  "transient_crater_diameter": 100.4,
  "energy_joules": 3.14e+12,
  "model": "Pi-Scaling Prototype Model v0.1",
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

### Pi-Scaling Prototype Model v0.1

The model uses a simplified **pi-scaling** approach (gravity-dominated regime):

```
D_tc = K₁ · d · (ρ_i / ρ_t)^(1/3) · (v² / (g · a))^μ
```

Where:
- `d` = impactor diameter
- `ρ_i / ρ_t` = density ratio (impactor / target)
- `v` = impact velocity
- `g` = surface gravity
- `a` = impactor radius
- `K₁ ≈ 1.03`, `μ ≈ 0.22` (approximate coefficients)

Angle correction uses `sin(θ)^(2μ)` scaling.

Final crater diameter ≈ 1.2 × transient crater diameter.

**References (simplified from):**
- Schmidt, R.M. & Housen, K.R. (1987)
- Holsapple, K.A. (1993)

---

## Input Validation

| Parameter | Valid Range | Error |
|-----------|-----------|-------|
| Impactor Diameter | > 0 | `422 Unprocessable Entity` |
| Impact Velocity | > 0 | `422 Unprocessable Entity` |
| Impact Angle | (0, 90] | `422 Unprocessable Entity` |

**Soft warnings** (non-blocking) are returned when parameters exceed the model's comfortable range:
- Diameter > 100 km or < 0.01 m
- Velocity > 100 km/s or < 1 km/s
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
