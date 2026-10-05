"""
Tests for the API endpoints.

Uses FastAPI TestClient (via httpx) so no real server is needed.
Database uses a temporary file to avoid polluting the main DB.
"""

import pytest
import tempfile
import os
import json

from fastapi.testclient import TestClient

# Override DB path BEFORE importing the app
_temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["LUNALAB_DB_PATH"] = _temp_db.name

from backend.api import app
from database.models import init_db


@pytest.fixture(scope="module")
def client():
    """Create a test client with a fresh temporary database."""
    init_db(_temp_db.name)
    with TestClient(app) as c:
        yield c
    # Cleanup
    os.unlink(_temp_db.name)


# ---------------------------------------------------------------------------
# POST /api/simulations
# ---------------------------------------------------------------------------

class TestCreateSimulation:
    def test_valid_simulation(self, client):
        res = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        assert res.status_code == 200
        data = res.json()
        assert "experiment_id" in data
        assert data["crater_diameter"] > 0
        assert data["crater_depth"] > 0
        assert "model" in data

    def test_invalid_diameter_rejected(self, client):
        res = client.post("/api/simulations", json={
            "impactor_diameter": -5,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        assert res.status_code == 422

    def test_zero_velocity_rejected(self, client):
        res = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 0,
            "impact_angle": 45,
        })
        assert res.status_code == 422

    def test_invalid_angle_rejected(self, client):
        res = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 100,
        })
        assert res.status_code == 422

    def test_missing_field_rejected(self, client):
        res = client.post("/api/simulations", json={
            "impactor_diameter": 10,
        })
        assert res.status_code == 422

    def test_optional_fields_accepted(self, client):
        res = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
            "impactor_density": 3500,
            "target_density": 2800,
            "surface_gravity": 1.62,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["experiment_id"] > 0

    def test_warnings_returned_for_extreme_values(self, client):
        res = client.post("/api/simulations", json={
            "impactor_diameter": 500_000,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        assert res.status_code == 200
        data = res.json()
        assert len(data.get("warnings", [])) > 0


# ---------------------------------------------------------------------------
# GET /api/experiments
# ---------------------------------------------------------------------------

class TestGetExperiments:
    def test_list_experiments(self, client):
        # Ensure at least one experiment exists
        client.post("/api/simulations", json={
            "impactor_diameter": 5,
            "impact_velocity": 15000,
            "impact_angle": 60,
        })
        res = client.get("/api/experiments")
        assert res.status_code == 200
        data = res.json()
        assert "experiments" in data
        assert len(data["experiments"]) >= 1

    def test_get_single_experiment(self, client):
        # Create one
        create = client.post("/api/simulations", json={
            "impactor_diameter": 8,
            "impact_velocity": 18000,
            "impact_angle": 50,
        })
        eid = create.json()["experiment_id"]

        res = client.get(f"/api/experiments/{eid}")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == eid
        assert data["impactor_diameter"] == 8
        assert data["impact_velocity"] == 18000

    def test_get_nonexistent_experiment_404(self, client):
        res = client.get("/api/experiments/999999")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/experiments/{id}/profile
# ---------------------------------------------------------------------------

class TestGetProfile:
    def test_profile_returns_xy(self, client):
        create = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        eid = create.json()["experiment_id"]

        res = client.get(f"/api/experiments/{eid}/profile")
        assert res.status_code == 200
        data = res.json()
        assert "x" in data
        assert "y" in data
        assert len(data["x"]) > 50


# ---------------------------------------------------------------------------
# POST /api/experiments/compare
# ---------------------------------------------------------------------------

class TestCompareExperiments:
    def test_compare_two_experiments(self, client):
        # Create two experiments
        r1 = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        r2 = client.post("/api/simulations", json={
            "impactor_diameter": 20,
            "impact_velocity": 25000,
            "impact_angle": 60,
        })
        id1 = r1.json()["experiment_id"]
        id2 = r2.json()["experiment_id"]

        res = client.post("/api/experiments/compare", json={
            "experiment_ids": [id1, id2],
        })
        assert res.status_code == 200
        data = res.json()
        assert len(data["experiments"]) == 2
        # Each should have a profile
        for exp in data["experiments"]:
            assert "profile" in exp
            assert "x" in exp["profile"]

    def test_compare_requires_two_ids(self, client):
        res = client.post("/api/experiments/compare", json={
            "experiment_ids": [1],
        })
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /api/experiments/{id}
# ---------------------------------------------------------------------------

class TestDeleteExperiment:
    def test_delete_experiment(self, client):
        create = client.post("/api/simulations", json={
            "impactor_diameter": 3,
            "impact_velocity": 12000,
            "impact_angle": 30,
        })
        eid = create.json()["experiment_id"]

        res = client.delete(f"/api/experiments/{eid}")
        assert res.status_code == 200

        # Should be gone
        res2 = client.get(f"/api/experiments/{eid}")
        assert res2.status_code == 404

    def test_delete_nonexistent_404(self, client):
        res = client.delete("/api/experiments/999999")
        assert res.status_code == 404
