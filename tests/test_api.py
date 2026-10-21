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
        login_as(c, "api-tester")
        yield c
    # Cleanup
    os.unlink(_temp_db.name)


def login_as(c, username):
    """Register a user and send its token on every request from this client."""
    res = c.post("/api/auth/register", json={"username": username, "password": "pass1234"})
    if res.status_code != 200:
        res = c.post("/api/auth/login", json={"username": username, "password": "pass1234"})
    c.headers["Authorization"] = f"Bearer {res.json()['token']}"
    return res.json()["user"]


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


# ---------------------------------------------------------------------------
# GET /api/experiments/{id}/elevation
# ---------------------------------------------------------------------------

class TestGetElevation:
    def test_elevation_returns_200(self, client):
        create = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        eid = create.json()["experiment_id"]
        res = client.get(f"/api/experiments/{eid}/elevation")
        assert res.status_code == 200

    def test_elevation_response_has_required_keys(self, client):
        create = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        eid = create.json()["experiment_id"]
        data = client.get(f"/api/experiments/{eid}/elevation").json()

        for key in ("x", "y", "z", "z_min", "z_max", "width", "height", "gen_time_ms"):
            assert key in data, f"Missing key: {key}"

    def test_elevation_grid_shape_is_1024x512(self, client):
        create = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        eid = create.json()["experiment_id"]
        data = client.get(f"/api/experiments/{eid}/elevation").json()

        assert data["width"]  == 1024
        assert data["height"] == 512
        assert len(data["x"]) == 1024
        assert len(data["y"]) == 512
        assert len(data["z"]) == 512
        # Spot-check a few rows (checking all 512 rows would be slow)
        for row_idx in (0, 255, 511):
            assert len(data["z"][row_idx]) == 1024, \
                f"Row {row_idx} has wrong length: {len(data['z'][row_idx])}"

    def test_elevation_z_min_is_negative(self, client):
        """Crater floor must be below the surrounding terrain."""
        create = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        eid = create.json()["experiment_id"]
        data = client.get(f"/api/experiments/{eid}/elevation").json()
        assert data["z_min"] < 0

    def test_elevation_z_max_is_positive(self, client):
        """Rim must be above the surrounding terrain."""
        create = client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        eid = create.json()["experiment_id"]
        data = client.get(f"/api/experiments/{eid}/elevation").json()
        assert data["z_max"] > 0

    def test_elevation_nonexistent_experiment_404(self, client):
        res = client.get("/api/experiments/999999/elevation")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Access control & input limits
# ---------------------------------------------------------------------------

SIM = {"impactor_diameter": 10, "impact_velocity": 20000, "impact_angle": 45}


class TestAccessControl:
    def test_endpoints_require_login(self):
        with TestClient(app) as anon:
            assert anon.post("/api/simulations", json=SIM).status_code == 401
            assert anon.get("/api/experiments").status_code == 401
            assert anon.get("/api/experiments/1").status_code == 401
            assert anon.delete("/api/experiments/1").status_code == 401
            assert anon.post("/api/challenges/tycho_crater/run", json={"simulation": SIM}).status_code == 401

    def test_public_endpoints_stay_open(self):
        with TestClient(app) as anon:
            assert anon.get("/api/challenges").status_code == 200
            assert anon.get("/api/moon/elevation", params={"lat": 0, "lon": 0}).status_code == 200

    def test_other_users_cannot_see_or_delete(self, client):
        eid = client.post("/api/simulations", json=SIM).json()["experiment_id"]
        with TestClient(app) as other:
            login_as(other, "intruder")
            assert other.get(f"/api/experiments/{eid}").status_code == 404
            assert other.get(f"/api/experiments/{eid}/profile").status_code == 404
            assert other.delete(f"/api/experiments/{eid}").status_code == 404
            ids = [e["id"] for e in other.get("/api/experiments").json()["experiments"]]
            assert eid not in ids
        assert client.get(f"/api/experiments/{eid}").status_code == 200

    def test_legacy_unowned_experiments_are_shared(self, client):
        from database.models import save_experiment
        legacy_id = save_experiment({
            "impactor_diameter": 1, "impact_velocity": 1, "impact_angle": 45,
            "impactor_density": 3000, "target_density": 2550, "surface_gravity": 1.62,
            "crater_diameter": 10, "crater_depth": 2, "rim_height": 0.4, "ejecta_volume": 1,
            "transient_crater_diameter": 8, "energy_joules": 1, "model_name": "legacy",
        }, db_path=_temp_db.name)
        assert client.get(f"/api/experiments/{legacy_id}").status_code == 200


class TestInputLimits:
    @pytest.mark.parametrize("field,value", [
        ("impactor_diameter", 1e100),
        ("impactor_diameter", 1_000_001),
        ("impact_velocity", 100_001),
    ])
    def test_values_above_limit_rejected_not_500(self, client, field, value):
        assert client.post("/api/simulations", json={**SIM, field: value}).status_code == 422

    def test_values_at_limit_accepted(self, client):
        res = client.post("/api/simulations", json={**SIM, "impactor_diameter": 1_000_000, "impact_velocity": 100_000})
        assert res.status_code == 200

    def test_azimuth_is_normalised(self, client):
        res = client.post("/api/simulations", json={**SIM, "impact_azimuth": -90})
        assert res.json()["impact_azimuth"] == 270

    def test_latitude_without_longitude_rejected(self, client):
        assert client.post("/api/simulations", json={**SIM, "latitude": 10}).status_code == 422
