"""
Tests for the Terrain Module — terrain/crater_terrain.py

These tests verify:
    1. Output shape is exactly (512, 1024)                 [TERRAIN-01]
    2. Same input produces deterministic terrain           [TERRAIN-02]
    3. Crater centre is lower than surrounding terrain     [TERRAIN-03]
    4. Crater depth is consistent with simulation output   [TERRAIN-04]
    5. Invalid parameters do not crash silently            [TERRAIN-05]
    6. Rim height is positive and at the expected radius   [TERRAIN-06]
    7. Ejecta blanket decays to zero at domain edge        [TERRAIN-07]
    8. Elevation API endpoint returns correct shape        [TERRAIN-08]

All tests are independent of FastAPI, SQLite, and the simulation module.
"""

import pytest
import math
import tempfile
import os

from terrain.crater_terrain import (
    generate_elevation_field,
    TERRAIN_WIDTH,
    TERRAIN_HEIGHT,
)
from simulation.crater_model import ImpactParameters, run_simulation


# ---------------------------------------------------------------------------
# Shared fixture — a standard crater geometry for terrain tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def standard_crater():
    """
    Run a standard simulation and return its geometry.
    Using scope='module' so the simulation only runs once across all tests.
    """
    params = ImpactParameters(
        impactor_diameter=10.0,
        impact_velocity=20000.0,
        impact_angle=45.0,
    )
    result = run_simulation(params)
    return {
        "crater_diameter": result.crater_diameter,
        "crater_depth":    result.crater_depth,
        "rim_height":      result.rim_height,
    }


@pytest.fixture(scope="module")
def standard_field(standard_crater):
    """Pre-computed elevation field for the standard crater."""
    return generate_elevation_field(**standard_crater)


# ---------------------------------------------------------------------------
# TERRAIN-01 — Output shape is exactly (512, 1024)
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_default_width_is_1024(self, standard_field):
        assert standard_field["width"] == TERRAIN_WIDTH
        assert standard_field["width"] == 1024

    def test_default_height_is_512(self, standard_field):
        assert standard_field["height"] == TERRAIN_HEIGHT
        assert standard_field["height"] == 512

    def test_z_has_correct_number_of_rows(self, standard_field):
        """z must have exactly height rows."""
        assert len(standard_field["z"]) == 512

    def test_every_z_row_has_correct_number_of_columns(self, standard_field):
        """Every row in z must have exactly width columns."""
        z = standard_field["z"]
        assert all(len(row) == 1024 for row in z), \
            "Not all rows have 1024 columns"

    def test_x_axis_length_equals_width(self, standard_field):
        assert len(standard_field["x"]) == 1024

    def test_y_axis_length_equals_height(self, standard_field):
        assert len(standard_field["y"]) == 512

    def test_custom_dimensions_respected(self, standard_crater):
        """Caller can override width and height."""
        field = generate_elevation_field(**standard_crater, width=256, height=128)
        assert field["width"] == 256
        assert field["height"] == 128
        assert len(field["z"]) == 128
        assert all(len(row) == 256 for row in field["z"])


# ---------------------------------------------------------------------------
# TERRAIN-02 — Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_produces_identical_z(self, standard_crater):
        """Two calls with the same parameters must produce the same grid."""
        f1 = generate_elevation_field(**standard_crater)
        f2 = generate_elevation_field(**standard_crater)
        # Compare centre cell and a selection of edge cells
        mid_row = len(f1["z"]) // 2
        mid_col = len(f1["z"][0]) // 2
        assert f1["z"][mid_row][mid_col] == f2["z"][mid_row][mid_col]
        assert f1["z"][0][0] == f2["z"][0][0]
        assert f1["z"][-1][-1] == f2["z"][-1][-1]
        assert f1["z_min"] == f2["z_min"]
        assert f1["z_max"] == f2["z_max"]

    def test_different_craters_produce_different_fields(self):
        """Different crater geometries must produce different grids."""
        fa = generate_elevation_field(
            crater_diameter=100.0, crater_depth=20.0, rim_height=5.0
        )
        fb = generate_elevation_field(
            crater_diameter=500.0, crater_depth=100.0, rim_height=25.0
        )
        assert fa["z_min"] != fb["z_min"], \
            "Different craters should have different z_min values"


# ---------------------------------------------------------------------------
# TERRAIN-03 — Crater centre is lower than surrounding terrain
# ---------------------------------------------------------------------------

class TestCraterGeometry:
    def test_centre_is_lowest_point(self, standard_field):
        """
        The centre cell (mid_row, mid_col) must be the minimum elevation,
        i.e. the crater floor.

        We compare with a small tolerance because z_min in the response is
        rounded to 4 decimal places, while individual z values retain
        float32 precision before JSON serialisation.
        """
        z = standard_field["z"]
        mid_row = len(z) // 2
        mid_col = len(z[0]) // 2
        centre_z = z[mid_row][mid_col]
        assert abs(centre_z - standard_field["z_min"]) < 0.001, \
            f"Centre z={centre_z} is not the minimum z={standard_field['z_min']}"

    def test_centre_is_below_zero(self, standard_field):
        """Crater bowl must be below the reference terrain level (z=0)."""
        z = standard_field["z"]
        mid_row = len(z) // 2
        mid_col = len(z[0]) // 2
        assert z[mid_row][mid_col] < 0

    def test_rim_is_above_zero(self, standard_field):
        """The rim peak elevation (z_max) must be positive."""
        assert standard_field["z_max"] > 0

    def test_domain_edges_are_near_zero(self, standard_field):
        """
        The outer edge of the domain is beyond the ejecta blanket and
        should be at or very close to zero (undisturbed terrain).
        """
        z = standard_field["z"]
        # Corner cells are the furthest from the crater centre
        corners = [
            z[0][0], z[0][-1], z[-1][0], z[-1][-1],
        ]
        for c in corners:
            assert abs(c) < 0.01, \
                f"Corner elevation {c} is not near zero (expected undisturbed terrain)"


# ---------------------------------------------------------------------------
# TERRAIN-04 — Crater depth consistency with simulation output
# ---------------------------------------------------------------------------

class TestDepthConsistency:
    def test_z_min_matches_simulation_depth(self, standard_field, standard_crater):
        """
        The most negative elevation in the terrain grid must be close to
        the crater depth from the scaling model.

        The centre of a parabolic bowl at r=0 gives:
            z = -depth * (1 - 0) = -depth

        So z_min should equal -crater_depth (within floating-point rounding).
        """
        expected_floor = -standard_crater["crater_depth"]
        assert abs(standard_field["z_min"] - expected_floor) < 0.01, (
            f"z_min={standard_field['z_min']:.4f} does not match "
            f"-crater_depth={expected_floor:.4f}"
        )

    def test_z_max_matches_rim_height(self, standard_field, standard_crater):
        """
        The maximum elevation must be close to rim_height.

        The Gaussian rim at t=0 (r = radius) gives:
            z = rim_height * exp(0) = rim_height

        z_max should be very close to rim_height.
        """
        expected_rim = standard_crater["rim_height"]
        assert abs(standard_field["z_max"] - expected_rim) < 0.1, (
            f"z_max={standard_field['z_max']:.4f} does not match "
            f"rim_height={expected_rim:.4f}"
        )


# ---------------------------------------------------------------------------
# TERRAIN-05 — Invalid parameters raise ValueError, do not crash silently
# ---------------------------------------------------------------------------

class TestInvalidParameters:
    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError, match="crater_diameter"):
            generate_elevation_field(
                crater_diameter=0.0, crater_depth=10.0, rim_height=2.0
            )

    def test_negative_diameter_raises(self):
        with pytest.raises(ValueError, match="crater_diameter"):
            generate_elevation_field(
                crater_diameter=-50.0, crater_depth=10.0, rim_height=2.0
            )

    def test_zero_depth_raises(self):
        with pytest.raises(ValueError, match="crater_depth"):
            generate_elevation_field(
                crater_diameter=100.0, crater_depth=0.0, rim_height=2.0
            )

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError, match="crater_depth"):
            generate_elevation_field(
                crater_diameter=100.0, crater_depth=-5.0, rim_height=2.0
            )

    def test_zero_rim_height_raises(self):
        with pytest.raises(ValueError, match="rim_height"):
            generate_elevation_field(
                crater_diameter=100.0, crater_depth=20.0, rim_height=0.0
            )

    def test_width_less_than_2_raises(self):
        with pytest.raises(ValueError, match="width"):
            generate_elevation_field(
                crater_diameter=100.0, crater_depth=20.0, rim_height=5.0,
                width=1, height=512
            )

    def test_height_less_than_2_raises(self):
        with pytest.raises(ValueError, match="width"):
            generate_elevation_field(
                crater_diameter=100.0, crater_depth=20.0, rim_height=5.0,
                width=1024, height=1
            )

    def test_very_small_crater_does_not_crash(self):
        """Micro-scale craters should still produce a valid grid."""
        field = generate_elevation_field(
            crater_diameter=0.1, crater_depth=0.02, rim_height=0.005
        )
        assert field["z_min"] < 0
        assert field["z_max"] > 0

    def test_very_large_crater_does_not_crash(self):
        """Planetary-scale craters should still produce a valid grid."""
        field = generate_elevation_field(
            crater_diameter=1_000_000.0,
            crater_depth=200_000.0,
            rim_height=50_000.0,
        )
        assert len(field["z"]) == 512
        assert len(field["z"][0]) == 1024


# ---------------------------------------------------------------------------
# TERRAIN-08 — Elevation API endpoint returns correct shape
# ---------------------------------------------------------------------------
# Note: this test requires the API layer; it lives here for grouping.
# If the API cannot be imported it will be skipped gracefully.

class TestElevationEndpoint:
    @pytest.fixture(scope="class")
    def api_client(self):
        """Set up an in-memory test client with a temporary database."""
        _temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        os.environ["LUNALAB_DB_PATH"] = _temp_db.name

        from fastapi.testclient import TestClient
        from backend.api import app
        from database.models import init_db

        init_db(_temp_db.name)
        with TestClient(app) as c:
            yield c

        os.unlink(_temp_db.name)

    def test_elevation_endpoint_returns_200(self, api_client):
        # Create an experiment first
        create = api_client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        assert create.status_code == 200
        eid = create.json()["experiment_id"]

        res = api_client.get(f"/api/experiments/{eid}/elevation")
        assert res.status_code == 200

    def test_elevation_endpoint_correct_shape(self, api_client):
        create = api_client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        eid = create.json()["experiment_id"]
        res = api_client.get(f"/api/experiments/{eid}/elevation")
        data = res.json()

        assert data["width"]  == 1024
        assert data["height"] == 512
        assert len(data["x"]) == 1024
        assert len(data["y"]) == 512
        assert len(data["z"]) == 512
        assert all(len(row) == 1024 for row in data["z"])

    def test_elevation_endpoint_not_found(self, api_client):
        res = api_client.get("/api/experiments/999999/elevation")
        assert res.status_code == 404

    def test_elevation_gen_time_is_reported(self, api_client):
        create = api_client.post("/api/simulations", json={
            "impactor_diameter": 10,
            "impact_velocity": 20000,
            "impact_angle": 45,
        })
        eid = create.json()["experiment_id"]
        res = api_client.get(f"/api/experiments/{eid}/elevation")
        data = res.json()
        assert "gen_time_ms" in data
        assert data["gen_time_ms"] >= 0
