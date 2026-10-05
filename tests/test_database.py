"""
Tests for the Database Module.

Tests CRUD operations independently using a temporary database.
"""

import pytest
import tempfile
import os
import json

from database.models import (
    init_db,
    save_experiment,
    get_experiment,
    get_all_experiments,
    get_experiments_by_ids,
    delete_experiment,
)


@pytest.fixture
def db_path():
    """Create a temporary database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


def _sample_experiment_data(**overrides):
    """Generate a sample experiment dict for testing."""
    data = {
        "impactor_diameter": 10.0,
        "impact_velocity": 20000.0,
        "impact_angle": 45.0,
        "impactor_density": 3000.0,
        "target_density": 2500.0,
        "surface_gravity": 1.62,
        "crater_diameter": 120.5,
        "crater_depth": 24.1,
        "rim_height": 6.025,
        "ejecta_volume": 50000.0,
        "transient_crater_diameter": 100.4,
        "energy_joules": 3.14e12,
        "model_name": "Pi-Scaling Prototype Model v0.1",
        "warnings_json": "[]",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Save & retrieve
# ---------------------------------------------------------------------------

class TestSaveAndRetrieve:
    def test_save_returns_id(self, db_path):
        data = _sample_experiment_data()
        eid = save_experiment(data, db_path)
        assert isinstance(eid, int)
        assert eid >= 1

    def test_retrieve_saved_experiment(self, db_path):
        data = _sample_experiment_data()
        eid = save_experiment(data, db_path)
        exp = get_experiment(eid, db_path)
        assert exp is not None
        assert exp["id"] == eid
        assert exp["impactor_diameter"] == 10.0
        assert exp["crater_diameter"] == 120.5
        assert exp["model_name"] == "Pi-Scaling Prototype Model v0.1"

    def test_retrieve_nonexistent_returns_none(self, db_path):
        exp = get_experiment(9999, db_path)
        assert exp is None

    def test_auto_generated_timestamp(self, db_path):
        data = _sample_experiment_data()
        eid = save_experiment(data, db_path)
        exp = get_experiment(eid, db_path)
        assert exp["created_at"] is not None
        assert len(exp["created_at"]) > 0


# ---------------------------------------------------------------------------
# List all experiments
# ---------------------------------------------------------------------------

class TestListExperiments:
    def test_list_empty(self, db_path):
        exps = get_all_experiments(db_path)
        assert exps == []

    def test_list_multiple(self, db_path):
        save_experiment(_sample_experiment_data(impactor_diameter=5), db_path)
        save_experiment(_sample_experiment_data(impactor_diameter=10), db_path)
        save_experiment(_sample_experiment_data(impactor_diameter=15), db_path)
        exps = get_all_experiments(db_path)
        assert len(exps) == 3

    def test_list_ordered_by_date(self, db_path):
        save_experiment(_sample_experiment_data(created_at="2024-01-01T00:00:00"), db_path)
        save_experiment(_sample_experiment_data(created_at="2024-06-01T00:00:00"), db_path)
        exps = get_all_experiments(db_path)
        # Newest first
        assert exps[0]["created_at"] >= exps[1]["created_at"]


# ---------------------------------------------------------------------------
# Get by IDs
# ---------------------------------------------------------------------------

class TestGetByIds:
    def test_get_multiple_by_ids(self, db_path):
        id1 = save_experiment(_sample_experiment_data(impactor_diameter=5), db_path)
        id2 = save_experiment(_sample_experiment_data(impactor_diameter=10), db_path)
        save_experiment(_sample_experiment_data(impactor_diameter=15), db_path)

        exps = get_experiments_by_ids([id1, id2], db_path)
        assert len(exps) == 2

    def test_get_by_ids_empty_list(self, db_path):
        exps = get_experiments_by_ids([], db_path)
        assert exps == []


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

class TestDeleteExperiment:
    def test_delete_existing(self, db_path):
        eid = save_experiment(_sample_experiment_data(), db_path)
        assert delete_experiment(eid, db_path) is True
        assert get_experiment(eid, db_path) is None

    def test_delete_nonexistent(self, db_path):
        assert delete_experiment(9999, db_path) is False
