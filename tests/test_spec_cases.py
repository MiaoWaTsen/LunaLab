"""
Test cases from the LunaLab test plan (section 4), one test per case id.

EP = equivalence partitioning, BVA = boundary value analysis.  Cases that
need a real browser (B-09 front end, B-12, B-13) live in tests/e2e.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from simulation.crater_model import classify_crater, ImpactParameters, run_simulation

BASE = {"impactor_diameter": 10, "impact_velocity": 18000, "impact_angle": 45}


@pytest.fixture(scope="module")
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def client(db_path):
    import database.models as models
    original = models.DB_PATH
    models.DB_PATH = db_path
    with TestClient(app) as c:
        token = c.post("/api/auth/register", json={"username": "spec-runner", "password": "pass1234"}).json()["token"]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c
    models.DB_PATH = original


def simulate(client, **overrides):
    return client.post("/api/simulations", json={**BASE, **overrides})


def warning_fields(res):
    return {w["field"] for w in res.json().get("warnings", [])}


# ---------------------------------------------------------------------------
# Flow A — impact simulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,angle,status,warns", [
    ("A-01", 0, 422, None),
    ("A-02", 0.001, 200, True),
    ("A-03a", 9.99, 200, True),
    ("A-03b", 10, 200, False),
    ("A-05a", 90, 200, False),
    ("A-05b", 90.01, 422, None),
    ("A-06", -30, 422, None),
])
def test_impact_angle_partitions(client, case, angle, status, warns):
    res = simulate(client, impact_angle=angle)
    assert res.status_code == status, case
    if warns is not None:
        assert ("impact_angle" in warning_fields(res)) is warns, case


@pytest.mark.parametrize("case,angle,asymmetric", [("A-04a", 44.99, True), ("A-04b", 45, False)])
def test_oblique_terrain_threshold(client, case, angle, asymmetric):
    eid = simulate(client, impact_angle=angle).json()["experiment_id"]
    field = client.get(f"/api/experiments/{eid}/elevation").json()
    assert field["is_asymmetric"] is asymmetric, case


@pytest.mark.parametrize("case,velocity,status,warns", [
    ("A-07", 0, 422, None),
    ("A-08a", 2379.99, 200, True),
    ("A-08b", 2380, 200, False),
    ("A-09a", 72000, 200, False),
    ("A-09b", 72000.01, 200, True),
])
def test_impact_velocity_partitions(client, case, velocity, status, warns):
    res = simulate(client, impact_velocity=velocity)
    assert res.status_code == status, case
    if warns is not None:
        assert ("impact_velocity" in warning_fields(res)) is warns, case


@pytest.mark.parametrize("case,diameter,status,warns", [
    ("A-10", 0, 422, None),
    ("A-11a", 0.00999, 200, True),
    ("A-11b", 0.01, 200, False),
    ("A-12a", 100000, 200, False),
    ("A-12b", 100000.01, 200, True),
    ("A-13", 1e100, 422, None),
])
def test_impactor_diameter_partitions(client, case, diameter, status, warns):
    res = simulate(client, impactor_diameter=diameter)
    assert res.status_code == status, case
    if warns is not None:
        assert ("impactor_diameter" in warning_fields(res)) is warns, case


def test_A14_non_numeric_diameter_rejected(client):
    assert simulate(client, impactor_diameter="abc").status_code == 422


@pytest.mark.parametrize("case,diameter,expected", [
    ("A-15a", 15_000, "simple"),
    ("A-15b", 15_000.01, "complex"),
    ("A-16a", 299_999, "complex"),
    ("A-16b", 300_000, "basin"),
])
def test_crater_type_boundaries(case, diameter, expected):
    assert classify_crater(diameter) == expected, case


def test_A16_basin_warning_from_simulation():
    r = run_simulation(ImpactParameters(40_000, 20_000, 45))
    assert r.crater_type == "basin"
    assert any(w["field"] == "crater_diameter" for w in r.warnings)


def test_A17_latitude_out_of_range(client):
    assert simulate(client, latitude=91, longitude=0).status_code == 422


def test_A18_tycho_reference(client):
    body = simulate(client, impactor_diameter=6000, impact_velocity=16000).json()
    assert body["crater_type"] == "complex"
    assert 70_000 < body["crater_diameter"] < 100_000
    assert 4_000 < body["crater_depth"] < 6_000


# ---------------------------------------------------------------------------
# Flow B — login and sessions
# ---------------------------------------------------------------------------

def register(client, username, password="pass1234"):
    return client.post("/api/auth/register", json={"username": username, "password": password},
                       headers={"Authorization": ""})


@pytest.mark.parametrize("case,username,status", [
    ("B-01a", "a", 422),
    ("B-01b", "ab", 200),
    ("B-02a", "x" * 30, 200),
    ("B-02b", "y" * 31, 422),
    ("B-03", "  ", 400),
])
def test_username_length(client, case, username, status):
    res = register(client, username)
    assert res.status_code == status, case
    if status == 200:
        assert res.json()["token"]


@pytest.mark.parametrize("case,length,status", [
    ("B-04a", 3, 422), ("B-04b", 4, 200), ("B-05a", 100, 200), ("B-05b", 101, 422),
])
def test_password_length(client, case, length, status):
    assert register(client, f"pw-{case}", "p" * length).status_code == status, case


def test_B06_duplicate_username(client):
    register(client, "dupe-user")
    res = register(client, "dupe-user")
    assert res.status_code == 400
    assert "already taken" in res.json()["detail"]


def test_B07_wrong_password_and_unknown_user_look_the_same(client):
    register(client, "real-user")
    wrong = client.post("/api/auth/login", json={"username": "real-user", "password": "nope1234"})
    unknown = client.post("/api/auth/login", json={"username": "ghost-user", "password": "nope1234"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def _set_expiry(db_path, token, expires_at):
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE user_sessions SET expires_at = ? WHERE token = ?", (expires_at.isoformat(), token))
    conn.commit()
    conn.close()


def _get_expiry(db_path, token):
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT expires_at FROM user_sessions WHERE token = ?", (token,)).fetchone()
    conn.close()
    return None if row is None else datetime.fromisoformat(row[0])


def test_B08_activity_one_second_before_expiry_extends_session(client, db_path):
    token = register(client, "almost-idle").json()["token"]
    _set_expiry(db_path, token, datetime.now(timezone.utc) + timedelta(seconds=1))
    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    remaining = _get_expiry(db_path, token) - datetime.now(timezone.utc)
    assert timedelta(minutes=29) < remaining <= timedelta(minutes=30)


def test_B09_expired_session_is_rejected_and_deleted(client, db_path):
    token = register(client, "fully-idle").json()["token"]
    _set_expiry(db_path, token, datetime.now(timezone.utc))
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert _get_expiry(db_path, token) is None


def test_B10_history_without_token(client):
    assert client.get("/api/challenges/history", headers={"Authorization": ""}).status_code == 401


def test_B11_logged_out_token_rejected(client):
    token = register(client, "leaver").json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    client.post("/api/auth/logout", headers=h)
    assert client.get("/api/auth/me", headers=h).status_code == 401
