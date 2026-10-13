"""
Tests for User Authentication and Gamification / Challenge Systems.
"""

import pytest
import tempfile
import os
from fastapi.testclient import TestClient

from backend.api import app
from database.models import (
    init_db,
    create_user,
    authenticate_user,
    create_session,
    get_user_by_session,
    add_user_xp,
    get_rank_for_level,
)
from simulation.challenges import (
    CHALLENGES,
    evaluate_challenge,
    get_challenge_by_id,
)

@pytest.fixture
def auth_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def client(auth_db, monkeypatch):
    monkeypatch.setattr("database.models.DB_PATH", auth_db)
    with TestClient(app) as c:
        yield c



def test_user_creation_and_auth(auth_db):
    user = create_user("CommanderNeil", "apollo11pass", db_path=auth_db)
    assert user["id"] is not None
    assert user["username"] == "CommanderNeil"
    assert user["xp"] == 0
    assert user["level"] == 1

    # Verify password authentication
    auth_user = authenticate_user("CommanderNeil", "apollo11pass", db_path=auth_db)
    assert auth_user is not None
    assert auth_user["id"] == user["id"]

    # Wrong password
    assert authenticate_user("CommanderNeil", "wrongpass", db_path=auth_db) is None
    # Nonexistent user
    assert authenticate_user("Nobody", "apollo11pass", db_path=auth_db) is None


def test_duplicate_username_fails(auth_db):
    create_user("BuzzAldrin", "secret123", db_path=auth_db)
    with pytest.raises(ValueError, match="already taken"):
        create_user("BuzzAldrin", "secret456", db_path=auth_db)


def test_session_lifecycle(auth_db):
    user = create_user("MichaelCollins", "columbia1969", db_path=auth_db)
    token = create_session(user["id"], db_path=auth_db)
    assert token is not None

    session_user = get_user_by_session(token, db_path=auth_db)
    assert session_user is not None
    assert session_user["username"] == "MichaelCollins"


def test_user_xp_level_progression(auth_db):
    user = create_user("CarlSagan", "cosmos123", db_path=auth_db)
    assert user["level"] == 1

    # Add XP
    res = add_user_xp(user["id"], 500, db_path=auth_db)
    assert res["xp"] == 500
    assert res["level"] >= 3
    assert "🚀" in res["rank_title"] or "🔬" in res["rank_title"]


# ---------------------------------------------------------------------------
# Challenge Evaluation Tests
# ---------------------------------------------------------------------------

def test_challenge_listing():
    assert len(CHALLENGES) >= 4
    tycho = get_challenge_by_id("tycho_crater")
    assert tycho is not None
    assert tycho["target_diameter"] == 85000.0


def test_challenge_evaluation_accuracy():
    tycho = get_challenge_by_id("tycho_crater")
    # Simulate high accuracy match
    result = {"crater_diameter": 84500.0, "crater_depth": 4750.0, "energy_joules": 1e18}
    params = {"impact_angle": 45.0}

    evaluation = evaluate_challenge("tycho_crater", result, params)
    assert evaluation["passed"] is True
    assert evaluation["grade"] in ["S", "A"]
    assert evaluation["score"] >= 80.0
    assert evaluation["xp_gain"] >= 100


def test_challenge_grazing_penalty():
    # Grazing skimmer expects low angle <= 20
    result = {"crater_diameter": 2000.0, "crater_depth": 350.0, "energy_joules": 1e14}
    # Steep angle -> penalized
    evaluation = evaluate_challenge("grazing_skimmer", result, {"impact_angle": 80.0})
    assert evaluation["score"] < 60.0
    assert evaluation["grade"] in ["B", "C"]


# ---------------------------------------------------------------------------
# HTTP API Auth & Challenge Endpoints Tests
# ---------------------------------------------------------------------------

def test_api_auth_register_and_login(client):
    username = f"user_{os.urandom(4).hex()}"
    # Register
    reg_res = client.post("/api/auth/register", json={"username": username, "password": "mypassword123"})
    assert reg_res.status_code == 200
    reg_data = reg_res.json()
    token = reg_data["token"]
    assert token is not None
    assert reg_data["user"]["username"] == username

    # Get /api/auth/me
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["username"] == username

    # Login
    login_res = client.post("/api/auth/login", json={"username": username, "password": "mypassword123"})
    assert login_res.status_code == 200
    assert login_res.json()["token"] is not None

    # Logout
    logout_res = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_res.status_code == 200


def test_api_challenge_run_endpoint(client):
    res = client.get("/api/challenges")
    assert res.status_code == 200
    challenges = res.json()["challenges"]
    assert len(challenges) > 0

    # Run challenge simulation
    payload = {
        "simulation": {
            "impactor_diameter": 25.0,
            "impact_velocity": 18000.0,
            "impact_angle": 45.0,
            "impactor_density": 3000.0,
            "target_density": 2500.0,
            "surface_gravity": 1.62,
        }
    }
    run_res = client.post("/api/challenges/apollo_tranquillity/run", json=payload)
    assert run_res.status_code == 200
    data = run_res.json()
    assert "evaluation" in data
    assert "simulation" in data
    assert data["evaluation"]["grade"] in ["S", "A", "B", "C"]
