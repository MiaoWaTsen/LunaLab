"""
Shared fixtures.

`db_backend` runs a test once per database engine: SQLite (local development
and tests) and PostgreSQL (production).  The production data loss came from
testing on a different engine than the one deployed, so the database tests
run on both.

The PostgreSQL run needs TEST_DATABASE_URL.  Locally it is skipped without
one; in CI (CI=true) a missing URL fails instead, so the PostgreSQL run can
never be skipped silently.
"""

import os

import pytest

# Tests drive jobs themselves (backend.jobs.run_next_job), so that nothing
# depends on a background thread's timing.  Set before the app is imported;
# the E2E server subprocess inherits it.
os.environ.setdefault("LUNALAB_JOB_WORKER", "0")

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "").strip() or None

# schema_migrations too: with it left behind, init_db would see every
# migration as applied and recreate nothing
_TABLES = "jobs, challenge_records, user_sessions, users, experiments, schema_migrations"


@pytest.fixture(params=["sqlite", "postgres"])
def db_backend(request, monkeypatch, tmp_path):
    """Yield the db_path to pass to database functions: a SQLite file, or None for PostgreSQL."""
    # Imported here, not at the top: importing database.models reads
    # LUNALAB_DB_PATH, which test modules (test_api.py) set at collection.
    from database.models import get_connection, init_db

    if request.param == "sqlite":
        path = str(tmp_path / "test.db")
        init_db(path)
        yield path
        return

    if TEST_DATABASE_URL is None:
        if os.environ.get("CI"):
            pytest.fail("TEST_DATABASE_URL is not set: the PostgreSQL tests must run in CI")
        pytest.skip("TEST_DATABASE_URL is not set")

    # db_path=None plus DATABASE_URL is exactly how production selects PostgreSQL
    monkeypatch.setattr("database.models.DATABASE_URL", TEST_DATABASE_URL)
    conn = get_connection()
    conn.executescript(f"DROP TABLE IF EXISTS {_TABLES} CASCADE")
    conn.commit()
    conn.close()
    init_db()
    yield None
