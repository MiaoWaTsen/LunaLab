"""
Tests for the schema migration runner (database/migrations.py), on SQLite
and PostgreSQL (see the db_backend fixture in conftest.py).
"""

import pytest

from database import migrations
from database.migrations import MIGRATIONS, Migration, applied_versions, migrate, table_columns
from database.models import get_connection, init_db, save_experiment, get_experiment

ALL_TABLES = "challenge_records, user_sessions, users, experiments, schema_migrations"

# The schema of the very first release: no location, owner or session expiry
LEGACY_SQL = """
CREATE TABLE experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    impactor_diameter   REAL NOT NULL,
    impact_velocity     REAL NOT NULL,
    impact_angle        REAL NOT NULL,
    impactor_density    REAL NOT NULL,
    target_density      REAL NOT NULL,
    surface_gravity     REAL NOT NULL,
    crater_diameter     REAL NOT NULL,
    crater_depth        REAL NOT NULL,
    rim_height          REAL NOT NULL,
    ejecta_volume       REAL NOT NULL,
    transient_crater_diameter REAL NOT NULL,
    energy_joules       REAL NOT NULL,
    model_name          TEXT NOT NULL,
    warnings_json       TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    UNIQUE NOT NULL,
    password_hash   TEXT    NOT NULL,
    salt            TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    xp              INTEGER DEFAULT 0,
    level           INTEGER DEFAULT 1,
    rank_title      TEXT    DEFAULT 'Cadet Researcher'
);
CREATE TABLE user_sessions (
    token           TEXT    PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    created_at      TEXT    NOT NULL
);
"""

EXPERIMENT = {
    "impactor_diameter": 10.0, "impact_velocity": 20000.0, "impact_angle": 45.0,
    "impactor_density": 3000.0, "target_density": 2500.0, "surface_gravity": 1.62,
    "crater_diameter": 120.5, "crater_depth": 24.1, "rim_height": 6.025,
    "ejecta_volume": 50000.0, "transient_crater_diameter": 100.4,
    "energy_joules": 3.14e12, "model_name": "test", "warnings_json": "[]",
}


class Db:
    """The database under test, on whichever engine db_backend selected."""

    def __init__(self, db_path):
        self.path = db_path
        self.postgres = db_path is None

    def connect(self):
        return get_connection(self.path)

    def run(self, fn):
        conn = self.connect()
        try:
            result = fn(conn)
            conn.commit()
            return result
        finally:
            conn.close()

    def versions(self):
        return self.run(applied_versions)

    def columns(self, table):
        return self.run(lambda conn: table_columns(conn, table, self.postgres))

    def count(self, table):
        return self.run(lambda conn: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])

    def migrate(self, migration_list=None):
        conn = self.connect()
        try:
            return migrate(conn, self.postgres, migration_list)
        finally:
            conn.close()

    def reset_to_legacy(self):
        def build(conn):
            for table in ALL_TABLES.split(", "):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            migrations.run_script(conn, LEGACY_SQL, self.postgres)
        self.run(build)


@pytest.fixture
def db(db_backend):
    return Db(db_backend)


def _add_notes_table(conn, postgres):
    conn.execute("CREATE TABLE notes (body TEXT)")


def _add_notes_then_fail(conn, postgres):
    conn.execute("CREATE TABLE notes (body TEXT)")
    conn.execute("INSERT INTO notes (body) VALUES (?)", ("half-done",))
    raise RuntimeError("migration failed half-way")


@pytest.fixture
def without_notes(db):
    """`notes` is created by the test migrations; make sure it never outlives a test."""
    drop = lambda conn: conn.execute("DROP TABLE IF EXISTS notes")
    db.run(drop)
    yield
    db.run(drop)


# ---------------------------------------------------------------------------
# A new database
# ---------------------------------------------------------------------------

def test_new_database_records_every_migration(db):
    assert db.versions() == [m.version for m in MIGRATIONS]


def test_new_database_has_the_full_schema(db):
    assert {"impact_azimuth", "latitude", "longitude", "surface_elevation", "user_id"} <= set(db.columns("experiments"))
    assert "expires_at" in db.columns("user_sessions")
    assert db.columns("users") and db.columns("challenge_records")


def test_running_again_applies_nothing(db):
    eid = save_experiment(dict(EXPERIMENT), db.path)
    assert db.migrate() == []
    init_db(db.path)
    assert db.versions() == [m.version for m in MIGRATIONS]
    assert get_experiment(eid, db.path)["crater_diameter"] == 120.5


def test_migration_versions_are_ascending_from_one():
    assert [m.version for m in MIGRATIONS] == list(range(1, len(MIGRATIONS) + 1))


# ---------------------------------------------------------------------------
# Databases that predate the schema_migrations table
# ---------------------------------------------------------------------------

def test_first_release_database_is_upgraded_and_keeps_its_data(db):
    db.reset_to_legacy()

    def seed(conn):
        columns = ", ".join(EXPERIMENT)
        marks = ", ".join("?" for _ in EXPERIMENT)
        conn.execute(
            f"INSERT INTO experiments (created_at, {columns}) VALUES (?, {marks})",
            ("2025-01-01T00:00:00+00:00", *EXPERIMENT.values()),
        )
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            ("ada", "hash", "salt", "2025-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO user_sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            ("old-token", 1, "2025-01-01T00:00:00+00:00"),
        )
    db.run(seed)

    init_db(db.path)

    assert db.versions() == [m.version for m in MIGRATIONS]
    assert {"impact_azimuth", "latitude", "longitude", "surface_elevation", "user_id"} <= set(db.columns("experiments"))
    assert "expires_at" in db.columns("user_sessions")
    assert db.columns("challenge_records")
    assert db.count("experiments") == 1 and db.count("users") == 1
    # Sessions from before expiry existed are dropped: users log in again
    assert db.count("user_sessions") == 0


def test_current_database_without_a_record_keeps_everything(db):
    """Production as it is today: the full schema, but no schema_migrations table."""
    eid = save_experiment(dict(EXPERIMENT), db.path)

    def seed_and_forget(conn):
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            ("ada", "hash", "salt", "2025-01-01T00:00:00+00:00"),
        )
        user_id = conn.execute("SELECT id FROM users WHERE username = ?", ("ada",)).fetchone()["id"]
        conn.execute(
            "INSERT INTO user_sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            ("live-token", user_id, "2025-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00"),
        )
        conn.execute("DROP TABLE schema_migrations")
    db.run(seed_and_forget)

    init_db(db.path)

    assert db.versions() == [m.version for m in MIGRATIONS]
    assert get_experiment(eid, db.path)["crater_diameter"] == 120.5
    assert db.count("users") == 1
    assert db.count("user_sessions") == 1  # live sessions survive


# ---------------------------------------------------------------------------
# Adding migrations later
# ---------------------------------------------------------------------------

def test_a_new_migration_is_applied_once(db, without_notes):
    extended = MIGRATIONS + [Migration(len(MIGRATIONS) + 1, "add notes", _add_notes_table)]

    assert db.migrate(extended) == [len(MIGRATIONS) + 1]
    assert db.columns("notes") == ["body"]
    assert db.migrate(extended) == []  # CREATE TABLE notes would fail if it ran twice
    assert db.versions() == [m.version for m in extended]


def test_a_failed_migration_leaves_the_database_untouched(db, without_notes):
    eid = save_experiment(dict(EXPERIMENT), db.path)
    n = len(MIGRATIONS)
    broken = MIGRATIONS + [
        Migration(n + 1, "add notes", _add_notes_table),
        Migration(n + 2, "fails", lambda conn, postgres: conn.execute("SELECT * FROM no_such_table")),
    ]

    with pytest.raises(Exception):
        db.migrate(broken)

    # The step before the failing one is rolled back with it
    assert db.versions() == [m.version for m in MIGRATIONS]
    assert db.columns("notes") == []
    assert get_experiment(eid, db.path)["crater_diameter"] == 120.5


def test_a_migration_that_raises_half_way_is_rolled_back(db, without_notes):
    broken = MIGRATIONS + [Migration(len(MIGRATIONS) + 1, "half", _add_notes_then_fail)]

    with pytest.raises(RuntimeError, match="half-way"):
        db.migrate(broken)

    assert db.columns("notes") == []
    assert db.versions() == [m.version for m in MIGRATIONS]


@pytest.mark.parametrize("versions", [[1, 1], [2, 1], [1, 3, 2]])
def test_duplicate_or_unordered_versions_are_rejected(db, versions):
    bad = [Migration(v, f"m{v}", _add_notes_table) for v in versions]
    with pytest.raises(ValueError, match="unique and ascending"):
        db.migrate(bad)
    assert db.versions() == [m.version for m in MIGRATIONS]
