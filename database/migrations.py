"""
Schema migrations — numbered, recorded, applied once.

The schema_migrations table lists the versions a database has had applied.
migrate() applies the missing ones in order, inside one transaction: if any
step fails, the database is left exactly as it was.

To change the schema, append a Migration to MIGRATIONS with the next
version number.  Never edit or reorder one that has been released: databases
that already recorded it will not run it again.

Migrations are written in SQLite's dialect, with ? placeholders;
run_script() translates the column types for PostgreSQL.

Version 1 is the schema as it stood when this table was introduced.  It is
written to be safe on a database of any earlier vintage (CREATE TABLE IF NOT
EXISTS, columns added only when missing), because those databases, including
the production one, carry no record of what they already have.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

# Arbitrary constant: serialises concurrent migrate() calls on PostgreSQL
_ADVISORY_LOCK_ID = 5_862_014

_CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    applied_at  TEXT    NOT NULL
)
"""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable  # apply(conn, postgres: bool) -> None


# ---------------------------------------------------------------------------
# Helpers for writing migrations
# ---------------------------------------------------------------------------

def postgres_dialect(sql: str) -> str:
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    # SQLite REAL is 64-bit; PostgreSQL REAL is only 32-bit.
    return re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql)


def run_script(conn, script: str, postgres: bool) -> None:
    """Run ;-separated statements one by one, so they stay in the open transaction."""
    if postgres:
        script = postgres_dialect(script)
    for statement in script.split(";"):
        if statement.strip():
            conn.execute(statement)


def table_columns(conn, table: str, postgres: bool) -> list[str]:
    if postgres:
        rows = conn.execute(
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = ?",
            (table,),
        ).fetchall()
    else:
        rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [row["name"] for row in rows]


# ---------------------------------------------------------------------------
# 1 — baseline
# ---------------------------------------------------------------------------

_BASELINE_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
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
    warnings_json       TEXT NOT NULL DEFAULT '[]',
    impact_azimuth      REAL DEFAULT 0.0,
    latitude            REAL,
    longitude           REAL,
    surface_elevation   REAL,
    user_id             INTEGER
);

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    UNIQUE NOT NULL,
    password_hash   TEXT    NOT NULL,
    salt            TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    xp              INTEGER DEFAULT 0,
    level           INTEGER DEFAULT 1,
    rank_title      TEXT    DEFAULT 'Cadet Researcher'
);

CREATE TABLE IF NOT EXISTS user_sessions (
    token           TEXT    PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    created_at      TEXT    NOT NULL,
    expires_at      TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS challenge_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER,
    challenge_id        TEXT    NOT NULL,
    score               REAL    NOT NULL,
    grade               TEXT    NOT NULL,
    diameter_error_pct  REAL    NOT NULL,
    depth_error_pct     REAL    NOT NULL,
    experiment_id       INTEGER,
    created_at          TEXT    NOT NULL
);
"""

# Columns added to `experiments` before migrations were numbered
_BASELINE_EXPERIMENT_COLUMNS = {
    "impact_azimuth": "REAL DEFAULT 0.0",
    "latitude": "REAL",
    "longitude": "REAL",
    "surface_elevation": "REAL",
    "user_id": "INTEGER",
}


def _baseline(conn, postgres: bool) -> None:
    run_script(conn, _BASELINE_SQL, postgres)

    existing = table_columns(conn, "experiments", postgres)
    for column, definition in _BASELINE_EXPERIMENT_COLUMNS.items():
        if column not in existing:
            run_script(conn, f"ALTER TABLE experiments ADD COLUMN {column} {definition}", postgres)

    if "expires_at" not in table_columns(conn, "user_sessions", postgres):
        # Pre-existing sessions have no expiry, so drop them (users log in again).
        conn.execute("ALTER TABLE user_sessions ADD COLUMN expires_at TEXT")
        conn.execute("DELETE FROM user_sessions")


# ---------------------------------------------------------------------------
# The list.  Append only.
# ---------------------------------------------------------------------------

MIGRATIONS: list[Migration] = [
    Migration(1, "baseline", _baseline),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def applied_versions(conn) -> list[int]:
    rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
    return [row["version"] for row in rows]


def migrate(conn, postgres: bool, migrations: list[Migration] | None = None) -> list[int]:
    """
    Apply the migrations this database has not had yet, in one transaction.

    Returns the versions applied by this call.  Commits on success; on any
    error rolls everything back and re-raises.
    """
    migrations = MIGRATIONS if migrations is None else migrations
    versions = [m.version for m in migrations]
    if versions != sorted(set(versions)):
        raise ValueError(f"Migration versions must be unique and ascending, got {versions}.")

    newly_applied = []
    try:
        # Take a write lock first, so two processes starting together cannot
        # both apply the same migration.  PostgreSQL has already opened a
        # transaction; SQLite needs an explicit one for DDL to be part of it.
        if postgres:
            conn.execute("SELECT pg_advisory_xact_lock(?)", (_ADVISORY_LOCK_ID,))
        else:
            conn.execute("BEGIN IMMEDIATE")

        conn.execute(_CREATE_MIGRATIONS_TABLE)
        done = set(applied_versions(conn))
        for migration in migrations:
            if migration.version in done:
                continue
            migration.apply(conn, postgres)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.name, datetime.now(timezone.utc).isoformat()),
            )
            newly_applied.append(migration.version)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return newly_applied
