"""
Database module — SQLite models & CRUD operations.

This module handles all data persistence for LunaLab experiments.
It is decoupled from the simulation logic and the web API.
"""

import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Database path — configurable via environment variable
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lunalab.db",
)

DB_PATH = os.environ.get("LUNALAB_DB_PATH", _DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
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
    warnings_json       TEXT NOT NULL DEFAULT '[]'
);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Create tables if they don't exist."""
    conn = get_connection(db_path)
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def save_experiment(data: dict, db_path: Optional[str] = None) -> int:
    """
    Insert a new experiment record.

    Parameters
    ----------
    data : dict
        Must contain all the column keys except 'id' and 'created_at'.
        'created_at' is auto-generated if not provided.

    Returns
    -------
    int
        The newly created experiment ID.
    """
    conn = get_connection(db_path)
    created_at = data.get("created_at", datetime.now(timezone.utc).isoformat())
    cursor = conn.execute(
        """
        INSERT INTO experiments (
            created_at,
            impactor_diameter, impact_velocity, impact_angle,
            impactor_density, target_density, surface_gravity,
            crater_diameter, crater_depth, rim_height,
            ejecta_volume, transient_crater_diameter,
            energy_joules, model_name, warnings_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            data["impactor_diameter"],
            data["impact_velocity"],
            data["impact_angle"],
            data["impactor_density"],
            data["target_density"],
            data["surface_gravity"],
            data["crater_diameter"],
            data["crater_depth"],
            data["rim_height"],
            data["ejecta_volume"],
            data["transient_crater_diameter"],
            data["energy_joules"],
            data["model_name"],
            data.get("warnings_json", "[]"),
        ),
    )
    conn.commit()
    experiment_id = cursor.lastrowid
    conn.close()
    return experiment_id


def get_experiment(experiment_id: int, db_path: Optional[str] = None) -> Optional[dict]:
    """Retrieve a single experiment by ID."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def get_all_experiments(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all experiments ordered by creation time descending."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_experiments_by_ids(
    ids: list[int], db_path: Optional[str] = None
) -> list[dict]:
    """Retrieve multiple experiments by their IDs."""
    if not ids:
        return []
    conn = get_connection(db_path)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM experiments WHERE id IN ({placeholders})", ids
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_experiment(experiment_id: int, db_path: Optional[str] = None) -> bool:
    """Delete an experiment. Returns True if a row was deleted."""
    conn = get_connection(db_path)
    cursor = conn.execute(
        "DELETE FROM experiments WHERE id = ?", (experiment_id,)
    )
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted
