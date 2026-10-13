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

import hashlib
import secrets


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Generate salt and PBKDF2-HMAC-SHA256 password hash."""
    if not salt:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verify password against salt and expected hash."""
    test_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(test_hash, expected_hash)


def get_rank_for_level(level: int) -> str:
    """Calculate military/scientific commander rank title based on level."""
    if level >= 10:
        return "Chief Lunar Director 🌟"
    elif level >= 7:
        return "Planetary Defense Commander 🛡️"
    elif level >= 5:
        return "Senior Impact Physicist 🔬"
    elif level >= 3:
        return "Lunar Exploration Officer 🚀"
    elif level >= 2:
        return "Surface Field Specialist 🌕"
    return "Cadet Researcher 👨‍🚀"


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
    """Create tables if they don't exist, and apply migrations."""
    conn = get_connection(db_path)
    conn.executescript(_CREATE_TABLE_SQL)
    
    # Migrations: Add new columns if missing
    cursor = conn.execute("PRAGMA table_info(experiments);")
    columns = [row["name"] for row in cursor.fetchall()]
    
    if "impact_azimuth" not in columns:
        conn.execute("ALTER TABLE experiments ADD COLUMN impact_azimuth REAL DEFAULT 0.0;")
    if "latitude" not in columns:
        conn.execute("ALTER TABLE experiments ADD COLUMN latitude REAL;")
    if "longitude" not in columns:
        conn.execute("ALTER TABLE experiments ADD COLUMN longitude REAL;")
    if "surface_elevation" not in columns:
        conn.execute("ALTER TABLE experiments ADD COLUMN surface_elevation REAL;")
    if "user_id" not in columns:
        conn.execute("ALTER TABLE experiments ADD COLUMN user_id INTEGER;")
        
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
            energy_joules, model_name, warnings_json,
            impact_azimuth, latitude, longitude, surface_elevation,
            user_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            data.get("impact_azimuth", 0.0),
            data.get("latitude"),
            data.get("longitude"),
            data.get("surface_elevation"),
            data.get("user_id"),
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


# ---------------------------------------------------------------------------
# User Authentication & Profiles
# ---------------------------------------------------------------------------

def create_user(username: str, password: str, db_path: Optional[str] = None) -> dict:
    """Register a new user with PBKDF2 hashed password."""
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters.")

    pwd_hash, salt = hash_password(password)
    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO users (username, password_hash, salt, created_at, xp, level, rank_title)
            VALUES (?, ?, ?, ?, 0, 1, 'Cadet Researcher 👨‍🚀')
            """,
            (username, pwd_hash, salt, created_at),
        )
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"Username '{username}' is already taken.")

    user = get_user_by_id(user_id, db_path)
    conn.close()
    return user


def authenticate_user(username: str, password: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Verify username and password, returns user dict if valid."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    conn.close()
    if not row:
        return None

    if verify_password(password, row["salt"], row["password_hash"]):
        user_dict = dict(row)
        user_dict.pop("password_hash", None)
        user_dict.pop("salt", None)
        return user_dict
    return None


def get_user_by_id(user_id: int, db_path: Optional[str] = None) -> Optional[dict]:
    """Retrieve user by ID without sensitive hash fields."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT id, username, created_at, xp, level, rank_title FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def create_session(user_id: int, db_path: Optional[str] = None) -> str:
    """Generate session token and store in user_sessions."""
    token = secrets.token_urlsafe(32)
    created_at = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO user_sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, created_at),
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_session(token: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Lookup user by session token."""
    if not token:
        return None
    conn = get_connection(db_path)
    row = conn.execute(
        """
        SELECT u.id, u.username, u.created_at, u.xp, u.level, u.rank_title
        FROM user_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.token = ?
        """,
        (token,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def delete_session(token: str, db_path: Optional[str] = None) -> bool:
    """Invalidate session token (logout)."""
    conn = get_connection(db_path)
    cursor = conn.execute("DELETE FROM user_sessions WHERE token = ?", (token,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def add_user_xp(user_id: int, xp_gain: int, db_path: Optional[str] = None) -> dict:
    """Award XP, update level and military/scientific rank title."""
    conn = get_connection(db_path)
    row = conn.execute("SELECT xp, level FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("User not found.")

    new_xp = max(0, row["xp"] + xp_gain)
    # Level formula: Level 1 = 0-100 XP, Level 2 = 100-250 XP, etc. Level = 1 + floor(sqrt(xp / 100))
    import math
    new_level = max(1, 1 + int(math.isqrt(new_xp // 100)))
    new_rank = get_rank_for_level(new_level)

    conn.execute(
        "UPDATE users SET xp = ?, level = ?, rank_title = ? WHERE id = ?",
        (new_xp, new_level, new_rank, user_id),
    )
    conn.commit()
    conn.close()
    return {"user_id": user_id, "xp": new_xp, "level": new_level, "rank_title": new_rank, "xp_gain": xp_gain}


# ---------------------------------------------------------------------------
# Challenge Records
# ---------------------------------------------------------------------------

def save_challenge_record(data: dict, db_path: Optional[str] = None) -> int:
    """Store completed scientific challenge score & grade."""
    created_at = data.get("created_at", datetime.now(timezone.utc).isoformat())
    conn = get_connection(db_path)
    cursor = conn.execute(
        """
        INSERT INTO challenge_records (
            user_id, challenge_id, score, grade, diameter_error_pct, depth_error_pct, experiment_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("user_id"),
            data["challenge_id"],
            data["score"],
            data["grade"],
            data["diameter_error_pct"],
            data["depth_error_pct"],
            data.get("experiment_id"),
            created_at,
        ),
    )
    conn.commit()
    rec_id = cursor.lastrowid
    conn.close()
    return rec_id


def get_user_challenge_records(user_id: Optional[int] = None, limit: int = 20, db_path: Optional[str] = None) -> list[dict]:
    """Retrieve challenge history records."""
    conn = get_connection(db_path)
    if user_id is not None:
        rows = conn.execute(
            "SELECT * FROM challenge_records WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM challenge_records ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

