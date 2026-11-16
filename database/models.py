"""
Database module — SQLite / PostgreSQL models & CRUD operations.

This module handles all data persistence for LunaLab experiments.
It is decoupled from the simulation logic and the web API.

Backend selection:
  * DATABASE_URL set (e.g. a Supabase connection string) -> PostgreSQL.
    Use this in production: Render's free tier wipes the local disk on every
    restart, redeploy and spin-down, so a SQLite file there loses all data.
  * otherwise -> a local SQLite file (LUNALAB_DB_PATH or ./lunalab.db).
    Tests and local development use this.
"""

import sqlite3
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from database.migrations import migrate


# ---------------------------------------------------------------------------
# Database path — configurable via environment variable
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lunalab.db",
)

DB_PATH = os.environ.get("LUNALAB_DB_PATH", _DEFAULT_DB_PATH)

# PostgreSQL connection string (Supabase: use the Session pooler URI, since
# Render has no IPv6 and Supabase's direct connection is IPv6-only).
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip() or None

# Session idle timeout: a token expires after this many minutes without use.
SESSION_TTL_MINUTES = int(os.environ.get("LUNALAB_SESSION_TTL_MINUTES", "30"))


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

def _use_postgres(db_path: Optional[str]) -> bool:
    # An explicit db_path always means a SQLite file (used by tests).
    return DATABASE_URL is not None and db_path is None


class _PgRow:
    """Row that behaves like sqlite3.Row: row["col"], row[0], dict(row), unpacking."""

    __slots__ = ("_cols", "_vals")

    def __init__(self, cols, vals):
        self._cols = cols
        self._vals = tuple(vals)

    def keys(self):
        return list(self._cols)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._vals[key]
        return self._vals[self._cols.index(key)]

    def __iter__(self):
        return iter(self._vals)

    def __len__(self):
        return len(self._vals)


class _PgCursor:
    def __init__(self, cur, lastrowid=None):
        self._cur = cur
        self.lastrowid = lastrowid
        self.rowcount = cur.rowcount
        self._cols = [d.name for d in cur.description] if cur.description else []

    def fetchone(self):
        row = self._cur.fetchone()
        return None if row is None else _PgRow(self._cols, row)

    def fetchall(self):
        return [_PgRow(self._cols, r) for r in self._cur.fetchall()]


class _PgConnection:
    """Minimal sqlite3-style wrapper over psycopg, so the CRUD code stays the same."""

    def __init__(self, url: str):
        import psycopg  # imported lazily: only needed when DATABASE_URL is set

        # prepare_threshold=None: Supabase's pooler can't keep prepared statements.
        self._conn = psycopg.connect(url, prepare_threshold=None)

    def execute(self, sql: str, params=()):
        sql = sql.replace("?", "%s")
        is_insert = sql.lstrip().upper().startswith("INSERT")
        if is_insert and "RETURNING" not in sql.upper():
            sql = sql.rstrip().rstrip(";") + " RETURNING *"
        cur = self._conn.cursor()
        cur.execute(sql, tuple(params))
        lastrowid = None
        if is_insert and cur.description:
            cols = [d.name for d in cur.description]
            row = cur.fetchone()
            if row is not None and "id" in cols:
                lastrowid = row[cols.index("id")]
        return _PgCursor(cur, lastrowid)

    def executescript(self, script: str) -> None:
        for stmt in script.split(";"):
            if stmt.strip():
                self._conn.execute(stmt)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


def _integrity_errors() -> tuple:
    errors = [sqlite3.IntegrityError]
    if DATABASE_URL is not None:
        import psycopg

        errors.append(psycopg.IntegrityError)
    return tuple(errors)


def get_connection(db_path: Optional[str] = None):
    """Return a new connection: PostgreSQL if DATABASE_URL is set, else SQLite."""
    if _use_postgres(db_path):
        return _PgConnection(DATABASE_URL)
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Bring the schema up to date (see database/migrations.py)."""
    conn = get_connection(db_path)
    try:
        migrate(conn, _use_postgres(db_path))
    finally:
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
    except _integrity_errors():
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


def _session_expiry(now: datetime) -> str:
    return (now + timedelta(minutes=SESSION_TTL_MINUTES)).isoformat()


def create_session(user_id: int, db_path: Optional[str] = None) -> str:
    """Generate session token and store in user_sessions."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO user_sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now.isoformat(), _session_expiry(now)),
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_session(token: str, db_path: Optional[str] = None) -> Optional[dict]:
    """
    Lookup user by session token.

    Expired sessions are deleted and return None.  A valid session has its
    expiry pushed forward, so it only times out after SESSION_TTL_MINUTES
    of inactivity.
    """
    if not token:
        return None
    now = datetime.now(timezone.utc)
    conn = get_connection(db_path)
    row = conn.execute(
        """
        SELECT u.id, u.username, u.created_at, u.xp, u.level, u.rank_title,
               s.expires_at AS session_expires_at
        FROM user_sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.token = ?
        """,
        (token,),
    ).fetchone()
    if not row:
        conn.close()
        return None

    expires_at = row["session_expires_at"]
    if not expires_at or datetime.fromisoformat(expires_at) <= now:
        conn.execute("DELETE FROM user_sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        return None

    conn.execute(
        "UPDATE user_sessions SET expires_at = ? WHERE token = ?",
        (_session_expiry(now), token),
    )
    conn.commit()
    conn.close()
    user = dict(row)
    user.pop("session_expires_at")
    return user


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
    # Level = 1 + floor(sqrt(xp / 100)): Lv1 < 100 XP, Lv2 < 400, Lv3 < 900, ...
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


def count_user_experiments_since(user_id: int, since_iso: str, db_path: Optional[str] = None) -> int:
    """Number of experiments a user created at or after an ISO-8601 UTC timestamp."""
    conn = get_connection(db_path)
    (count,) = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE user_id = ? AND created_at >= ?",
        (user_id, since_iso),
    ).fetchone()
    conn.close()
    return count


def get_best_challenge_grade(user_id: int, challenge_id: str, db_path: Optional[str] = None) -> Optional[str]:
    """Best grade (S > A > B > C) a user has reached on a challenge, or None."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT DISTINCT grade FROM challenge_records WHERE user_id = ? AND challenge_id = ?",
        (user_id, challenge_id),
    ).fetchall()
    conn.close()
    grades = {r["grade"] for r in rows}
    for grade in ("S", "A", "B", "C"):
        if grade in grades:
            return grade
    return None


def get_user_challenge_records(user_id: Optional[int] = None, limit: int = 20, db_path: Optional[str] = None) -> list[dict]:
    """Retrieve challenge history records (all users when user_id is None)."""
    conn = get_connection(db_path)
    base_sql = """
        SELECT r.*, u.username
        FROM challenge_records r
        LEFT JOIN users u ON r.user_id = u.id
    """
    if user_id is not None:
        rows = conn.execute(
            base_sql + " WHERE r.user_id = ? ORDER BY r.created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            base_sql + " ORDER BY r.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

