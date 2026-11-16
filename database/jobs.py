"""
Background jobs — a durable queue in the database.

A job is a row.  Its status moves

    queued ──claim──▶ running ──▶ succeeded | failed | cancelled
       ▲                 │
       └──── requeue ────┘   (the worker stopped: shutdown, or no heartbeat)

and every transition is a single UPDATE guarded by the status it expects
(and, once running, by the worker that owns the job), so two workers can
never both claim a job and a worker that lost a job cannot overwrite it.

The queue lives in the database rather than in memory because the web
process is not durable: the host restarts and suspends it at will.  A
running job proves it is alive by updating heartbeat_at; one whose
heartbeat is older than STALE_AFTER is assumed to have died with its
worker and goes back to the queue, at most MAX_ATTEMPTS times in all.

Timestamps are UTC ISO-8601 strings, as elsewhere in the schema; they
compare correctly as text.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from database.models import get_connection

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "succeeded", "failed", "cancelled"
ACTIVE = (QUEUED, RUNNING)
FINISHED = (SUCCEEDED, FAILED, CANCELLED)

# A running job silent for this long is taken to have lost its worker.
# Handlers must report progress more often than this.
STALE_AFTER = timedelta(minutes=2)
# How many times a job may be started before a dying worker is blamed on the job
MAX_ATTEMPTS = 3

_SUMMARY_COLUMNS = (
    "id, user_id, type, params_json, status, progress, error, attempts, "
    "cancel_requested, worker_id, created_at, started_at, heartbeat_at, finished_at"
)


def _now(now: Optional[datetime]) -> datetime:
    return now or datetime.now(timezone.utc)


def _to_job(row) -> dict:
    job = dict(zip(row.keys(), row))
    job["params"] = json.loads(job.pop("params_json"))
    if "result_json" in job:
        raw = job.pop("result_json")
        job["result"] = None if raw is None else json.loads(raw)
    job["cancel_requested"] = bool(job["cancel_requested"])
    return job


def _fetch(conn, job_id: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return None if row is None else _to_job(row)


# ---------------------------------------------------------------------------
# Submitting and reading
# ---------------------------------------------------------------------------

def create_job(user_id: int, job_type: str, params: dict,
               db_path: Optional[str] = None, now: Optional[datetime] = None) -> dict:
    """Queue a job and return it."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO jobs (user_id, type, params_json, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, job_type, json.dumps(params), QUEUED, _now(now).isoformat()),
        )
        conn.commit()
        return _fetch(conn, cursor.lastrowid)
    finally:
        conn.close()


def get_job(job_id: int, db_path: Optional[str] = None) -> Optional[dict]:
    """A job with its result, or None."""
    conn = get_connection(db_path)
    try:
        return _fetch(conn, job_id)
    finally:
        conn.close()


def list_user_jobs(user_id: int, limit: int = 50, db_path: Optional[str] = None) -> list[dict]:
    """A user's jobs, newest first, without their (possibly large) results."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM jobs WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [_to_job(row) for row in rows]
    finally:
        conn.close()


def count_active_jobs(user_id: int, db_path: Optional[str] = None) -> int:
    """How many of a user's jobs are queued or running."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE user_id = ? AND status IN (?, ?)",
            (user_id, *ACTIVE),
        ).fetchone()
        return row["n"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The worker's side
# ---------------------------------------------------------------------------

def recover_stale_jobs(db_path: Optional[str] = None, now: Optional[datetime] = None) -> None:
    """Settle running jobs whose worker has gone silent."""
    now = _now(now)
    stale = (now - STALE_AFTER).isoformat()
    silent = "status = ? AND heartbeat_at < ?"
    conn = get_connection(db_path)
    try:
        # The user had already asked for it to stop
        conn.execute(
            f"UPDATE jobs SET status = ?, finished_at = ?, worker_id = NULL "
            f"WHERE {silent} AND cancel_requested = 1",
            (CANCELLED, now.isoformat(), RUNNING, stale),
        )
        # Started MAX_ATTEMPTS times and never finished: stop retrying
        conn.execute(
            f"UPDATE jobs SET status = ?, error = ?, finished_at = ?, worker_id = NULL "
            f"WHERE {silent} AND attempts >= ?",
            (FAILED, f"The worker stopped {MAX_ATTEMPTS} times while running this job.",
             now.isoformat(), RUNNING, stale, MAX_ATTEMPTS),
        )
        conn.execute(
            f"UPDATE jobs SET status = ?, progress = 0, worker_id = NULL WHERE {silent}",
            (QUEUED, RUNNING, stale),
        )
        conn.commit()
    finally:
        conn.close()


def claim_next_job(worker_id: str, db_path: Optional[str] = None,
                   now: Optional[datetime] = None) -> Optional[dict]:
    """Take the oldest queued job for this worker, or None if the queue is empty."""
    now = _now(now)
    recover_stale_jobs(db_path, now)
    conn = get_connection(db_path)
    try:
        # Another worker may take the job between the SELECT and the UPDATE;
        # then the UPDATE matches nothing and we look again.
        for _ in range(10):
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY id LIMIT 1", (QUEUED,)
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, worker_id = ?, attempts = attempts + 1, progress = 0, "
                "started_at = ?, heartbeat_at = ? WHERE id = ? AND status = ?",
                (RUNNING, worker_id, now.isoformat(), now.isoformat(), row["id"], QUEUED),
            )
            conn.commit()
            if cursor.rowcount == 1:
                return _fetch(conn, row["id"])
        return None
    finally:
        conn.close()


def heartbeat(job_id: int, worker_id: str, progress: float,
              db_path: Optional[str] = None, now: Optional[datetime] = None) -> str:
    """
    Record that the job is alive and how far it is.

    Returns what the worker should do next: "continue", "cancel" (the user
    asked), or "lost" (the job is no longer this worker's: stop, and write
    nothing more).
    """
    progress = min(1.0, max(0.0, float(progress)))
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "UPDATE jobs SET heartbeat_at = ?, progress = ? WHERE id = ? AND worker_id = ? AND status = ?",
            (_now(now).isoformat(), progress, job_id, worker_id, RUNNING),
        )
        conn.commit()
        if cursor.rowcount != 1:
            return "lost"
        row = conn.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return "cancel" if row["cancel_requested"] else "continue"
    finally:
        conn.close()


def finish_job(job_id: int, worker_id: str, status: str, result: Optional[dict] = None,
               error: Optional[str] = None, db_path: Optional[str] = None,
               now: Optional[datetime] = None) -> bool:
    """End a running job.  False if it was no longer this worker's to end."""
    if status not in FINISHED:
        raise ValueError(f"status must be one of {FINISHED}.")
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "UPDATE jobs SET status = ?, result_json = ?, error = ?, finished_at = ?, "
            "progress = CASE WHEN ? = 1 THEN 1 ELSE progress END "
            "WHERE id = ? AND worker_id = ? AND status = ?",
            (status, None if result is None else json.dumps(result, allow_nan=False), error,
             _now(now).isoformat(), int(status == SUCCEEDED), job_id, worker_id, RUNNING),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def release_job(job_id: int, worker_id: str, db_path: Optional[str] = None) -> bool:
    """Put a running job back in the queue (the worker is shutting down); not an attempt."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "UPDATE jobs SET status = ?, worker_id = NULL, progress = 0, attempts = attempts - 1 "
            "WHERE id = ? AND worker_id = ? AND status = ?",
            (QUEUED, job_id, worker_id, RUNNING),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Cancelling and cleaning up
# ---------------------------------------------------------------------------

def request_cancel(job_id: int, db_path: Optional[str] = None,
                   now: Optional[datetime] = None) -> Optional[dict]:
    """
    Cancel a job: at once if it is still queued; if it is running, by asking
    its worker, which stops at its next progress report.  A finished job is
    left as it is.  Returns the job, or None if there is no such job.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "UPDATE jobs SET status = ?, finished_at = ? WHERE id = ? AND status = ?",
            (CANCELLED, _now(now).isoformat(), job_id, QUEUED),
        )
        if cursor.rowcount != 1:
            # Not queued (any more): if a worker has it, leave it a note
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND status = ?", (job_id, RUNNING)
            )
        conn.commit()
        return _fetch(conn, job_id)
    finally:
        conn.close()


def delete_finished_jobs(finished_before: datetime, db_path: Optional[str] = None) -> int:
    """Delete finished jobs older than a cutoff; returns how many."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "DELETE FROM jobs WHERE status IN (?, ?, ?) AND finished_at < ?",
            (*FINISHED, finished_before.isoformat()),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
