"""
Background jobs — running what database/jobs.py queues.

    JobType      what a kind of job accepts (a pydantic model) and how to run it
    JobContext   handed to the running job: report progress, learn to stop
    run_next_job claim one job, run it to an end state; everything a test needs
    Worker       a thread that calls run_next_job in a loop

The worker runs inside the web process: the free hosting tier has no
separate worker service.  That is why the queue is in the database and why
a job can be interrupted at any moment: see database/jobs.py for how a job
survives its worker.

A handler is `run(params, ctx) -> dict`.  It must call ctx.progress()
regularly (well inside database.jobs.STALE_AFTER): that is its heartbeat,
and the only point at which it can be cancelled or interrupted.
"""

import logging
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from pydantic import BaseModel

from database import jobs as queue

log = logging.getLogger("lunalab.jobs")

# Finished jobs are kept this long, then deleted by the worker
RETENTION = timedelta(days=7)
_CLEANUP_EVERY_SECONDS = 3600.0


@dataclass(frozen=True)
class JobType:
    params_model: type[BaseModel]
    run: Callable  # run(params: params_model, ctx: JobContext) -> dict (JSON-serialisable)
    description: str = ""


class JobCancelled(Exception):
    """The user cancelled the job."""


class JobInterrupted(Exception):
    """The worker is shutting down; the job goes back to the queue."""


class JobLost(Exception):
    """The job is no longer this worker's (it was presumed dead and requeued)."""


class JobContext:
    def __init__(self, job: dict, worker_id: str, db_path: Optional[str] = None,
                 should_stop: Callable[[], bool] = lambda: False,
                 heartbeat_interval: float = 1.0):
        self.job = job
        self._worker_id = worker_id
        self._db_path = db_path
        self._should_stop = should_stop
        self._interval = heartbeat_interval
        self._last_beat: Optional[float] = None

    def progress(self, fraction: float) -> None:
        """
        Report progress (0-1).  Raises to unwind the handler when the job
        should not continue; handlers let those exceptions propagate.
        """
        if self._should_stop():
            raise JobInterrupted()
        # The database write is rate-limited; cancellation is noticed at the same rate
        now = time.monotonic()
        if self._last_beat is not None and now - self._last_beat < self._interval:
            return
        self._last_beat = now
        outcome = queue.heartbeat(self.job["id"], self._worker_id, fraction, self._db_path)
        if outcome == "cancel":
            raise JobCancelled()
        if outcome == "lost":
            raise JobLost()


def _error_message(error: Exception) -> str:
    # ValueError (which includes pydantic's ValidationError) carries a message
    # written for the user; anything else is a bug whose text stays in the log.
    if isinstance(error, ValueError):
        return str(error)[:500]
    return f"Internal error ({type(error).__name__})."


def run_next_job(worker_id: str, job_types: dict[str, JobType], db_path: Optional[str] = None,
                 should_stop: Callable[[], bool] = lambda: False,
                 heartbeat_interval: float = 1.0) -> Optional[dict]:
    """
    Claim the oldest queued job and run it until it ends, one way or another.

    Returns the job as it stands afterwards, or None if nothing was queued.
    Never raises because of the job itself.
    """
    job = queue.claim_next_job(worker_id, db_path)
    if job is None:
        return None
    job_id = job["id"]
    ctx = JobContext(job, worker_id, db_path, should_stop, heartbeat_interval)
    try:
        job_type = job_types.get(job["type"])
        if job_type is None:
            raise ValueError(f"Unknown job type '{job['type']}'.")
        # Validated at submission, but the job may have waited across a deploy
        params = job_type.params_model.model_validate(job["params"])
        result = job_type.run(params, ctx)
        if not queue.finish_job(job_id, worker_id, queue.SUCCEEDED, result=result, db_path=db_path):
            log.warning("job %s finished but was no longer ours; result discarded", job_id)
    except JobCancelled:
        queue.finish_job(job_id, worker_id, queue.CANCELLED, db_path=db_path)
    except JobInterrupted:
        queue.release_job(job_id, worker_id, db_path)
    except JobLost:
        log.warning("job %s was taken from this worker; stopping", job_id)
    except Exception as error:
        log.exception("job %s (%s) failed", job_id, job["type"])
        queue.finish_job(job_id, worker_id, queue.FAILED, error=_error_message(error), db_path=db_path)
    return queue.get_job(job_id, db_path)


class Worker:
    """Runs queued jobs on a background thread, one at a time."""

    def __init__(self, job_types: dict[str, JobType], db_path: Optional[str] = None,
                 poll_interval: float = 30.0, heartbeat_interval: float = 1.0):
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(4)}"
        self._job_types = job_types
        self._db_path = db_path
        # notify() wakes the worker at once for jobs submitted in this process.
        # Polling only finds stale jobs and other processes' submissions, so it
        # can be slow, which spares the database a connection every few seconds.
        self._poll_interval = poll_interval
        self._heartbeat_interval = heartbeat_interval
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_cleanup: Optional[float] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="lunalab-job-worker", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        """A job was just queued."""
        self._wake.set()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop; a running job goes back to the queue at its next progress report."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = run_next_job(self.worker_id, self._job_types, self._db_path,
                                   self._stop.is_set, self._heartbeat_interval)
                if job is None:
                    self._cleanup()
            except Exception:
                # The database being unreachable must not kill the worker
                log.exception("job worker: error outside a job; retrying after a pause")
                job = None
            if job is None:
                self._wake.wait(self._poll_interval)
                self._wake.clear()

    def _cleanup(self) -> None:
        now = time.monotonic()
        if self._last_cleanup is not None and now - self._last_cleanup < _CLEANUP_EVERY_SECONDS:
            return
        self._last_cleanup = now
        deleted = queue.delete_finished_jobs(datetime.now(timezone.utc) - RETENTION, self._db_path)
        if deleted:
            log.info("deleted %d finished jobs older than %s", deleted, RETENTION)
