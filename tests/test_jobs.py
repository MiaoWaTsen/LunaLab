"""
Tests for background jobs: the queue (database/jobs.py), the runner and
worker thread (backend/jobs.py), the first job type and the API.

The queue and API tests run on SQLite and PostgreSQL (db_backend fixture).
Time is passed in rather than waited for, and jobs are run by calling
run_next_job directly; only the two Worker tests involve a real thread.
"""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from backend.api import app
from backend.job_types import JOB_TYPES, InferenceBatchParams
from backend.jobs import JobType, Worker, run_next_job
from database import jobs as queue

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
LATER = T0 + queue.STALE_AFTER + timedelta(seconds=1)   # the first worker now looks dead


@pytest.fixture
def db(db_backend):
    return db_backend


def submit(db, user_id=1, job_type="echo", params=None, now=T0):
    return queue.create_job(user_id, job_type, params or {"n": 1}, db, now)


# ---------------------------------------------------------------------------
# Queue: submitting and claiming
# ---------------------------------------------------------------------------

def test_new_job_is_queued_with_its_params(db):
    job = submit(db, params={"n": 3, "label": "α"})
    assert job["status"] == "queued"
    assert job["params"] == {"n": 3, "label": "α"}
    assert job["progress"] == 0 and job["attempts"] == 0
    assert job["result"] is None and job["error"] is None
    assert queue.get_job(job["id"], db) == job


def test_missing_job_is_none(db):
    assert queue.get_job(12345, db) is None


def test_claim_takes_the_oldest_and_marks_it_running(db):
    first, second = submit(db), submit(db)
    claimed = queue.claim_next_job("w1", db, T0)
    assert claimed["id"] == first["id"]
    assert claimed["status"] == "running" and claimed["worker_id"] == "w1"
    assert claimed["attempts"] == 1
    assert claimed["started_at"] == claimed["heartbeat_at"] == T0.isoformat()
    assert queue.claim_next_job("w2", db, T0)["id"] == second["id"]
    assert queue.claim_next_job("w3", db, T0) is None


def test_empty_queue_claims_nothing(db):
    assert queue.claim_next_job("w1", db, T0) is None


def test_concurrent_workers_each_claim_a_different_job(db):
    ids = {submit(db)["id"] for _ in range(12)}
    claimed, errors = [], []

    def work(name):
        try:
            while (job := queue.claim_next_job(name, db)) is not None:
                claimed.append(job["id"])
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not errors
    assert sorted(claimed) == sorted(ids)   # every job once, none twice


# ---------------------------------------------------------------------------
# Queue: heartbeat and finishing
# ---------------------------------------------------------------------------

def test_heartbeat_records_progress(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    beat = T0 + timedelta(seconds=30)
    assert queue.heartbeat(job["id"], "w1", 0.4, db, beat) == "continue"
    stored = queue.get_job(job["id"], db)
    assert stored["progress"] == pytest.approx(0.4)
    assert stored["heartbeat_at"] == beat.isoformat()


@pytest.mark.parametrize("reported, stored", [(-0.5, 0.0), (1.7, 1.0)])
def test_progress_is_clamped(db, reported, stored):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    queue.heartbeat(job["id"], "w1", reported, db)
    assert queue.get_job(job["id"], db)["progress"] == stored


def test_heartbeat_from_another_worker_is_lost(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    assert queue.heartbeat(job["id"], "intruder", 0.9, db) == "lost"
    assert queue.get_job(job["id"], db)["progress"] == 0


def test_finish_stores_the_result(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    end = T0 + timedelta(seconds=5)
    assert queue.finish_job(job["id"], "w1", "succeeded", result={"answer": 42}, db_path=db, now=end)
    stored = queue.get_job(job["id"], db)
    assert stored["status"] == "succeeded" and stored["result"] == {"answer": 42}
    assert stored["progress"] == 1 and stored["finished_at"] == end.isoformat()


def test_failed_job_keeps_the_progress_it_reached(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    queue.heartbeat(job["id"], "w1", 0.3, db)
    queue.finish_job(job["id"], "w1", "failed", error="boom", db_path=db)
    stored = queue.get_job(job["id"], db)
    assert stored["status"] == "failed" and stored["error"] == "boom"
    assert stored["progress"] == pytest.approx(0.3)


def test_only_the_owner_can_finish_and_only_once(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    assert not queue.finish_job(job["id"], "intruder", "succeeded", result={"x": 1}, db_path=db)
    assert queue.get_job(job["id"], db)["status"] == "running"
    assert queue.finish_job(job["id"], "w1", "succeeded", result={"x": 2}, db_path=db)
    assert not queue.finish_job(job["id"], "w1", "failed", error="late", db_path=db)
    assert queue.get_job(job["id"], db)["result"] == {"x": 2}


def test_finish_rejects_a_non_final_status_and_nan(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    with pytest.raises(ValueError):
        queue.finish_job(job["id"], "w1", "running", db_path=db)
    with pytest.raises(ValueError):
        queue.finish_job(job["id"], "w1", "succeeded", result={"x": float("nan")}, db_path=db)
    assert queue.get_job(job["id"], db)["status"] == "running"


# ---------------------------------------------------------------------------
# Queue: a worker that dies
# ---------------------------------------------------------------------------

def test_silent_job_goes_back_to_the_queue_for_another_worker(db):
    job = submit(db)
    queue.claim_next_job("dead", db, T0)
    queue.heartbeat(job["id"], "dead", 0.6, db, T0)

    again = queue.claim_next_job("alive", db, LATER)
    assert again["id"] == job["id"] and again["worker_id"] == "alive"
    assert again["attempts"] == 2 and again["progress"] == 0

    # The first worker was only slow: it must not touch the job any more
    assert queue.heartbeat(job["id"], "dead", 0.9, db) == "lost"
    assert not queue.finish_job(job["id"], "dead", "succeeded", result={"stale": True}, db_path=db)
    assert queue.get_job(job["id"], db)["result"] is None


def test_job_with_a_recent_heartbeat_is_left_alone(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    almost = T0 + queue.STALE_AFTER - timedelta(seconds=1)
    assert queue.claim_next_job("w2", db, almost) is None
    assert queue.get_job(job["id"], db)["worker_id"] == "w1"


def test_heartbeats_keep_a_long_job_alive(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    queue.heartbeat(job["id"], "w1", 0.5, db, LATER - timedelta(seconds=10))
    assert queue.claim_next_job("w2", db, LATER) is None


def test_job_that_keeps_killing_its_worker_fails(db):
    job = submit(db)
    now = T0
    for attempt in range(1, queue.MAX_ATTEMPTS + 1):
        claimed = queue.claim_next_job(f"w{attempt}", db, now)
        assert claimed["id"] == job["id"] and claimed["attempts"] == attempt
        now += queue.STALE_AFTER + timedelta(seconds=1)

    assert queue.claim_next_job("one-more", db, now) is None
    stored = queue.get_job(job["id"], db)
    assert stored["status"] == "failed"
    assert str(queue.MAX_ATTEMPTS) in stored["error"]
    assert stored["finished_at"] == now.isoformat()


def test_silent_job_the_user_cancelled_is_cancelled_not_retried(db):
    job = submit(db)
    queue.claim_next_job("dead", db, T0)
    queue.request_cancel(job["id"], db)
    assert queue.claim_next_job("alive", db, LATER) is None
    assert queue.get_job(job["id"], db)["status"] == "cancelled"


def test_released_job_is_queued_again_without_costing_an_attempt(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    queue.heartbeat(job["id"], "w1", 0.5, db)
    assert queue.release_job(job["id"], "w1", db)
    stored = queue.get_job(job["id"], db)
    assert stored["status"] == "queued" and stored["attempts"] == 0
    assert stored["progress"] == 0 and stored["worker_id"] is None
    assert not queue.release_job(job["id"], "w1", db)   # no longer running


# ---------------------------------------------------------------------------
# Queue: cancelling, listing, cleaning up
# ---------------------------------------------------------------------------

def test_cancelling_a_queued_job_is_immediate(db):
    job = submit(db)
    cancelled = queue.request_cancel(job["id"], db, T0)
    assert cancelled["status"] == "cancelled" and cancelled["finished_at"] == T0.isoformat()
    assert queue.claim_next_job("w1", db, T0) is None


def test_cancelling_a_running_job_asks_its_worker(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    asked = queue.request_cancel(job["id"], db)
    assert asked["status"] == "running" and asked["cancel_requested"] is True
    assert queue.heartbeat(job["id"], "w1", 0.5, db) == "cancel"


def test_cancelling_a_finished_job_changes_nothing(db):
    job = submit(db)
    queue.claim_next_job("w1", db, T0)
    queue.finish_job(job["id"], "w1", "succeeded", result={"x": 1}, db_path=db)
    after = queue.request_cancel(job["id"], db)
    assert after["status"] == "succeeded" and after["cancel_requested"] is False


def test_cancelling_a_missing_job_is_none(db):
    assert queue.request_cancel(12345, db) is None


def test_listing_is_per_user_newest_first_and_without_results(db):
    a1 = submit(db, user_id=1)
    submit(db, user_id=2)
    a2 = submit(db, user_id=1)
    queue.claim_next_job("w1", db, T0)
    queue.finish_job(a1["id"], "w1", "succeeded", result={"big": "x" * 1000}, db_path=db)

    listed = queue.list_user_jobs(1, db_path=db)
    assert [job["id"] for job in listed] == [a2["id"], a1["id"]]
    assert all("result" not in job for job in listed)
    assert queue.list_user_jobs(1, limit=1, db_path=db)[0]["id"] == a2["id"]


def test_active_count_covers_queued_and_running_only(db):
    done, running, _queued = submit(db), submit(db), submit(db)
    submit(db, user_id=2)
    queue.claim_next_job("w1", db, T0)
    queue.finish_job(done["id"], "w1", "succeeded", result={}, db_path=db)
    queue.claim_next_job("w1", db, T0)
    assert queue.get_job(running["id"], db)["status"] == "running"
    assert queue.count_active_jobs(1, db) == 2
    assert queue.count_active_jobs(2, db) == 1
    assert queue.count_active_jobs(3, db) == 0


def test_cleanup_deletes_only_old_finished_jobs(db):
    old, recent, active = submit(db), submit(db), submit(db)
    queue.claim_next_job("w1", db, T0)
    queue.finish_job(old["id"], "w1", "succeeded", result={}, db_path=db, now=T0)
    queue.claim_next_job("w1", db, T0)
    queue.finish_job(recent["id"], "w1", "failed", error="x", db_path=db, now=T0 + timedelta(days=5))

    assert queue.delete_finished_jobs(T0 + timedelta(days=1), db) == 1
    assert queue.get_job(old["id"], db) is None
    assert queue.get_job(recent["id"], db) is not None
    assert queue.get_job(active["id"], db)["status"] == "queued"   # however old, never deleted


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EchoParams(BaseModel):
    n: int


def job_types(run):
    return {"echo": JobType(params_model=EchoParams, run=run)}


def run(db, handler, **kwargs):
    return run_next_job("w1", job_types(handler), db, heartbeat_interval=0, **kwargs)


def test_runner_stores_what_the_handler_returns(db):
    submit(db, params={"n": 4})
    seen = {}

    def handler(params, ctx):
        seen["params"] = params
        ctx.progress(0.5)
        return {"squared": params.n ** 2}

    job = run(db, handler)
    assert seen["params"] == EchoParams(n=4)
    assert job["status"] == "succeeded" and job["result"] == {"squared": 16}
    assert job["progress"] == 1


def test_runner_with_an_empty_queue_does_nothing(db):
    assert run(db, lambda params, ctx: {}) is None


def test_handler_value_error_reaches_the_user(db):
    submit(db)

    def handler(params, ctx):
        raise ValueError("crater larger than the Moon")

    job = run(db, handler)
    assert job["status"] == "failed" and job["error"] == "crater larger than the Moon"
    assert job["result"] is None


def test_unexpected_error_is_not_shown_to_the_user(db):
    submit(db)

    def handler(params, ctx):
        raise RuntimeError("secret internal detail /srv/app/config")

    job = run(db, handler)
    assert job["status"] == "failed"
    assert job["error"] == "Internal error (RuntimeError)."


def test_unknown_job_type_fails(db):
    submit(db, job_type="retired-type")
    job = run(db, lambda params, ctx: {})
    assert job["status"] == "failed" and "retired-type" in job["error"]


def test_params_that_no_longer_validate_fail_the_job(db):
    submit(db, params={"n": "not a number"})
    job = run(db, lambda params, ctx: {})
    assert job["status"] == "failed" and "n" in job["error"]


def test_result_that_is_not_valid_json_fails_the_job(db):
    submit(db)
    job = run(db, lambda params, ctx: {"x": float("inf")})
    assert job["status"] == "failed" and job["result"] is None


def test_cancelled_while_running_stops_at_the_next_progress_report(db):
    submitted = submit(db)
    steps = []

    def handler(params, ctx):
        for step in range(5):
            steps.append(step)
            if step == 1:
                queue.request_cancel(submitted["id"], db)
            ctx.progress(step / 5)
        return {"finished": True}

    job = run(db, handler)
    assert steps == [0, 1]
    assert job["status"] == "cancelled" and job["result"] is None
    assert job["finished_at"] is not None


def test_shutdown_puts_the_job_back_in_the_queue(db):
    submit(db)
    stopping = threading.Event()
    steps = []

    def handler(params, ctx):
        for step in range(5):
            steps.append(step)
            if step == 2:
                stopping.set()
            ctx.progress(step / 5)
        return {}

    job = run(db, handler, should_stop=stopping.is_set)
    assert steps == [0, 1, 2]
    assert job["status"] == "queued" and job["attempts"] == 0

    stopping.clear()
    assert run(db, handler)["status"] == "succeeded"   # the next worker completes it


def test_worker_presumed_dead_does_not_overwrite_the_job(db):
    submitted = submit(db)

    def handler(params, ctx):
        # While this handler was stuck, the job was given to another worker
        assert queue.claim_next_job("w2", db, datetime.now(timezone.utc) + queue.STALE_AFTER * 2)
        ctx.progress(0.5)
        return {"from": "w1"}   # pragma: no cover - progress() raises

    run(db, handler)
    job = queue.get_job(submitted["id"], db)
    assert job["status"] == "running" and job["worker_id"] == "w2"
    assert job["result"] is None


def test_progress_writes_are_rate_limited(db, monkeypatch):
    submit(db)
    beats = []
    real = queue.heartbeat
    monkeypatch.setattr(queue, "heartbeat", lambda *a, **k: beats.append(a) or real(*a, **k))

    def handler(params, ctx):
        for step in range(100):
            ctx.progress(step / 100)
        return {}

    job = run_next_job("w1", job_types(handler), db, heartbeat_interval=3600)
    assert job["status"] == "succeeded"
    assert len(beats) == 1


# ---------------------------------------------------------------------------
# The inference_batch job type
# ---------------------------------------------------------------------------

def test_inference_batch_infers_each_crater(db):
    submit(db, job_type="inference_batch", params={"crater_diameters": [18.8, 2000.0]})
    job = run_next_job("w1", JOB_TYPES, db, heartbeat_interval=0)

    assert job["status"] == "succeeded", job["error"]
    result = job["result"]
    assert result["count"] == 2
    small, large = result["craters"]
    assert small["crater_diameter"] == 18.8 and small["material"] == "regolith"
    assert large["crater_diameter"] == 2000.0 and large["material"] == "rock"
    for crater in (small, large):
        assert crater["effective_sample_size"] > 100
        for name in ("impactor_diameter", "velocity", "mass", "kinetic_energy"):
            p = crater["posterior"][name]
            assert 0 < p["p5"] < p["p50"] < p["p95"]
    # A bigger crater needs a bigger impactor
    assert large["posterior"]["impactor_diameter"]["p50"] > small["posterior"]["impactor_diameter"]["p50"]


def test_inference_batch_agrees_with_the_single_inference_endpoint(client, ada, db):
    single = client.post("/api/inference", json={"crater_diameter": 18.8}, headers=ada).json()
    submit(db, job_type="inference_batch", params={"crater_diameters": [18.8]})
    batch = run_next_job("w1", JOB_TYPES, db, heartbeat_interval=0)["result"]["craters"][0]

    assert batch["effective_sample_size"] == single["effective_sample_size"]
    for name, summary in batch["posterior"].items():
        assert summary == single["parameters"][name]["posterior"]


def test_inference_batch_is_reproducible(db):
    params = {"crater_diameters": [50.0], "diameter_error": 0.1}
    submit(db, job_type="inference_batch", params=params)
    submit(db, job_type="inference_batch", params=params)
    first = run_next_job("w1", JOB_TYPES, db, heartbeat_interval=0)
    second = run_next_job("w1", JOB_TYPES, db, heartbeat_interval=0)
    assert first["result"] == second["result"]


@pytest.mark.parametrize("params", [
    {"crater_diameters": []},
    {"crater_diameters": [10.0] * 501},
    {"crater_diameters": [-5.0]},
    {"crater_diameters": [10.0], "diameter_error": 0.0},
    {"crater_diameters": [10.0], "material": "cheese"},
    {},
])
def test_inference_batch_rejects_bad_params(params):
    with pytest.raises(ValidationError):
        InferenceBatchParams.model_validate(params)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def wait_for(condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out waiting for the worker")


def test_worker_thread_runs_jobs_as_they_arrive(db):
    worker = Worker(job_types(lambda params, ctx: {"n": params.n}), db, poll_interval=60)
    worker.start()
    try:
        ids = []
        for n in (1, 2):
            ids.append(submit(db, params={"n": n}, now=None)["id"])
            worker.notify()   # poll_interval is 60 s: only notify() can explain a prompt run
        finished = [wait_for(lambda i=i: (j := queue.get_job(i, db))["status"] == "succeeded" and j)
                    for i in ids]
        assert [job["result"] for job in finished] == [{"n": 1}, {"n": 2}]
    finally:
        worker.stop()


def test_stopping_the_worker_requeues_the_job_it_was_running(db):
    started = threading.Event()

    def handler(params, ctx):
        while True:
            started.set()
            ctx.progress(0.1)
            time.sleep(0.01)

    job = submit(db, now=None)
    worker = Worker(job_types(handler), db, heartbeat_interval=0)
    worker.start()
    assert started.wait(20)
    worker.stop()

    stored = queue.get_job(job["id"], db)
    assert stored["status"] == "queued" and stored["attempts"] == 0


def test_worker_survives_a_database_error(db, monkeypatch):
    calls = []
    real = queue.claim_next_job

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("database unreachable")
        return real(*args, **kwargs)

    monkeypatch.setattr(queue, "claim_next_job", flaky)
    job = submit(db, now=None)
    worker = Worker(job_types(lambda params, ctx: {"ok": True}), db, poll_interval=0.05)
    worker.start()
    try:
        wait_for(lambda: queue.get_job(job["id"], db)["status"] == "succeeded")
    finally:
        worker.stop()
    assert len(calls) >= 2


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

BATCH = {"type": "inference_batch", "params": {"crater_diameters": [18.8]}}


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setattr("database.models.DB_PATH", db)
    with TestClient(app) as c:
        yield c


def sign_up(client, username):
    response = client.post("/api/auth/register", json={"username": username, "password": "moonrock"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


@pytest.fixture
def ada(client):
    return sign_up(client, "ada")


def test_jobs_require_sign_in(client):
    assert client.post("/api/jobs", json=BATCH).status_code == 401
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs/1").status_code == 401
    assert client.post("/api/jobs/1/cancel").status_code == 401


def test_job_types_are_listed_with_their_schema(client):
    types = client.get("/api/jobs/types").json()
    batch = next(t for t in types if t["type"] == "inference_batch")
    assert "crater_diameters" in batch["params_schema"]["properties"]


def test_submit_then_poll_until_the_result_is_ready(client, ada, db):
    response = client.post("/api/jobs", json=BATCH, headers=ada)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued" and job["result"] is None
    # Defaults are filled in, so the stored params say exactly what will run
    assert job["params"] == {"crater_diameters": [18.8], "diameter_error": 0.05, "material": None}

    assert client.get(f"/api/jobs/{job['id']}", headers=ada).json()["status"] == "queued"

    run_next_job("w1", JOB_TYPES, db, heartbeat_interval=0)

    done = client.get(f"/api/jobs/{job['id']}", headers=ada).json()
    assert done["status"] == "succeeded" and done["progress"] == 1
    assert done["result"]["count"] == 1


def test_job_replies_do_not_expose_internals(client, ada):
    job = client.post("/api/jobs", json=BATCH, headers=ada).json()
    assert {"worker_id", "user_id", "attempts", "heartbeat_at"}.isdisjoint(job)


def test_unknown_type_and_bad_params_are_rejected(client, ada):
    unknown = client.post("/api/jobs", json={"type": "mine-bitcoin", "params": {}}, headers=ada)
    assert unknown.status_code == 422 and "mine-bitcoin" in unknown.json()["detail"]

    bad = client.post("/api/jobs", json={"type": "inference_batch", "params": {"crater_diameters": []}},
                      headers=ada)
    assert bad.status_code == 422
    assert bad.json()["detail"][0]["loc"] == ["crater_diameters"]
    assert client.get("/api/jobs", headers=ada).json()["jobs"] == []


def test_active_jobs_per_user_are_limited(client, ada):
    from backend.routers.jobs import MAX_ACTIVE_JOBS_PER_USER

    ids = [client.post("/api/jobs", json=BATCH, headers=ada).json()["id"]
           for _ in range(MAX_ACTIVE_JOBS_PER_USER)]
    assert client.post("/api/jobs", json=BATCH, headers=ada).status_code == 429

    # Another user is unaffected, and cancelling frees a slot
    assert client.post("/api/jobs", json=BATCH, headers=sign_up(client, "grace")).status_code == 202
    assert client.post(f"/api/jobs/{ids[0]}/cancel", headers=ada).json()["status"] == "cancelled"
    assert client.post("/api/jobs", json=BATCH, headers=ada).status_code == 202


def test_users_cannot_see_or_cancel_each_others_jobs(client, ada):
    grace = sign_up(client, "grace")
    job = client.post("/api/jobs", json=BATCH, headers=ada).json()

    assert client.get(f"/api/jobs/{job['id']}", headers=grace).status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/cancel", headers=grace).status_code == 404
    assert client.get("/api/jobs", headers=grace).json()["jobs"] == []
    assert client.get(f"/api/jobs/{job['id']}", headers=ada).json()["status"] == "queued"


def test_list_is_newest_first_without_results(client, ada, db):
    first = client.post("/api/jobs", json=BATCH, headers=ada).json()
    second = client.post("/api/jobs", json=BATCH, headers=ada).json()
    run_next_job("w1", JOB_TYPES, db, heartbeat_interval=0)

    jobs = client.get("/api/jobs", headers=ada).json()["jobs"]
    assert [job["id"] for job in jobs] == [second["id"], first["id"]]
    assert [job["status"] for job in jobs] == ["queued", "succeeded"]
    assert all("result" not in job for job in jobs)


def test_cancel_a_running_job_through_the_api(client, ada, db):
    job = client.post("/api/jobs", json=BATCH, headers=ada).json()
    queue.claim_next_job("w1", db)

    reply = client.post(f"/api/jobs/{job['id']}/cancel", headers=ada).json()
    assert reply["status"] == "running" and reply["cancel_requested"] is True
    assert queue.heartbeat(job["id"], "w1", 0.5, db) == "cancel"


def test_missing_job_is_404(client, ada):
    assert client.get("/api/jobs/99999", headers=ada).status_code == 404
    assert client.post("/api/jobs/99999/cancel", headers=ada).status_code == 404


def test_app_starts_a_worker_only_when_enabled(db, monkeypatch):
    monkeypatch.setattr("database.models.DB_PATH", db)
    with TestClient(app):
        assert app.state.job_worker is None   # conftest turns it off for tests

    monkeypatch.setenv("LUNALAB_JOB_WORKER", "1")
    with TestClient(app) as client:
        worker = app.state.job_worker
        assert worker is not None
        headers = sign_up(client, "ada")
        job = client.post("/api/jobs", json=BATCH, headers=headers).json()
        done = wait_for(lambda: (j := client.get(f"/api/jobs/{job['id']}", headers=headers).json())
                        ["status"] not in ("queued", "running") and j)
        assert done["status"] == "succeeded", done
    assert not worker._thread.is_alive()   # stopped with the app
