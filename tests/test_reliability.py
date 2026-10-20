import asyncio
import os
import shutil
import sqlite3
import time

import pytest

from app import main
from app.jobs import JobStore
from tests.conftest import KEY, assert_manual_review, upload, wait_done


def wait_for(predicate, timeout: float = 10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def hold(tmp_path, monkeypatch):
    flag = tmp_path / "hold"
    flag.touch()
    monkeypatch.setenv("SLOW_FLAG", str(flag))
    return flag


def status(client, job_id):
    return client.app.state.store.get(job_id)["status"]


def seed_job(store: JobStore, videos, source: str = "api") -> str:
    job_id = store.new_id()
    store.job_dir(job_id).mkdir(parents=True)
    shutil.copy(videos["vfr"], store.video_path(job_id))
    store.create(job_id, "vfr.mov", videos["vfr"].stat().st_size, 4.0, source)
    return job_id


def test_worker_survives_failing_iteration(client, videos, monkeypatch):
    store = client.app.state.store
    original = store.claim_next
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return original()

    monkeypatch.setattr(store, "claim_next", flaky)
    store.wakeup.set()
    wait_for(lambda: calls["n"] >= 1)
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert body["status"] == "done"
    assert client.app.state.worker.alive


def test_unserializable_result_fails_job_and_worker_continues(make_client, videos):
    client = make_client(detector="tests.unserializable_detector")
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert_manual_review(body, "result_invalid")
    assert client.app.state.worker.alive
    assert wait_done(client, upload(client, videos["vfr"]).json()["job_id"])["error"] == "result_invalid"


def test_healthz_503_when_worker_dead(client):
    assert client.get("/healthz").json()["worker_alive"] is True
    client.app.state.worker.stop()
    r = client.get("/healthz")
    assert r.status_code == 503
    assert r.json()["worker_alive"] is False


def test_claim_fails_jobs_past_deadline(tmp_path, videos):
    store = JobStore(tmp_path / "data", deadline_sec=285)
    late = seed_job(store, videos)
    fresh = seed_job(store, videos)
    store._write("update jobs set created_at = created_at - 286 where id = ?", (late,))
    claimed = store.claim_next()
    assert claimed["id"] == fresh
    assert store.get(late)["status"] == "failed"
    assert store.get(late)["error"] == "deadline"


def test_budget_is_capped_by_upload_deadline(make_client, videos, hold):
    client = make_client(detector="tests.slow_detector", job_deadline_sec=1.5, job_timeout_sec=270, post_budget_sec=0, min_budget_sec=0)
    started = time.time()
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert body["error"] == "timeout"
    assert time.time() - started < 3


def test_detector_runs_in_child_and_is_killed_on_timeout(make_client, videos, hold):
    client = make_client(detector="tests.slow_detector", job_timeout_sec=0.5)
    detector = client.app.state.detector
    first_pid = detector.pid
    assert first_pid != os.getpid()
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert body["error"] == "timeout"
    assert not pid_alive(first_pid)
    assert detector.pid not in (None, first_pid)
    hold.unlink()
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert body["status"] == "done"


def test_requeue_once_then_crashed(tmp_path, make_client, videos):
    store = JobStore(tmp_path / "data")
    once = seed_job(store, videos)
    twice = seed_job(store, videos)
    store._write("update jobs set created_at = created_at + 1 where id = ?", (twice,))
    store.claim_next()
    store._write("update jobs set status = 'running', attempts = 2 where id = ?", (twice,))
    assert store.get(once)["attempts"] == 1
    store.close()

    client = make_client()
    assert wait_done(client, once)["status"] == "done"
    body = wait_done(client, twice)
    assert_manual_review(body, "crashed")
    assert body["message"] == "The server restarted twice while processing this video."
    assert client.app.state.store.get(once)["attempts"] == 2


def test_old_database_is_migrated(tmp_path):
    (tmp_path / "data").mkdir()
    db = sqlite3.connect(tmp_path / "data" / "jobs.db")
    db.execute("create table jobs (id text primary key, status text not null, filename text, size integer, duration_sec real,"
               " created_at real not null, started_at real, finished_at real, progress real not null default 0,"
               " result text, raw text, error text)")
    db.execute("insert into jobs (id, status, created_at) values ('old', 'running', 0)")
    db.commit()
    db.close()
    store = JobStore(tmp_path / "data")
    assert store.get("old")["attempts"] == 0
    assert store.get("old")["source"] == "api"
    assert store.requeue_running() == (1, 0)


def test_content_length_rejected_before_body_is_read():
    sent = []

    async def downstream(scope, receive, send):
        raise AssertionError("request reached the app")

    async def receive():
        raise AssertionError("body was read")

    async def send(message):
        sent.append(message)

    middleware = main.UploadLimit(downstream, limit=1024)
    scope = {"type": "http", "method": "POST", "path": "/api/v1/analyze", "headers": [(b"content-length", str(10 * 1024 * 1024).encode())]}
    asyncio.run(middleware(scope, receive, send))
    assert sent[0]["status"] == 413


def test_content_length_over_limit_via_http(make_client, videos):
    client = make_client(max_upload_bytes=1024)
    r = client.post("/app/analyze", content=b"x" * (200 * 1024), headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413


def test_app_queue_cap_and_api_priority(make_client, videos, hold):
    client = make_client(detector="tests.slow_detector")
    blocker = upload(client, videos["vfr"], key=None, url="/app/analyze").json()["job_id"]
    wait_for(lambda: status(client, blocker) == "running")
    app_jobs = [upload(client, videos["vfr"], key=None, url="/app/analyze").json()["job_id"] for _ in range(3)]
    r = upload(client, videos["vfr"], key=None, url="/app/analyze")
    assert r.status_code == 429
    api_job = upload(client, videos["vfr"]).json()["job_id"]
    assert client.get(f"/app/jobs/{api_job}").json()["queue_position"] == 1
    assert client.get(f"/app/jobs/{app_jobs[0]}").json()["queue_position"] == 2
    hold.unlink()
    for job_id in [blocker, api_job, *app_jobs]:
        assert wait_done(client, job_id, url="/app/jobs/{}")["status"] == "done"
    store = client.app.state.store
    assert store.get(api_job)["started_at"] < min(store.get(j)["started_at"] for j in app_jobs)


def test_upload_io_runs_off_event_loop(client, videos, monkeypatch):
    seen = []

    def off_loop(fn):
        def wrapper(*args):
            try:
                asyncio.get_running_loop()
                seen.append("loop")
            except RuntimeError:
                seen.append("thread")
            return fn(*args)

        return wrapper

    monkeypatch.setattr(main, "probe", off_loop(main.probe))
    monkeypatch.setattr(main, "save_upload", off_loop(main.save_upload))
    assert upload(client, videos["vfr"]).status_code == 200
    assert seen == ["thread", "thread"]


def test_purge_removes_orphan_directories(tmp_path, videos):
    store = JobStore(tmp_path / "data")
    known = seed_job(store, videos)
    old_orphan = store.job_dir("orphan-old")
    new_orphan = store.job_dir("orphan-new")
    old_orphan.mkdir()
    new_orphan.mkdir()
    two_days = time.time() - 2 * 86400
    os.utime(old_orphan, (two_days, two_days))
    os.utime(store.job_dir(known), (two_days, two_days))
    assert store.purge_older_than(86400) == 1
    assert not old_orphan.exists()
    assert new_orphan.exists()
    assert store.job_dir(known).exists()


def test_detector_receives_remaining_budget():
    from app.detector_proc import DetectorProcess

    proc = DetectorProcess("tests.budget_detector", pdeathsig=False)
    proc.start()
    try:
        assert proc.wait_ready(30)
        assert 40 < proc.analyze("unused.mov", 42)["budget_sec"] <= 42
        assert proc.analyze("unused.mov", 5)["budget_sec"] <= 5
    finally:
        proc.stop()
