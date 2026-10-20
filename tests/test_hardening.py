import asyncio
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import ROOT, Settings
from app.jobs import JobStore, Worker
from app.main import create_app
from tests.conftest import KEY, upload, wait_done
from tests.live import alive, group_members


def wait_for(predicate, timeout: float = 10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met")


@pytest.fixture
def hold(tmp_path, monkeypatch):
    flag = tmp_path / "hold"
    flag.touch()
    monkeypatch.setenv("SLOW_FLAG", str(flag))
    return flag


def seed(store: JobStore, videos, age_sec: float = 0) -> str:
    job_id = store.new_id()
    store.job_dir(job_id).mkdir(parents=True)
    shutil.copy(videos["vfr"], store.video_path(job_id))
    store.create(job_id, "vfr.mov", 1, 4.0)
    if age_sec:
        store._write("update jobs set created_at = created_at - ? where id = ?", (age_sec, job_id))
    return job_id


def test_chunked_body_limit_in_middleware():
    reads = {"n": 0}

    async def endless_receive():
        reads["n"] += 1
        return {"type": "http.request", "body": b"x" * 65536, "more_body": True}

    async def app(scope, receive, send):
        while True:
            await receive()

    middleware = main.UploadLimit(app, limit=1024 * 1024)
    scope = {"type": "http", "method": "POST", "path": "/app/analyze", "headers": []}
    with pytest.raises(main.ApiError) as exc:
        asyncio.run(middleware(scope, endless_receive, None))
    assert exc.value.status == 413
    assert reads["n"] == (1024 * 1024 + main.MULTIPART_SLACK) // 65536 + 1


def test_chunked_upload_gets_413(make_client):
    client = make_client(max_upload_bytes=1024 * 1024)
    sent = {"bytes": 0}

    def body():
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="big.mov"\r\n\r\n'
        for _ in range(200):
            sent["bytes"] += 65536
            yield b"\0" * 65536

    r = client.post("/app/analyze", content=body(), headers={"Content-Type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert r.json()["error"] == "file_too_large"
    assert not any((client.app.state.store.data_dir / "jobs").iterdir())


def test_timeout_kills_whole_detector_group(make_client, videos, tmp_path, monkeypatch):
    pidfile = tmp_path / "grandchild.pid"
    monkeypatch.setenv("GRANDCHILD_PIDFILE", str(pidfile))
    client = make_client(detector="tests.forking_detector", job_timeout_sec=2, post_budget_sec=0, min_budget_sec=0)
    first = client.app.state.detector.pid
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert body["error"] == "timeout"
    grandchild = int(pidfile.read_text())
    assert not alive(first) and not alive(grandchild)
    assert group_members(first) == []
    assert client.app.state.detector.pid != first


def test_app_queue_cap_is_atomic(make_client, videos, hold):
    client = make_client(detector="tests.slow_detector")
    blocker = upload(client, videos["vfr"]).json()["job_id"]
    wait_for(lambda: client.app.state.store.get(blocker)["status"] == "running")
    with ThreadPoolExecutor(10) as pool:
        codes = list(pool.map(lambda _: upload(client, videos["vfr"], key=None, url="/app/analyze").status_code, range(10)))
    assert codes.count(200) == 3
    assert codes.count(429) == 7
    assert client.app.state.store.queued_count("app") == 3
    hold.unlink()


def test_failed_upload_releases_slot(make_client, videos, tmp_path, hold):
    client = make_client(detector="tests.slow_detector")
    blocker = upload(client, videos["vfr"]).json()["job_id"]
    wait_for(lambda: client.app.state.store.get(blocker)["status"] == "running")
    garbage = tmp_path / "garbage.mov"
    garbage.write_bytes(os.urandom(4096))
    for _ in range(5):
        assert upload(client, garbage, key=None, url="/app/analyze").status_code == 400
    assert [upload(client, videos["vfr"], key=None, url="/app/analyze").status_code for _ in range(4)] == [200, 200, 200, 429]
    hold.unlink()


def test_loading_child_is_not_killed_on_ready_timeout(make_client, videos, monkeypatch):
    monkeypatch.setenv("LOAD_DELAY_SEC", "3")
    client = make_client(detector="tests.slow_load_detector", job_timeout_sec=0.5, post_budget_sec=0, min_budget_sec=0)
    detector = client.app.state.detector
    pid = detector.pid
    assert client.get("/healthz").status_code == 503
    assert client.get("/healthz").json()["status"] == "model_not_loaded"
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert body["error"] == "detector_not_ready"
    assert detector.pid == pid and alive(pid)
    wait_for(lambda: client.get("/healthz").status_code == 200, timeout=10)
    assert detector.pid == pid
    assert wait_done(client, upload(client, videos["vfr"]).json()["job_id"])["status"] == "done"


def test_claim_fails_job_below_min_budget(tmp_path, videos):
    store = JobStore(tmp_path / "data", deadline_sec=285, post_budget_sec=15, min_budget_sec=20)
    short = seed(store, videos, age_sec=251)
    enough = seed(store, videos, age_sec=249)
    claimed = store.claim_next()
    assert claimed["id"] == enough
    assert store.get(short)["status"] == "failed"
    assert store.get(short)["error"] == "deadline"


def test_detector_budget_reserves_post_processing(tmp_path, videos):
    store = JobStore(tmp_path / "data", deadline_sec=285, post_budget_sec=15, min_budget_sec=20)
    seen = {}

    def process(job, progress, timings, budget):
        seen["budget"] = budget
        return {"verdict": "no_violation", "violations": []}, {"events": []}

    worker = Worker(store, process, timeout_sec=270, retention_sec=3600)
    seed(store, videos)
    worker.run_one(store.claim_next())
    assert 254 < seen["budget"] <= 270
    old = seed(store, videos, age_sec=200)
    worker.run_one(store.claim_next())
    assert 69 < seen["budget"] <= 70
    assert store.get(old)["status"] == "done"


def test_result_after_deadline_is_failed(tmp_path, videos):
    store = JobStore(tmp_path / "data", deadline_sec=2, post_budget_sec=0, min_budget_sec=0)

    def process(job, progress, timings, budget):
        time.sleep(2.2)
        return {"verdict": "no_violation", "violations": []}, {"events": []}

    worker = Worker(store, process, timeout_sec=270, retention_sec=3600)
    job_id = seed(store, videos)
    worker.run_one(store.claim_next())
    assert store.get(job_id)["status"] == "failed"
    assert store.get(job_id)["error"] == "deadline"


def test_graceful_shutdown_keeps_job_running_and_requeues(tmp_path, videos, hold):
    settings = Settings(api_key=KEY, data_dir=tmp_path / "data", detector="tests.slow_detector")
    client = TestClient(create_app(settings))
    client.__enter__()
    job_id = upload(client, videos["vfr"]).json()["job_id"]
    store = client.app.state.store
    wait_for(lambda: store.get(job_id)["status"] == "running")
    client.__exit__(None, None, None)
    row = JobStore(tmp_path / "data").get(job_id)
    assert row["status"] == "running"
    assert row["error"] is None
    assert row["attempts"] == 0
    hold.unlink()
    with TestClient(create_app(settings)) as again:
        assert wait_done(again, job_id)["status"] == "done"
        assert again.app.state.store.get(job_id)["attempts"] == 1


def test_healthz_503_when_model_not_loaded(client):
    assert client.get("/healthz").status_code == 200
    os.kill(client.app.state.detector.pid, 9)
    wait_for(lambda: client.get("/healthz").status_code == 503)
    body = client.get("/healthz").json()
    assert body["model_loaded"] is False and body["worker_alive"] is True


def test_worker_loop_death_exits_process(tmp_path):
    store = JobStore(tmp_path / "data")
    worker = Worker(store, lambda *a: None, timeout_sec=1, retention_sec=3600)
    exits = []
    worker.exit = exits.append

    def explode():
        raise KeyboardInterrupt()

    store.claim_next = explode
    hook = threading.excepthook
    threading.excepthook = lambda args: None
    try:
        worker.start()
        worker.join(timeout=5)
    finally:
        threading.excepthook = hook
    assert exits == [1]


def test_requested_stop_does_not_exit(tmp_path):
    store = JobStore(tmp_path / "data")
    worker = Worker(store, lambda *a: None, timeout_sec=1, retention_sec=3600)
    exits = []
    worker.exit = exits.append
    worker.start()
    worker.stop()
    assert exits == []


def assert_clean_error(body: dict) -> None:
    text = json.dumps(body, ensure_ascii=False)
    assert re.fullmatch(r"[a-z_0-9]+", body["error"])
    assert "Traceback" not in text and "/" not in body.get("detail", "") + body.get("message", "")
    assert "\n" not in text and "Errno" not in text


def test_errors_are_codes_without_paths(make_client, videos, tmp_path):
    client = make_client(detector="tests.failing_detector")
    garbage = tmp_path / "garbage.mov"
    garbage.write_bytes(os.urandom(4096))
    r = upload(client, garbage)
    assert r.status_code == 400 and r.json()["error"] == "invalid_video"
    assert_clean_error(r.json())
    for r, code in [
        (client.get("/api/v1/jobs/nope", headers={"X-API-Key": KEY}), "job_not_found"),
        (client.get("/nope"), "not_found"),
        (upload(client, videos["vfr"], key="bad"), "unauthorized"),
        (upload(client, videos["vfr"], name="a.avi"), "wrong_extension"),
    ]:
        assert r.json()["error"] == code
        assert_clean_error(r.json())
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert body["error"] == "detector_failed"
    assert_clean_error(body)
    store = client.app.state.store
    legacy = seed(store, videos)
    store.claim_next()
    store.fail(legacy, "RuntimeError: cannot open /srv/data/jobs/x/input.mov\nTraceback ...")
    body = client.get(f"/api/v1/jobs/{legacy}", headers={"X-API-Key": KEY}).json()
    assert body["error"] == "processing_failed"
    assert_clean_error(body)


def test_unhandled_error_hides_traceback(tmp_path):
    app = create_app(Settings(api_key=KEY, data_dir=tmp_path / "data"))
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.store.queue_length = lambda: 1 / 0
        r = client.get("/healthz")
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "detail": "Internal server error."}


def test_non_finite_confidence_is_dropped(make_client, videos):
    client = make_client(detector="tests.nan_detector")
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    raw = client.get(f"/api/v1/jobs/{body['job_id']}", headers={"X-API-Key": KEY}).text
    assert len(body["violations"]) == 1
    assert body["violations"][0]["confidence"] == 0.9
    assert "NaN" not in raw and "Infinity" not in raw and "null" not in raw.replace('"closest_type_id":null', "")


def test_app_uploads_disabled(make_client, videos):
    client = make_client(app_uploads_enabled=False)
    r = upload(client, videos["vfr"], key=None, url="/app/analyze")
    assert r.status_code == 503
    assert r.json() == {"error": "uploads_disabled", "detail": "Uploads from the web UI are turned off. Existing results are still available."}
    job_id = upload(client, videos["vfr"]).json()["job_id"]
    assert wait_done(client, job_id, url="/app/jobs/{}")["status"] == "done"
    assert client.get(f"/app/jobs/{job_id}/video").status_code == 200
    assert client.get("/healthz").json()["app_uploads_enabled"] is False


@pytest.mark.parametrize("value,expected", [("false", False), ("0", False), ("no", False), ("true", True), ("1", True), ("", True)])
def test_app_uploads_env_flag(monkeypatch, value, expected):
    monkeypatch.setenv("APP_UPLOADS_ENABLED", value)
    assert Settings().app_uploads_enabled is expected


def test_dockerfile_uses_uv_image_without_cache():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/" in dockerfile
    assert "UV_NO_CACHE=1" in dockerfile
    assert "pip install" not in dockerfile
