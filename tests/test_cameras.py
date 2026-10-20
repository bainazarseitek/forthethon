import json
import time
from pathlib import Path

import av
import pytest
from PIL import Image

from app import cameras as cameras_module
from app.cameras import CameraHub
from app.jobs import STREAM_BUDGET_SEC, JobStore, Worker


def make_clip(path: Path, seconds: float = 1.5, fps: int = 10, shade: int = 0) -> Path:
    with av.open(str(path), "w", format="mov") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        for i in range(int(seconds * fps)):
            image = Image.new("RGB", (160, 120), (shade, (i * 20) % 255, 80))
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def config(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(cameras_module, "MIN_SEGMENT_SEC", 0.5)
    clips = [str(make_clip(tmp_path / f"clip{i}.mov", shade=i * 100)) for i in range(2)]
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"cameras": [{"id": "c1", "name": "Касса 1", "playlist": clips}]}), encoding="utf-8")
    return path


def wait_for(predicate, timeout: float = 10):
    started = time.time()
    while time.time() - started < timeout:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met")


def segments(hub: CameraHub, state: str | None = None) -> list[dict]:
    return [s for s in hub.status()["segments"] if state is None or s["state"] == state]


def test_stream_jobs_wait_behind_app_and_api(tmp_path):
    store = JobStore(tmp_path)
    for job_id, source in [("s", "stream"), ("a", "app"), ("j", "api")]:
        store.create(job_id, f"{job_id}.mov", 1, 1.0, source=source)
        time.sleep(0.01)
    assert store.queue_position(store.get("s")) == 2
    assert [store.claim_next()["id"] for _ in range(3)] == ["j", "a", "s"]


def test_each_clip_becomes_a_queued_segment_that_stays_live(tmp_path, config):
    store = JobStore(tmp_path / "data")
    hub = CameraHub(store, config, idle_sec=60)
    hub.start()
    try:
        assert wait_for(lambda: hub.previews()[0][1])
        wait_for(lambda: len(segments(hub, "skipped")) >= 1 and segments(hub, "queued"))
    finally:
        hub.stop()
    # No worker runs here, so the first clip is still waiting when the second ends and
    # gets replaced: the queue never holds more than one segment per camera.
    skipped, queued = segments(hub, "skipped")[-1], segments(hub, "queued")[0]
    assert skipped["reason"] == "busy" and store.status_of(skipped["job_id"]) is None
    assert queued["clip"] == "clip1" and 1.3 < queued["duration_sec"] < 1.7
    job = store.get(queued["job_id"])
    assert job["source"] == "stream" and job["filename"].startswith("Касса 1 · ")
    assert store.queued_count("stream") == 1


def test_segments_are_skipped_while_a_jury_video_waits(tmp_path, config):
    store = JobStore(tmp_path / "data")
    store.create("jury", "jury.mov", 1, 10.0, source="api")
    hub = CameraHub(store, config, idle_sec=60)
    hub.start()
    try:
        wait_for(lambda: segments(hub, "skipped"))
    finally:
        hub.stop()
    assert {s["reason"] for s in segments(hub, "skipped")} == {"jury"}
    assert store.queued_count("stream") == 0


def test_segments_are_skipped_shortly_after_a_jury_video(tmp_path, config):
    store = JobStore(tmp_path / "data")
    store.create("jury", "jury.mov", 1, 10.0, source="api")
    store.claim_next()
    store.finish("jury", {"verdict": "clean", "violations": []}, {})
    hub = CameraHub(store, config, idle_sec=60)
    hub.start()
    try:
        wait_for(lambda: segments(hub, "skipped"))
    finally:
        hub.stop()
    assert {s["reason"] for s in segments(hub, "skipped")} == {"jury"}
    assert not store.busy_or_recent("api", 0)


def test_stream_jobs_get_a_short_detector_budget(tmp_path):
    store = JobStore(tmp_path)
    budgets = {}
    worker = Worker(store, lambda job, progress, timings, budget: budgets.setdefault(job["source"], budget) and ({}, {}), 270, 3600)
    for job_id, source in [("s", "stream"), ("a", "app")]:
        store.create(job_id, f"{job_id}.mov", 1, 1.0, source=source)
        worker.run_one(store.claim_next())
    assert budgets["stream"] == STREAM_BUDGET_SEC and budgets["app"] > 250


def test_cameras_stop_when_nobody_watches(tmp_path, config):
    hub = CameraHub(JobStore(tmp_path / "data"), config, idle_sec=0.2)
    hub.start()
    wait_for(lambda: not hub.running, timeout=6)
    assert hub.stopped_reason == "idle"
    assert all(not c.thread.is_alive() for c in hub.cameras)


def test_camera_endpoints_analyse_segments_and_send_previews(make_client, config):
    client = make_client(cameras_config=config)
    assert client.get("/app/cameras").json()["cameras"][0]["state"] == "off"
    assert client.post("/app/cameras/start").json()["running"] is True
    with client.websocket_connect("/app/cameras/preview") as ws:
        camera_id, _, jpeg = ws.receive_bytes().partition(b"\n")
    assert camera_id == b"c1" and jpeg[:2] == b"\xff\xd8"
    done = wait_for(lambda: [s for s in client.get("/app/cameras").json()["segments"] if s["state"] == "done"])
    job = client.get(f"/app/jobs/{done[0]['job_id']}").json()
    assert job["status"] == "done" and job["filename"].startswith("Касса 1")
    for v in done[0]["violations"]:
        assert v["at"] == pytest.approx(done[0]["started_at"] + v["timecode_sec"])
        frame = client.get(f"/app/cameras/evidence/{done[0]['job_id']}/{v['index']}")
        assert frame.status_code == 200 and frame.content[:2] == b"\xff\xd8"
    assert client.post("/app/cameras/stop").json()["running"] is False
    assert client.get("/healthz").json()["cameras_running"] is False


def test_camera_start_errors(make_client, tmp_path):
    assert client_error(make_client(cameras_enabled=False)) == (503, "cameras_disabled")
    assert client_error(make_client(cameras_config=tmp_path / "missing.json")) == (503, "cameras_unavailable")


def client_error(client) -> tuple[int, str]:
    response = client.post("/app/cameras/start")
    return response.status_code, response.json()["error"]
