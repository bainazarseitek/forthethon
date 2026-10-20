import base64
import io
import json
import os
import re

from PIL import Image

from app.frames import probe

from tests.conftest import KEY, assert_manual_review, upload, wait_done


def assert_section10(body: dict) -> None:
    assert isinstance(body["job_id"], str)
    assert body["status"] == "done"
    assert body["verdict"] in ("violation", "no_violation")
    assert isinstance(body["violations"], list)
    assert (body["verdict"] == "violation") == bool(body["violations"])
    for v in body["violations"]:
        assert isinstance(v["type"], str) and v["type"]
        assert isinstance(v["timecode_sec"], float)
        assert isinstance(v["description"], str) and v["description"]
        assert 1 <= len(v["evidence_frames"]) <= 3
        assert v["timecode_sec"] == v["evidence_frames"][0]["timestamp_sec"]
        for frame in v["evidence_frames"]:
            assert set(frame) == {"timestamp_sec", "image_base64"}
            assert isinstance(frame["timestamp_sec"], float)
            raw = base64.b64decode(frame["image_base64"], validate=True)
            assert raw[:3] == b"\xff\xd8\xff"
            assert Image.open(io.BytesIO(raw)).format == "JPEG"


def test_missing_key(client, videos):
    assert upload(client, videos["vfr"], key=None).status_code == 401
    assert client.get("/api/v1/jobs/whatever").status_code == 401


def test_wrong_key(client, videos):
    assert upload(client, videos["vfr"], key="nope").status_code == 401
    assert client.get("/api/v1/jobs/whatever", headers={"X-API-Key": "nope"}).status_code == 401


def test_right_key(client, videos):
    r = upload(client, videos["vfr"])
    assert r.status_code == 200
    assert r.json()["status"] == "processing"
    assert client.get(f"/api/v1/jobs/{r.json()['job_id']}", headers={"X-API-Key": KEY}).status_code == 200


def test_unset_server_key_rejects_everything(make_client, videos):
    client = make_client(api_key="")
    assert upload(client, videos["vfr"], key="").status_code == 401


def test_wrong_extension(client, videos):
    r = upload(client, videos["vfr"], name="video.avi")
    assert r.status_code == 400


def test_empty_file(client, tmp_path):
    empty = tmp_path / "empty.mov"
    empty.write_bytes(b"")
    assert upload(client, empty).status_code == 400


def test_garbage_bytes(client, tmp_path):
    garbage = tmp_path / "garbage.mov"
    garbage.write_bytes(os.urandom(64 * 1024))
    r = upload(client, garbage)
    assert r.status_code == 400
    assert not any((client.app.state.store.data_dir / "jobs").iterdir())


def test_too_large(make_client, videos):
    client = make_client(max_upload_bytes=1024)
    assert upload(client, videos["vfr"]).status_code == 413


def test_unknown_job(client):
    assert client.get("/api/v1/jobs/deadbeef", headers={"X-API-Key": KEY}).status_code == 404
    assert client.get("/app/jobs/deadbeef").status_code == 404


def test_lifecycle_violation(make_client, videos):
    client = make_client(threshold=0.0, threshold_other=0.0)
    r = upload(client, videos["vfr"])
    assert r.json() == {"job_id": r.json()["job_id"], "status": "processing"}
    body = wait_done(client, r.json()["job_id"])
    assert_section10(body)
    assert body["verdict"] == "violation"
    json.dumps(body)
    for v in body["violations"]:
        assert v["rule"]["document"]
        assert len(v["boxes"]) == len(v["evidence_frames"])


def test_lifecycle_clean(make_client, videos):
    client = make_client(threshold=1.0, threshold_other=1.0)
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert_section10(body)
    assert body["verdict"] == "no_violation"
    assert body["violations"] == []


def test_detector_exception_fails_job(make_client, videos):
    client = make_client(detector="tests.failing_detector")
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert_manual_review(body, "detector_failed")
    assert len(body["violations"][0]["evidence_frames"]) == 1
    assert body["message"] == "The detector failed on this video."
    assert "boom" not in json.dumps(body)


def test_watchdog_timeout(make_client, videos, monkeypatch):
    monkeypatch.setenv("STUB_DELAY_SEC", "3")
    client = make_client(job_timeout_sec=0.3)
    body = wait_done(client, upload(client, videos["vfr"]).json()["job_id"])
    assert_manual_review(body, "timeout")
    assert body["message"] == "The detector did not finish in time and was stopped."


def test_app_endpoints(make_client, videos):
    client = make_client(threshold=0.0, threshold_other=0.0)
    r = upload(client, videos["vfr"], key=None, url="/app/analyze")
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    body = wait_done(client, job_id, url="/app/jobs/{}")
    assert_section10({k: v for k, v in body.items() if k not in ("queue_position", "progress", "filename", "duration_sec", "video_url")})
    assert body["queue_position"] == 0
    assert body["progress"] == 1
    assert body["video_url"] == f"/app/jobs/{job_id}/video"
    assert all("rule" in v and "boxes" in v for v in body["violations"])


def test_queue_position(make_client, videos, monkeypatch):
    monkeypatch.setenv("STUB_DELAY_SEC", "0.5")
    client = make_client()
    ids = [upload(client, videos["vfr"], key=None, url="/app/analyze").json()["job_id"] for _ in range(3)]
    positions = [client.get(f"/app/jobs/{i}").json()["queue_position"] for i in ids]
    assert positions[2] > positions[1] >= 1
    for i in ids:
        wait_done(client, i, url="/app/jobs/{}")


def test_video_remux_and_range(client, videos):
    job_id = upload(client, videos["vfr"], key=None, url="/app/analyze").json()["job_id"]
    full = client.get(f"/app/jobs/{job_id}/video")
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    assert full.content[4:8] == b"ftyp"
    part = client.get(f"/app/jobs/{job_id}/video", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert len(part.content) == 100
    assert part.headers["content-range"] == f"bytes 0-99/{len(full.content)}"
    tail = client.get(f"/app/jobs/{job_id}/video", headers={"Range": "bytes=100-"})
    assert tail.status_code == 206
    assert full.content[100:] == tail.content


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["model_loaded"] is True
    assert body["queue_length"] == 0


def test_remux_keeps_rotation(client, videos, tmp_path):
    job_id = upload(client, videos["rotated"], key=None, url="/app/analyze").json()["job_id"]
    out = tmp_path / "out.mp4"
    out.write_bytes(client.get(f"/app/jobs/{job_id}/video").content)
    assert probe(out).rotation == 90


def test_static_compression_and_cache(client):
    index = client.get("/", headers={"Accept-Encoding": "gzip"})
    assert index.status_code == 200
    assert index.headers["cache-control"] == "no-cache"
    script = re.search(r'src="(/assets/[^"]+\.js)"', index.text).group(1)
    js = client.get(script, headers={"Accept-Encoding": "gzip"})
    assert js.status_code == 200
    assert js.headers["content-encoding"] == "gzip"
    assert js.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert "content-encoding" not in client.get(script, headers={"Accept-Encoding": "identity"}).headers


def test_video_is_never_gzipped(client, videos):
    job_id = upload(client, videos["vfr"], key=None, url="/app/analyze").json()["job_id"]
    body = wait_done(client, job_id, url="/app/jobs/{}")
    assert body["status"] == "done"
    full = client.get(f"/app/jobs/{job_id}/video", headers={"Accept-Encoding": "gzip"})
    assert full.status_code == 200 and "content-encoding" not in full.headers
    part = client.get(f"/app/jobs/{job_id}/video", headers={"Accept-Encoding": "gzip", "Range": "bytes=0-99"})
    assert part.status_code == 206 and "content-encoding" not in part.headers
    assert part.content == full.content[:100]
