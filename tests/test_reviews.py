from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

KEY = "test-key"
RESULT = {
    "verdict": "violation",
    "violations": [
        {"type": "Наличные оставлены без присмотра", "type_id": "cash_unattended", "timecode_sec": 11.0, "confidence": 0.97,
         "description": "x", "evidence_frames": [], "rule": {}, "boxes": []},
    ],
}


def done_job(store, job_id: str, source: str = "app") -> None:
    store.create(job_id, f"{job_id}.mov", 100, 20.0, source)
    store._write("update jobs set status = 'running' where id = ?", (job_id,))
    assert store.finish(job_id, RESULT, {"events": []})


def make(tmp_path):
    app = create_app(Settings(api_key=KEY, detector="stub", data_dir=tmp_path))
    return TestClient(app), app.state.store


ADDED = {"id": "r1", "type_id": "drink_in_cash_zone", "t_start_sec": 5.0, "t_end_sec": 8.0, "comment": "на столе кружка рядом с деньгами"}


def test_review_round_trip_with_model_snapshot(tmp_path):
    client, store = make(tmp_path)
    done_job(store, "job1")
    assert client.get("/app/jobs/job1/review").json() == {"job_id": "job1", "version": 0, "decisions": {}, "added": [], "updated_at": None}
    body = {"version": 0, "decisions": {"event-0": {"status": "accepted", "comment": "верно"}}, "added": [ADDED]}
    saved = client.put("/app/jobs/job1/review", json=body).json()
    assert saved["version"] == 1 and saved["added"] == [ADDED]
    assert saved["decisions"]["event-0"]["model_event"] == {
        "type_id": "cash_unattended", "type": "Наличные оставлены без присмотра", "timecode_sec": 11.0, "confidence": 0.97,
    }
    assert client.get("/app/jobs/job1/review").json()["version"] == 1


def test_stale_version_is_a_conflict_not_an_overwrite(tmp_path):
    client, store = make(tmp_path)
    done_job(store, "job1")
    assert client.put("/app/jobs/job1/review", json={"version": 0, "added": [ADDED]}).status_code == 200
    r = client.put("/app/jobs/job1/review", json={"version": 0, "added": []})
    assert r.status_code == 409 and r.json()["error"] == "review_conflict"
    assert client.get("/app/jobs/job1/review").json()["added"] == [ADDED]


def test_invalid_reviews_are_rejected(tmp_path):
    client, store = make(tmp_path)
    done_job(store, "job1")
    for added in (
        dict(ADDED, type_id="made_up"),  # not in the catalog
        dict(ADDED, t_start_sec=50.0, t_end_sec=None),  # after the 20 s video
        dict(ADDED, t_end_sec=2.0),  # ends before it starts
        dict(ADDED, comment=""),  # the reviewer must say what is happening
    ):
        r = client.put("/app/jobs/job1/review", json={"version": 0, "added": [added]})
        assert r.status_code == 422 and r.json()["error"] == "invalid_review", added
    assert client.put("/app/jobs/job1/review", json={"version": 0, "decisions": {"event-0": {"status": "maybe"}}}).status_code == 422
    assert client.get("/app/jobs/nope/review").status_code == 404


def test_export_is_team_only_excludes_jury_jobs_and_outlives_the_video(tmp_path):
    client, store = make(tmp_path)
    done_job(store, "ui")
    done_job(store, "jury", source="api")
    for job in ("ui", "jury"):
        assert client.put(f"/app/jobs/{job}/review", json={"version": 0, "added": [ADDED]}).status_code == 200
    assert client.get("/api/v1/reviews/export").status_code == 401
    store.purge_older_than(-1)  # job rows and videos are gone after retention
    videos = client.get("/api/v1/reviews/export", headers={"X-API-Key": KEY}).json()["videos"]
    assert [v["job_id"] for v in videos] == ["ui"]  # test videos must never become training data
    (event,) = videos[0]["events"]
    assert event == {"type_id": "drink_in_cash_zone", "t_start_sec": 5.0, "t_end_sec": 8.0, "key_t_sec": [5.0],
                     "labeled": "reviewer", "comment": ADDED["comment"]}


def test_catalog_lists_every_type_for_the_form(tmp_path):
    client, _ = make(tmp_path)
    types = client.get("/app/catalog").json()["types"]
    ids = [t["type_id"] for t in types]
    assert "cash_unattended" in ids and "other" in ids and len(ids) == len(set(ids))
    assert all(t["name"] for t in types)
