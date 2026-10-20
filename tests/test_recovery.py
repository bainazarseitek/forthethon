import shutil

from app.jobs import JobStore
from tests.conftest import wait_done


def test_processing_job_finishes_after_restart(tmp_path, make_client, videos):
    store = JobStore(tmp_path / "data")
    job_id = store.new_id()
    store.job_dir(job_id).mkdir(parents=True)
    shutil.copy(videos["vfr"], store.video_path(job_id))
    store.create(job_id, "vfr.mov", videos["vfr"].stat().st_size, 4.0)
    assert store.claim_next()["status"] == "running"
    store.close()

    client = make_client()
    body = wait_done(client, job_id)
    assert body["status"] == "done"
    assert body["verdict"] in ("violation", "no_violation")


def test_purge_removes_old_jobs(tmp_path, videos):
    store = JobStore(tmp_path / "data")
    job_id = store.new_id()
    store.job_dir(job_id).mkdir(parents=True)
    store.create(job_id, "vfr.mov", 1, 4.0)
    store.claim_next()
    store.finish(job_id, {"verdict": "no_violation", "violations": []}, {"events": []})
    assert store.purge_older_than(3600) == 0
    assert store.purge_older_than(-1) == 1
    assert store.get(job_id) is None
    assert not store.job_dir(job_id).exists()
