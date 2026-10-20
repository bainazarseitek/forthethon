import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.conftest import upload, wait_done

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1, takes about 5 minutes")


def test_concurrent_uploads_finish_within_300s(make_client, videos, monkeypatch):
    jobs = int(os.environ.get("LOAD_JOBS", 5))
    monkeypatch.setenv("STUB_DELAY_SEC", os.environ.get("LOAD_DELAY_SEC", "50"))
    client = make_client()
    with ThreadPoolExecutor(jobs) as pool:
        ids = list(pool.map(lambda _: upload(client, videos["vfr"]).json()["job_id"], range(jobs)))
    with ThreadPoolExecutor(jobs) as pool:
        bodies = list(pool.map(lambda i: wait_done(client, i, timeout=330), ids))
    store = client.app.state.store
    rows = []
    for job_id, body in zip(ids, bodies):
        job = store.get(job_id)
        rows.append((job_id, job["status"], job["error"] or "", job["finished_at"] - job["created_at"],
                     (job["started_at"] or job["finished_at"]) - job["created_at"]))
    print(f"\n{'job':<18}{'status':<8}{'error':<10}{'queue_wait_s':>13}{'upload_to_final_s':>19}")
    for job_id, state, error, total, wait in sorted(rows, key=lambda r: r[3]):
        print(f"{job_id:<18}{state:<8}{error:<10}{wait:>13.1f}{total:>19.1f}")
    assert all(total < 300 for *_, total, _ in rows)
