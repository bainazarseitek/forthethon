import json
import logging
import os
import secrets
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from app.detector_proc import DetectorError, DetectorNotReady, DetectorTimeout

log = logging.getLogger("jobs")

SCHEMA = """
create table if not exists jobs (
    id text primary key,
    status text not null,
    source text not null default 'api',
    attempts integer not null default 0,
    filename text,
    size integer,
    duration_sec real,
    created_at real not null,
    started_at real,
    finished_at real,
    progress real not null default 0,
    result text,
    raw text,
    error text
)
"""

# Reviewer feedback: decisions on model events and violations the reviewer added. Kept
# when the job and its video are purged (labels only), and never used to change the
# model automatically.
REVIEWS = """
create table if not exists reviews (
    job_id text primary key,
    source text not null,
    filename text,
    duration_sec real,
    video_sha256 text,
    version integer not null,
    payload text not null,
    updated_at real not null
)
"""

MIGRATIONS = {
    "source": "alter table jobs add column source text not null default 'api'",
    "attempts": "alter table jobs add column attempts integer not null default 0",
}

# Jury API first, then the web UI, then camera segments, which only use idle time.
RANKS = {"api": 0, "app": 1}
PRIORITY = "(case source when 'api' then 0 when 'app' then 1 else 2 end)"
MAX_ATTEMPTS = 2
STREAM_BUDGET_SEC = 90


def rank(source: str) -> int:
    return RANKS.get(source, 2)


class JobStore:
    def __init__(self, data_dir: Path, deadline_sec: float = 285, post_budget_sec: float = 15, min_budget_sec: float = 20):
        self.data_dir = Path(data_dir)
        self.deadline_sec = deadline_sec
        self.post_budget_sec = post_budget_sec
        self.min_budget_sec = min_budget_sec
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.data_dir / "jobs.db", check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("pragma journal_mode=wal")
        self._db.execute(SCHEMA)
        self._db.execute(REVIEWS)
        columns = {row["name"] for row in self._db.execute("pragma table_info(jobs)")}
        for column, sql in MIGRATIONS.items():
            if column not in columns:
                self._db.execute(sql)
        self.wakeup = threading.Event()

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def _write(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            return self._db.execute(sql, params).rowcount

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / "jobs" / job_id

    def video_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "input.mov"

    def new_id(self) -> str:
        return secrets.token_hex(8)

    def deadline(self, job: dict) -> float:
        return job["created_at"] + self.deadline_sec

    def create(self, job_id: str, filename: str, size: int, duration_sec: float, source: str = "api") -> None:
        self._write(
            "insert into jobs (id, status, source, filename, size, duration_sec, created_at) values (?, 'queued', ?, ?, ?, ?, ?)",
            (job_id, source, filename, size, duration_sec, time.time()),
        )
        self.wakeup.set()

    def reserve(self, source: str, cap: int | None = None) -> str | None:
        job_id = self.new_id()
        with self._lock:
            if cap is not None:
                taken = self._db.execute(
                    "select count(*) from jobs where source = ? and status in ('uploading', 'queued')", (source,)
                ).fetchone()[0]
                if taken >= cap:
                    return None
            self._db.execute(
                "insert into jobs (id, status, source, created_at) values (?, 'uploading', ?, ?)", (job_id, source, time.time())
            )
        return job_id

    def activate(self, job_id: str, filename: str, size: int, duration_sec: float) -> None:
        self._write(
            "update jobs set status = 'queued', filename = ?, size = ?, duration_sec = ?, created_at = ? where id = ? and status = 'uploading'",
            (filename, size, duration_sec, time.time(), job_id),
        )
        self.wakeup.set()

    def discard(self, job_id: str) -> None:
        self._write("delete from jobs where id = ? and status = 'uploading'", (job_id,))
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def budget(self, job: dict, timeout_sec: float, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return min(timeout_sec, self.deadline(job) - now - self.post_budget_sec)

    def get(self, job_id: str) -> dict | None:
        rows = self._query("select * from jobs where id = ? and status != 'uploading'", (job_id,))
        if not rows:
            return None
        job = dict(rows[0])
        job["result"] = json.loads(job["result"]) if job["result"] else None
        job["raw"] = json.loads(job["raw"]) if job["raw"] else None
        return job

    def queue_position(self, job: dict) -> int:
        if job["status"] != "queued":
            return 0
        rows = self._query(
            f"select count(*) from jobs where status = 'running'"
            f" or (status = 'queued' and ({PRIORITY}, created_at, id) < (?, ?, ?))",
            (rank(job["source"]), job["created_at"], job["id"]),
        )
        return int(rows[0][0])

    def queue_length(self) -> int:
        return int(self._query("select count(*) from jobs where status in ('queued', 'running')")[0][0])

    def queued_count(self, source: str) -> int:
        return int(self._query("select count(*) from jobs where status = 'queued' and source = ?", (source,))[0][0])

    def active_count(self, source: str) -> int:
        return int(self._query("select count(*) from jobs where status in ('uploading', 'queued', 'running') and source = ?", (source,))[0][0])

    def busy_or_recent(self, source: str, within_sec: float) -> bool:
        rows = self._query(
            "select count(*) from jobs where source = ? and (status in ('uploading', 'queued', 'running') or created_at > ?)",
            (source, time.time() - within_sec),
        )
        return int(rows[0][0]) > 0

    def status_of(self, job_id: str) -> str | None:
        rows = self._query("select status from jobs where id = ?", (job_id,))
        return rows[0]["status"] if rows else None

    def cancel_queued(self, job_id: str) -> bool:
        if not self._write("delete from jobs where id = ? and status = 'queued'", (job_id,)):
            return False
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return True

    def remove(self, job_id: str) -> None:
        if self._write("delete from jobs where id = ? and status in ('queued', 'done', 'failed')", (job_id,)):
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def drop_queued(self, source: str) -> int:
        rows = self._query("select id from jobs where source = ? and status = 'queued'", (source,))
        for row in rows:
            self.cancel_queued(row["id"])
        return len(rows)

    def claim_next(self) -> dict | None:
        with self._lock:
            while True:
                row = self._db.execute(
                    f"select id, created_at from jobs where status = 'queued' order by {PRIORITY}, created_at, id limit 1"
                ).fetchone()
                if row is None:
                    return None
                now = time.time()
                if row["created_at"] + self.deadline_sec - now - self.post_budget_sec < self.min_budget_sec:
                    self._db.execute(
                        "update jobs set status = 'failed', finished_at = ?, error = 'deadline' where id = ?", (now, row["id"])
                    )
                    log.info(json.dumps({"event": "job", "id": row["id"], "status": "failed", "error": "deadline"}))
                    continue
                self._db.execute(
                    "update jobs set status = 'running', started_at = ?, progress = 0.05, attempts = attempts + 1 where id = ?",
                    (now, row["id"]),
                )
                break
        return self.get(row["id"])

    def set_progress(self, job_id: str, progress: float) -> None:
        self._write("update jobs set progress = ? where id = ? and status = 'running'", (progress, job_id))

    def finish(self, job_id: str, result: dict, raw: dict) -> bool:
        return 1 == self._write(
            "update jobs set status = 'done', progress = 1, finished_at = ?, result = ?, raw = ? where id = ? and status = 'running'",
            (time.time(), json.dumps(result, ensure_ascii=False), json.dumps(raw, ensure_ascii=False), job_id),
        )

    def release(self, job_id: str) -> None:
        self._write("update jobs set attempts = max(attempts - 1, 0) where id = ? and status = 'running'", (job_id,))

    def fail(self, job_id: str, error: str) -> bool:
        return 1 == self._write(
            "update jobs set status = 'failed', finished_at = ?, error = ? where id = ? and status = 'running'",
            (time.time(), error, job_id),
        )

    def requeue_running(self) -> tuple[int, int]:
        for row in self._query("select id from jobs where status = 'uploading'"):
            self.discard(row["id"])
        with self._lock:
            crashed = self._db.execute(
                "update jobs set status = 'failed', finished_at = ?, error = 'crashed' where status = 'running' and attempts >= ?",
                (time.time(), MAX_ATTEMPTS),
            ).rowcount
            requeued = self._db.execute(
                "update jobs set status = 'queued', started_at = null, progress = 0 where status = 'running'"
            ).rowcount
        if requeued:
            self.wakeup.set()
        return requeued, crashed

    def purge_older_than(self, max_age_sec: float) -> int:
        cutoff = time.time() - max_age_sec
        rows = self._query("select id from jobs where created_at < ? and status in ('done', 'failed')", (cutoff,))
        for row in rows:
            shutil.rmtree(self.job_dir(row["id"]), ignore_errors=True)
            self._write("delete from jobs where id = ?", (row["id"],))
        known = {row["id"] for row in self._query("select id from jobs")}
        orphans = 0
        for path in (self.data_dir / "jobs").iterdir():
            if path.name not in known and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                orphans += 1
        return len(rows) + orphans

    def get_review(self, job_id: str) -> dict | None:
        rows = self._query("select * from reviews where job_id = ?", (job_id,))
        return _review(rows[0]) if rows else None

    def save_review(self, job: dict, payload: dict, base_version: int, video_sha256: str | None) -> dict:
        """Store a review if the reviewer started from the current version, else raise ReviewConflict."""
        with self._lock:
            row = self._db.execute("select version, video_sha256 from reviews where job_id = ?", (job["id"],)).fetchone()
            current = row["version"] if row else 0
            if base_version != current:
                raise ReviewConflict(current)
            self._db.execute(
                "insert or replace into reviews (job_id, source, filename, duration_sec, video_sha256, version, payload, updated_at) "
                "values (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job["id"],
                    job["source"],
                    job["filename"],
                    job["duration_sec"],
                    (row["video_sha256"] if row else None) or video_sha256,
                    current + 1,
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
        return self.get_review(job["id"])

    def reviews(self, source: str) -> list[dict]:
        return [_review(row) for row in self._query("select * from reviews where source = ? order by updated_at", (source,))]

    def close(self) -> None:
        self._db.close()


class ReviewConflict(Exception):
    def __init__(self, current: int):
        super().__init__(f"review changed, current version {current}")
        self.current = current


def _review(row: sqlite3.Row) -> dict:
    review = dict(row)
    review.update(json.loads(review.pop("payload")))
    return review


class Worker:
    def __init__(
        self,
        store: JobStore,
        process: Callable[[dict, Callable[[float], None], dict, float], tuple[dict, dict]],
        timeout_sec: float,
        retention_sec: float,
    ):
        self.store = store
        self.process = process
        self.timeout_sec = timeout_sec
        self.retention_sec = retention_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="worker", daemon=True)
        self._last_purge = 0.0
        self.exit = os._exit

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()
        self.store.wakeup.set()

    def join(self, timeout: float = 5) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def stop(self) -> None:
        self.request_stop()
        self.join()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._maybe_purge()
                    job = self.store.claim_next()
                    if job is None:
                        self.store.wakeup.wait(timeout=5)
                        self.store.wakeup.clear()
                        continue
                    self.run_one(job)
                except Exception:
                    log.exception("worker iteration failed")
                    self._stop.wait(1)
        finally:
            if not self._stop.is_set():
                log.critical("worker loop died, exiting so the restart policy brings the service back")
                self.exit(1)

    def _maybe_purge(self) -> None:
        if time.time() - self._last_purge > 600:
            self._last_purge = time.time()
            removed = self.store.purge_older_than(self.retention_sec)
            if removed:
                log.info(json.dumps({"event": "purge", "removed": removed}))

    def run_one(self, job: dict) -> None:
        job_id = job["id"]
        timings: dict = {}
        budget = self.store.budget(job, self.timeout_sec)
        if job["source"] == "stream":
            budget = min(budget, STREAM_BUDGET_SEC)
        try:
            result, raw = self.process(job, lambda p: self.store.set_progress(job_id, p), timings, budget)
        except Exception as exc:
            if self.stopping:
                log.info("job %s interrupted by shutdown, left running for requeue: %r", job_id, exc)
                self.store.release(job_id)
                return
            if isinstance(exc, DetectorNotReady):
                code = "detector_not_ready"
            elif isinstance(exc, DetectorTimeout):
                code = "timeout"
            elif isinstance(exc, DetectorError):
                code = "detector_failed"
            else:
                code = "processing_failed"
            log.error("job %s failed with %s: %s: %s", job_id, code, type(exc).__name__, exc, exc_info=code == "processing_failed")
            self.store.fail(job_id, code)
        else:
            if time.time() > self.store.deadline(job):
                log.error("job %s finished after its deadline", job_id)
                self.store.fail(job_id, "deadline")
            else:
                try:
                    self.store.finish(job_id, result, raw)
                except Exception:
                    log.exception("job %s could not be saved", job_id)
                    self.store.fail(job_id, "result_invalid")
        final = self.store.get(job_id) or {}
        log.info(
            json.dumps(
                {
                    "event": "job",
                    "id": job_id,
                    "source": job["source"],
                    "attempt": job["attempts"],
                    "size": job["size"],
                    "duration_sec": job["duration_sec"],
                    "queue_wait_sec": round(job["started_at"] - job["created_at"], 3),
                    "budget_sec": round(budget, 3),
                    "detector_sec": timings.get("detector_sec"),
                    "total_sec": round((final.get("finished_at") or time.time()) - job["created_at"], 3),
                    "status": final.get("status"),
                    "verdict": (final.get("result") or {}).get("verdict"),
                    "error": final.get("error"),
                }
            )
        )
