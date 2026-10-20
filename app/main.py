import asyncio
import hashlib
import hmac
import logging
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import av
from fastapi import Depends, FastAPI, Header, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware

from app import pipeline
from app.cameras import PREVIEW_FPS, CameraHub, CamerasUnavailable
from app.config import Settings
from app.detector_proc import DetectorProcess
from app.frames import VideoError, probe
from app.jobs import JobStore, ReviewConflict, Worker

ALLOWED_EXTENSIONS = {".mov", ".mp4"}
STATIC_DIR = Path(__file__).resolve().parent / "static"
MULTIPART_SLACK = 64 * 1024
UPLOAD_PATHS = {"/api/v1/analyze", "/app/analyze"}

JOB_ERRORS = {
    "timeout": "The detector did not finish in time and was stopped.",
    "deadline": "The job could not finish within the time limit after upload.",
    "crashed": "The server restarted twice while processing this video.",
    "detector_not_ready": "The detection model was still loading.",
    "detector_failed": "The detector failed on this video.",
    "result_invalid": "The detector returned a result that could not be saved.",
    "processing_failed": "Processing failed.",
}

class Decision(BaseModel):
    status: Literal["accepted", "rejected", "uncertain", "unreviewed"]
    reason: str | None = Field(default=None, max_length=300)
    comment: str | None = Field(default=None, max_length=2000)


class AddedViolation(BaseModel):
    """A violation the model missed, added by the reviewer at a frame of the video."""

    id: str = Field(min_length=1, max_length=64)
    type_id: str = Field(min_length=1, max_length=64)
    t_start_sec: float = Field(ge=0)
    t_end_sec: float | None = Field(default=None, ge=0)
    comment: str = Field(min_length=1, max_length=2000)


class ReviewIn(BaseModel):
    version: int = Field(ge=0, description="version the reviewer started from; 0 for a new review")
    decisions: dict[str, Decision] = Field(default_factory=dict, max_length=200)
    added: list[AddedViolation] = Field(default_factory=list, max_length=50)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def response(self) -> JSONResponse:
        return JSONResponse({"error": self.code, "detail": self.message}, status_code=self.status)


def too_large(limit: int) -> ApiError:
    return ApiError(413, "file_too_large", f"The file is larger than {limit // (1024 * 1024)} MB.")


class UploadLimit:
    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if not (scope["type"] == "http" and scope["method"] == "POST" and scope["path"] in UPLOAD_PATHS):
            await self.app(scope, receive, send)
            return
        allowed = self.limit + MULTIPART_SLACK
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > allowed:
            await too_large(self.limit).response()(scope, receive, send)
            return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > allowed:
                    raise too_large(self.limit)
            return message

        await self.app(scope, limited_receive, send)


class CachedStatic(StaticFiles):
    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        immutable = self.get_path(scope).startswith("assets/")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if immutable else "no-cache"
        return response


class TooLarge(Exception):
    pass


def save_upload(source, target: Path, limit: int) -> int:
    size = 0
    with target.open("wb") as out:
        while chunk := source.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise TooLarge()
            out.write(chunk)
    return size


def job_error(code: str | None) -> tuple[str, str]:
    code = code if code in JOB_ERRORS else "processing_failed"
    return code, JOB_ERRORS[code]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    rules = pipeline.Rules.load(settings.rules_path)
    detector = DetectorProcess(settings.detector)
    remux_lock = threading.Lock()

    def process(job: dict, progress, timings: dict, budget: float) -> tuple[dict, dict]:
        def timed(path: str) -> dict:
            started = time.monotonic()
            try:
                return detector.analyze(path, budget)
            finally:
                timings["detector_sec"] = round(time.monotonic() - started, 3)

        return pipeline.run(str(store.video_path(job["id"])), timed, rules, settings, progress)

    store = JobStore(settings.data_dir, settings.job_deadline_sec, settings.post_budget_sec, settings.min_budget_sec)
    worker = Worker(store, process, settings.job_timeout_sec, settings.retention_sec)
    cameras = CameraHub(store, settings.cameras_config, settings.cameras_idle_sec, settings.timezone)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not settings.api_key:
            log.warning("API_KEY is not set, every /api/v1 request will be rejected")
        requeued, crashed = store.requeue_running()
        if requeued or crashed:
            log.info("startup: requeued %d jobs, failed %d crashed jobs", requeued, crashed)
        if dropped := store.drop_queued("stream"):
            log.info("startup: dropped %d camera segments left from the last run", dropped)
        detector.start()
        if not await run_in_threadpool(detector.wait_ready, settings.job_timeout_sec):
            log.error("detector %s is not loaded yet, /healthz reports 503 until it is", settings.detector)
        worker.start()
        yield
        cameras.stop("shutdown")
        worker.request_stop()
        detector.stop()
        worker.join()

    app = FastAPI(title="fortethon", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(UploadLimit, limit=settings.max_upload_bytes)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, f"http_{exc.status_code}")
        return JSONResponse({"error": code, "detail": str(exc.detail).splitlines()[0][:200]}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse({"error": "internal_error", "detail": "Internal server error."}, status_code=500)

    app.state.store = store
    app.state.settings = settings
    app.state.worker = worker
    app.state.detector = detector
    app.state.cameras = cameras

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if not settings.api_key or not x_api_key or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise ApiError(401, "unauthorized", "Invalid or missing X-API-Key.")

    async def accept_upload(request: Request, source: str, cap: int | None = None) -> dict:
        job_id = store.reserve(source, cap)
        if job_id is None:
            raise ApiError(429, "queue_full", "Too many videos are waiting. Try again in a few minutes.")
        try:
            async with request.form(max_files=1) as form:
                file = form.get("file")
                if not isinstance(file, StarletteUploadFile):
                    raise ApiError(422, "file_missing", "The multipart field 'file' is required.")
                name = file.filename or ""
                if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
                    raise ApiError(400, "wrong_extension", "Expected a .mov file.")
                store.job_dir(job_id).mkdir(parents=True)
                target = store.video_path(job_id)
                try:
                    size = await run_in_threadpool(save_upload, file.file, target, settings.max_upload_bytes)
                except TooLarge as exc:
                    raise too_large(settings.max_upload_bytes) from exc
                if size == 0:
                    raise ApiError(400, "empty_file", "The file is empty.")
                try:
                    info = await run_in_threadpool(probe, target)
                except VideoError as exc:
                    log.info("upload %s rejected, cannot decode: %s", job_id, exc)
                    raise ApiError(400, "invalid_video", "The file could not be decoded as a video.") from exc
        except BaseException:
            store.discard(job_id)
            raise
        store.activate(job_id, name, size, info.duration_sec)
        return {"job_id": job_id, "status": "processing"}

    def manual_review(job: dict) -> dict:
        duration = job["duration_sec"] or 0.0
        event = {
            "type_id": "other",
            "confidence": 0.0,
            "observation": "Автоматический анализ не завершён за отведённое время. Нарушения не подтверждены и не исключены, видео требует ручной проверки",
            "frames": [{"t_sec": duration / 2, "boxes": []}],
            "t_start_sec": 0.0,
            "t_end_sec": duration,
        }
        try:
            violation = pipeline.build_violation(event, str(store.video_path(job["id"])), rules, settings)
        except Exception:
            log.exception("manual review frame failed for job %s", job["id"])
            _, description, rule = rules.describe(event)
            violation = {"timecode_sec": 0.0, "description": description, "evidence_frames": [], "type_id": "other", "confidence": 0.0, "rule": rule, "boxes": []}
        violation["type"] = "Требуется ручная проверка"
        return {"verdict": "violation", "violations": [violation]}

    def public_view(job: dict) -> dict:
        status = {"queued": "processing", "running": "processing"}.get(job["status"], job["status"])
        body: dict = {"job_id": job["id"], "status": status}
        if status == "done":
            body.update(job["result"])
        elif status == "failed":
            body["status"] = "done"
            body.update(manual_review(job))
            body["error"], body["message"] = job_error(job["error"])
        return body

    def load_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise ApiError(404, "job_not_found", "No job with this id. Jobs are deleted 24 hours after upload.")
        return job

    @app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
    async def analyze(request: Request) -> dict:
        return await accept_upload(request, "api")

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def get_job(job_id: str) -> dict:
        return public_view(load_job(job_id))

    @app.post("/app/analyze")
    async def app_analyze(request: Request) -> dict:
        if not settings.app_uploads_enabled:
            raise ApiError(503, "uploads_disabled", "Uploads from the web UI are turned off. Existing results are still available.")
        return await accept_upload(request, "app", settings.app_queue_cap)

    @app.get("/app/jobs/{job_id}")
    def app_job(job_id: str, request: Request) -> dict:
        job = load_job(job_id)
        body = public_view(job)
        if body.get("violations"):
            body["violations"] = sorted(body["violations"], key=lambda v: v["timecode_sec"])
        body.update(
            {
                "queue_position": store.queue_position(job),
                "progress": round(job["progress"], 3),
                "filename": job["filename"],
                "duration_sec": job["duration_sec"],
                "video_url": str(request.url_for("app_video", job_id=job_id).path),
            }
        )
        return body

    @app.get("/app/catalog")
    def catalog() -> dict:
        """Violation types for the reviewer's "add a violation" form, in catalog order."""
        return {"types": [{"type_id": t["type_id"], "name": t["name"], "tier": t.get("tier")} for t in rules.types.values()]}

    def app_violations(job: dict) -> list[dict]:
        # Same order as GET /app/jobs/{id}, so the UI's "event-<i>" keys point at the same events.
        return sorted(public_view(job).get("violations") or [], key=lambda v: v["timecode_sec"])

    def review_view(job_id: str, review: dict | None) -> dict:
        if review is None:
            return {"job_id": job_id, "version": 0, "decisions": {}, "added": [], "updated_at": None}
        return {k: review[k] for k in ("job_id", "version", "decisions", "added", "updated_at")}

    @app.get("/app/jobs/{job_id}/review")
    def get_review(job_id: str) -> dict:
        review = store.get_review(job_id)
        if review is None:
            load_job(job_id)
        return review_view(job_id, review)

    @app.put("/app/jobs/{job_id}/review")
    async def put_review(job_id: str, request: Request) -> dict:
        job = load_job(job_id)
        if public_view(job)["status"] != "done":
            raise ApiError(409, "job_not_done", "The video is still being processed.")
        try:
            body = ReviewIn.model_validate(await request.json())
        except (ValidationError, ValueError) as exc:
            raise ApiError(422, "invalid_review", str(exc).splitlines()[0][:200])
        duration = job["duration_sec"] or 0.0
        for item in body.added:
            if item.type_id not in rules.types:
                raise ApiError(422, "invalid_review", f"Unknown violation type {item.type_id!r}.")
            end = item.t_end_sec if item.t_end_sec is not None else item.t_start_sec
            if end < item.t_start_sec or (duration and end > duration + 1):
                raise ApiError(422, "invalid_review", f"Time of {item.id!r} is outside the video.")
        model_events = dict(enumerate(app_violations(job)))
        decisions = {}
        for key, decision in body.decisions.items():
            index = int(key.removeprefix("event-")) if key.removeprefix("event-").isdigit() else None
            event = model_events.get(index) if index is not None else None
            snapshot = {k: event.get(k) for k in ("type_id", "type", "timecode_sec", "confidence")} if event else None
            decisions[key] = decision.model_dump(exclude_none=True) | {"model_event": snapshot}
        payload = {"decisions": decisions, "added": [item.model_dump() for item in body.added]}
        existing = store.get_review(job_id)
        sha = existing and existing["video_sha256"] or await run_in_threadpool(sha256_file, store.video_path(job_id))
        try:
            review = await run_in_threadpool(store.save_review, job, payload, body.version, sha)
        except ReviewConflict as exc:
            raise ApiError(409, "review_conflict", f"The review was changed elsewhere; reload version {exc.current}.")
        return review_view(job_id, review)

    @app.get("/api/v1/reviews/export", dependencies=[Depends(require_key)])
    def export_reviews() -> dict:
        """Reviewer feedback from the web UI in the ground_truth.json event format.

        For the team only (API key). Jury API jobs are never exported: test videos must
        not be used for training. Nothing here changes the model automatically.
        """
        videos = []
        for review in store.reviews("app"):
            events = [
                {
                    "type_id": a["type_id"],
                    "t_start_sec": a["t_start_sec"],
                    "t_end_sec": a["t_end_sec"] if a["t_end_sec"] is not None else a["t_start_sec"],
                    "key_t_sec": [a["t_start_sec"]],
                    "labeled": "reviewer",
                    "comment": a["comment"],
                }
                for a in review["added"]
            ]
            model = [{"event_key": k, **(d.get("model_event") or {}), **{f: d.get(f) for f in ("status", "reason", "comment")}} for k, d in review["decisions"].items()]
            videos.append(
                {
                    "job_id": review["job_id"],
                    "file": review["filename"],
                    "video_sha256": review["video_sha256"],
                    "duration_sec": review["duration_sec"],
                    "reviewed_at": review["updated_at"],
                    "events": events,
                    "model_events": model,
                }
            )
        return {"note": "reviewer feedback; labels only, videos are purged after 24 h; not used to change the model", "videos": videos}

    @app.get("/app/jobs/{job_id}/video", name="app_video")
    def app_video(job_id: str) -> FileResponse:
        load_job(job_id)
        source = store.video_path(job_id)
        target = store.job_dir(job_id) / "video.mp4"
        if not source.exists():
            raise ApiError(404, "video_not_found", "The video for this job is no longer stored.")
        with remux_lock:
            if not target.exists():
                remux(source, target)
        return FileResponse(target, media_type="video/mp4")

    def cameras_view() -> dict:
        if cameras.running:
            cameras.touch()
        return {"enabled": settings.cameras_enabled, **cameras.status()}

    @app.get("/app/cameras")
    def cameras_status() -> dict:
        return cameras_view()

    @app.post("/app/cameras/start")
    def cameras_start() -> dict:
        if not settings.cameras_enabled:
            raise ApiError(503, "cameras_disabled", "Camera streams are turned off on this server.")
        try:
            cameras.start()
        except CamerasUnavailable as exc:
            log.error("cameras cannot start: %s", exc)
            raise ApiError(503, "cameras_unavailable", "The camera list on the server is missing or invalid.") from exc
        return cameras_view()

    @app.post("/app/cameras/stop")
    def cameras_stop() -> dict:
        cameras.stop()
        return cameras_view()

    @app.get("/app/cameras/evidence/{job_id}/{index}")
    def cameras_evidence(job_id: str, index: int) -> Response:
        image = cameras.evidence(job_id, index)
        if image is None:
            raise ApiError(404, "frame_not_found", "No evidence frame for this segment.")
        return Response(image, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400, immutable"})

    @app.websocket("/app/cameras/preview")
    async def cameras_preview(websocket: WebSocket) -> None:
        # One socket carries every camera: "<camera id>\n<jpeg>" per frame. An open socket
        # counts as someone watching, which keeps the cameras from stopping as idle.
        await websocket.accept()
        closed = asyncio.ensure_future(websocket.receive())
        sent: dict[str, int] = {}
        try:
            while not closed.done():
                if cameras.running:
                    cameras.touch()
                for camera_id, preview in cameras.previews():
                    if preview and sent.get(camera_id) != preview[0]:
                        sent[camera_id] = preview[0]
                        await websocket.send_bytes(camera_id.encode() + b"\n" + preview[1])
                await asyncio.sleep(0.5 / PREVIEW_FPS)
        except Exception:
            pass
        finally:
            closed.cancel()

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        alive = worker.alive
        loaded = detector.loaded
        healthy = alive and loaded
        body = {
            "status": "ok" if healthy else ("worker_dead" if not alive else "model_not_loaded"),
            "detector": settings.detector,
            "model_loaded": loaded,
            "worker_alive": alive,
            "detector_pid": detector.pid,
            "queue_length": store.queue_length(),
            "app_uploads_enabled": settings.app_uploads_enabled,
            "cameras_running": cameras.running,
        }
        return JSONResponse(body, status_code=200 if healthy else 503)

    if (STATIC_DIR / "index.html").exists():
        app.mount("/", CachedStatic(directory=STATIC_DIR, html=True), name="static")

    return app


def remux(source: Path, target: Path) -> None:
    partial = target.with_suffix(".part.mp4")
    try:
        with av.open(str(source)) as src, av.open(str(partial), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
            stream = src.streams.video[0]
            out = dst.add_stream_from_template(stream)
            for packet in src.demux(stream):
                if packet.dts is None:
                    continue
                packet.stream = out
                dst.mux(packet)
    except av.FFmpegError as exc:
        partial.unlink(missing_ok=True)
        log.error("remux of %s failed: %s", source, exc)
        raise ApiError(500, "remux_failed", "The video could not be prepared for playback.") from exc
    partial.replace(target)
