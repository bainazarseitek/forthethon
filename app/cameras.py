"""Camera streams for the live view.

A camera is either a network stream (rtsp:// or http:// from a real camera) or, for the
demo, a playlist of recorded clips played back at real speed so it behaves like one. Each
camera thread reads the stream, keeps a small preview frame and records segments; every
finished segment becomes a normal job with source 'stream', which the worker takes only
when no jury or UI job is waiting.
"""

import base64
import io
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import av

from app.config import ROOT
from app.jobs import JobStore

log = logging.getLogger("cameras")

PREVIEW_FPS = 8
PREVIEW_HEIGHT = 480
PREVIEW_QUALITY = 70
HISTORY_PER_CAMERA = 40
MIN_SEGMENT_SEC = 3.0
RECONNECT_SEC = 5.0
OPEN_TIMEOUT_SEC = 10.0
JURY_QUIET_SEC = 600


class CamerasUnavailable(Exception):
    pass


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    playlist: tuple[Path, ...] = ()
    url: str | None = None

    @property
    def kind(self) -> str:
        return "playlist" if self.playlist else "stream"


def load_config(path: Path) -> tuple[list[CameraConfig], float]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CamerasUnavailable(f"camera config {path} cannot be read: {exc}") from exc
    cameras = []
    for item in data.get("cameras", []):
        playlist = tuple(p if p.is_absolute() else ROOT / p for p in map(Path, item.get("playlist", [])))
        url = item.get("url")
        if bool(playlist) == bool(url):
            raise CamerasUnavailable(f"camera {item.get('id')}: set exactly one of playlist and url")
        missing = [str(p) for p in playlist if not p.is_file()]
        if missing:
            raise CamerasUnavailable(f"camera {item['id']}: missing clips {missing}")
        cameras.append(CameraConfig(item["id"], item["name"], playlist, url))
    if not cameras:
        raise CamerasUnavailable(f"camera config {path} lists no cameras")
    return cameras, float(data.get("segment_sec", 30))


def preview_jpeg(frame: av.VideoFrame) -> bytes:
    height = min(PREVIEW_HEIGHT, frame.height)
    width = max(2, round(frame.width * height / frame.height / 2) * 2)
    image = frame.reformat(width=width, height=height, format="rgb24").to_image()
    rotation = int(frame.rotation or 0) % 360
    if rotation:
        image = image.rotate(rotation, expand=True)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=PREVIEW_QUALITY)
    return buffer.getvalue()


@dataclass
class Segment:
    job_id: str
    camera_id: str
    camera_name: str
    clip: str | None
    started_at: float
    ended_at: float | None = None
    duration_sec: float = 0.0
    state: str = "recording"  # recording, queued, running, done, failed, skipped
    reason: str | None = None
    verdict: str | None = None
    analysed_at: float | None = None
    violations: list[dict] = field(default_factory=list)

    @property
    def final(self) -> bool:
        return self.state in ("done", "failed", "skipped")

    def view(self) -> dict:
        return {
            "job_id": self.job_id,
            "camera_id": self.camera_id,
            "clip": self.clip,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_sec": round(self.duration_sec, 3),
            "state": self.state,
            "reason": self.reason,
            "verdict": self.verdict,
            "analysed_at": self.analysed_at,
            "violations": self.violations,
        }


class Recorder:
    """Copies the stream's packets into one segment file without re-encoding."""

    def __init__(self, segment: Segment, path: Path, template: av.video.stream.VideoStream):
        self.segment = segment
        self.path = path
        self.time_base = template.time_base
        self.container = av.open(str(path), "w", format="mov")
        self.stream = self.container.add_stream_from_template(template)
        self.base: int | None = None
        self.end = 0

    @property
    def seconds(self) -> float:
        return float(self.end * self.time_base)

    def mux(self, packet: av.Packet) -> None:
        if self.base is None:
            # The first keyframe is shown at 0; reordered frames can decode slightly earlier.
            self.base = packet.pts if packet.pts is not None else packet.dts
        packet.dts -= self.base
        if packet.pts is not None:
            packet.pts -= self.base
            self.end = max(self.end, packet.pts + (packet.duration or 0))
        packet.stream = self.stream
        self.container.mux(packet)

    def close(self) -> None:
        try:
            self.container.close()
        except av.FFmpegError as exc:
            log.warning("segment %s did not close cleanly: %s", self.segment.job_id, exc)


class Camera:
    def __init__(self, config: CameraConfig, hub: "CameraHub", stop: threading.Event):
        self.config = config
        self.hub = hub
        self.stop = stop
        self.state = "connecting"  # connecting, live, error
        self.error: str | None = None
        self.clip: str | None = None
        self.preview: tuple[int, bytes] | None = None
        self.recording_since: float | None = None
        self._seq = 0
        self.thread = threading.Thread(target=self._run, name=f"camera-{config.id}", daemon=True)

    def view(self) -> dict:
        return {
            "id": self.config.id,
            "name": self.config.name,
            "kind": self.config.kind,
            "state": self.state,
            "error": self.error,
            "clip": self.clip,
            "recording_since": self.recording_since,
        }

    def _run(self) -> None:
        sources = self.config.playlist or (self.config.url,)
        while not self.stop.is_set():
            for source in sources:
                if self.stop.is_set():
                    return
                try:
                    self._play(str(source))
                except Exception as exc:
                    self.state, self.error = "error", str(exc)[:200]
                    log.warning("camera %s: %s failed: %s", self.config.id, source, exc)
                    self.stop.wait(RECONNECT_SEC)

    def _play(self, source: str) -> None:
        paced = self.config.url is None
        options = {"rtsp_transport": "tcp"} if source.startswith("rtsp") else {}
        with av.open(source, options=options, timeout=OPEN_TIMEOUT_SEC) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            self.state, self.error = "live", None
            self.clip = Path(source).stem if paced else None
            clock: tuple[float, float] | None = None
            next_preview = 0.0
            recorder: Recorder | None = None
            try:
                for packet in container.demux(stream):
                    if self.stop.is_set():
                        break
                    if packet.dts is None:
                        continue
                    t = float(packet.dts * stream.time_base)
                    if clock is None:
                        clock = (time.time(), t)
                    if paced:
                        wait = clock[0] + (t - clock[1]) - time.time()
                        if wait > 0 and self.stop.wait(wait):
                            break
                    if recorder is None:
                        if not packet.is_keyframe:
                            continue
                        recorder = self.hub.open_segment(self, stream)
                    elif not paced and packet.is_keyframe and recorder.seconds >= self.hub.segment_sec:
                        self.hub.close_segment(recorder)
                        recorder = self.hub.open_segment(self, stream)
                    try:
                        frames = packet.decode()
                    except av.FFmpegError:
                        frames = []
                    for frame in frames:
                        if time.time() >= next_preview:
                            next_preview = time.time() + 1 / PREVIEW_FPS
                            self._seq += 1
                            self.preview = (self._seq, preview_jpeg(frame))
                    recorder.mux(packet)
            finally:
                if recorder is not None:
                    # A clip that played to its end is a whole segment; a stop cuts it short.
                    if self.stop.is_set():
                        self.hub.discard_segment(recorder)
                    else:
                        self.hub.close_segment(recorder)
                self.recording_since = None


class CameraHub:
    def __init__(self, store: JobStore, config_path: Path, idle_sec: float = 120, timezone: str = "Asia/Almaty"):
        self.store = store
        self.config_path = Path(config_path)
        self.idle_sec = idle_sec
        try:
            self.tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            self.tz = None
        self.segment_sec = 30.0
        self.cameras: list[Camera] = []
        self.started_at: float | None = None
        self.stopped_reason: str | None = None
        self._segments: dict[str, list[Segment]] = {}
        self._stop = threading.Event()
        self._stop.set()
        self._lock = threading.RLock()
        self._last_seen = 0.0

    @property
    def running(self) -> bool:
        # Keep reporting the hub as running while camera threads are still
        # shutting down. This prevents callers from observing a stopped hub
        # before its readers and recorders have actually finished cleanup.
        return not self._stop.is_set() or any(camera.thread.is_alive() for camera in self.cameras)

    def touch(self) -> None:
        self._last_seen = time.time()

    def configured(self) -> list[CameraConfig]:
        return load_config(self.config_path)[0]

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            configs, self.segment_sec = load_config(self.config_path)
            for history in self._segments.values():
                for old in history:
                    self.store.remove(old.job_id)
            self._stop = stop = threading.Event()
            self.cameras = [Camera(c, self, stop) for c in configs]
            self._segments = {c.id: [] for c in configs}
            self.started_at = time.time()
            self.stopped_reason = None
            self.touch()
            for camera in self.cameras:
                camera.thread.start()
            threading.Thread(target=self._watch_idle, args=(stop,), name="camera-idle", daemon=True).start()
            log.info(json.dumps({"event": "cameras_start", "cameras": [c.id for c in configs]}))

    def stop(self, reason: str = "stopped") -> None:
        with self._lock:
            if not self.running:
                return
            self._stop.set()
            self.stopped_reason = reason
            cameras = list(self.cameras)
        for camera in cameras:
            camera.thread.join(timeout=5)
        log.info(json.dumps({"event": "cameras_stop", "reason": reason}))

    def _watch_idle(self, stop: threading.Event) -> None:
        # Nobody watching means nobody needs the GPU spent on this: stop on our own.
        while not stop.wait(2):
            if time.time() - self._last_seen > self.idle_sec:
                self.stop("idle")
                return

    def local_time(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, self.tz).strftime("%H:%M:%S")

    def open_segment(self, camera: Camera, template) -> Recorder:
        job_id = self.store.reserve("stream")
        assert job_id is not None
        self.store.job_dir(job_id).mkdir(parents=True)
        now = time.time()
        segment = Segment(job_id, camera.config.id, camera.config.name, camera.clip, now)
        camera.recording_since = now
        with self._lock:
            history = self._segments.setdefault(camera.config.id, [])
            history.append(segment)
            while len(history) > HISTORY_PER_CAMERA:
                self._refresh(history[0])
                if not history[0].final:
                    break
                old = history.pop(0)
                self.store.remove(old.job_id)
        return Recorder(segment, self.store.video_path(job_id), template)

    def _forget(self, segment: Segment) -> None:
        self.store.discard(segment.job_id)
        with self._lock:
            history = self._segments.get(segment.camera_id, [])
            if segment in history:
                history.remove(segment)

    def discard_segment(self, recorder: Recorder) -> None:
        recorder.close()
        self._forget(recorder.segment)

    def close_segment(self, recorder: Recorder) -> None:
        segment = recorder.segment
        recorder.close()
        segment.ended_at = time.time()
        segment.duration_sec = recorder.seconds
        if segment.duration_sec < MIN_SEGMENT_SEC:
            self._forget(segment)
            return
        if self.store.busy_or_recent("api", JURY_QUIET_SEC):
            # Jury videos come first: while one waits or runs, or arrived recently, segments are not analysed.
            self.store.discard(segment.job_id)
            segment.state, segment.reason = "skipped", "jury"
            return
        with self._lock:
            for older in self._segments.get(segment.camera_id, []):
                # Stay live: a segment still waiting is replaced by the newer one.
                if older.state == "queued" and self.store.cancel_queued(older.job_id):
                    older.state, older.reason = "skipped", "busy"
        name = f"{segment.camera_name} · {self.local_time(segment.started_at)}"
        size = recorder.path.stat().st_size
        self.store.activate(segment.job_id, name, size, segment.duration_sec)
        segment.state = "queued"

    def _refresh(self, segment: Segment) -> None:
        if segment.final or segment.state == "recording":
            return
        status = self.store.status_of(segment.job_id)
        if status in ("queued", "running"):
            segment.state = status
            return
        job = self.store.get(segment.job_id) if status else None
        if job is None or job["status"] == "failed":
            segment.state, segment.reason = "failed", (job or {}).get("error") or "missing"
            return
        segment.state = "done"
        segment.verdict = job["result"]["verdict"]
        segment.analysed_at = job["finished_at"]
        segment.violations = sorted(
            (
                {
                    "index": i,
                    "type": v["type"],
                    "type_id": v.get("type_id"),
                    "confidence": v.get("confidence"),
                    "timecode_sec": v["timecode_sec"],
                    "at": segment.started_at + v["timecode_sec"],
                    "has_frame": bool(v.get("evidence_frames")),
                }
                for i, v in enumerate(job["result"]["violations"])
            ),
            key=lambda v: v["timecode_sec"],
        )

    def status(self) -> dict:
        with self._lock:
            segments = [s for history in self._segments.values() for s in history]
            for segment in segments:
                self._refresh(segment)
            segments.sort(key=lambda s: s.started_at, reverse=True)
            cameras = [c.view() for c in self.cameras] if self.running else None
        if cameras is None:
            try:
                cameras = [{"id": c.id, "name": c.name, "kind": c.kind, "state": "off", "error": None, "clip": None, "recording_since": None} for c in self.configured()]
            except CamerasUnavailable:
                cameras = []
        return {
            "running": self.running,
            "started_at": self.started_at,
            "stopped_reason": self.stopped_reason,
            "now": time.time(),
            "idle_stop_sec": self.idle_sec,
            "jury_waiting": self.store.active_count("api") > 0,
            "cameras": cameras,
            "segments": [s.view() for s in segments],
        }

    def previews(self) -> list[tuple[str, tuple[int, bytes] | None]]:
        return [(c.config.id, c.preview) for c in self.cameras] if self.running else []

    def evidence(self, job_id: str, index: int) -> bytes | None:
        with self._lock:
            known = any(s.job_id == job_id for history in self._segments.values() for s in history)
        job = self.store.get(job_id) if known else None
        if not job or job["status"] != "done":
            return None
        violations = job["result"]["violations"]
        if not 0 <= index < len(violations) or not violations[index].get("evidence_frames"):
            return None
        return base64.b64decode(violations[index]["evidence_frames"][0]["image_base64"])
