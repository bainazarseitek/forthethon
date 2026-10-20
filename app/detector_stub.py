import bisect
import hashlib
import json
import os
import time
from pathlib import Path

from app.config import Settings
from app.frames import frame_times

OBSERVATION = "заглушка детектора: синтетическое событие для проверки сервиса, содержимое кадра не анализировалось"


def _catalog() -> list[str]:
    rules = json.loads(Path(Settings().rules_path).read_text(encoding="utf-8"))
    return [t["type_id"] for t in rules["types"] if t["type_id"] != "other"]


def _digest(path: str) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def _snap(times: list[float], t: float) -> float:
    i = bisect.bisect_left(times, t)
    candidates = times[max(0, i - 1) : i + 1]
    return round(min(candidates, key=lambda x: abs(x - t)), 3)


def _event(h: bytes, i: int, times: list[float], catalog: list[str], confidence: float) -> dict:
    duration = times[-1]
    type_id = catalog[h[i] % len(catalog)]
    closest = None
    if h[i + 8] % 5 == 0:
        type_id, closest = "other", type_id
    start = duration * h[i + 16] / 256
    end = min(duration, start + 1.0 + h[i + 24] % 3)
    count = 1 + h[i + 4] % 3
    step = (end - start) / count
    keys = sorted({_snap(times, start + step * k) for k in range(count)})
    frames = [
        {"t_sec": t, "boxes": [{"x": 0.25 + 0.1 * k, "y": 0.4, "w": 0.2, "h": 0.15, "label": "заглушка"}]}
        for k, t in enumerate(keys)
    ]
    return {
        "type_id": type_id,
        "closest_type_id": closest,
        "t_start_sec": round(start, 3),
        "t_end_sec": round(end, 3),
        "confidence": confidence,
        "observation": OBSERVATION,
        "frames": frames,
    }


def analyze(video_path: str) -> dict:
    delay = float(os.environ.get("STUB_DELAY_SEC") or 0)
    if delay:
        time.sleep(delay)
    times = frame_times(video_path)
    if not times:
        raise RuntimeError("no frames")
    h = _digest(video_path)
    catalog = _catalog()
    count = h[31] % 3
    events = [_event(h, i, times, catalog, round(0.55 + h[i + 12] % 40 / 100, 2)) for i in range(count)]
    events.append(_event(h, 3, times, catalog, 0.2))
    return {"events": events}
