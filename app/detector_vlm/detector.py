import base64
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import av
from PIL import Image

from app.detector_vlm import boxes as box_module
from app.detector_vlm import crops
from app.detector_vlm.state import apply_verify, detail_states, frame_states, state_events, verify
from app.detector_vlm.unseen import camera_blocked, keyframe_textures

VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:10301/v1/chat/completions")
FPS = float(os.environ.get("VLM_FPS", "2"))
MAX_FRAMES = int(os.environ.get("VLM_MAX_FRAMES", "300"))
FULL_SIDE = 960
STATE_FRAMES = int(os.environ.get("VLM_STATE_FRAMES", "100"))
STATE_STEP = float(os.environ.get("VLM_STATE_STEP", "1"))
ACTIONS = os.environ.get("VLM_ACTIONS", "1") == "1"
HFLIP = os.environ.get("VLM_HFLIP", "0") == "1"
TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "200"))
BOXES_MIN_LEFT = float(os.environ.get("VLM_BOXES_MIN_LEFT", "40"))
BOX_TYPES = {
    "safe_left_open",
    "key_left_in_safe",
    "workstation_left_unlocked",
    "cash_unattended",
    "stamp_left_on_desk",
    "previous_cash_on_desk",
    "bag_accepted_unchecked",
    "stored_without_count",
    "drink_in_cash_zone",
    "personal_items_in_cash_zone",
    "door_left_open",
}
RESERVE_SEC = float(os.environ.get("VLM_RESERVE_SEC", "10"))
COST_PER_SEC = float(os.environ.get("VLM_COST_PER_SEC", "0.3"))
COST_BASE_SEC = float(os.environ.get("VLM_COST_BASE_SEC", "45"))
COST_SAFETY_SEC = float(os.environ.get("VLM_COST_SAFETY_SEC", "15"))
DEGRADED_SHARE = float(os.environ.get("VLM_DEGRADED_SHARE", "0.6"))
MODE = os.environ.get("VLM_MODE", "rules")
MIN_CONF = float(os.environ.get("VLM_MIN_CONF", "0.5"))
CUTS: dict[str, float] = {"serving_two_clients": 0.6, "safe_left_open": 0.9}


def load() -> None:
    deadline = time.monotonic() + float(os.environ.get("VLM_WAIT", "900"))
    while True:
        try:
            with urllib.request.urlopen(VLM_URL.replace("/chat/completions", "/models"), timeout=5):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def _upright(frame: av.VideoFrame, side: int) -> Image.Image:
    image = frame.to_image()
    rotation = int(frame.rotation or 0) % 360
    if rotation:
        image = image.rotate(rotation, expand=True)
    image.thumbnail((side, side))
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if HFLIP else image


def sample_grid(path: str, fps: float = FPS, side: int = FULL_SIDE) -> tuple[float, list[tuple[float, Image.Image]]]:
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if duration * fps > MAX_FRAMES:
            fps = MAX_FRAMES / duration
        step = 1.0 / fps
        grid: list[tuple[float, Image.Image]] = []
        previous = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.time)
            target = len(grid) * step
            while t >= target:
                if previous is not None and target - previous[0] < t - target:
                    grid.append((previous[0], _upright(previous[1], side)))
                else:
                    grid.append((t, _upright(frame, side)))
                target = len(grid) * step
            previous = (t, frame)
        if previous is not None and (not grid or grid[-1][0] < previous[0] - step / 2):
            grid.append((previous[0], _upright(previous[1], side)))
    return fps, grid


def _image_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _post(body: dict) -> dict:
    request = urllib.request.Request(VLM_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def _event(type_id: str, p: float, t: float, observation: str, times: list[float] | None = None) -> dict:
    times = times or [t]
    return {
        "type_id": type_id,
        "closest_type_id": None,
        "t_start_sec": times[0],
        "t_end_sec": times[-1],
        "confidence": round(p, 4),
        "observation": observation,
        "frames": [{"t_sec": x, "boxes": []} for x in times[:3]],
    }


def _actions(path: str, fine: bool) -> list[dict]:
    from app.detector_vlm.actions import detect_actions

    return detect_actions(path, fine)


def _ground(frame: Image.Image) -> dict:
    try:
        return crops.ground(frame)
    except (OSError, ValueError, KeyError):
        return {}


def _blocked(path: str) -> list[dict]:
    return camera_blocked(keyframe_textures(path))


def entry_time(states: list[dict]) -> float | None:
    away = 0
    for s in states:
        if s["cashier_present"] < 0.5:
            away += 1
        else:
            return s["t"] if away >= 3 else None
    return None


def _door(path: str, states: list[dict]) -> list[dict]:
    from app.detector_vlm.actions import door_after_entry

    entry = entry_time(states)
    return door_after_entry(path, entry) if entry is not None else []


def _duration(path: str) -> float:
    with av.open(path) as container:
        stream = container.streams.video[0]
        if stream.duration:
            return float(stream.duration * stream.time_base)
        return float(container.duration or 0) / 1e6


def cost(duration: float) -> float:
    return COST_PER_SEC * duration + COST_BASE_SEC + COST_SAFETY_SEC


def plan(duration: float, budget_sec: float | None) -> dict:
    left = budget_sec - RESERVE_SEC if budget_sec is not None else None
    if left is None or left >= cost(duration):
        return {"step": STATE_STEP, "detail": True, "fine": True}
    if left >= DEGRADED_SHARE * cost(duration):
        return {"step": STATE_STEP * 2, "detail": True, "fine": False}
    return {"step": STATE_STEP * 3, "detail": False, "fine": False}


def calibrate(event: dict) -> dict:
    cut = CUTS.get(event["type_id"])
    if cut is None or event.get("source") == "actions":
        return event
    p = event["confidence"]
    p = 0.5 + 0.5 * (p - cut) / (1 - cut) if p >= cut else 0.5 * p / cut
    return dict(event, confidence=round(min(1.0, p), 4))


def always(events: list[dict], states: list[dict]) -> list[dict]:
    if any(e["confidence"] >= 0.5 for e in events):
        return events
    if events:
        top = max(events, key=lambda e: e["confidence"])
        return [dict(top, confidence=0.5)] + [e for e in events if e is not top]
    if not states:
        return events
    s = max(states, key=lambda s: min(s.get("client_present", 0), s.get("cash_on_desk", 0)))
    observation = "клиент у стола, на столе купюры, предъявление документа клиентом в кадре не видно"
    return [_event("no_id_check", 0.5, s["t"], observation)]


PRIORITY = [
    "no_id_check",
    "count_outside_client_view",
    "bag_accepted_unchecked",
    "stored_without_count",
    "cash_hand_to_hand",
    "serving_two_clients",
    "safe_left_open",
    "cash_unattended",
    "key_left_in_safe",
    "stamp_left_on_desk",
    "workstation_left_unlocked",
    "left_during_service",
    "previous_cash_on_desk",
    "drink_in_cash_zone",
    "personal_items_in_cash_zone",
    "second_person_in_cash_zone",
    "door_left_open",
    "personal_phone_in_cash_zone",
    "camera_view_blocked",
    "collection_in_view_of_clients",
]


def merge(events: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for e in events:
        if e["confidence"] >= 0.05 and e["confidence"] > best.get(e["type_id"], {}).get("confidence", -1):
            best[e["type_id"]] = e
    rank = {t: i for i, t in enumerate(PRIORITY)}
    return sorted(best.values(), key=lambda e: (e["confidence"] < 0.5, rank.get(e["type_id"], len(rank)), -e["confidence"]))


def analyze(path: str, budget_sec: float | None = None) -> dict:
    deadline = time.monotonic() + budget_sec - RESERVE_SEC if budget_sec is not None else None
    mode = plan(_duration(path), budget_sec)
    pool = ThreadPoolExecutor(4)
    actions_future = pool.submit(_actions, path, mode["fine"]) if ACTIONS else None
    _, grid = sample_grid(path, fps=1.0 / mode["step"], side=FULL_SIDE)
    if not grid:
        raise ValueError("no decodable frames")
    if len(grid) > STATE_FRAMES:
        stride = len(grid) / STATE_FRAMES
        grid = [grid[int(i * stride)] for i in range(STATE_FRAMES)] + [grid[-1]]
    blocked_future = pool.submit(_blocked, path)
    boxes_future = pool.submit(_ground, grid[0][1]) if mode["detail"] else None
    states = frame_states(_post, _image_url, grid)
    door_future = pool.submit(_door, path, states) if ACTIONS else None
    detail = detail_states(grid, states, boxes_future.result()) if boxes_future else {}
    raw = state_events(states, detail)
    checks = verify(_post, _image_url, grid, raw)
    events = [_event(e["type_id"], e["confidence"], e["t"], e["observation"], e.get("times")) for e in apply_verify(raw, checks)]
    try:
        events += [dict(e, source="unseen") for e in blocked_future.result(timeout=10)]
    except Exception:
        pass
    actions_error, door = None, []
    for future, source in ((actions_future, "actions"), (door_future, "door")):
        if future is None:
            continue
        try:
            wait = None if deadline is None else max(1.0, deadline - time.monotonic())
            found = [dict(e, source=source) for e in future.result(timeout=wait)]
        except Exception as exc:
            actions_error = (actions_error or "") + f"{source}: {exc!r} "
            continue
        events += found
        if source == "door":
            door = found
    pool.shutdown(wait=False, cancel_futures=True)
    events = merge([calibrate(e) for e in events])
    if MODE == "always":
        events = always(events, states)
    events = [e for e in events if e["confidence"] >= MIN_CONF]
    left = None if deadline is None else deadline - time.monotonic()
    boxes_status: dict = {"skipped": "low budget", "left": round(left, 1)} if left is not None else {}
    if left is None or left >= BOXES_MIN_LEFT:
        boxed = [e for e in events if e["type_id"] in BOX_TYPES]
        box_module.add_boxes(path, boxed, (detail or {}).get("boxes"), None if left is None else left - BOXES_MIN_LEFT + 10)
        boxes_status = dict(box_module.LAST, boxed=sum(bool(f["boxes"]) for e in boxed for f in e["frames"]))
    report = {
        "boxes": boxes_status,
        "door": door,
        "plan": mode,
        "actions_error": actions_error,
        "states": states,
        "detail": detail,
        "verify": checks,
    }
    return {"events": events, "report": report}
