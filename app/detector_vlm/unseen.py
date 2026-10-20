import math

import av
from PIL import Image, ImageFilter, ImageOps, ImageStat

CELLS = 4
SIDE = 96
RATIO = 0.35
FRAC = 0.45
BLANK_SD = 1.5
MIN_SEC = 3.0
COLLECTION_MIN = 0.5


def texture(image: Image.Image) -> tuple[float, list[float]]:
    gray = image.convert("L")
    if gray.size != (SIDE, SIDE):
        gray = gray.resize((SIDE, SIDE))
    sd = ImageStat.Stat(gray).stddev[0]
    edges = ImageOps.autocontrast(gray, cutoff=1).filter(ImageFilter.FIND_EDGES)
    step = SIDE // CELLS
    cells = [
        ImageStat.Stat(edges.crop((j * step + 1, i * step + 1, (j + 1) * step - 1, (i + 1) * step - 1))).mean[0]
        for i in range(CELLS)
        for j in range(CELLS)
    ]
    return sd, cells


def keyframe_textures(path: str) -> list[tuple[float, float, list[float]]]:
    rows = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            if frame.pts is not None:
                rows.append((float(frame.time), *texture(frame.reformat(width=SIDE, height=SIDE, format="gray").to_image())))
    return rows


def blocked_scores(rows: list[tuple[float, float, list[float]]]) -> list[tuple[float, float]]:
    if not rows:
        return []
    n = len(rows[0][2])
    ref = [sorted(r[2][k] for r in rows)[len(rows) // 2] for k in range(n)]
    out = []
    for t, sd, cells in rows:
        frac = sum(c < RATIO * max(r, 1.0) for c, r in zip(cells, ref)) / n
        out.append((t, 1.0 if sd < BLANK_SD else frac))
    return out


def _event(type_id: str, p: float, t0: float, t1: float, t: float, observation: str) -> dict:
    return {
        "type_id": type_id,
        "closest_type_id": None,
        "t_start_sec": round(t0, 3),
        "t_end_sec": round(t1, 3),
        "confidence": round(p, 4),
        "observation": observation,
        "frames": [{"t_sec": round(t, 3), "boxes": []}],
    }


def camera_blocked(rows: list[tuple[float, float, list[float]]]) -> list[dict]:
    events, run = [], []
    for t, s in blocked_scores(rows) + [(math.inf, 0.0)]:
        if s >= FRAC:
            run.append((t, s))
            continue
        if run and run[-1][0] - run[0][0] >= MIN_SEC - 0.1:
            p = min(1.0, 0.5 + 0.5 * min(x for _, x in run))
            observation = "большая часть кадра закрыта предметом или объектив затемнён, обзор кассы перекрыт"
            events.append(_event("camera_view_blocked", p, run[0][0], run[-1][0], run[0][0], observation))
        run = []
    return events


def collection_score(w: dict) -> float:
    return min(max(w.get("bag_in", 0.0), w.get("bag_open", 0.0)), w.get("second_person", 0.0), w.get("client", 0.0))


def collection_in_view(windows: list[dict]) -> list[dict]:
    best = max(windows, key=collection_score, default=None)
    if best is None or collection_score(best) < COLLECTION_MIN:
        return []
    observation = "кассир принимает или вскрывает инкассаторскую сумку, у стола одновременно находятся ещё два человека"
    t = (best["t0"] + best["t1"]) / 2
    return [_event("collection_in_view_of_clients", collection_score(best), best["t0"], best["t1"], t, observation)]


def detect_unseen(path: str, facts: dict | None = None) -> list[dict]:
    windows = (facts or {}).get("windows") or []
    events = camera_blocked(keyframe_textures(path)) + collection_in_view(windows)
    return sorted(events, key=lambda e: e["t_start_sec"])


def _selfcheck() -> None:
    rows = [(float(t), 40.0, [10.0] * 16) for t in range(12)]
    rows[4:8] = [(float(t), 40.0, [1.0] * 12 + [10.0] * 4) for t in range(4, 8)]
    assert [round(x, 2) for _, x in blocked_scores(rows)[3:9]] == [0.0, 0.75, 0.75, 0.75, 0.75, 0.0]
    assert [(e["t_start_sec"], e["t_end_sec"]) for e in camera_blocked(rows)] == [(4.0, 7.0)]
    assert camera_blocked(rows[:4] + rows[5:]) == []
    assert collection_in_view([{"t0": 1, "t1": 3.5, "bag_in": 0.9, "second_person": 0.8, "client": 0.95}])[0]["confidence"] == 0.8
    assert collection_in_view([{"t0": 1, "t1": 3.5, "bag_in": 0.9, "second_person": 0.3, "client": 0.95}]) == []
