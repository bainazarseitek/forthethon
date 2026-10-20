import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageChops

from app.detector_vlm import crops

STATE_WORKERS = int(os.environ.get("VLM_STATE_WORKERS", "24"))

FIELDS = {
    "cashier_present": "кассир (работник банка) находится в кассе у своего стола или у сейфа",
    "cashier_at_safe": "кассир стоит или сидит прямо у сейфа и работает с ним",
    "client_present": "клиент сидит или стоит у стола напротив кассира",
    "second_person_cashier_side": "кроме кассира внутри кассы на стороне кассира (рядом с кассиром или за его столом, не на стуле клиента напротив) стоит ещё один человек",
    "safe_door_open": "дверца сейфа открыта",
    "cash_on_desk": "на столе лежат купюры или пачки денег",
    "multiple_cash_stacks": "на столе лежат две или больше отдельные стопки купюр",
    "bag_on_desk": "на столе лежит инкассаторская сумка или мешок",
    "drink_or_food": "стакан, кружка, бутылка или еда на кассовом столе или в руках кассира",
    "personal_items_on_desk": "женская сумка, рюкзак, куртка или кошелёк лежат на кассовом столе",
}

UPSCALE = float(os.environ.get("VLM_UPSCALE", "3"))
GATE = float(os.environ.get("VLM_GATE", "0.01"))
GATE_SIDE = 240
GATE_LEVEL = 25


def _prompt(fields: dict, what: str) -> str:
    return (
        f"Тебе покажут {what} с камеры видеонаблюдения в кассе банка. Для каждого пункта ответь true, только если это ясно видно, иначе false.\n"
        + "\n".join(f"- {k}: {v}" for k, v in fields.items())
        + "\nИзображение:"
    )


def _probs(choice: dict, fields: dict) -> dict[str, float]:
    text, out = "", {}
    for token in (choice.get("logprobs") or {}).get("content") or []:
        word = token["token"].strip()
        if word in ("true", "false"):
            m = re.search(r'"(\w+)"\s*:\s*$', text)
            if m and m.group(1) in fields:
                out[m.group(1)] = min(1.0, sum(math.exp(c["logprob"]) for c in token["top_logprobs"] if c["token"].strip() == "true"))
        text += token["token"]
    parsed = json.loads(choice["message"]["content"])
    return {k: out.get(k, float(bool(parsed.get(k)))) for k in fields}


def ask(post, image_url, image: Image.Image, fields: dict, what: str = "кадр") -> dict[str, float]:
    body = {
        "model": "vlm",
        "temperature": 0,
        "max_tokens": 300,
        "logprobs": True,
        "top_logprobs": 5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": _prompt(fields, what)}, {"type": "image_url", "image_url": {"url": image_url(image)}}],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "state",
                "schema": {"type": "object", "properties": {k: {"type": "boolean"} for k in fields}, "required": list(fields)},
            },
        },
    }
    return _probs(post(body)["choices"][0], fields)


def zoom(image: Image.Image, box: tuple[float, float, float, float], margin: float, side: int) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    crop = image.crop((int(max(0, x1 - mx) * w), int(max(0, y1 - my) * h), int(min(1, x2 + mx) * w), int(min(1, y2 + my) * h)))
    scale = side / max(crop.size)
    return crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.BICUBIC) if scale > 1 else crop


def _gray(image: Image.Image) -> Image.Image:
    small = image.convert("L")
    small.thumbnail((GATE_SIDE, GATE_SIDE))
    return small


def changed(a: Image.Image, b: Image.Image) -> float:
    histogram = ImageChops.difference(a, b).histogram()
    return sum(histogram[GATE_LEVEL + 1 :]) / sum(histogram)


def anchors(frames: list[tuple[float, Image.Image]], gate: float = GATE) -> list[int]:
    out, anchor, grays = [], 0, [_gray(image) for _, image in frames]
    for i, gray in enumerate(grays):
        if i == 0 or i == len(grays) - 1 or gray.size != grays[anchor].size or changed(grays[anchor], gray) >= gate:
            anchor = i
        out.append(anchor)
    return out


def frame_states(post, image_url, frames: list[tuple[float, Image.Image]]) -> list[dict]:
    source = anchors(frames) if GATE > 0 else list(range(len(frames)))
    asked = sorted(set(source))
    with ThreadPoolExecutor(STATE_WORKERS) as pool:
        answers = dict(zip(asked, pool.map(lambda i: ask(post, image_url, frames[i][1], FIELDS), asked)))
    return [{"t": t, **answers[source[i]]} for i, (t, _) in enumerate(frames)]


AWAY_FACTS = {
    "cashier_absent": "cashier_absent",
    "cash_on_desk": "desk_items",
    "stamps_or_journal_on_desk": "stamps_v9",
    "laptop_screen_active": "laptop_open_app",
    "key_in_safe_lock": "key_in_safe",
}
SAFE_FACTS = {"safe_door_open": "safe_open"}
TWO_MAX_FRAMES = 6


def _fact(frame: Image.Image, boxes: dict, name: str) -> float:
    target = crops.QUESTIONS[name][0]
    if target != "full" and target not in boxes:
        return crops.ask(crops.crop(frame, (0, 0, frame.width, frame.height), pad=0, max_scale=2.0), crops.QUESTIONS[name][3])
    return crops.fact(frame, boxes, name)


def detail_states(frames: list[tuple[float, Image.Image]], states: list[dict], boxes: dict) -> dict:
    if not frames:
        return {}
    smoothed = smooth(states)
    away = [r for r in _runs(smoothed, lambda s: s["cashier_present"] < AWAY) if _span(r) >= MIN_AWAY_SEC or r[-1]["t"] == states[-1]["t"]]
    by_t = dict(frames)
    away_ts = sorted({t for r in away for t in {r[min(1, len(r) - 1)]["t"], r[len(r) // 2]["t"], r[-1]["t"]}})
    safe_ts = sorted({t for t, _ in frames[:: max(1, len(frames) // 8)]} | set(away_ts))
    two = sorted(
        (s for s in smoothed if s["client_present"] >= 0.5 and s["second_person_cashier_side"] >= 0.3),
        key=lambda s: -s["second_person_cashier_side"],
    )[:TWO_MAX_FRAMES]
    jobs = [("desk", t, key, name) for t in away_ts for key, name in AWAY_FACTS.items()]
    jobs += [("safe", t, key, name) for t in safe_ts for key, name in SAFE_FACTS.items()]
    jobs += [("two", s["t"], key, key) for s in two for key in ("client_seated", "person_standing_at_desk")]
    with ThreadPoolExecutor(STATE_WORKERS) as pool:
        answers = list(pool.map(lambda j: _fact(by_t[j[1]], boxes, j[3]), jobs))
    out: dict = {"boxes": boxes, "desk": {}, "safe": {}, "two": {}}
    for (kind, t, key, _), p in zip(jobs, answers):
        out[kind].setdefault(str(t), {})[key] = p
    return out


def smooth(states: list[dict]) -> list[dict]:
    out = []
    for i, s in enumerate(states):
        near = states[max(0, i - 1) : i + 2]
        out.append({"t": s["t"], **{k: sum(n.get(k, 0.0) for n in near) / len(near) for k in FIELDS}})
    return out


def _runs(states: list[dict], pred) -> list[list[dict]]:
    runs, cur = [], []
    for s in states:
        if pred(s):
            cur.append(s)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _span(run: list[dict]) -> float:
    return run[-1]["t"] - run[0]["t"]


def _persistent(states: list[dict], key: str, k: int = 2) -> tuple[float, float]:
    best, t = 0.0, 0.0
    for i in range(len(states) - k + 1):
        p = min(s.get(key, 0.0) for s in states[i : i + k])
        if p > best:
            best, t = p, states[i]["t"]
    return best, t


AWAY = 0.3
MIN_AWAY_SEC = 2.0
ARRIVAL_SEC = 4.0
SAFE_OPEN = 0.9


def _detail(detail: dict, kind: str, run: list[dict], key: str, strict: bool = False) -> tuple[float, float] | None:
    ts = [s["t"] for s in run]
    hits = [(a[key], float(t)) for t, a in (detail.get(kind) or {}).items() if key in a and ts[0] - 1e-6 <= float(t) <= ts[-1] + 1e-6]
    if not hits:
        return None
    values = [p for p, _ in hits]
    return (min(values) if strict else sum(values) / len(values)), max(t for _, t in hits)


def state_events(raw: list[dict], detail: dict | None = None) -> list[dict]:
    if not raw:
        return []
    detail = detail or {}
    states = smooth(raw)
    events = []

    def add(type_id: str, p: float, t: float, observation: str) -> None:
        if p > 0:
            events.append({"type_id": type_id, "confidence": round(p, 4), "t": t, "observation": observation})

    for run in _runs(states, lambda s: s["cashier_present"] < AWAY):
        if (_span(run) < MIN_AWAY_SEC and run[-1] is not states[-1]) or (run[0] is states[0] and run[-1] is not states[-1]):
            continue
        away = 1 - max(s["cashier_present"] for s in run)
        absent = _detail(detail, "desk", run, "cashier_absent")
        if absent is not None:
            away = absent[0]
        client = _persistent(run, "client_present")
        if client[0] > 0.5:
            add("left_during_service", min(away, client[0]), run[0]["t"], "кассир ушёл из кассы, клиент остаётся у стола")
            continue
        for type_id, kind, key, text in [
            ("cash_unattended", "desk", "cash_on_desk", "кассира нет в кассе, на столе остались купюры"),
            ("safe_left_open", "safe", "safe_door_open", "кассира нет в кассе, дверца сейфа открыта"),
            ("key_left_in_safe", "desk", "key_in_safe_lock", "кассира нет в кассе, в замке сейфа остался ключ"),
            (
                "workstation_left_unlocked",
                "desk",
                "laptop_screen_active",
                "кассира нет в кассе, экран ноутбука показывает открытое окно программы",
            ),
            ("stamp_left_on_desk", "desk", "stamps_or_journal_on_desk", "кассира нет в кассе, на столе остались штампы или журнал"),
        ]:
            found = _detail(detail, kind, run, key, strict=key == "safe_door_open")
            if found is not None:
                add(type_id, min(away, found[0]), run[0]["t"], text)

    safe = sorted((float(t), a["safe_door_open"]) for t, a in (detail.get("safe") or {}).items() if "safe_door_open" in a)
    if len(safe) >= 3:
        values = [p for _, p in safe]
        p = min(sum(values) / len(values), min(values[-3:]))
        tail = len(safe)
        while tail > 0 and safe[tail - 1][1] >= SAFE_OPEN:
            tail -= 1
        t = safe[min(tail, len(safe) - 1)][0]
        add("safe_left_open", p, t, "дверца сейфа остаётся открытой до конца записи")

    for run in _runs(states, lambda s: s["second_person_cashier_side"] >= 0.5 and s["client_present"] < 0.5):
        after = [s for s in states if run[-1]["t"] < s["t"] <= run[-1]["t"] + ARRIVAL_SEC]
        if len(run) < 3 or any(s["client_present"] >= 0.5 for s in after):
            continue
        p = min(min(s["second_person_cashier_side"], 1 - s["client_present"]) for s in run)
        add("second_person_in_cash_zone", p, run[0]["t"], "в кассе на стороне кассира находится второй человек, клиента у стола нет")

    for run in _runs(states, lambda s: s["client_present"] < AWAY):
        if run[0] is states[0] or run[-1] is states[-1] or _span(run) < MIN_AWAY_SEC:
            continue
        i = states.index(run[-1]) + 1
        arrival = states[i : i + 3]
        cash = min(min(s["cash_on_desk"] for s in run), max(s["cash_on_desk"] for s in arrival))
        add(
            "previous_cash_on_desk",
            min(cash, max(s["client_present"] for s in arrival)),
            arrival[0]["t"],
            "предыдущий клиент ушёл, купюры остались на столе, подходит следующий клиент",
        )

    two = sorted(
        ((p["client_seated"] * p["person_standing_at_desk"], float(t)) for t, p in (detail.get("two") or {}).items()), reverse=True
    )
    if len(two) >= 2:
        add(
            "serving_two_clients",
            two[1][0],
            min(two[0][1], two[1][1]),
            "у стола сидит клиент, рядом у стола кассира стоит ещё один посетитель",
        )

    for type_id, key, text, k in [
        ("drink_in_cash_zone", "drink_or_food", "на кассовом столе или в руках кассира стакан или напиток", 2),
        ("personal_items_in_cash_zone", "personal_items_on_desk", "на кассовом столе лежат сумка или куртка", 2),
    ]:
        p, t = _persistent(states, key, k)
        add(type_id, p, t, text)
    return retime(events, raw)


LEAVE_LEAD = 3.0
AWAY_TYPES = {"left_during_service", "cash_unattended", "key_left_in_safe", "workstation_left_unlocked", "stamp_left_on_desk"}
PERSIST_KEYS = {"drink_in_cash_zone": "drink_or_food", "personal_items_in_cash_zone": "personal_items_on_desk"}


def _index(raw: list[dict], t: float) -> int:
    return max((j for j, s in enumerate(raw) if s["t"] <= t + 1e-6), default=0)


def departure(raw: list[dict], t: float) -> float:
    i = _index(raw, t)
    while i + 1 < len(raw) and raw[i]["cashier_present"] >= 0.9:
        i += 1
    while i > 0 and raw[i - 1]["cashier_present"] < 0.9:
        i -= 1
    return raw[i]["t"]


def _onset(series: list[tuple[float, float]], t: float, level: float) -> float:
    i = max((j for j, (x, _) in enumerate(series) if x <= t + 1e-6), default=0)
    while i > 0 and series[i - 1][1] >= level:
        i -= 1
    return series[i][0]


def retime(events: list[dict], raw: list[dict]) -> list[dict]:
    smoothed = smooth(raw)
    out = []
    for e in events:
        t = e["t"]
        if e["type_id"] in AWAY_TYPES or (e["type_id"] == "safe_left_open" and e["observation"].startswith("кассира нет")):
            start = max(0.0, departure(raw, t) - LEAVE_LEAD)
        elif e["type_id"] in PERSIST_KEYS:
            start = _onset([(x["t"], x[PERSIST_KEYS[e["type_id"]]]) for x in smoothed], t, 0.5)
        elif e["type_id"] == "second_person_in_cash_zone":
            start = _onset([(x["t"], x["second_person_cashier_side"]) for x in raw], t, 0.3)
        else:
            start = t
        times: list[float] = []
        for x in sorted({start, t}):
            if not times or x - times[-1] >= 0.5:
                times.append(x)
        out.append(dict(e, t=times[0], times=times))
    return out


VERIFY = {
    "drink_in_cash_zone": "на кассовом столе стоит стакан, кружка или бутылка с напитком",
    "personal_items_in_cash_zone": "на кассовом столе лежит личная сумка, рюкзак, куртка или кошелёк",
}
VERIFY_MIN = 0.3


def verify(post, image_url, frames: list[tuple[float, Image.Image]], events: list[dict]) -> dict:
    jobs, seen = [], set()
    for e in sorted(events, key=lambda e: -e["confidence"]):
        if e["type_id"] not in VERIFY or e["confidence"] < VERIFY_MIN or e["type_id"] in seen:
            continue
        seen.add(e["type_id"])
        question = VERIFY[e["type_id"]]
        near = [f for f in frames if e["t"] - 0.01 <= f[0] <= e["t"] + 3.01][:3] or frames[:1]
        for t, image in near:
            jobs.append((e["type_id"], zoom(image, (0, 0, 1, 1), 0, int(max(image.size) * UPSCALE)), question))
    with ThreadPoolExecutor(STATE_WORKERS) as pool:
        answers = list(pool.map(lambda j: ask(post, image_url, j[1], {"answer": j[2]})["answer"], jobs))
    out: dict[str, list[float]] = {}
    for (type_id, *_), p in zip(jobs, answers):
        out.setdefault(type_id, []).append(p)
    return out


def apply_verify(events: list[dict], checks: dict | None) -> list[dict]:
    if checks is None:
        return events
    result = []
    for e in events:
        ps = sorted(checks.get(e["type_id"]) or [])
        if ps:
            e = dict(e, confidence=round(min(e["confidence"], ps[len(ps) // 2]), 4))
        elif e["type_id"] in VERIFY:
            e = dict(e, confidence=round(min(e["confidence"], VERIFY_MIN), 4))
        result.append(e)
    return result
