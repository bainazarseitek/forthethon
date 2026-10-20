import base64
import http.client
import io
import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import av
from PIL import Image

from app.detector_vlm.unseen import collection_in_view

VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:10301/v1/chat/completions")
FPS = float(os.environ.get("ACT_FPS", "2"))
WINDOW = float(os.environ.get("ACT_WINDOW", "3"))
STRIDE = float(os.environ.get("ACT_STRIDE", "1.5"))
SIDE = int(os.environ.get("ACT_SIDE", "848"))
WIN_SIDE = int(os.environ.get("ACT_WIN_SIDE", "640"))
WORKERS = int(os.environ.get("ACT_WORKERS", "24"))
TIMEOUT = float(os.environ.get("ACT_TIMEOUT", "120"))
CARD_STEP = float(os.environ.get("ACT_CARD_STEP", "2"))
CARD_SCALE = float(os.environ.get("ACT_CARD_SCALE", "1.5"))

INTRO = (
    "Отрывок 3 секунды с камеры под потолком в кассе банка. За столом сидит кассир, напротив через стол может сидеть клиент. "
    "На столе ноутбук, телефон и чёрная счётная машинка для купюр. Рядом серый металлический сейф. Смотри внимательно на руки. "
    "Для каждого пункта ответь true, только если это видно в отрывке."
)

FACTS = {
    "client": "напротив кассира через стол сидит или стоит другой человек",
    "client_gives": "человек напротив протягивает руку над столом и отдаёт кассиру или кладёт перед ним небольшой предмет или бумаги",
    "machine": "кассир кладёт бумаги в чёрную счётную машинку, достаёт их из неё, или машинка прогоняет бумаги",
    "to_client": "кассир протягивает купюры человеку напротив, и тот берёт их прямо из руки кассира, не со стола",
    "safe": "кассир открывает серый сейф или кладёт в него что-то",
    "bag_in": "человек напротив передаёт кассиру тканевую сумку или мешок или кладёт её на стол перед кассиром",
    "bag_open": "кассир вскрывает тканевую сумку или достаёт из неё пачки денег",
    "second_person": "кроме человека, сидящего напротив кассира, у стола стоит ещё один человек, и кассир берёт у него предмет",
    "to_pocket": "кассир кладёт купюры в свой карман, рукав, сумку или под стол",
    "to_other": "кассир передаёт деньги работнику банка или постороннему, а не человеку, сидящему напротив",
}

CARD_FACTS = {
    "card": "на столе или в руке кассира или клиента видна небольшая белая карточка (удостоверение), отдельно от купюр",
}


def _prompt(intro: str, facts: dict[str, str]) -> str:
    return intro + "\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items())


def _schema(facts: dict[str, str]) -> dict:
    return {"type": "object", "properties": {k: {"type": "boolean"} for k in facts}, "required": list(facts)}


def _upright(frame: av.VideoFrame) -> Image.Image:
    image = frame.to_image()
    rotation = int(getattr(frame, "rotation", 0) or 0) % 360
    if rotation:
        image = image.rotate(rotation, expand=True)
    image.thumbnail((SIDE, SIDE))
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if os.environ.get("VLM_HFLIP") == "1" else image


def sample(path: str, fps: float = FPS) -> list[tuple[float, Image.Image]]:
    out: list[tuple[float, Image.Image]] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.time)
            if t + 1e-6 >= len(out) / fps:
                out.append((t, _upright(frame)))
    return out


def encode_mp4(frames: list[Image.Image], fps: float) -> bytes:
    buffer = io.BytesIO()
    w, h = frames[0].size
    w, h = w - w % 2, h - h % 2
    with av.open(buffer, mode="w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(image.convert("RGB").crop((0, 0, w, h)))):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buffer.getvalue()


def _post(body: dict) -> dict:
    data = json.dumps(body).encode()
    deadline = time.monotonic() + float(os.environ.get("ACT_WAIT", "600"))
    while True:
        try:
            request = urllib.request.Request(VLM_URL, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read())
        except (OSError, http.client.HTTPException) as exc:
            if getattr(exc, "code", 500) < 500 or time.monotonic() > deadline:
                raise
            time.sleep(3)


def _probs(choice: dict, facts: dict[str, str]) -> dict[str, float]:
    text, out = "", {}
    for token in (choice.get("logprobs") or {}).get("content") or []:
        word = token["token"].strip()
        if word in ("true", "false"):
            m = re.search(r'"(\w+)"\s*:\s*$', text)
            if m and m.group(1) in facts:
                out[m.group(1)] = sum(math.exp(c["logprob"]) for c in token["top_logprobs"] if c["token"].strip() == "true")
        text += token["token"]
    parsed = json.loads(choice["message"]["content"])
    return {k: round(out.get(k, float(bool(parsed.get(k)))), 4) for k in facts}


def _ask(media: dict, text: str, facts: dict[str, str]) -> dict[str, float]:
    body = {
        "model": "vlm",
        "temperature": 0,
        "max_tokens": 300,
        "logprobs": True,
        "top_logprobs": 5,
        "messages": [{"role": "user", "content": [media, {"type": "text", "text": text}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "facts", "schema": _schema(facts)}},
    }
    return _probs(_post(body)["choices"][0], facts)


def ask_window(frames: list[Image.Image]) -> dict[str, float]:
    frames = [im.resize((round(im.width * WIN_SIDE / max(im.size)), round(im.height * WIN_SIDE / max(im.size)))) for im in frames]
    video = base64.b64encode(encode_mp4(frames, FPS)).decode("ascii")
    return _ask({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video}"}}, _prompt(INTRO, FACTS), FACTS)


def ask_card(image: Image.Image) -> dict[str, float]:
    image = image.resize((round(image.width * CARD_SCALE), round(image.height * CARD_SCALE)))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return _ask(
        {"type": "image_url", "image_url": {"url": url}},
        _prompt("Кадр с камеры под потолком в кассе банка. Ответь true, только если это ясно видно.", CARD_FACTS),
        CARD_FACTS,
    )


def _mid(w: dict) -> float:
    return (w["t0"] + w["t1"]) / 2


def sessions(windows: list[dict]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start, gap = None, 0
    for i, w in enumerate(windows):
        if w["client"] >= 0.5:
            if start is None:
                start = i
            gap, last = 0, i
        elif start is not None:
            gap += 1
            if gap > 1:
                out.append((start, last))
                start = None
    if start is not None:
        out.append((start, last))
    return out


def intake(windows: list[dict], a: int, b: int) -> tuple[int, int, int] | None:
    m = next((i for i in range(a, b + 1) if windows[i]["machine"] >= 0.5), None)
    if m is None:
        return None
    gives = [i for i in range(a, m + 1) if windows[i]["client_gives"] >= 0.5]
    last = max(range(a, m + 1), key=lambda i: windows[i]["client_gives"])
    first = gives[0] if gives else last
    return m, first, last


def _second(values: list[float]) -> float:
    return sorted(values)[-2] if len(values) > 1 else 0.0


def bag_score(windows: list[dict], a: int, b: int) -> float:
    return _second([w["bag_in"] * w["client"] for w in windows[a : b + 1]])


def card_times(windows: list[dict]) -> list[float]:
    times: set[float] = set()

    def span(t: float, end: float, cap: int) -> None:
        k = 0
        while t <= end and k < cap:
            times.add(round(t, 2))
            t += CARD_STEP
            k += 1

    for a, b in sessions(windows):
        found = intake(windows, a, b)
        if found:
            m, first, _ = found
            span(windows[a]["t0"], min(windows[first]["t0"] - 0.1, windows[a]["t0"] + CARD_STEP), 2)
            span(windows[m]["t1"], windows[b]["t1"] + CARD_STEP, 6)
        if bag_score(windows, a, b) >= 0.2:
            span(windows[a]["t0"], windows[b]["t1"], 10)
    return sorted(times)


def window_facts(path: str, fine: bool = True) -> dict:
    grid = sample(path)
    if not grid:
        raise ValueError("no decodable frames")
    end = grid[-1][0]
    spans = []
    for i in range(max(1, math.ceil((end - WINDOW) / STRIDE) + 1)):
        picked = [(t, im) for t, im in grid if i * STRIDE <= t < i * STRIDE + WINDOW]
        if len(picked) >= 2:
            spans.append(picked)

    def window(i: int) -> dict:
        picked = spans[i]
        return {"t0": round(picked[0][0], 3), "t1": round(picked[-1][0], 3), **ask_window([im for _, im in picked])}

    def card(t: float) -> dict:
        ft, image = min(grid, key=lambda g: abs(g[0] - t))
        return {"t": round(ft, 3), **ask_card(image)}

    def busy(w: dict) -> bool:
        return w["client_gives"] >= 0.3 or w["bag_in"] >= 0.3 or w["second_person"] >= 0.2 or w["to_client"] >= 0.1

    with ThreadPoolExecutor(WORKERS) as pool:
        coarse = dict(zip(range(0, len(spans), 2), pool.map(window, range(0, len(spans), 2))))
        fill = [i for i in range(1, len(spans), 2) if fine and any(busy(coarse[j]) for j in (i - 1, i + 1) if j in coarse)]
        coarse.update(zip(fill, pool.map(window, fill)))
        windows = [coarse[i] for i in sorted(coarse)]
        cards = list(pool.map(card, card_times(windows)))
    return {"windows": windows, "cards": cards}


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


def _top2(values: list[float]) -> float:
    values = sorted(values, reverse=True)[:2]
    return sum(values) / len(values) if values else 0.0


def derive(facts: dict) -> list[dict]:
    windows, cards = facts["windows"], facts["cards"]
    events: list[dict] = []

    def add(type_id: str, p: float, w: dict, t: float, observation: str) -> None:
        events.append(_event(type_id, p, w["t0"], w["t1"], t, observation))

    def card_between(t0: float, t1: float) -> float:
        return _top2([c["card"] for c in cards if t0 <= c["t"] <= t1])

    for a, b in sessions(windows):
        found = intake(windows, a, b)
        bag = bag_score(windows, a, b)
        if found:
            m, first, g = found
            base = max([c["card"] for c in cards if windows[a]["t0"] - 0.5 <= c["t"] < windows[first]["t0"]] or [0.0])
            card = max(0.0, card_between(windows[m]["t1"], windows[b]["t1"] + CARD_STEP) - base)
            p = windows[g]["client_gives"] * max(w["machine"] for w in windows[m : m + 3]) * (1 - card)
            if bag < 0.5:
                add(
                    "no_id_check",
                    p,
                    windows[first],
                    windows[first]["t1"],
                    "клиент кладёт купюры на стол, кассир берёт их и начинает пересчёт; удостоверение или карточка на столе не видны",
                )
            after = [i for i in range(b + 1, min(len(windows), b + 6)) if windows[i]["client"] < 0.5]
            if after:
                i = max(after, key=lambda k: windows[k]["machine"] * (1 - windows[k]["client"]))
                p = windows[i]["machine"] * (1 - windows[i]["client"]) * windows[g]["client_gives"]
                add(
                    "count_outside_client_view",
                    p,
                    windows[i],
                    _mid(windows[b]),
                    "клиент уходит от стола, кассир продолжает работать с его купюрами на счётной машинке",
                )
        if bag >= 0.2:
            k = max(range(a, b + 1), key=lambda i: windows[i]["bag_in"] * windows[i]["client_gives"])
            p = bag * (1 - card_between(windows[a]["t0"], windows[b]["t1"]))
            add(
                "bag_accepted_unchecked",
                p,
                windows[k],
                _mid(windows[k]),
                "второе лицо передаёт кассиру сумку, документ или карточку перед этим кассиру не показывает",
            )

    spans = sessions(windows)
    for (a1, b1), (a2, _) in zip(spans, spans[1:]):
        gives = max(w["client_gives"] for w in windows[a1 : b1 + 1])
        stored = max(w["safe"] for w in windows[a1 : a2 + 1])
        add(
            "previous_cash_on_desk",
            (1 - stored) ** 2 * gives,
            windows[a2],
            _mid(windows[a2]),
            "подходит следующий клиент, купюры предыдущего клиента не убраны в сейф и остаются на столе",
        )

    opened = [i for i, w in enumerate(windows) if w["bag_open"] >= 0.5]
    runs = [i for i, w in enumerate(windows) if w["safe"] >= 0.5 and (i == 0 or windows[i - 1]["safe"] < 0.5)]
    late = [i for i in runs if opened and i > opened[0]]
    if late and len(opened) >= 3:
        s = late[-1]
        counted = _top2([w["machine"] for w in windows[opened[0] :]])
        p = max(w["safe"] for w in windows[s : s + 4]) * max(windows[i]["bag_open"] for i in opened) * (1 - counted)
        add(
            "stored_without_count",
            p,
            windows[s],
            windows[s]["t0"],
            "кассир достаёт пачки из вскрытой сумки и убирает их в сейф, счётная машинка не используется",
        )

    for i, w in enumerate(windows):
        p = w["second_person"] * w["client"] * w["client_gives"]
        if p >= 0.05:
            add("serving_two_clients", p, w, _mid(w), "кассир берёт предмет у второго человека, пока первый клиент сидит у стола")
        for key, type_id, text in [
            ("to_client", "cash_hand_to_hand", "кассир передаёт купюры клиенту из руки в руку над столом, не кладя их на стол"),
            ("to_pocket", "cash_to_pocket", "кассир убирает купюры в карман, сумку или под стол"),
            ("to_other", "cash_to_non_client", "кассир передаёт деньги человеку, который не является клиентом у стола"),
        ]:
            if w[key] >= 0.2:
                add(type_id, w[key], w, _mid(w), text)

    best: list[dict] = []
    for e in sorted(events, key=lambda e: -e["confidence"]):
        t = e["frames"][0]["t_sec"]
        if e["confidence"] >= 0.05 and not any(x["type_id"] == e["type_id"] and abs(x["frames"][0]["t_sec"] - t) < 6 for x in best):
            best.append(e)
    return sorted(best, key=lambda e: e["t_start_sec"])


DOOR_INTRO = (
    "Отрывок 3 секунды с камеры под потолком в кассе банка. На столе ноутбук и чёрная счётная машинка для купюр. "
    "Для каждого пункта ответь true, только если это видно в отрывке."
)
DOOR_FACTS = {
    "closes_door": "кассир оборачивается к стеклянной входной двери и закрывает её за собой (тянет створку, дверь захлопывается)",
    "door_open": "стеклянная входная дверь в кассу стоит открытой: створка отведена от проёма",
    "enters": "кассир входит в кассу через стеклянную дверь",
}


def door_after_entry(path: str, entry: float) -> list[dict]:
    grid = [(t, im) for t, im in sample(path) if t < entry + 13]
    spans, t = [], 0.0
    while t + 2 <= entry + 10:
        picked = [im for tt, im in grid if t <= tt < t + WINDOW]
        if len(picked) >= 2:
            spans.append(picked)
        t += STRIDE
    if not spans:
        return []
    with ThreadPoolExecutor(WORKERS) as pool:
        closes = [r["closes_door"] for r in pool.map(lambda fr: _door_window(fr), spans)]
    w = {"t0": entry, "t1": entry + 1}
    return [
        _event(
            "door_left_open", 1 - max(closes), w["t0"], w["t1"], entry + 1, "кассир входит в кассу и не закрывает за собой стеклянную дверь"
        )
    ]


def _door_window(frames: list[Image.Image]) -> dict[str, float]:
    frames = [im.resize((round(im.width * WIN_SIDE / max(im.size)), round(im.height * WIN_SIDE / max(im.size)))) for im in frames]
    video = base64.b64encode(encode_mp4(frames, FPS)).decode("ascii")
    return _ask({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video}"}}, _prompt(DOOR_INTRO, DOOR_FACTS), DOOR_FACTS)


def detect_actions(path: str, fine: bool = True) -> list[dict]:
    facts = window_facts(path, fine)
    events = derive(facts)
    try:
        events += collection_in_view(facts["windows"])
    except Exception:
        pass
    return events
