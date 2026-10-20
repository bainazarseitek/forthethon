import base64
import io
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait

import av

URL = os.environ.get("VLM_URL", "http://127.0.0.1:10301/v1/chat/completions")
TIMEOUT = float(os.environ.get("BOXES_TIMEOUT", "30"))
BUDGET = float(os.environ.get("BOXES_BUDGET", "10"))
WORKERS = int(os.environ.get("BOXES_WORKERS", "2"))
MAX_AREA = 0.6
MIN_SIDE = 0.01

ROLES = "Кассир сидит за столом лицом к экрану ноутбука, обычно ближе к камере и спиной к ней. Клиент сидит или стоит по другую сторону стола лицом к кассиру. "
PEOPLE = ("кассир", "клиент", "второй клиент", "посторонний")
CASHIER = ("кассир", "сотрудник банка за столом лицом к экрану ноутбука, обычно ближе к камере и спиной к ней")
CLIENT = ("клиент", "посетитель по другую сторону стола лицом к кассиру и камере")
CASH = (
    "купюры",
    "купюры или пачки денег на столе, даже небольшие: светлые или цветные бумажные прямоугольники. ноутбук, телефон, клавиатура и счётная машинка не считаются",
)
CASH_HANDS = ("купюры в руках", "руки кассира и клиента над столом, между которыми передаются купюры, вместе с купюрами")
SAFE = ("сейф", "светло-серый металлический сейф, высокий ящик на тумбе у стены. чёрная счётная машинка на столе это не сейф")
OUTSIDER = (
    "посторонний",
    "человек внутри кабинета кассы, который не кассир: стоит у стола, у сейфа или проходит по кабинету. люди за стеклянной стеной в коридоре не считаются",
)
FIXED_LABELS = {"desk": "стол кассира", "laptop": "ноутбук", "safe": "сейф", "door": "дверь"}

TARGETS = {
    "cash_unattended": ([CASH], "desk"),
    "previous_cash_on_desk": ([CASH], "desk"),
    "cash_mixed_between_clients": ([CASH], "desk"),
    "cash_returned_without_act": ([CASH_HANDS], "desk"),
    "rejected_notes_unchecked": ([CASH], "desk"),
    "stamp_left_on_desk": ([("штампы", "штампы и печати на столе: небольшие цветные цилиндры или футляры")], "desk"),
    "workstation_left_unlocked": (
        [("ноутбук", "ноутбук на столе кассира вместе с экраном и клавиатурой. счётная машинка и телефон это не ноутбук")],
        "laptop",
    ),
    "key_left_in_safe": ([SAFE + ("ключ в сейфе",)], "safe"),
    "safe_left_open": ([SAFE + ("сейф открыт",)], "safe"),
    "stored_without_count": ([CASH, SAFE], "safe"),
    "door_left_open": ([("дверь", "стеклянная входная дверь в кассу вместе с открытой створкой", "дверь открыта")], "door"),
    "no_id_check": ([CLIENT, CASHIER], None),
    "serving_two_clients": ([CLIENT, ("второй клиент", "ещё один посетитель у стола, кроме кассира и первого клиента"), CASHIER], None),
    "cash_hand_to_hand": ([CASH_HANDS, CLIENT, CASHIER], None),
    "cash_to_non_client": ([CASH_HANDS], None),
    "cash_to_pocket": ([("руки кассира", "руки кассира с купюрами")], None),
    "count_outside_client_view": ([("купюры у кассира", "руки кассира с купюрами, которые он пересчитывает")], None),
    "collection_in_view_of_clients": ([CASH, CLIENT], None),
    "second_person_in_cash_zone": ([OUTSIDER], None),
    "left_during_service": ([CLIENT], "desk"),
    "drink_in_cash_zone": ([("стакан", "стакан, чашка или бутылка с напитком")], None),
    "personal_phone_in_cash_zone": ([("телефон", "смартфон в руке кассира или на столе. стационарный телефон не считается")], None),
    "personal_items_in_cash_zone": ([("личные вещи", "личные вещи кассира в кассовой зоне: сумка, пакет, смартфон, косметичка")], None),
    "bag_accepted_unchecked": ([("сумка", "сумка или пакет, принятые от клиента")], None),
}


def _prompt(labels):
    lines = "\n".join(f"{t[0]}: {t[1]}." for t in labels)
    names = ", ".join(t[0] for t in labels)
    roles = ROLES if any(t[0] in PEOPLE for t in labels) else ""
    return (
        "Это кадр с потолочной камеры в кассе банка. Найди на кадре объекты и верни их рамки "
        "в координатах 0-1000 как JSON-список "
        '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}, ...]. '
        f"{roles}Рамка должна плотно охватывать только сам объект.\n"
        f"{lines}\n"
        f"Верни рамки только для меток: {names}. Для каждой метки одна рамка. "
        "Если объекта нет в кадре, не включай его. Ответь только JSON."
    )


def _post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())


def _img(im):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=90)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}


def _box(x1, y1, x2, y2, label):
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
    w, h = x2 - x1, y2 - y1
    if w < MIN_SIDE or h < MIN_SIDE or w * h > MAX_AREA:
        return None
    return {"x": round(x1, 4), "y": round(y1, 4), "w": round(w, 4), "h": round(h, 4), "label": label}


def parse(text, labels):
    names = [t[0] for t in labels]
    shown = {t[0]: t[2] if len(t) > 2 else t[0] for t in labels}
    out = {}
    for m in re.finditer(r'"label"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*\[([^\]]*)\]', text):
        name = m.group(1).strip().lower()
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(2))
        if name not in names or name in out or len(nums) != 4:
            continue
        x1, y1, x2, y2 = (float(v) / 1000 for v in nums)
        box = _box(x1, y1, x2, y2, shown[name])
        if box:
            out[name] = box
    return [out[n] for n in names if n in out]


def ground(frame, labels):
    body = {
        "model": "vlm",
        "temperature": 0,
        "max_tokens": 120,
        "messages": [{"role": "user", "content": [_img(frame), {"type": "text", "text": _prompt(labels)}]}],
    }
    return parse(_post(body)["choices"][0]["message"]["content"], labels)


def _fixed(grounding, key, label, size):
    bb = (grounding or {}).get(key)
    if not bb or not size:
        return []
    W, H = size
    box = _box(bb[0] / W, bb[1] / H, bb[2] / W, bb[3] / H, label)
    return [box] if box else []


def _upright(frame):
    image = frame.to_image()
    rotation = int(frame.rotation or 0) % 360
    return image.rotate(rotation, expand=True) if rotation else image


def frames_at(path, times):
    out = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        for t in sorted(set(times)):
            container.seek(int(max(0.0, t - 1.0) / stream.time_base), stream=stream, backward=True)
            best = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                ft = float(frame.time)
                if best is None or abs(ft - t) < abs(best[0] - t):
                    best = (ft, frame)
                if ft >= t:
                    break
            if best:
                out[t] = _upright(best[1])
    return out


def _roles(boxes, labels):
    by = {b["label"]: b for b in boxes}
    client, cashier = by.get("клиент"), by.get("кассир")
    if client and cashier and client["y"] + client["h"] > cashier["y"] + cashier["h"]:
        client["label"], cashier["label"] = "кассир", "клиент"
    order = [t[2] if len(t) > 2 else t[0] for t in labels]
    return sorted(boxes, key=lambda b: order.index(b["label"]))


def _frame_boxes(image, type_id, grounding, size):
    labels, fixed = TARGETS[type_id]
    fixed_label = FIXED_LABELS.get(fixed)
    if fixed in ("laptop",) and grounding and grounding.get(fixed):
        return _fixed(grounding, fixed, fixed_label, size)
    try:
        boxes = [b for b in _roles(ground(image, labels), labels) if b["label"] != "кассир" or type_id == "no_id_check"]
    except Exception:
        boxes = []
    return boxes[:2] or _fixed(grounding, fixed, fixed_label, size)


LAST = {}


def add_boxes(path, events, grounding=None, budget=None):
    budget = BUDGET if budget is None else min(BUDGET, budget)
    LAST.clear()
    LAST.update(budget=round(budget, 1), requests=0, done=0, error=None)
    try:
        jobs = [(e, f) for e in events if e.get("type_id") in TARGETS for f in e.get("frames", [])[:3]]
        if not jobs:
            return events
        images = frames_at(path, [f["t_sec"] for _, f in jobs])
        size = next(iter(images.values())).size if images else None
        pool = ThreadPoolExecutor(WORKERS)
        futures = {
            pool.submit(_frame_boxes, images[f["t_sec"]], e["type_id"], grounding, size): (e, f) for e, f in jobs if f["t_sec"] in images
        }
        done, _ = wait(futures, timeout=budget)
        LAST.update(requests=len(futures), done=len(done))
        pool.shutdown(wait=False, cancel_futures=True)
        for fut, (e, f) in futures.items():
            if fut in done:
                try:
                    f["boxes"] = fut.result()
                except Exception as exc:
                    LAST["error"] = repr(exc)
            if not f.get("boxes"):
                fixed = TARGETS[e["type_id"]][1]
                f["boxes"] = _fixed(grounding, fixed, FIXED_LABELS.get(fixed), size)
    except Exception as exc:
        LAST["error"] = repr(exc)
    return events


if __name__ == "__main__":
    assert parse('[{"label": "сейф", "bbox_2d": [100, 200, 300, 400]}, {"label": "x", "bbox_2d": [1,2,3,4]}]', [SAFE]) == [
        {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2, "label": "сейф"}
    ]
    assert parse('[{"label": "сейф", "bbox_2d": [0, 0, 1000, 1000]}]', [SAFE]) == []
    assert parse("garbage", [SAFE]) == []
    assert [
        b["label"] for b in _roles([{"label": "клиент", "y": 0.4, "h": 0.3}, {"label": "кассир", "y": 0.3, "h": 0.2}], [CLIENT, CASHIER])
    ] == ["клиент", "кассир"]
    t = time.monotonic()
    ev = [{"type_id": "safe_left_open", "frames": [{"t_sec": 9.0, "boxes": []}]}]
    print(add_boxes("/workspace/backend/assets/videos/Х-20.mov", ev), round(time.monotonic() - t, 2))
