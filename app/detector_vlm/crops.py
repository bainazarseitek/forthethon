import base64
import io
import json
import math
import os
import re
import urllib.request

from PIL import Image

URL = os.environ.get("VLM_URL", "http://127.0.0.1:10301/v1/chat/completions")
OBJECTS = ("desk", "laptop", "safe", "door")
GROUND_PROMPT = (
    "Это кадр с потолочной камеры в кассе банка. Найди на кадре объекты и верни их рамки "
    "в координатах 0-1000 как JSON-список "
    '[{"label": "desk", "bbox_2d": [x1, y1, x2, y2]}, ...]. '
    "desk: деревянная столешница стола кассира в середине кадра, на которой стоят ноутбук и счётная машинка. "
    "Белая столешница на переднем плане внизу кадра это не стол кассира, её не выделяй. "
    "laptop: ноутбук на столе кассира, вместе с экраном и клавиатурой. Счётная машинка и стационарный телефон это не ноутбук. "
    "safe: серый металлический сейф на тумбе у стены, весь корпус с дверцей. "
    "door: стеклянная входная дверь в кассу. "
    "Если объекта нет в кадре, не включай его. Ответь только JSON."
)
PAD = 0.15
TARGET_TOKENS = 1200
MAX_SCALE = 8.0


def _post(body, timeout=120):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _img(im):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=90)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}


def ground(frame):
    body = {
        "model": "vlm",
        "temperature": 0,
        "max_tokens": 400,
        "messages": [{"role": "user", "content": [_img(frame), {"type": "text", "text": GROUND_PROMPT}]}],
    }
    text = _post(body)["choices"][0]["message"]["content"]
    items = []
    for m in re.finditer(r'"label"\s*:\s*"(\w+)"\s*,\s*"bbox_2d"\s*:\s*\[([^\]]*)\]', text):
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(2))
        items.append({"label": m.group(1), "bbox_2d": nums})
    W, H = frame.size
    out = {}
    for it in items:
        lab, bb = it.get("label"), it.get("bbox_2d")
        if lab not in OBJECTS or lab in out or not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bb)
        if x2 <= x1 or y2 <= y1:
            continue
        out[lab] = (x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000)
    return out


def crop(frame, box, pad=PAD, target_tokens=TARGET_TOKENS, max_scale=MAX_SCALE):
    W, H = frame.size
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    im = frame.crop((int(max(0, x1 - dx)), int(max(0, y1 - dy)), int(min(W, x2 + dx)), int(min(H, y2 + dy))))
    s = min(max_scale, math.sqrt(target_tokens * 1024 / (im.width * im.height)))
    return im.resize((max(32, round(im.width * s)), max(32, round(im.height * s))), Image.LANCZOS)


def ask(crop_image, question):
    body = {
        "model": "vlm",
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 10,
        "messages": [
            {"role": "user", "content": [_img(crop_image), {"type": "text", "text": question + "\nОтветь одним словом: да или нет."}]}
        ],
    }
    tl = _post(body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    y = sum(math.exp(t["logprob"]) for t in tl if t["token"].strip().lower() in ("да", "yes"))
    n = sum(math.exp(t["logprob"]) for t in tl if t["token"].strip().lower() in ("нет", "no"))
    return y / max(y + n, 1e-9)


QUESTIONS = {
    "stamps_v9": ("desk", 8.0, 0.5, "На столе кассира лежат штампы (цилиндры в футлярах) или журнал в обложке?"),
    "cashier_absent": (
        "full",
        8.0,
        0.5,
        "В кабинете кассы сейчас нет ни одного человека: кассира не видно ни за столом, ни у сейфа, ни у счётной машинки, ни у двери? Люди за стеклянной стеной в коридоре не считаются.",
    ),
    "desk_items": (
        "desk",
        8.0,
        0.5,
        "Кроме ноутбука, стационарного телефона, счётных машинок и проводов, на столешнице лежит ещё какой-нибудь предмет (купюры, пачка денег, сумка, бумаги)?",
    ),
    "stamps_journal": (
        "desk",
        8.0,
        0.3,
        "На столе стоят штампы: небольшие цветные цилиндры в футлярах (например бирюзовый и тёмно-зелёный), или лежит закрытая книга-журнал в тёмной обложке? Ноутбук, телефон, счётные машинки, купюры и тканевая сумка не считаются.",
    ),
    "drink_food": ("desk", 8.0, 0.5, "На столе кассира стоит стакан, чашка или бутылка с напитком, или лежит еда?"),
    "phone_in_hand": (
        "desk",
        8.0,
        0.5,
        "Кассир держит в руке смартфон: плоский чёрный блестящий прямоугольник с экраном? Купюры и пачки денег светлые или розовые и бумажные, это не смартфон. Карточка, документ и стационарный телефон на столе тоже не считаются.",
    ),
    "key_in_safe": ("safe", 8.0, 0.5, "В замке дверцы серого сейфа торчит ключ?"),
    "safe_open": (
        "safe",
        3.0,
        0.5,
        "Серый металлический сейф стоит на тумбе у стены. Его дверца открыта: створка отведена в сторону или видна внутренняя полость сейфа? Если сейф выглядит цельным закрытым ящиком, ответ нет.",
    ),
    "laptop_open_app": (
        "laptop",
        8.0,
        0.5,
        "Экран ноутбука на столе показывает открытое белое окно программы с таблицей или текстом? Если на экране синяя или цветная картинка-заставка, экран тёмный или крышка закрыта, ответ нет.",
    ),
    "client_seated": ("full", 8.0, 0.6, "На стуле для посетителя по другую сторону стола, напротив кассира, сидит человек?"),
    "person_standing_at_desk": (
        "full",
        8.0,
        0.6,
        "Внутри кабинета вплотную у стола кассира стоит человек (стоит, а не сидит)? Кассир не считается. Люди за стеклянной стеной и дверью в коридоре не считаются.",
    ),
}


def fact(frame, boxes, name):
    target, max_scale, _, text = QUESTIONS[name]
    if target == "full":
        image = crop(frame, (0, 0, frame.width, frame.height), pad=0, max_scale=max_scale)
    else:
        image = crop(frame, boxes[target], max_scale=max_scale)
    return ask(image, text)


def two_clients(frame, boxes):
    return fact(frame, boxes, "client_seated") * fact(frame, boxes, "person_standing_at_desk")
