import json
import re
import sys

src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()

NUM = r"\d+(?:\.\d+)?"


def parse_t(cell):
    bold = [float(x) for x in re.findall(r"\*\*(" + NUM + r")", cell)]
    bold += [float(x) for x in re.findall(r"–\*\*(" + NUM + r")", cell)]
    nums = [float(x) for x in re.findall(NUM, cell)]
    return min(nums), max(nums), sorted(set(bold)), "**" in cell


def parse_label(line):
    if "без нарушения" in line:
        return None
    m = re.search(r"(" + NUM + r") с", line)
    return float(m.group(1))


sections = re.split(r"^## ", text, flags=re.M)[1:]
videos = {}
for sec in sections:
    header = sec.splitlines()[0].strip()
    vid = header.split()[0]
    lab = re.search(r"^- label: (.*)$", sec, re.M)
    if not lab:
        continue
    label_t = parse_label(lab.group(1))
    rows = []
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[1] not in ("ok", "viol", "viol-unlabeled", "?", ""):
            continue
        if not re.search(r"\d", cells[0]) or cells[0] == "t":
            continue
        t0, t1, bold_t, has_bold = parse_t(cells[0])
        km = re.search(r"ключевой момент (" + NUM + r") с", cells[3])
        if km:
            bold_t, has_bold = [float(km.group(1))], True
        rows.append({"t0": t0, "t1": t1, "bold": bold_t, "has_bold": has_bold,
                     "kind": cells[1], "type_id": cells[2], "seen": cells[3], "rule": cells[4]})
    videos[vid] = {"label_t": label_t, "rows": rows, "header": header, "sec": sec}

videos["Х-48"] = dict(videos["Х-5"], header="Х-48")

result = []
for vid, d in videos.items():
    events = {}
    for r in d["rows"]:
        if r["kind"] not in ("viol", "viol-unlabeled") or not r["type_id"]:
            continue
        labeled = r["kind"] == "viol"
        key = (r["type_id"], labeled, None if labeled else r["t0"])
        e = events.setdefault(key, {"type_id": r["type_id"], "labeled": labeled, "t_start_sec": r["t0"],
                                    "t_end_sec": r["t1"], "key_t_sec": []})
        e["t_start_sec"] = min(e["t_start_sec"], r["t0"])
        e["t_end_sec"] = max(e["t_end_sec"], r["t1"])
        e["key_t_sec"] += r["bold"] if r["bold"] else ([r["t0"]] if r["has_bold"] else [])
    evs = sorted(events.values(), key=lambda e: e["t_start_sec"])
    for e in evs:
        e["key_t_sec"] = sorted(set(e["key_t_sec"])) or [e["t_start_sec"]]
    lt = d["label_t"]
    primary = None
    labeled_evs = [e for e in evs if e["labeled"]]
    if lt is not None and labeled_evs:
        def dist(e):
            if e["t_start_sec"] - 1.5 <= lt <= e["t_end_sec"] + 1.5:
                return 0 if lt in [round(k) for k in e["key_t_sec"]] or any(abs(k - lt) <= 1 for k in e["key_t_sec"]) else 0.5
            return min(abs(lt - e["t_start_sec"]), abs(lt - e["t_end_sec"]))
        primary = min(labeled_evs, key=dist)
    for e in evs:
        e["primary"] = e is primary
    ok_steps = sorted({t.strip() for r in d["rows"] if r["kind"] == "ok" for t in r["type_id"].split(",") if t.strip()})
    pair = re.search(r"чист\w* пар\w*(?: к)?:? \*{0,2}((?:[XХ]-\d+(?:, )?)+)", d["sec"])
    item = {
        "video": vid,
        "file": f"assets/videos/{vid}.mov",
        "verdict": "clean" if lt is None else "violation",
        "label_t_sec": lt,
        "events": evs,
        "ok_steps": ok_steps,
    }
    if pair:
        item["pair"] = [p.strip() for p in pair.group(1).split(",") if p.strip()]
    if vid in ("Х-47", "Х-48"):
        item["same_file_as"] = {"Х-47": "Х-4", "Х-48": "Х-5"}[vid]
    result.append(item)

bad = [r["video"] for r in result if r["verdict"] == "violation" and not any(e["primary"] for e in r["events"])]
assert not bad, bad
assert all(not any(e["labeled"] for e in r["events"]) for r in result if r["verdict"] == "clean")
assert all("," not in s and s for r in result for s in r["ok_steps"] + r.get("pair", []))

doc = {
    "source": "docs/annotations.md",
    "notes": [
        "label_t_sec: метка организаторов (assets/open_labels.xlsx). primary: событие, которое совпадает с меткой",
        "labeled: false значит нарушение есть на видео, но организаторы его не разметили. в eval не штраф за находку и не пропуск",
        "для обучения и few-shot все события равноправные нарушения. labeled и verdict только для сверки с метками организаторов, модели их не показывать",
        "в одном видео может быть несколько событий. при eval засчитывать попадание в любое событие видео",
        "Х-47 побайтно совпадает с Х-4, Х-48 с Х-5. не класть копии одновременно в few-shot и в тест",
        "время в секундах по pts кадра",
    ],
    "videos": result,
}
json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(len(result), "videos")
for r in result:
    print(r["video"], r["verdict"], r["label_t_sec"], [(e["type_id"], e["t_start_sec"], e["t_end_sec"], e["key_t_sec"], e["primary"]) for e in r["events"]])
