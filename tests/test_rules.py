import json

import pytest

from app import detector_stub
from app.config import ROOT
from app.frames import frame_times
from app.pipeline import Rules

RULES = Rules.load(ROOT / "data" / "rules.json")
RAW = json.loads((ROOT / "data" / "rules.json").read_text(encoding="utf-8"))


def event(type_id: str, closest: str | None = None) -> dict:
    return {"type_id": type_id, "closest_type_id": closest, "observation": "кассир кладёт купюры на стол.", "frames": []}


@pytest.mark.parametrize("type_id", [t["type_id"] for t in RAW["types"]])
def test_every_type_renders_citation(type_id):
    entry, text, rule = RULES.describe(event(type_id))
    assert entry["type_id"] == type_id
    assert text.startswith("Кассир кладёт купюры на стол. ")
    assert ".." not in text
    assert entry["rule"].rstrip(".") in text
    for ref in entry["refs"]:
        if ref["doc"] == "BANK":
            assert "внутренние правила банка" in text
        else:
            assert RULES.documents[ref["doc"]]["title"] in text
            if ref["clause"] and type_id != "other":
                assert ref["clause"] in text
    assert rule["document"]
    assert "fallback" not in json.dumps(rule, ensure_ascii=False)


def test_other_has_no_clause():
    _, text, rule = RULES.describe(event("other"))
    assert text.endswith(f"({RULES.documents['S1']['title']}).")
    assert rule == {"document": RULES.documents["S1"]["title"], "clause": None, "url": RULES.documents["S1"]["url"]}


def test_other_with_closest_adds_analogy():
    _, text, _ = RULES.describe(event("other", "cash_unattended"))
    assert "По аналогии: " in text
    assert "п. 20 ч. 2" in text


def test_unknown_type_is_other():
    entry, text, _ = RULES.describe(event("made_up", "cash_unattended"))
    assert entry["type_id"] == "other"
    assert "По аналогии" not in text


def test_primary_ref_skips_bank():
    rule = RULES.primary_ref([{"doc": "BANK", "clause": None}, {"doc": "IS53", "clause": "п. 71 пп. 9"}])
    assert rule["document"] == RULES.documents["IS53"]["title"]
    assert rule["clause"] == "п. 71 пп. 9"
    rule = RULES.primary_ref([{"doc": "BANK", "clause": None}])
    assert rule == {"document": "внутренние правила банка", "clause": None, "url": None}


def test_catalog_cites_only_rk_acts():
    assert all(ref["doc"] != "BANK" for t in RULES.types.values() for ref in t["refs"])


def test_stub_is_deterministic_and_uses_catalog(videos):
    first = detector_stub.analyze(str(videos["vfr"]))
    assert first == detector_stub.analyze(str(videos["vfr"]))
    times = {round(t, 3) for t in frame_times(videos["vfr"])}
    assert first["events"]
    for e in first["events"]:
        assert e["type_id"] in RULES.types
        assert e["closest_type_id"] is None or (e["type_id"] == "other" and e["closest_type_id"] in RULES.types)
        assert 0 <= e["confidence"] <= 1
        assert e["t_start_sec"] <= e["t_end_sec"]
        assert 1 <= len(e["frames"]) <= 3
        assert all(f["t_sec"] in times for f in e["frames"])


def test_detector_order_is_kept(videos):
    from app.config import Settings
    from app.pipeline import run

    settings = Settings()
    events = [
        {"type_id": "no_id_check", "t_start_sec": 1.0, "t_end_sec": 1.2, "confidence": 0.9, "observation": "a", "frames": [{"t_sec": 1.0, "boxes": []}]},
        {"type_id": "cash_unattended", "t_start_sec": 0.2, "t_end_sec": 0.4, "confidence": 0.9, "observation": "b", "frames": [{"t_sec": 0.2, "boxes": []}]},
    ]
    result, _ = run(str(videos["vfr"]), lambda _: {"events": events}, Rules.load(settings.rules_path), settings)
    assert [v["type_id"] for v in result["violations"]] == ["no_id_check", "cash_unattended"]


def test_observation_is_capitalized():
    _, text, _ = RULES.describe({**event("cash_unattended"), "observation": " кассира нет в кассе."})
    assert text.startswith("Кассира нет в кассе. ")
