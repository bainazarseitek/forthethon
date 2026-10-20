import importlib.util

from app.config import ROOT

spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
eval_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_script)


def test_byte_copies_counted_once():
    entries = eval_script.load_ground_truth(ROOT / "data" / "ground_truth.json")
    names = [e["video"] for e in entries]
    assert "Х-47" not in names and "Х-48" not in names
    assert len(names) == 40
    x4 = next(e for e in entries if e["video"] == "Х-4")
    assert {e["type_id"] for e in x4["events"]} == {"previous_cash_on_desk", "cash_mixed_between_clients"}


def test_resolves_latin_and_cyrillic_names(tmp_path):
    (tmp_path / "X-3.mov").write_bytes(b"x")
    assert eval_script.resolve({"file": "assets/videos/Х-3.mov"}, tmp_path) == tmp_path / "X-3.mov"


def test_summary_metrics():
    gt_violation = {"verdict": "violation", "events": [{"type_id": "cash_unattended", "labeled": True, "key_t_sec": [10.0]}]}
    gt_clean_unlabeled = {"verdict": "clean", "events": [{"type_id": "no_id_check", "labeled": False, "key_t_sec": [8.0]}]}
    gt_clean = {"verdict": "clean", "events": []}
    hit = {"status": "done", "latency_sec": 4, "violations": [{"type_id": "cash_unattended", "timecode_sec": 10.5, "confidence": 0.9}]}
    rows = [
        {"video": "a", "gt": gt_violation, "result": hit},
        {"video": "b", "gt": gt_clean_unlabeled, "result": {**hit, "latency_sec": 10}},
        {"video": "c", "gt": gt_clean, "result": {"status": "done", "latency_sec": 2, "violations": []}},
        {"video": "d", "gt": gt_violation, "result": {"status": "failed", "error": "timeout", "latency_sec": 301}},
    ]
    report = eval_script.summarize(rows, {})
    assert report["verdict_accuracy"] == 0.5
    assert report["false_positives"] == ["b"]
    assert report["false_positives_on_unlabeled_events"] == ["b"]
    assert report["false_negatives"] == []
    assert report["not_done"] == [("d", "failed", "timeout")]
    assert report["type_accuracy_labeled"] == 1.0
    assert report["timecode_error_sec"]["max"] == 2.5
    assert report["latency_sec"]["over_300"] == 1
    assert eval_script.sweep(rows)[-1]["fp"] == 0
