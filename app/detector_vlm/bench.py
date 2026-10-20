import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.detector_vlm.detector import MODE, _event, always, analyze, calibrate, load, merge
from app.detector_vlm.state import apply_verify, state_events

ROOT = Path(__file__).resolve().parents[2]


def run(out_dir: Path, videos_dir: Path, names: list[str], workers: int) -> None:
    gt = {v["video"]: v for v in json.loads((ROOT / "data/ground_truth.json").read_text())["videos"]}
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> None:
        target = out_dir / f"{name}.json"
        if target.exists():
            return
        path = next(p for p in (videos_dir / f"{name}.mov", videos_dir / f"{name.translate(str.maketrans('XХ', 'ХX'))}.mov") if p.exists())
        started = time.monotonic()
        try:
            result = analyze(str(path))
        except Exception as exc:
            result = {"events": [], "error": repr(exc)}
        result["latency_sec"] = round(time.monotonic() - started, 2)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, names or list(gt)))
    score(out_dir, gt)


def rederive(r: dict) -> list[dict]:
    report = r.get("report", {})
    events = [
        _event(e["type_id"], e["confidence"], e["t"], e["observation"], e.get("times"))
        for e in apply_verify(state_events(report.get("states") or [], report.get("detail")), report.get("verify"))
    ]
    events = merge(
        [calibrate(e) for e in events + [e for e in r["events"] if e.get("source") in ("actions", "unseen")] + report.get("door", [])]
    )
    return always(events, report.get("states") or []) if MODE == "always" else events


def score(out_dir: Path, gt: dict, threshold: float = 0.5, again: bool = False) -> None:
    rows, fp, fn, type_hit, type_total, lat, errors = [], [], [], 0, 0, [], []
    for name, v in gt.items():
        f = out_dir / f"{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if again:
            r["events"] = rederive(r)
        lat.append(r.get("latency_sec", 0))
        ev = [e for e in r["events"] if e["confidence"] >= (0.7 if e["type_id"] == "other" else threshold)]
        pred = "violation" if ev else "clean"
        truth = "violation" if v["events"] else "clean"
        if pred == "violation" and truth == "clean":
            fp.append(name)
        if pred == "clean" and truth == "violation":
            fn.append(name)
        keys = [k for e in v["events"] for k in e["key_t_sec"]]
        if keys:
            errors += [min(abs(e["frames"][0]["t_sec"] - k) for k in keys) for e in ev]
        gt_types = {e["type_id"] for e in v["events"]}
        if gt_types:
            type_total += 1
            type_hit += bool(gt_types & {e["type_id"] for e in ev})
        rows.append(
            f"{name:6} gt={sorted(gt_types)} pred={[(e['type_id'], round(e['confidence'], 2), e['frames'][0]['t_sec']) for e in ev]}"
        )
    print("\n".join(rows))
    n = len(rows)
    print(
        f"n={n} acc={(n - len(fp) - len(fn)) / max(n, 1):.3f} FP={fp} FN={fn} type_hit={type_hit}/{type_total} tc_med={sorted(errors)[len(errors) // 2] if errors else None:} lat_max={max(lat or [0])} lat_med={sorted(lat)[len(lat) // 2] if lat else 0}"
    )


if __name__ == "__main__":
    if sys.argv[1] == "rescore":
        gt = {v["video"]: v for v in json.loads((ROOT / "data/ground_truth.json").read_text())["videos"]}
        score(Path(sys.argv[2]), gt, float(sys.argv[3]) if len(sys.argv) > 3 else 0.5, again=True)
        sys.exit()
    load()
    run(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[4:], int(sys.argv[3]))
