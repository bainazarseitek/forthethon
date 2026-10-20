import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.detector_vlm.actions import derive, window_facts

ROOT = Path(__file__).resolve().parents[2]
TYPES = [
    "no_id_check",
    "cash_hand_to_hand",
    "cash_returned_without_act",
    "count_outside_client_view",
    "rejected_notes_unchecked",
    "stored_without_count",
    "bag_accepted_unchecked",
    "serving_two_clients",
    "cash_to_pocket",
    "cash_to_non_client",
]


def ground_truth() -> dict:
    return {v["video"]: v for v in json.loads((ROOT / "data/ground_truth.json").read_text())["videos"]}


def facts(out_dir: Path, videos_dir: Path, names: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> None:
        target = out_dir / f"{name}.json"
        if target.exists():
            return
        path = next(p for p in (videos_dir / f"{name}.mov", videos_dir / f"{name.translate(str.maketrans('XХ', 'ХX'))}.mov") if p.exists())
        started = time.monotonic()
        result = window_facts(str(path))
        result["latency_sec"] = round(time.monotonic() - started, 2)
        target.write_text(json.dumps(result, ensure_ascii=False))
        print(name, result["latency_sec"], flush=True)

    with ThreadPoolExecutor(int(os.environ.get("BENCH_VIDEOS", "1"))) as pool:
        list(pool.map(one, names or list(ground_truth())))


def score(out_dir: Path, threshold: float = 0.5, verbose: bool = False) -> None:
    gt = ground_truth()
    stats = {t: {"hit": 0, "gt": 0, "fp_videos": 0, "neg_videos": 0} for t in TYPES}
    lat = []
    for name, v in gt.items():
        f = out_dir / f"{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        lat.append(r["latency_sec"])
        events = [e for e in derive(r) if e["confidence"] >= threshold]
        for t in TYPES:
            truth = [e for e in v["events"] if e["type_id"] == t]
            pred = [e["frames"][0]["t_sec"] for e in events if e["type_id"] == t]
            if truth:
                for e in truth:
                    stats[t]["gt"] += 1
                    stats[t]["hit"] += any(abs(p - k) <= 2.0 for k in e["key_t_sec"] for p in pred)
            else:
                stats[t]["neg_videos"] += 1
                stats[t]["fp_videos"] += bool(pred)
        if verbose:
            print(
                f"{name:6} gt={[(e['type_id'], e['key_t_sec']) for e in v['events'] if e['type_id'] in TYPES]} pred={[(e['type_id'], e['confidence'], e['frames'][0]['t_sec']) for e in events]}"
            )
    print(f"threshold={threshold} videos={len(lat)} lat_max={max(lat or [0])} lat_med={sorted(lat)[len(lat) // 2] if lat else 0}")
    for t, s in stats.items():
        print(f"{t:28} recall {s['hit']}/{s['gt']}  fp_videos {s['fp_videos']}/{s['neg_videos']}")


if __name__ == "__main__":
    if sys.argv[1] == "facts":
        facts(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
    else:
        score(Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5, True)
