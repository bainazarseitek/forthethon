import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
LOOKALIKE = str.maketrans("XХ", "ХX")


def load_ground_truth(path: Path) -> list[dict]:
    videos = json.loads(path.read_text(encoding="utf-8"))["videos"]
    by_name = {v["video"]: dict(v, events=list(v["events"])) for v in videos}
    merged = []
    for v in videos:
        original = v.get("same_file_as")
        if original and original in by_name:
            target = by_name[original]
            target["events"] += v["events"]
            if v["verdict"] == "violation":
                target["verdict"] = "violation"
            target.setdefault("copies", []).append(v["video"])
        else:
            merged.append(by_name[v["video"]])
    return merged


def resolve(entry: dict, videos_dir: Path | None) -> Path | None:
    name = Path(entry["file"]).name
    candidates = [ROOT / entry["file"]]
    if videos_dir:
        candidates += [videos_dir / name, videos_dir / name.translate(LOOKALIKE)]
    return next((p for p in candidates if p.exists()), None)


def run_one(client: httpx.Client, url: str, key: str, path: Path, timeout: float) -> dict:
    started = time.monotonic()
    with path.open("rb") as f:
        r = client.post(f"{url}/api/v1/analyze", headers={"X-API-Key": key}, files={"file": (path.name, f, "video/quicktime")})
    if r.status_code != 200:
        return {"status": "upload_error", "error": f"{r.status_code} {r.text[:200]}", "latency_sec": time.monotonic() - started}
    job_id = r.json()["job_id"]
    while time.monotonic() - started < timeout:
        body = client.get(f"{url}/api/v1/jobs/{job_id}", headers={"X-API-Key": key}).json()
        if body.get("status") != "processing":
            body["latency_sec"] = time.monotonic() - started
            return body
        time.sleep(1)
    return {"job_id": job_id, "status": "client_timeout", "latency_sec": time.monotonic() - started}


def type_id_of(violation: dict, names: dict[str, str]) -> str:
    return violation.get("type_id") or names.get(violation.get("type", ""), "other")


def predicted_verdict(result: dict, threshold: float | None = None) -> str | None:
    if result.get("status") != "done":
        return None
    violations = result["violations"]
    if threshold is not None:
        violations = [v for v in violations if v.get("confidence", 1) >= threshold]
    return "violation" if violations else "clean"


def summarize(rows: list[dict], names: dict[str, str]) -> dict:
    done = [r for r in rows if r["result"].get("status") == "done"]
    correct = [r for r in rows if predicted_verdict(r["result"]) == r["gt"]["verdict"]]
    fp = [r["video"] for r in done if r["gt"]["verdict"] == "clean" and predicted_verdict(r["result"]) == "violation"]
    fn = [r["video"] for r in done if r["gt"]["verdict"] == "violation" and predicted_verdict(r["result"]) == "clean"]
    fp_unlabeled = [r["video"] for r in done if r["video"] in fp and any(not e["labeled"] for e in r["gt"]["events"])]

    type_hits, type_total, errors = 0, 0, []
    for r in done:
        preds = [(v["timecode_sec"], type_id_of(v, names)) for v in r["result"]["violations"]]
        if not preds:
            continue
        for event in (e for e in r["gt"]["events"] if e["labeled"]):
            type_total += 1
            key = event["key_t_sec"][0]
            nearest = min(preds, key=lambda p: abs(p[0] - key))
            type_hits += nearest[1] == event["type_id"]
        keys = [k for e in r["gt"]["events"] for k in e["key_t_sec"]]
        if keys:
            errors += [min(abs(t - k) for k in keys) for t, _ in preds]

    latencies = [r["result"]["latency_sec"] for r in rows]
    return {
        "videos": len(rows),
        "done": len(done),
        "not_done": [(r["video"], r["result"].get("status"), r["result"].get("error")) for r in rows if r not in done],
        "verdict_accuracy": round(len(correct) / len(rows), 3) if rows else None,
        "false_positives": fp,
        "false_positives_on_unlabeled_events": fp_unlabeled,
        "false_negatives": fn,
        "type_accuracy_labeled": round(type_hits / type_total, 3) if type_total else None,
        "type_labeled_events": type_total,
        "timecode_error_sec": {
            "p50": round(statistics.median(errors), 3) if errors else None,
            "mean": round(statistics.mean(errors), 3) if errors else None,
            "max": round(max(errors), 3) if errors else None,
        },
        "latency_sec": {
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
            "over_300": sum(t > 300 for t in latencies),
        },
    }


def sweep(rows: list[dict]) -> list[dict]:
    out = []
    for step in range(1, 20):
        threshold = step / 20
        verdicts = [(predicted_verdict(r["result"], threshold), r["gt"]["verdict"]) for r in rows]
        out.append(
            {
                "threshold": threshold,
                "accuracy": round(sum(p == g for p, g in verdicts) / len(verdicts), 3),
                "fp": sum(p == "violation" and g == "clean" for p, g in verdicts),
                "fn": sum(p == "clean" and g == "violation" for p, g in verdicts),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "data" / "ground_truth.json")
    parser.add_argument("--rules", type=Path, default=ROOT / "data" / "rules.json")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "assets" / "videos")
    parser.add_argument("--timeout", type=float, default=330)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    names = {t["name"]: t["type_id"] for t in json.loads(args.rules.read_text(encoding="utf-8"))["types"]}
    entries, missing = [], []
    for gt in load_ground_truth(args.ground_truth):
        path = resolve(gt, args.videos_dir)
        (entries if path else missing).append((gt, path))
    if missing:
        print(f"skipping {len(missing)} videos not found locally", file=sys.stderr)
    if not entries:
        print("no videos found", file=sys.stderr)
        return 1

    url = args.url.rstrip("/")
    with httpx.Client(timeout=120) as client, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [(gt, pool.submit(run_one, client, url, args.key, path, args.timeout)) for gt, path in entries]
        rows = [{"video": gt["video"], "gt": gt, "result": f.result()} for gt, f in futures]

    report = summarize(rows, names)
    if args.sweep:
        report["threshold_sweep"] = sweep(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        slim = [
            {"video": r["video"], "gt_verdict": r["gt"]["verdict"], **{k: v for k, v in r["result"].items() if k != "violations"},
             "violations": [{k: v for k, v in x.items() if k != "evidence_frames"} for x in r["result"].get("violations", [])]}
            for r in rows
        ]
        args.out.write_text(json.dumps({"report": report, "rows": slim}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
