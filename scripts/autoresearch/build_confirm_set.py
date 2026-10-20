"""Build the fresh confirmation window set "confirm-v1" (train/validation videos only).

Rows: annotation rows of TRAIN/VALIDATION videos in the *current* docs/annotations.md (parsed with
scripts/build_review.annotations, nothing written to review/generated except our own dir) that are
NOT referenced by data/action_pilot.json row_mapping. Row indices are 0-based into each clip's
annotation rows; every frozen interval is checked to match the current row at the same index by
start/end (so the current numbering equals the frozen snapshot's numbering for mapped rows).

Labels: fixed type -> class table written BEFORE any model output on these rows was seen. It uses
only the `types` field (comma separated). Semantics follow the frozen class notes: machine = a
machine episode (count / unload); safe = putting into or locking the safe; document = receiving /
inspecting / returning a document; laptop = laptop data entry or locking the workstation; leave =
the cashier is away from the workstation. Violation *states* that do not name an observable
cashier action (e.g. key_left_in_safe, safe_left_open, stored_without_count, door_left_open) map
to nothing, because the frozen set itself labels such rows inconsistently.

    hand_count                -> hand_count
    machine_count             -> machine
    machine_unload            -> machine
    cash_to_safe              -> safe
    safe_locked               -> safe
    items_to_safe             -> safe
    id_check                  -> document
    document_handover         -> document
    document_return           -> document
    data_entry                -> laptop
    workstation_locked        -> laptop
    workstation_left_unlocked -> leave
    cash_unattended           -> leave
    anything else             -> (none)

A row is in the primary set iff its types map to exactly one class. Rows whose types map to two
different classes (e.g. "machine_unload, hand_count"), map to none, or have empty types are excluded
and listed in confirm_set.json["excluded_rows"].

Windows: identical rules to the frozen set (run_action_pilot.make_windows / prepare): 2 s windows,
16 frames, 1 s stride, 0.125 s inward margin, max 8 per row (linspace subsample), frames from real
source PTS by floor lookup with the same asserts. The window/PTS code is a verbatim copy of those
functions (make_windows cannot be called directly because it asserts all six classes per split);
it is verified here by regenerating every frozen window of the involved videos from the frozen
row_mapping and comparing start/end/source_frames exactly. Any fresh window overlapping in time a
frozen window of the same video (review/generated/action-5090/prepared.json) is dropped.

Scenario groups: connected components of prepared.json pairs; unpaired videos are their own group.
Only ffprobe (timestamps) is run, only on train/validation videos. Test videos are never touched.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import subprocess  # noqa: E402

from build_review import annotations  # noqa: E402
from run_protocol_pilot import digest, save_json  # noqa: E402

OUT = ROOT / "review/generated/autoresearch/confirm-v1"
FROZEN = ROOT / "review/generated/action-5090/prepared.json"
TABLE = {
    "hand_count": "hand_count",
    "machine_count": "machine", "machine_unload": "machine",
    "cash_to_safe": "safe", "safe_locked": "safe", "items_to_safe": "safe",
    "id_check": "document", "document_handover": "document", "document_return": "document",
    "data_entry": "laptop", "workstation_locked": "laptop",
    "workstation_left_unlocked": "leave", "cash_unattended": "leave",
}
TEST = {"X-4", "X-5", "X-26", "X-27", "X-28", "X-29", "X-34", "X-35", "X-37", "X-47", "X-48"}


def starts_for(row, cfg):  # verbatim from run_action_pilot.make_windows
    low, high = row["start"] + cfg["boundary_margin"], row["end"] - cfg["boundary_margin"]
    starts = np.arange(low, high - cfg["window_seconds"] + 1e-7, cfg["stride_seconds"])
    if len(starts) > cfg["max_windows_per_row"]:
        starts = starts[np.linspace(0, len(starts) - 1, cfg["max_windows_per_row"]).round().astype(int)]
    return starts


def pts_frames(pts, w, cfg):  # verbatim from run_action_pilot.prepare
    targets = w["start"] + np.arange(cfg["num_frames"]) * cfg["window_seconds"] / cfg["num_frames"]
    indexes = np.searchsorted(pts, targets + 1e-7, side="right") - 1
    assert indexes.min() >= 0 and len(set(indexes)) == cfg["num_frames"]
    assert np.all(pts[indexes] <= targets + 1e-7) and max(targets - pts[indexes]) < .07
    assert pts[indexes[0]] >= w["annotation_start"] and pts[indexes[-1]] < w["annotation_end"]
    return indexes.tolist(), pts[indexes].tolist()


def probe(source):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)])
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in json.loads(out)["frames"]])
    assert np.all(np.diff(pts) > 0)
    return pts


def main():
    assert not OUT.exists(), "never overwrite an output dir"
    cfg = json.loads((ROOT / "data/action_pilot.json").read_text())
    frozen = json.loads(FROZEN.read_text())
    assert frozen["config"] == cfg
    clips = {c["id"]: c for c in json.loads((ROOT / "review/generated/manifest.json").read_text())["clips"]}
    ann = annotations()
    videos = cfg["splits"]["train"] + cfg["splits"]["validation"]
    assert not set(videos) & TEST
    split_of = {v: s for s in ("train", "validation") for v in cfg["splits"][s]}

    # Frozen row numbering == current numbering for every mapped row (X-8: hand [0], machine [1,2], safe [4,5]).
    assert cfg["row_mapping"]["X-8"] == {"hand_count": [0], "machine": [1, 2], "safe": [4, 5]}
    for it in frozen["intervals"]:
        r = ann[it["video"]]["rows"][it["row"]]
        assert (r["start"], r["end"]) == (it["start"], it["end"]), it
    mapped = {(v, i) for v, m in cfg["row_mapping"].items() for rows in m.values() for i in rows}

    # Groups from the frozen pair graph.
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b, _ in frozen["pairs"]:
        parent[find(a)] = find(b)

    rows, excluded, vocab = [], [], Counter()
    for v in videos:
        for i, r in enumerate(ann[v]["rows"]):
            if (v, i) in mapped:
                continue
            types = [t.strip() for t in r["types"].split(",") if t.strip()]
            vocab.update(types)
            classes = sorted({TABLE[t] for t in types if t in TABLE})
            base = dict(video=v, row=i, start=r["start"], end=r["end"], types=r["types"], kind=r["kind"])
            if not types:
                excluded.append(dict(base, reason="empty types"))
            elif len(classes) == 0:
                excluded.append(dict(base, reason="types map to no class"))
            elif len(classes) > 1 or any(t not in TABLE for t in types):
                excluded.append(dict(base, reason="types map to several classes / partly unmapped", classes=classes))
            else:
                rows.append(dict(base, label=classes[0], seen=r["seen"]))

    # Verify the copied window/PTS code against the frozen windows of the involved videos.
    involved = sorted({r["video"] for r in rows}, key=lambda v: int(v[2:]))
    pts_cache = {}
    for v in involved:
        assert digest(ROOT / clips[v]["source"]) == clips[v]["sha256"] == frozen["sources"].get(v, clips[v])["sha256"]
        pts_cache[v] = probe(ROOT / clips[v]["source"])
        for fw in [w for w in frozen["windows"] if w["video"] == v]:
            r = ann[v]["rows"][fw["row"]]
            assert any(abs(s - fw["start"]) < 1e-9 for s in starts_for(r, cfg))
            sf, sp = pts_frames(pts_cache[v], dict(start=fw["start"], annotation_start=r["start"], annotation_end=r["end"]), cfg)
            assert sf == fw["source_frames"] and np.allclose(sp, fw["source_pts"]), (v, fw["index"])
    verified = sum(w["video"] in involved for w in frozen["windows"])

    windows, dropped, too_short = [], [], []
    for r in rows:
        starts = starts_for(r, cfg)
        if not len(starts):
            too_short.append(dict(video=r["video"], row=r["row"], label=r["label"], start=r["start"], end=r["end"]))
            continue
        fz = [(w["start"], w["end"]) for w in frozen["windows"] if w["video"] == r["video"]]
        for s in starts:
            w = dict(video=r["video"], row=r["row"], split=split_of[r["video"]], label=r["label"], types=r["types"],
                     start=float(s), end=float(s + cfg["window_seconds"]), annotation_start=r["start"],
                     annotation_end=r["end"], seen=r["seen"], group=find(r["video"]))
            if any(w["start"] < b and a < w["end"] for a, b in fz):
                dropped.append(w)
                continue
            w["source_frames"], w["source_pts"] = pts_frames(pts_cache[r["video"]], w, cfg)
            w["index"] = len(windows)
            windows.append(w)

    count = lambda key: dict(sorted(Counter(key(w) for w in windows).items()))
    summary = dict(windows=len(windows), per_class=count(lambda w: w["label"]), per_video=count(lambda w: w["video"]),
                   per_group=count(lambda w: w["group"]), per_split=count(lambda w: w["split"]),
                   per_group_class=count(lambda w: f'{w["group"]}:{w["label"]}'),
                   primary_rows=len(rows), rows_too_short=len(too_short), windows_dropped_overlap=len(dropped),
                   excluded_rows=len(excluded), frozen_windows_reverified=verified)
    OUT.mkdir(parents=True)
    save_json(OUT / "confirm_set.json", dict(
        name="confirm-v1", type_table=TABLE, type_vocabulary=dict(vocab.most_common()), summary=summary,
        windows=windows, primary_rows=rows, rows_too_short=too_short, dropped_overlap=dropped, excluded_rows=excluded,
        sources={v: dict(file=clips[v]["source"], sha256=clips[v]["sha256"]) for v in involved},
        annotations_sha256=digest(ROOT / "docs/annotations.md"), frozen_prepared_sha256=digest(FROZEN),
        config_sha256=digest(ROOT / "data/action_pilot.json"), script_sha256=digest(__file__), test_touched=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
