"""Autoresearch evaluator: candidate vs champion with a noise check.

Usage: python3 scripts/autoresearch/evaluate.py candidates/<name>.py [--vs candidates/<champion>.py]

A candidate module defines
    fit_predict(train, test) -> list[str]
where train/test are dicts {"X": {encoder: ndarray}, "y": ndarray[str], "meta": list[window dict]}.
Any hyperparameter search must use only `train` (inner CV); never look at test["y"].

Only the 160 train+validation windows are loaded. Test windows/videos are never read.

Two protocols, both reported:
  cv  - leave-one-scenario-group-out over train+val (paired/related videos share a group), pooled macro F1.
  val - the frozen split: fit on train, score validation (comparable to the 5090 report).
Noise check: paired bootstrap over scenario groups (cv) of macro-F1 delta, 2000 resamples.
Verdict "win" needs cv delta CI95 lower bound > 0 and val delta >= 0.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
EXTRA = ROOT / "review/generated/autoresearch/features"  # optional extra encoders, same 160 rows
LEDGER = ROOT / "docs/autoresearch/ledger.jsonl"
LABELS = ["hand_count", "machine", "laptop", "document", "safe", "leave"]
CHAMPION = ROOT / "scripts/autoresearch/champion.txt"


def load():
    prepared = json.loads((SRC / "prepared.json").read_text())
    windows = prepared["windows"]
    assert all(w["split"] in ("train", "validation") for w in windows), "test window leaked into data"
    X = dict(np.load(SRC / "features.npz"))
    for f in sorted(EXTRA.glob("*.npz")) if EXTRA.exists() else []:
        X.update({f"{f.stem}:{k}": v for k, v in np.load(f).items()})
    assert all(len(v) == len(windows) for v in X.values())
    y = np.array([w["label"] for w in windows])
    # Scenario groups: connected components of the pair graph.
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b, _ in prepared["pairs"]:
        parent[find(a)] = find(b)
    groups = np.array([find(w["video"]) for w in windows])
    split = np.array([w["split"] for w in windows])
    return X, y, groups, split, windows


def subset(X, y, windows, idx):
    return {"X": {k: v[idx] for k, v in X.items()}, "y": y[idx], "meta": [windows[i] for i in idx]}


LABEL_KEYS = {"label", "seen", "row", "annotation_start", "annotation_end"}  # label-derived meta


def blind(test):
    """Hide labels from the candidate: no y, and no label-derived meta fields."""
    test["y"] = None
    test["meta"] = [{k: v for k, v in m.items() if k not in LABEL_KEYS} for m in test["meta"]]
    return test


def run(module_path, X, y, groups, split, windows):
    spec = importlib.util.spec_from_file_location(Path(module_path).stem, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cv = np.empty(len(y), dtype=object)
    for g in np.unique(groups):
        te, tr = np.where(groups == g)[0], np.where(groups != g)[0]
        cv[te] = mod.fit_predict(subset(X, y, windows, tr), blind(subset(X, y, windows, te)))
    tr, te = np.where(split == "train")[0], np.where(split == "validation")[0]
    val = np.array(mod.fit_predict(subset(X, y, windows, tr), blind(subset(X, y, windows, te))), dtype=object)
    return cv.astype(str), val.astype(str)


def f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--vs", help="baseline candidate (default: champion.txt)")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    base_path = a.vs or (ROOT / "scripts/autoresearch" / CHAMPION.read_text().strip())
    X, y, groups, split, windows = load()
    cand_cv, cand_val = run(a.candidate, X, y, groups, split, windows)
    base_cv, base_val = run(base_path, X, y, groups, split, windows)
    yval = y[split == "validation"]

    rng = np.random.default_rng(20261017)
    ug = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ug}
    deltas = []
    for _ in range(2000):
        idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        deltas.append(f1(y[idx], cand_cv[idx]) - f1(y[idx], base_cv[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    d_cv = f1(y, cand_cv) - f1(y, base_cv)
    d_val = f1(yval, cand_val) - f1(yval, base_val)
    verdict = "win" if lo > 0 and d_val >= 0 else ("loss" if hi < 0 else "noise")
    per_class = f1_score(y, cand_cv, labels=LABELS, average=None, zero_division=0)
    out = {
        "candidate": Path(a.candidate).name,
        "baseline": str(Path(base_path).name),
        "candidate_sha256": hashlib.sha256(Path(a.candidate).read_bytes()).hexdigest()[:12],
        "cv_f1": round(f1(y, cand_cv), 4), "base_cv_f1": round(f1(y, base_cv), 4),
        "val_f1": round(f1(yval, cand_val), 4), "base_val_f1": round(f1(yval, base_val), 4),
        "cv_delta": round(d_cv, 4), "cv_delta_ci95": [round(lo, 4), round(hi, 4)],
        "p_delta_le_0": round(float(np.mean(np.array(deltas) <= 0)), 4),
        "val_delta": round(d_val, 4),
        "cv_per_class_f1": dict(zip(LABELS, np.round(per_class, 3).tolist())),
        "verdict": verdict, "note": a.note,
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if not a.no_log:
        with LEDGER.open("a") as fh:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
