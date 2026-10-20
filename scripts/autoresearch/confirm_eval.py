"""Pre-registered confirmation of the champion on confirm-v1 (fresh windows no experiment has used).

Protocol (written before this script was run; run exactly once; does NOT touch ledger.jsonl):
  Models:   candidates/siglip_late_clip.py (champion) vs candidates/baseline_clip.py (baseline), imported
            unchanged, called via fit_predict(train, test).
  Train:    the 160 frozen train+val windows (review/generated/action-5090/prepared.json) with
            features.npz 'clip' and autoresearch features/siglip.npz 'frames' (key 'siglip:frames').
  Test:     the 21 confirm-v1 windows (confirm_set.json) with confirm_features.npz ('clip',
            'siglip:frames'); test dict has y=None and meta without label-derived fields.
  Primary:  for every fresh window, train on the frozen windows EXCLUDING the fresh window's scenario
            group (connected component of prepared.json pairs; unpaired videos are their own group).
            Implemented per fresh group (one fit per group; fits are deterministic, so identical to
            per-window fits).
  Secondary: train on all 160 frozen windows.
  Metrics:  macro F1 over the classes present in the fresh set (fixed list, zero_division=0;
            predictions of absent classes count as errors), accuracy, per-class F1, 6x6 confusion
            (rows = truth, cols = prediction, frozen class order), champion minus baseline.
  Noise:    paired bootstrap over the fresh set's scenario groups, 2000 resamples, seed 20261017,
            of delta macro F1 (same fixed class list) and delta accuracy; CI95 = 2.5/97.5 percentiles.
  Decision (primary only): "confirmed" if delta macro F1 > 0 and CI95 lower > 0; "consistent" if
            delta > 0 but CI includes 0; "not confirmed" if delta <= 0.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
CONF = ROOT / "review/generated/autoresearch/confirm-v1"
LABELS = ["hand_count", "machine", "laptop", "document", "safe", "leave"]
LABEL_KEYS = {"label", "seen", "row", "annotation_start", "annotation_end", "types", "group", "source_pts",
              "source_frames", "index"}
SEED = 20261017
MODELS = {"champion": "siglip_late_clip.py", "baseline": "baseline_clip.py"}


def load_module(name):
    path = ROOT / "scripts/autoresearch/candidates" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    out = CONF / "confirm_eval.json"
    assert not out.exists(), "pre-registered: run once"
    prepared = json.loads((SRC / "prepared.json").read_text())
    fw = prepared["windows"]
    assert len(fw) == 160 and all(w["split"] in ("train", "validation") for w in fw)
    Xf = {"clip": np.load(SRC / "features.npz")["clip"],
          "siglip:frames": np.load(ROOT / "review/generated/autoresearch/features/siglip.npz")["frames"]}
    assert all(len(v) == 160 for v in Xf.values())
    yf = np.array([w["label"] for w in fw])
    cs = json.loads((CONF / "confirm_set.json").read_text())
    cw = cs["windows"]
    Xc = dict(np.load(CONF / "confirm_features.npz"))
    assert all(len(v) == len(cw) for v in Xc.values()) and set(Xc) == set(Xf)
    yc = np.array([w["label"] for w in cw])

    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b, _ in prepared["pairs"]:
        parent[find(a)] = find(b)
    gf = np.array([find(w["video"]) for w in fw])
    gc = np.array([find(w["video"]) for w in cw])
    assert all(g == w["group"] for g, w in zip(gc, cw))

    def train_dict(idx):
        return {"X": {k: v[idx] for k, v in Xf.items()}, "y": yf[idx], "meta": [fw[i] for i in idx]}

    def test_dict(idx):
        return {"X": {k: v[idx] for k, v in Xc.items()}, "y": None,
                "meta": [{k: v for k, v in cw[i].items() if k not in LABEL_KEYS} for i in idx]}

    mods = {k: load_module(v) for k, v in MODELS.items()}
    preds = {"primary": {}, "secondary": {}}
    for name, mod in mods.items():
        p = np.empty(len(cw), dtype=object)
        excluded = {}
        for g in np.unique(gc):
            te, tr = np.where(gc == g)[0], np.where(gf != g)[0]
            excluded[str(g)] = int((gf == g).sum())
            p[te] = mod.fit_predict(train_dict(tr), test_dict(te))
        preds["primary"][name] = p.astype(str)
        preds["secondary"][name] = np.asarray(mod.fit_predict(train_dict(np.arange(160)), test_dict(np.arange(len(cw))))).astype(str)
    present = [c for c in LABELS if c in set(yc)]

    def mf1(y, p):
        return f1_score(y, p, labels=present, average="macro", zero_division=0)

    ug = np.unique(gc)
    members = {g: np.where(gc == g)[0] for g in ug}
    result = dict(protocol=__doc__, classes_present=present, n=len(cw), groups={str(g): int(len(members[g])) for g in ug},
                  frozen_windows_excluded_per_group=excluded, models=MODELS, protocols={})
    for proto, pr in preds.items():
        rows = {}
        for name, p in pr.items():
            rows[name] = dict(macro_f1=round(mf1(yc, p), 4), accuracy=round(accuracy_score(yc, p), 4),
                              correct=int((yc == p).sum()),
                              per_class_f1=dict(zip(present, np.round(f1_score(yc, p, labels=present, average=None, zero_division=0), 3).tolist())),
                              confusion=confusion_matrix(yc, p, labels=LABELS).tolist())
        rng = np.random.default_rng(SEED)
        dF, dA = [], []
        c, b = pr["champion"], pr["baseline"]
        for _ in range(2000):
            idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
            dF.append(mf1(yc[idx], c[idx]) - mf1(yc[idx], b[idx]))
            dA.append(accuracy_score(yc[idx], c[idx]) - accuracy_score(yc[idx], b[idx]))
        d_f1 = mf1(yc, c) - mf1(yc, b)
        d_acc = accuracy_score(yc, c) - accuracy_score(yc, b)
        lo, hi = np.percentile(dF, [2.5, 97.5])
        rows["delta"] = dict(macro_f1=round(d_f1, 4), macro_f1_ci95=[round(lo, 4), round(hi, 4)],
                             p_delta_le_0=round(float(np.mean(np.array(dF) <= 0)), 4),
                             accuracy=round(d_acc, 4), accuracy_ci95=np.round(np.percentile(dA, [2.5, 97.5]), 4).tolist())
        rows["predictions"] = [dict(video=w["video"], start=w["start"], truth=w["label"], champion=c[i], baseline=b[i])
                               for i, w in enumerate(cw)]
        result["protocols"][proto] = rows
    d = result["protocols"]["primary"]["delta"]
    result["decision"] = ("confirmed" if d["macro_f1"] > 0 and d["macro_f1_ci95"][0] > 0 else
                          "consistent" if d["macro_f1"] > 0 else "not confirmed")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ("protocol", "protocols")}, indent=1))
    for proto, rows in result["protocols"].items():
        print("==", proto)
        for k in ("champion", "baseline", "delta"):
            print(k, json.dumps({kk: vv for kk, vv in rows[k].items() if kk != "confusion"}))
        for k in ("champion", "baseline"):
            print(k, "confusion", rows[k]["confusion"])
        for r in rows["predictions"]:
            print(r)


if __name__ == "__main__":
    main()
