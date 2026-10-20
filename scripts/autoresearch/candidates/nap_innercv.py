"""NAP with K and C chosen by inner leave-one-scenario-group-out CV on the training fold (pre-registered).

Problem: sceneinv_nap (K=8, C=10 fixed a priori) reached CV 0.698, but its K curve is spiky (K=8 isolated).
Hypothesis (ONE evaluation, spec fixed before running): choosing K in {0,2,4,8,12,16} and C in {1,10} by
inner CV on the training fold only gives a gain over the champion that generalises without outer-fold peeking.

Each configuration is the sceneinv_nap pipeline: CLIP mean -> StandardScaler -> project out the top-K
within-class between-video directions (sceneinv_nap.nap_basis) -> balanced LR(C). K=0, C=10 == champion.
Inner groups: connected components of prepared.json 'pairs' (scenario metadata only; nothing else is read
from the file), mapped onto the training-fold videos; a video in no pair is its own group. No pair crosses
the train/val/test split, so this equals the components restricted to train videos (X-43/X-45 stay together
via train video X-44, as in the evaluator). Fallback: group = video if pairs are unavailable or < 3 groups.
Leave-one-group-out; scaler, NAP basis and LR refit on every inner-train part; predictions pooled; score =
macro F1 over the training-fold classes. Pick the best score; ties -> smaller K, then smaller C. Refit on the
whole training fold. Each call logs its choice to stderr (prefix NAP_INNERCV) for the chosen-K report.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

KS, CS = (0, 2, 4, 8, 12, 16), (1, 10)
GRID = [(k, c) for k in KS for c in CS]
_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sceneinv_nap", _here / "sceneinv_nap.py")
_nap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nap)


def _load_pairs():
    try:
        prepared = json.loads((_here.parents[2] / "review/generated/action-5090/prepared.json").read_text())
        return [(a, b) for a, b, _ in prepared["pairs"]]
    except (OSError, KeyError, ValueError):
        return []


PAIRS = _load_pairs()


def scenario_groups(videos):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b in PAIRS:
        parent[find(a)] = find(b)
    groups = np.array([find(v) for v in videos])
    return groups if len(set(groups)) >= 3 else np.asarray(videos)


def predict_configs(X, y, videos, Xt, configs):
    """Predictions on Xt per (K, C); scaler, NAP basis and LR use (X, y, videos) only."""
    sc = StandardScaler().fit(X)
    Z, Zt = sc.transform(X), sc.transform(Xt)
    V = _nap.nap_basis(Z, y, videos, max(KS))  # SVD rows are ordered, so V[:k] == nap_basis(..., k)
    out = {}
    for k, c in configs:
        Vk = V[:k]
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=20261017)
        lr.fit(Z - (Z @ Vk.T) @ Vk, y)
        out[k, c] = lr.predict(Zt - (Zt @ Vk.T) @ Vk)
    return out


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    videos = np.array([m["video"] for m in train["meta"]])
    groups = scenario_groups(videos)
    pred = {kc: np.empty(len(y), dtype=object) for kc in GRID}
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        for kc, p in predict_configs(X[tr], y[tr], videos[tr], X[te], GRID).items():
            pred[kc][te] = p
    labels = np.unique(y)
    score = {kc: f1_score(y, pred[kc].astype(str), labels=labels, average="macro", zero_division=0) for kc in GRID}
    best = max(GRID, key=lambda kc: (round(score[kc], 9), -kc[0], -kc[1]))
    log = {
        "held_out": sorted({m["video"] for m in test["meta"]}),
        "n_train": len(y), "n_inner_groups": len(np.unique(groups)),
        "K": best[0], "C": best[1], "inner_f1": round(score[best], 4),
        "grid": {f"K{k}_C{c}": round(s, 4) for (k, c), s in score.items()},
    }
    print("NAP_INNERCV " + json.dumps(log), file=sys.stderr, flush=True)
    return predict_configs(X, y, videos, test["X"]["clip"], [best])[best].tolist()
