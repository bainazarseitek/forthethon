"""Drop-one-scene-group bagging of the champion head (pre-registered, one evaluation, nothing tuned).

Problem: the champion (CLIP mean -> train-only StandardScaler -> balanced LR C=10) overfits the scene
identity of the few groups holding rare classes (hand_count: 2 train scenario groups).
Hypothesis: averaging heads that each never saw one train scenario group reduces reliance on any
single scene and improves cross-scene macro F1.

Spec (fixed before running):
- Train scenario groups = connected components of the full prepared.json pair graph (same union-find
  as evaluate.py, so X-43/X-45 stay one group via window-less X-44), looked up for the videos in
  `train`. Only the `pairs` key is read from disk; no windows or labels.
- One head per train group g: the champion pipeline (StandardScaler + LogisticRegression(C=10,
  class_weight="balanced", max_iter=3000, random_state=20261017)) fit on train windows outside g.
- Prediction = argmax of the unweighted mean of the heads' predict_proba over the train classes;
  a head missing a class gives it probability 0 (sklearn BaggingClassifier convention).
- No full-data head. Fallback with < 2 train groups: the plain champion.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PREPARED = Path(__file__).resolve().parents[3] / "review/generated/action-5090/prepared.json"
PAIRS = json.loads(PREPARED.read_text())["pairs"]


def scene_groups(videos):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b, _ in PAIRS:
        parent[find(a)] = find(b)
    return np.array([find(v) for v in videos])


def head():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017),
    )


def fit_predict(train, test):
    X, y, Xt = train["X"]["clip"], train["y"], test["X"]["clip"]
    groups = scene_groups([m["video"] for m in train["meta"]])
    classes, ug = np.unique(y), np.unique(groups)
    if len(ug) < 2:
        return head().fit(X, y).predict(Xt).tolist()
    P = np.zeros((len(Xt), len(classes)))
    for g in ug:
        keep = groups != g
        m = head().fit(X[keep], y[keep])
        P[:, np.searchsorted(classes, m.classes_)] += m.predict_proba(Xt)  # sum: same argmax as mean
    return classes[P.argmax(1)].tolist()
