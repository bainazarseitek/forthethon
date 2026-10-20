"""Late fusion over CLIP/VideoMAEv2/VideoMAE heads; encoder subset and C chosen by inner GroupKFold (by meta video) on train only."""
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ENCS = ["clip", "videomaev2", "videomae"]
SUBSETS = [s for r in (1, 2, 3) for s in combinations(ENCS, r) if "clip" in s]
CS = [0.1, 1, 10]


def _probs(Xtr, ytr, Xte, C, classes):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=20261017))
    m.fit(Xtr, ytr)
    out = np.zeros((len(Xte), len(classes)))
    out[:, [list(classes).index(c) for c in m.classes_]] = m.predict_proba(Xte)
    return out


def fit_predict(train, test):
    y = train["y"]
    classes = np.unique(y)
    groups = np.array([w["video"] for w in train["meta"]])
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(y, y, groups))
    # out-of-fold probabilities per (encoder, C), then score each subset by averaging
    oof = {}
    for e in ENCS:
        for C in CS:
            P = np.zeros((len(y), len(classes)))
            for tr, va in folds:
                P[va] = _probs(train["X"][e][tr], y[tr], train["X"][e][va], C, classes)
            oof[e, C] = P
    best = max(((s, C) for s in SUBSETS for C in CS),
               key=lambda sc: f1_score(y, classes[np.mean([oof[e, sc[1]] for e in sc[0]], 0).argmax(1)],
                                       average="macro", zero_division=0))
    s, C = best
    P = np.mean([_probs(train["X"][e], y, test["X"][e], C, classes) for e in s], 0)
    return classes[P.argmax(1)]
