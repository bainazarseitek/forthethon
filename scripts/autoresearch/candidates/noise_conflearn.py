"""Confident-learning filter of train windows. Pre-registered; constants fixed before running.

Problem: annotation intervals are broad and mix actions, so some train windows carry a label their content
does not match. Hypothesis: dropping the train windows least consistent with their given label, then refitting,
improves macro F1.

Model everywhere: CLIP mean -> train-only StandardScaler -> balanced LR C=10.
1. Out-of-fold probabilities on TRAIN only: GroupKFold(n_splits=min(5, #videos)) grouped by meta["video"];
   p_i = P(given label y_i | x_i) from the fold model (0 if y_i is absent from that fold's training part).
2. For each class with n_c train windows drop the k_c = min(ceil(n_c * 10 / 100), n_c - 2) windows with lowest p_i
   ("bottom 10%", same count as p <= class 10th percentile; never fewer than 2 left; ties broken by index).
3. Refit on the kept windows and predict. Test labels are never seen; nothing on disk changes.
"""
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C, PCT, MIN_KEEP = 10, 10, 2


def make():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=20261017),
    )


def given_label_proba(X, y, videos):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(videos)))).split(X, y, videos):
        model = make().fit(X[tr], y[tr])
        col = {c: j for j, c in enumerate(model.classes_)}
        proba = model.predict_proba(X[te])
        p[te] = [proba[r, col[y[i]]] if y[i] in col else 0.0 for r, i in enumerate(te)]
    return p


def keep_mask(y, p):
    keep = np.ones(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = max(0, min(math.ceil(len(idx) * PCT / 100), len(idx) - MIN_KEEP))
        keep[idx[np.argsort(p[idx], kind="stable")[:k]]] = False
    return keep


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    videos = np.array([m["video"] for m in train["meta"]])
    keep = keep_mask(y, given_label_proba(X, y, videos))
    return make().fit(X[keep], y[keep]).predict(test["X"]["clip"]).tolist()
