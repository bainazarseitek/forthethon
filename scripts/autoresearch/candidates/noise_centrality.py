"""Down-weight boundary (likely mixed) train windows. Pre-registered; constants fixed before running.

Problem: annotation intervals are broad and mix actions; windows near interval boundaries are more likely mixed.
Hypothesis: down-weighting train windows by closeness to their interval boundary improves macro F1.

CLIP mean -> train-only StandardScaler (unweighted, as baseline) -> balanced LR C=10 with sample weights
  gap_i = min(start_i - annotation_start_i, annotation_end_i - end_i), clipped at 0
  w_i   = 0.3 + 0.7 * gap_i / max(gap_j : j in the same annotation interval)
i.e. the most central window of each interval gets 1, a window touching the boundary 0.3, linear in between;
a single-window interval gets 1. Weights are rescaled to mean 1 within each class, so balanced class totals and
effective regularisation equal the baseline's: only within-class emphasis moves. Uses train meta/labels only;
test meta is ignored; nothing on disk changes.
"""
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

C, W_MIN = 10, 0.3


def centrality_weights(meta, y):
    gap = np.array([max(0.0, min(m["start"] - m["annotation_start"], m["annotation_end"] - m["end"])) for m in meta])
    intervals = defaultdict(list)
    for i, m in enumerate(meta):
        intervals[(m["video"], m["annotation_start"], m["annotation_end"], str(y[i]))].append(i)
    w = np.ones(len(y))
    for idx in intervals.values():
        top = gap[idx].max()
        if top > 0:
            w[idx] = W_MIN + (1 - W_MIN) * gap[idx] / top
    for c in np.unique(y):
        w[y == c] /= w[y == c].mean()
    return w


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=20261017)
    lr.fit(sc.transform(X), y, sample_weight=centrality_weights(train["meta"], y))
    return lr.predict(sc.transform(test["X"]["clip"])).tolist()
