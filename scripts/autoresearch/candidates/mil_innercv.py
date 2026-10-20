"""MIL-style sub-window LR; chunk size {1,4} and C {0.01,0.1,1,10} chosen by inner GroupKFold (by video, all
sub-samples of a window stay together) macro F1 on train only; mean log-prob aggregation."""
import importlib.util, os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

_spec = importlib.util.spec_from_file_location("mil_frame_lr", os.path.join(os.path.dirname(__file__), "mil_frame_lr.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
GRID = [(k, c) for k in (1, 4) for c in (0.01, 0.1, 1.0, 10.0)]


def fit_predict(train, test):
    X, y = train["X"]["clipframes:frames"], np.asarray(train["y"])
    groups = np.array([m["video"] for m in train["meta"]])
    labels = np.unique(y)
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups))
    best, best_s = (4, 10.0), -1
    for k, c in GRID:
        pred = np.empty(len(y), dtype=object)
        for tr, te in folds:
            pred[te] = _m.fit_apply(X[tr], y[tr], X[te], k, c)
        s = f1_score(y, pred.astype(str), labels=labels, average="macro", zero_division=0)
        if s > best_s:
            best, best_s = (k, c), s
    return _m.fit_apply(X, y, test["X"]["clipframes:frames"], *best).tolist()
