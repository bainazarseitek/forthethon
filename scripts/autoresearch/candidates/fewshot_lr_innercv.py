"""Standardized CLIP -> balanced LR; C and PCA dims chosen by inner leave-one-video-group-out CV (macro F1) on train only."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GRID = [(c, d) for c in (0.001, 0.01, 0.1, 1.0, 10.0) for d in (16, 64, None)]


def make(c, d):
    steps = [StandardScaler()] + ([PCA(n_components=d, random_state=0)] if d else [])
    return make_pipeline(*steps, LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=20261017))


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    groups = np.array([m["video"] for m in train["meta"]])
    labels = np.unique(y)
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups))
    best, best_s = GRID[-1], -1
    for c, d in GRID:
        pred = np.empty(len(y), dtype=object)
        for tr, te in folds:
            dd = min(d, len(tr) - 1) if d else None
            pred[te] = make(c, dd).fit(X[tr], y[tr]).predict(X[te])
        s = f1_score(y, pred.astype(str), labels=labels, average="macro", zero_division=0)
        if s > best_s:
            best, best_s = (c, d), s
    return make(*best).fit(X, y).predict(test["X"]["clip"]).tolist()
