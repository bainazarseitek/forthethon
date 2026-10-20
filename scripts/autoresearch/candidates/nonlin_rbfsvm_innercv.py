"""Nonlinear head, candidate 1 of 2 (pre-registered; constants fixed before running).

Problem: only linear heads (and LDA/NCM) were tried; classes like machine/document may be multi-modal across scenes.
Hypothesis: a nonlinear head on train-standardized CLIP window means improves macro F1.
Spec: StandardScaler (fit on train) -> SVC(kernel="rbf", gamma="scale", class_weight="balanced").
C from {1, 10, 100} chosen by inner GroupKFold on train only (groups = meta["video"], n_splits = min(5, #videos)),
score = pooled out-of-fold macro F1 over the train labels, lower C wins ties. The scaler is refit inside every
inner fold; the final model is refit on all of train with the chosen C. Nothing else is tuned.
"""
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CS = (1, 10, 100)


def make(c):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced"))


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    groups = np.array([m["video"] for m in train["meta"]])
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups))
    best, best_s = CS[0], -1.0
    for c in CS:
        pred = np.empty(len(y), dtype=object)
        for tr, te in folds:
            pred[te] = make(c).fit(X[tr], y[tr]).predict(X[te])
        s = f1_score(y, pred.astype(str), labels=np.unique(y), average="macro", zero_division=0)
        if s > best_s:
            best, best_s = c, s
    return make(best).fit(X, y).predict(test["X"]["clip"]).tolist()
