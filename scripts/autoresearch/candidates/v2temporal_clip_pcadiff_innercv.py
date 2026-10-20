"""CLIP mean + PCA-k of v2 per-dim mean |temporal-position diff| (PCA fit on train); k in {0,8,16,32} and C in {1,10} by inner GroupKFold (video groups) macro F1 on train only."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GRID = [(c, k) for c in (1.0, 10.0) for k in (0, 8, 16, 32)]


def motion(t):
    return np.abs(np.diff(t, axis=1)).mean(1)


def fit_pred(clip_tr, mot_tr, y, clip_te, mot_te, c, k):
    sc = StandardScaler().fit(clip_tr)
    a, b = [sc.transform(clip_tr)], [sc.transform(clip_te)]
    if k:
        m = make_pipeline(StandardScaler(), PCA(n_components=min(k, len(y) - 1), random_state=0), StandardScaler()).fit(mot_tr)
        a.append(m.transform(mot_tr)); b.append(m.transform(mot_te))
    lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=20261017).fit(np.hstack(a), y)
    return lr.predict(np.hstack(b))


def fit_predict(train, test):
    clip, mot, y = train["X"]["clip"], motion(train["X"]["v2tokens:temporal"]), train["y"]
    groups = np.array([m["video"] for m in train["meta"]])
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(clip, y, groups))
    labels = np.unique(y)
    best, best_s = (10.0, 0), -1
    for c, k in GRID:
        pred = np.empty(len(y), dtype=object)
        for tr, te in folds:
            pred[te] = fit_pred(clip[tr], mot[tr], y[tr], clip[te], mot[te], c, k)
        s = f1_score(y, pred.astype(str), labels=labels, average="macro", zero_division=0)
        if s > best_s + 1e-12:
            best, best_s = (c, k), s
    return fit_pred(clip, mot, y, test["X"]["clip"], motion(test["X"]["v2tokens:temporal"]), *best).tolist()
