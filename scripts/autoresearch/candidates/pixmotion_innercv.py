"""CLIP window mean + 38-d pixel motion; per-block train-only scaler; W in {0,1,3,10} x C in {0.1,1,10}
chosen by inner GroupKFold (by meta video, 5 folds) macro F1 on train only. W=0 = CLIP-only fallback."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

WS, CS, SEED = [0.0, 1.0, 3.0, 10.0], [0.1, 1, 10], 20261017


def _fit_pred(Xc, Xp, y, Xc2, Xp2, w, C):
    sc, sp = StandardScaler().fit(Xc), StandardScaler().fit(Xp)
    f = lambda a, b: np.hstack([sc.transform(a), w * sp.transform(b)])
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
    return m.fit(f(Xc, Xp), y).predict(f(Xc2, Xp2))


def fit_predict(train, test):
    Xc, Xp, y = train["X"]["clip"], train["X"]["pixmotion:feat"], train["y"]
    groups = np.array([w["video"] for w in train["meta"]])
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(y, y, groups))

    def score(w, C):
        p = np.empty(len(y), dtype=object)
        for tr, va in folds:
            p[va] = _fit_pred(Xc[tr], Xp[tr], y[tr], Xc[va], Xp[va], w, C)
        return f1_score(y, p.astype(str), average="macro", zero_division=0)

    w, C = max(((w, C) for w in WS for C in CS), key=lambda wc: score(*wc))
    return _fit_pred(Xc, Xp, y, test["X"]["clip"], test["X"]["pixmotion:feat"], w, C)
