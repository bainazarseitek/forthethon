"""(video,class)-pair weighting + nuisance-attribute projection (pre-registered, constants fixed before running).

CLIP mean -> train-only StandardScaler -> project out top K=8 within-class between-video directions
(as sceneinv_nap) -> LR C=10 with per-(video,class) equalising sample weights (as sceneinv_weights).
"""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

K, C = 8, 10
_here = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_w, _n = _load("sceneinv_weights"), _load("sceneinv_nap")


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    videos = np.array([m["video"] for m in train["meta"]])
    sc = StandardScaler().fit(X)
    Z, Zt = sc.transform(X), sc.transform(test["X"]["clip"])
    V = _n.nap_basis(Z, y, videos, K)
    P = lambda A: A - (A @ V.T) @ V
    lr = LogisticRegression(C=C, max_iter=3000, random_state=20261017)
    lr.fit(P(Z), y, sample_weight=_w.pair_weights(y, list(videos)))
    return lr.predict(P(Zt)).tolist()
