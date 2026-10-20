"""NAP K-ensemble (pre-registered, ONE evaluation; constants fixed before running).

Identical to sceneinv_nap (CLIP mean -> train-only StandardScaler -> project out the top-K
within-class between-video directions from sceneinv_nap.nap_basis, fit on TRAIN only ->
balanced LR C=10), except one head is fit per K in {4, 8, 12, 16} and their predicted class
probabilities are averaged with equal weight before argmax. Tests whether the NAP gain is
stable rather than resting on a lucky K (isolated K=8 spike in the K-sensitivity diagnostic).
"""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

KS, C = (4, 8, 12, 16), 10

_spec = importlib.util.spec_from_file_location("sceneinv_nap", Path(__file__).parent / "sceneinv_nap.py")
_nap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nap)


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    videos = np.array([m["video"] for m in train["meta"]])
    sc = StandardScaler().fit(X)
    Z, Zt = sc.transform(X), sc.transform(test["X"]["clip"])
    probs, classes = [], None
    for k in KS:
        V = _nap.nap_basis(Z, y, videos, k)
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=20261017)
        lr.fit(Z - Z @ V.T @ V, y)
        assert classes is None or list(lr.classes_) == list(classes)
        classes = lr.classes_
        probs.append(lr.predict_proba(Zt - Zt @ V.T @ V))
    return classes[np.mean(probs, axis=0).argmax(1)].tolist()
