"""CLIP window mean + pixel motion (38-d: 4x4 frame-diff + 4x4 Farneback magnitude grids + 6 desk-region stats).
Per-block train-only StandardScaler, pixel block x W=3 (fixed a priori, same W as iteration 2), balanced LR C=10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

W, C, SEED = 3.0, 10, 20261017


def fit_predict(train, test):
    sc = StandardScaler().fit(train["X"]["clip"])
    sp = StandardScaler().fit(train["X"]["pixmotion:feat"])
    f = lambda d: np.hstack([sc.transform(d["X"]["clip"]), W * sp.transform(d["X"]["pixmotion:feat"])])
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
    m.fit(f(train), train["y"])
    return m.predict(f(test))
