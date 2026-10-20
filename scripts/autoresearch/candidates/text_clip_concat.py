"""Pre-registered: CLIP image mean (standardized) + text similarity (24-d cliptext:sim_frames,
standardized, x W=3 fixed a priori), balanced LR C=10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

W = 3.0
KEY = "cliptext:sim_frames"


def fit_predict(train, test):
    a, b = StandardScaler().fit(train["X"]["clip"]), StandardScaler().fit(train["X"][KEY])
    f = lambda d: np.hstack([a.transform(d["X"]["clip"]), W * b.transform(d["X"][KEY])])
    m = LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017)
    m.fit(f(train), train["y"])
    return m.predict(f(test))
