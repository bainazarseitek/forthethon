"""Early fusion: per-encoder train-only StandardScaler on CLIP and VideoMAEv2, concat, balanced LR C=10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ENC = ["clip", "videomaev2"]


def fit_predict(train, test):
    scalers = {e: StandardScaler().fit(train["X"][e]) for e in ENC}
    feat = lambda d: np.hstack([scalers[e].transform(d["X"][e]) for e in ENC])
    m = LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017)
    m.fit(feat(train), train["y"])
    return m.predict(feat(test))
