"""DINOv2-base CLS window mean (L2) + scaler + balanced LR C=10. Constants pre-registered (C=10, seed), no tuning."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261017


def feat(d):
    m = d["X"]["dinov2:cls"].mean(1)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def fit_predict(train, test):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED))
    model.fit(feat(train), train["y"])
    return model.predict(feat(test))
