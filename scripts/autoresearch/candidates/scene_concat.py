"""Clip concatenated with scene-subtracted clip (pre-registered, one look):
[clip, L2(clip - scene_mean)] -> StandardScaler -> balanced LR C=10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feats(d):
    x = d["X"]["clip"] - d["X"]["clipscene:mean"]
    return np.hstack([d["X"]["clip"], x / np.linalg.norm(x, axis=1, keepdims=True)])


def fit_predict(train, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017))
    model.fit(feats(train), train["y"])
    return model.predict(feats(test))
