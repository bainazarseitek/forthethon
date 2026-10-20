"""Champion CLIP mean + per-dim std of per-frame L2 CLIP embeddings over 16 frames; scaler + balanced LR C=10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(f):
    mean = f.mean(1)
    mean = mean / np.linalg.norm(mean, axis=1, keepdims=True)  # identical to champion 'clip'
    return np.hstack([mean, f.std(1)])


def fit_predict(train, test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017),
    )
    model.fit(features(train["X"]["clipframes:frames"]), train["y"])
    return model.predict(features(test["X"]["clipframes:frames"]))
