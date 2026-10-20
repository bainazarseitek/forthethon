"""Champion CLIP mean + (second-half mean - first-half mean) of per-frame CLIP embeddings + 3 scalar motion stats; scaler + balanced LR C=10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(f):
    mean = f.mean(1)
    mean = mean / np.linalg.norm(mean, axis=1, keepdims=True)  # identical to champion 'clip'
    d = np.diff(f, axis=1)
    step = np.linalg.norm(d, axis=2)  # (n, 15) per-step embedding motion
    scalars = np.stack([step.mean(1), step.std(1), 1 - (f[:, :8].mean(1) * f[:, 8:].mean(1)).sum(1)], 1)
    return np.hstack([mean, f[:, 8:].mean(1) - f[:, :8].mean(1), scalars])


def fit_predict(train, test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017),
    )
    model.fit(features(train["X"]["clipframes:frames"]), train["y"])
    return model.predict(features(test["X"]["clipframes:frames"]))
