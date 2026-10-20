"""VideoMAEv2 temporal tokens alone: L2 mean of per-position fc_norm tokens + per-dim mean |temporal-position diff|; scaler + balanced LR C=10 (fixed a priori, same as champion)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KEY = "v2tokens:temporal"


def features(t):
    mean = t.mean(1)
    mean = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    return np.hstack([mean, np.abs(np.diff(t, axis=1)).mean(1)])


def fit_predict(train, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10, class_weight="balanced", max_iter=5000, random_state=20261017))
    model.fit(features(train["X"][KEY]), train["y"])
    return model.predict(features(test["X"][KEY]))
