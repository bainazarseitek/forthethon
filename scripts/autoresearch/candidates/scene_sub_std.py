"""Scene subtraction + per-video std scaling (pre-registered, one look):
x = L2((clip - scene_mean) / (scene_std + 0.005)); scene mean/std over all frames of the video at 2 fps,
label-free. EPS=0.005 fixed a priori (~10th percentile of per-dim std). StandardScaler, balanced LR C=10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 0.005


def feats(d):
    x = (d["X"]["clip"] - d["X"]["clipscene:mean"]) / (d["X"]["clipscene:std"] + EPS)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fit_predict(train, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017))
    model.fit(feats(train), train["y"])
    return model.predict(feats(test))
