"""Late fusion: mean of probabilities from CLIP-mean and DINOv2-CLS-mean (L2) heads, each scaler + balanced LR C=10. Pre-registered, no tuning."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261017


def feats(d):
    m = d["X"]["dinov2:cls"].mean(1)
    return [d["X"]["clip"], m / np.linalg.norm(m, axis=1, keepdims=True)]


def fit_predict(train, test):
    probs, classes = [], None
    for a, b in zip(feats(train), feats(test)):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED))
        m.fit(a, train["y"])
        classes = m.classes_
        probs.append(m.predict_proba(b))
    return classes[np.mean(probs, axis=0).argmax(1)]
