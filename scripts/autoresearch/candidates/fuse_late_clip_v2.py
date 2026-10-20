"""Late fusion: mean of predicted probabilities from per-encoder (CLIP, VideoMAEv2) scaler+balanced LR C=10 heads."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ENC = ["clip", "videomaev2"]


def fit_predict(train, test):
    probs, classes = [], None
    for e in ENC:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017))
        m.fit(train["X"][e], train["y"])
        classes = m.classes_
        probs.append(m.predict_proba(test["X"][e]))
    return classes[np.mean(probs, axis=0).argmax(1)]
