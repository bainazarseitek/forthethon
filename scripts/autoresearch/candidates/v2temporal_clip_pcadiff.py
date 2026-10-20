"""CLIP mean + compact v2 motion: PCA(16, fit on train only) of standardized v2 per-dim mean |temporal-position diff|; per-block scaler, balanced LR C=10 (constants fixed a priori)."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 16


def motion(t):
    return np.abs(np.diff(t, axis=1)).mean(1)


def fit_predict(train, test):
    clip_sc = StandardScaler().fit(train["X"]["clip"])
    mot = make_pipeline(StandardScaler(), PCA(n_components=K, random_state=0), StandardScaler())
    mot.fit(motion(train["X"]["v2tokens:temporal"]))

    def feats(d):
        return np.hstack([clip_sc.transform(d["X"]["clip"]), mot.transform(motion(d["X"]["v2tokens:temporal"]))])

    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=5000, random_state=20261017)
    lr.fit(feats(train), train["y"])
    return lr.predict(feats(test))
