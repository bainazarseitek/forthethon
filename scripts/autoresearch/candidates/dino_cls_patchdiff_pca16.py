"""DINOv2 CLS window mean (L2) + PCA(16) of per-dim mean |frame diff| of DINOv2 mean-patch tokens (scaler+PCA fit on train only);
scaler + balanced LR C=10. k=16, C=10 pre-registered (same as motion_pca16), no tuning."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 16
SEED = 20261017


def parts(d):
    m = d["X"]["dinov2:cls"].mean(1)
    return m / np.linalg.norm(m, axis=1, keepdims=True), np.abs(np.diff(d["X"]["dinov2:patch"], axis=1)).mean(1)


def fit_predict(train, test):
    m_tr, d_tr = parts(train)
    m_te, d_te = parts(test)
    red = make_pipeline(StandardScaler(), PCA(n_components=K, random_state=SEED)).fit(d_tr)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED))
    model.fit(np.hstack([m_tr, red.transform(d_tr)]), train["y"])
    return model.predict(np.hstack([m_te, red.transform(d_te)]))
