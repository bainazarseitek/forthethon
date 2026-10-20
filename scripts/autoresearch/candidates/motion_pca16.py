"""Champion CLIP mean + PCA(16) of per-dim mean |frame diff| (scaler+PCA fitted on train only); scaler + balanced LR C=10. k=16 fixed a priori."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 16
SEED = 20261017


def parts(f):
    mean = f.mean(1)
    mean = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    return mean, np.abs(np.diff(f, axis=1)).mean(1)


def fit_predict(train, test):
    m_tr, d_tr = parts(train["X"]["clipframes:frames"])
    m_te, d_te = parts(test["X"]["clipframes:frames"])
    red = make_pipeline(StandardScaler(), PCA(n_components=K, random_state=SEED)).fit(d_tr)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED),
    )
    model.fit(np.hstack([m_tr, red.transform(d_tr)]), train["y"])
    return model.predict(np.hstack([m_te, red.transform(d_te)]))
