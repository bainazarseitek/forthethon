"""MIL-style: each of 16 per-frame CLIP embeddings inherits window label (16x samples); scaler on train frames,
balanced LR C=10; window prediction = argmax of mean per-frame log-probs."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHUNK = 1  # frames per sub-sample
C = 10.0


def subs(f, chunk):
    n, t, d = f.shape
    s = f.reshape(n, t // chunk, chunk, d).mean(2)
    s = s / np.linalg.norm(s, axis=2, keepdims=True)
    return s  # (n, k, d)


def fit_apply(Xtr, ytr, Xte, chunk, c):
    s = subs(Xtr, chunk)
    k = s.shape[1]
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=20261017))
    model.fit(s.reshape(-1, s.shape[2]), np.repeat(ytr, k))
    st = subs(Xte, chunk)
    lp = model.predict_log_proba(st.reshape(-1, st.shape[2])).reshape(len(Xte), st.shape[1], -1).mean(1)
    return model.classes_[lp.argmax(1)]


def fit_predict(train, test):
    return fit_apply(train["X"]["clipframes:frames"], np.asarray(train["y"]),
                     test["X"]["clipframes:frames"], CHUNK, C).tolist()
