"""Nuisance-attribute projection of scene identity (pre-registered, constants fixed before running).

CLIP mean -> train-only StandardScaler. On TRAIN only: residualise each window by its class mean,
average residuals per (video, class) pair (equal weight per pair), PCA -> top K=8 "between-video,
within-class" directions; project them out of train and test features. Then balanced LR C=10.
Class means are removed first so the nuisance subspace captures scene identity, not class mix.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

K, C = 8, 10


def nap_basis(Z, y, videos, k):
    R = Z.copy()
    for c in np.unique(y):
        R[y == c] -= Z[y == c].mean(0)
    keys = sorted(set(zip(videos, y)))
    M = np.stack([R[(videos == v) & (y == c)].mean(0) for v, c in keys])
    M -= M.mean(0)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[: min(k, len(keys) - 1)]


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    videos = np.array([m["video"] for m in train["meta"]])
    sc = StandardScaler().fit(X)
    Z, Zt = sc.transform(X), sc.transform(test["X"]["clip"])
    V = nap_basis(Z, y, videos, K)
    P = lambda A: A - (A @ V.T) @ V
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=20261017)
    lr.fit(P(Z), y)
    return lr.predict(P(Zt)).tolist()
