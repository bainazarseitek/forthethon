"""Nearest class mean (prototype) with cosine on train-centered CLIP. No hyperparameters."""
import numpy as np


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    mu = X.mean(0)
    norm = lambda A: (A - mu) / np.linalg.norm(A - mu, axis=1, keepdims=True)
    Xt = norm(X)
    classes = np.unique(y)
    P = np.stack([Xt[y == c].mean(0) for c in classes])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return classes[np.argmax(norm(test["X"]["clip"]) @ P.T, axis=1)].tolist()
