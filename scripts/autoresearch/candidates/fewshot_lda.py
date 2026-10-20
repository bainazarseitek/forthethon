"""Shrinkage LDA (Ledoit-Wolf auto shrinkage, uniform priors) on standardized CLIP. No tuned hyperparameters."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(train, test):
    k = len(np.unique(train["y"]))
    model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(k, 1 / k)),
    )
    model.fit(train["X"]["clip"], train["y"])
    return model.predict(test["X"]["clip"]).tolist()
