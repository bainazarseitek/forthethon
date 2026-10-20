"""PRE-REGISTERED (single evaluation, written before running): blend of motion_pca16 + motion_scalars.

Spec (all constants inherited unchanged, nothing tuned):
  features = [ StandardScaler(CLIP mean, L2-normalized)                      # 768, as in both parents
             , StandardScaler(PCA16(StandardScaler(mean|frame diff|)))       # 16, motion_pca16.parts + K=16, SEED
             , W * StandardScaler(motion_scalars.motion_stats)              # 6, W=3 from motion_scalars
             ]
  All scalers / PCA fitted on `train` only. Classifier: LogisticRegression(C=10, class_weight="balanced",
  max_iter=3000, random_state=20261017). No inner CV, no search.
Functions/constants are imported from motion_pca16.py and motion_scalars.py by file path (as motion_twostage.py does).
Hypothesis: beats the champion (CV CI95 lower bound > 0 and val delta >= 0).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import motion_pca16 as P  # noqa: E402
import motion_scalars as S  # noqa: E402

SEED = P.SEED


def fit_predict(train, test):
    ftr, fte = train["X"]["clipframes:frames"], test["X"]["clipframes:frames"]
    m_tr, d_tr = P.parts(ftr)
    m_te, d_te = P.parts(fte)
    red = make_pipeline(StandardScaler(), PCA(n_components=P.K, random_state=SEED)).fit(d_tr)
    sa = StandardScaler().fit(np.hstack([m_tr, red.transform(d_tr)]))
    ss = StandardScaler().fit(S.motion_stats(ftr))
    feat = lambda m, d, f: np.hstack([sa.transform(np.hstack([m, red.transform(d)])), S.W * ss.transform(S.motion_stats(f))])
    model = LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED)
    model.fit(feat(m_tr, d_tr, ftr), train["y"])
    return model.predict(feat(m_te, d_te, fte))
