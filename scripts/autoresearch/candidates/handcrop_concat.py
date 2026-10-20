"""Full-frame CLIP mean ++ hand-crop CLIP mean; train-only StandardScaler, balanced LR C=10. Pre-registered, no tuning.

Hand crop (extract_handcrop.py): per source frame GroundingDINO "hands. banknotes." (box 0.3, text 0.25),
union of the top-2 hand boxes after NMS 0.5, each side padded 20%, full frame when no hand is detected;
CLIP ViT-L/14 action-pilot view, L2 per frame. Crop block = mean of the 16 crop embeddings re-L2'd (same
pooling as the champion's full-frame block). One StandardScaler over the 1536 dims = per-block
standardisation; both blocks are 768-d, so they carry equal weight.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261017


def feats(d):
    m = d["X"]["handcrop:frames"].mean(1)
    return np.hstack([d["X"]["clip"], m / np.linalg.norm(m, axis=1, keepdims=True)])


def fit_predict(train, test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED),
    )
    model.fit(feats(train), train["y"])
    return model.predict(feats(test))
