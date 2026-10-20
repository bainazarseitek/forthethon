"""Late fusion: mean of class probabilities from a full-frame CLIP-mean head and a hand-crop CLIP-mean head,
each train-only StandardScaler + balanced LR C=10. Pre-registered, no tuning.

Hand crop (extract_handcrop.py): per source frame GroundingDINO "hands. banknotes." (box 0.3, text 0.25),
union of the top-2 hand boxes after NMS 0.5, each side padded 20%, full frame when no hand is detected;
CLIP ViT-L/14 action-pilot view, L2 per frame; window block = mean of the 16 crop embeddings re-L2'd.
Unlike concatenation, each head keeps the champion's 768-d size at the same C.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261017


def feats(d):
    m = d["X"]["handcrop:frames"].mean(1)
    return [d["X"]["clip"], m / np.linalg.norm(m, axis=1, keepdims=True)]


def fit_predict(train, test):
    probs, classes = [], None
    for a, b in zip(feats(train), feats(test)):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED))
        m.fit(a, train["y"])
        assert classes is None or np.array_equal(classes, m.classes_)
        classes = m.classes_
        probs.append(m.predict_proba(b))
    return classes[np.mean(probs, axis=0).argmax(1)]
