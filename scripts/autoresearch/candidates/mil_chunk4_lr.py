"""MIL-style: 4 chunk means of 4 consecutive per-frame CLIP embeddings per window (4x samples), each inherits
window label; scaler + balanced LR C=10; window prediction = argmax of mean chunk log-probs."""
import importlib.util, os
import numpy as np

_spec = importlib.util.spec_from_file_location("mil_frame_lr", os.path.join(os.path.dirname(__file__), "mil_frame_lr.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)


def fit_predict(train, test):
    return _m.fit_apply(train["X"]["clipframes:frames"], np.asarray(train["y"]),
                        test["X"]["clipframes:frames"], 4, 10.0).tolist()
