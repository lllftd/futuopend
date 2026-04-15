from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_binary_calibrator(y_true: np.ndarray, raw_p: np.ndarray) -> IsotonicRegression | None:
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.clip(np.asarray(raw_p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if y.size < 100 or len(np.unique(y)) < 2:
        return None
    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib.fit(p, y.astype(np.float64))
    return calib


def apply_binary_calibrator(p: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if calibrator is None:
        return arr
    return np.clip(np.asarray(calibrator.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)
