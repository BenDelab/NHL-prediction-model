"""
Probability calibration — Platt scaling (sigmoid) vs isotonic regression.

Fix A: calibration method selection now uses a separate held-out evaluation set
(X_eval / y_eval) rather than the same data used to fit the calibrator.
This prevents the calibrator from always appearing to improve on its own training
data — isotonic regression in particular memorises small datasets perfectly,
giving a falsely low Brier score and being selected when it shouldn't be.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def _try_calibrate(model, X_cal, y_cal, method: str):
    """Fit one calibration method on (X_cal, y_cal). Returns fitted calibrator."""
    try:
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)
        return cal
    except (ValueError, TypeError):
        pass

    # sklearn >= 1.6 removed cv="prefit" — manual fallback
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    probs = model.predict_proba(X_cal)[:, 1]

    if method == "sigmoid":
        lr = LogisticRegression(max_iter=1000)
        lr.fit(probs.reshape(-1, 1), y_cal)

        class _SigmoidCal:
            def __init__(self, base, lr):
                self._base, self._lr = base, lr
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
            def predict_proba(self, X):
                p = self._base.predict_proba(X)[:, 1].reshape(-1, 1)
                return self._lr.predict_proba(p)

        return _SigmoidCal(model, lr)

    else:  # isotonic
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs, y_cal)

        class _IsotonicCal:
            def __init__(self, base, iso):
                self._base, self._iso = base, iso
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
            def predict_proba(self, X):
                p = self._base.predict_proba(X)[:, 1]
                cal_p = np.clip(self._iso.predict(p), 0, 1)
                return np.column_stack([1 - cal_p, cal_p])

        return _IsotonicCal(model, iso)


def calibrate(model, X_cal, y_cal, X_eval=None, y_eval=None):
    """
    Fit sigmoid and isotonic calibrators on (X_cal, y_cal).
    Select the best method by Brier score on (X_eval, y_eval).

    X_eval / y_eval should be a separate held-out set not used for fitting —
    this prevents leakage where the calibrator always looks better on its own
    training data. Falls back to X_cal if X_eval is not provided.
    """
    if X_eval is None or y_eval is None:
        X_eval, y_eval = X_cal, y_cal

    raw_brier  = brier_score_loss(y_eval, model.predict_proba(X_eval)[:, 1])
    best_model  = model
    best_brier  = raw_brier
    best_method = "none (raw)"

    for method in ("sigmoid", "isotonic"):
        try:
            cal = _try_calibrate(model, X_cal, y_cal, method)   # fit on X_cal
            b   = brier_score_loss(y_eval, cal.predict_proba(X_eval)[:, 1])  # eval on X_eval
            if b < best_brier:
                best_brier  = b
                best_model  = cal
                best_method = method
        except Exception:
            pass

    improvement = raw_brier - best_brier
    print(f"    calibration: {best_method} | Brier {raw_brier:.4f} → {best_brier:.4f} "
          f"({'−' if improvement >= 0 else '+'}{abs(improvement):.4f})")
    return best_model


def calibration_curve_data(y_true, y_prob, n_bins: int = 10):
    from sklearn.calibration import calibration_curve
    return calibration_curve(y_true, y_prob, n_bins=n_bins)
