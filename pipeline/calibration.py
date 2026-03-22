"""
Probability calibration — Platt scaling (sigmoid) vs isotonic regression.

Why calibrate?
  A model predicting "65% win probability" should be right ~65% of the time.
  Without calibration, tree models (RF, XGB, LGBM) often produce probabilities
  that are too confident or too conservative. Calibration corrects this bias,
  which matters when you want to compare confidence across predictions.

  - Sigmoid (Platt scaling): fits a logistic regression on the model's raw scores.
    More stable on smaller datasets.
  - Isotonic regression: non-parametric staircase fit, more flexible but needs
    more calibration data to avoid overfitting.

  We try both and keep whichever gives the lower Brier score on the held-out
  calibration set. If neither method improves on the raw model, the raw model
  is returned unchanged.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def _try_calibrate(model, X_cal, y_cal, method: str):
    """Attempt sklearn cv='prefit' calibration; fall back to manual Platt if removed."""
    try:
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)
        return cal
    except (ValueError, TypeError):
        pass

    # sklearn >= 1.6 removed cv="prefit" — implement Platt scaling manually
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


def calibrate(model, X_cal, y_cal):
    """
    Try sigmoid and isotonic calibration; return whichever has the lower Brier
    score on the calibration set. Falls back to the raw model if neither helps.
    """
    raw_brier = brier_score_loss(y_cal, model.predict_proba(X_cal)[:, 1])

    best_model  = model
    best_brier  = raw_brier
    best_method = "none (raw)"

    for method in ("sigmoid", "isotonic"):
        try:
            cal = _try_calibrate(model, X_cal, y_cal, method)
            b = brier_score_loss(y_cal, cal.predict_proba(X_cal)[:, 1])
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
