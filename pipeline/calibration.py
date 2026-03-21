"""Probability calibration: Platt scaling and isotonic regression."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def calibrate(model, X_cal, y_cal, method: str = "sigmoid"):
    """
    Wrap a trained model with calibration.

    method: 'sigmoid' (Platt scaling) or 'isotonic'
    X_cal/y_cal: held-out calibration set (NOT the test set)
    Returns a calibrated classifier.
    """
    # sklearn >= 1.6 removed cv="prefit"; use set_output to calibrate pre-fitted model
    try:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrated.fit(X_cal, y_cal)
        return calibrated
    except (ValueError, TypeError):
        pass

    # Fallback: Platt scaling via logistic regression on predicted probabilities
    probs = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(probs, y_cal)

    class _CalibratedModel:
        def __init__(self, base, scaler):
            self._base = base
            self._platt = scaler

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        def predict_proba(self, X):
            p = self._base.predict_proba(X)[:, 1].reshape(-1, 1)
            cal_p = self._platt.predict_proba(p)
            return cal_p

    return _CalibratedModel(model, platt)


def calibration_curve_data(y_true, y_prob, n_bins: int = 10):
    """Return fraction of positives and mean predicted probability per bin."""
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    return fraction_of_positives, mean_predicted_value
