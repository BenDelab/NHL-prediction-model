"""
Probability calibration via temperature scaling.

Why temperature scaling instead of isotonic / sigmoid:
────────────────────────────────────────────────────────
Isotonic regression fits a step function on the calibration set.  On NHL data
the step function collapses the entire upper tail (raw p > 0.55) to a single
plateau value (~0.55), destroying all discrimination among high-confidence
predictions.  Sigmoid (Platt) scaling collapses the spread in the opposite
direction, pushing everything below 0.54.

Temperature scaling is a single-parameter method:
    calibrated_p = sigmoid( logit(raw_p) / T )

where T > 1 reduces overconfidence and T < 1 increases it.  Crucially it
preserves the RANKING of predictions — a game the model rates higher stays
higher — while correcting the absolute probability scale.  This is the
correct approach when the goal is both good calibration AND discriminative
power for high-confidence picks.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


def _nll(T, raw_probs, y):
    """Negative log-likelihood of temperature-scaled probabilities."""
    p = expit(logit(np.clip(raw_probs, 1e-6, 1 - 1e-6)) / T)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def _fit_temperature(raw_probs, y):
    """Find the temperature T that minimises NLL on (raw_probs, y)."""
    result = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded",
                             args=(raw_probs, y))
    return float(result.x)


class TemperatureScaler:
    """Wraps a fitted model; applies temperature scaling at predict time."""

    def __init__(self, model, T: float):
        self._model = model
        self.T = T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        raw = self._model.predict_proba(X)[:, 1]
        cal = expit(logit(np.clip(raw, 1e-6, 1 - 1e-6)) / self.T)
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1 - cal, cal])


def calibrate(model, X_cal, y_cal, X_eval=None, y_eval=None):
    """
    Fit temperature scaling on (X_cal, y_cal).
    X_eval / y_eval are accepted for API compatibility but not used —
    temperature scaling has only one parameter so hold-out selection is
    unnecessary (no overfitting risk).

    Returns a TemperatureScaler wrapping the original model.
    """
    raw_probs = model.predict_proba(X_cal)[:, 1]
    T = _fit_temperature(raw_probs, np.array(y_cal))

    from sklearn.metrics import brier_score_loss
    raw_brier = brier_score_loss(y_cal, raw_probs)
    cal_probs = expit(logit(np.clip(raw_probs, 1e-6, 1 - 1e-6)) / T)
    cal_brier = brier_score_loss(y_cal, cal_probs)
    improvement = raw_brier - cal_brier

    print(f"    calibration: temperature T={T:.3f} | "
          f"Brier {raw_brier:.4f} → {cal_brier:.4f} "
          f"({'−' if improvement >= 0 else '+'}{abs(improvement):.4f})")

    return TemperatureScaler(model, T)


def calibrate_pooled(models_by_window: dict, X_cal_by_window: dict,
                     y_cal: np.ndarray) -> tuple[float, dict]:
    """
    Pool raw probabilities from ALL windows and models, fit ONE global
    temperature T, then wrap every model with that T.

    Why this beats per-window calibration on NHL data:
    - Each window sees ~26 cal games (150 × 0.70 / 4 windows).
      26 games = ±10 pp uncertainty on the win rate, making per-window T
      estimates noisy.  Pooling gives ~312 observations (26 × 4 windows × 3
      models) — enough for a stable T estimate.
    - Temperature scaling has one degree of freedom, so there is no
      overfitting risk from using a larger pool.

    Parameters
    ----------
    models_by_window : {window: {name: fitted_model}}
    X_cal_by_window  : {window: (X_cal, feature_cols)}
        X_cal should already be the per-window slice of calibration data.
    y_cal : 1-D array of true labels (same for every window since games are
        the same; broadcast from the first window)

    Returns
    -------
    T_global : float
        The fitted temperature.
    calibrated : {window: {name: TemperatureScaler}}
    """
    all_raw_probs = []
    all_labels    = []

    for w, models in models_by_window.items():
        X_cal, feat_cols = X_cal_by_window[w]
        for name, m in models.items():
            raw = m.predict_proba(X_cal)[:, 1]
            all_raw_probs.append(raw)
            all_labels.append(np.asarray(y_cal))

    raw_pool  = np.concatenate(all_raw_probs)
    y_pool    = np.concatenate(all_labels)

    T_global = _fit_temperature(raw_pool, y_pool)

    from sklearn.metrics import brier_score_loss
    raw_brier = brier_score_loss(y_pool, np.clip(raw_pool, 1e-7, 1 - 1e-7))
    cal_pool  = expit(logit(np.clip(raw_pool, 1e-6, 1 - 1e-6)) / T_global)
    cal_brier = brier_score_loss(y_pool, np.clip(cal_pool, 1e-7, 1 - 1e-7))
    improvement = raw_brier - cal_brier
    print(f"  [pooled] T={T_global:.3f} | "
          f"Brier {raw_brier:.4f} → {cal_brier:.4f} "
          f"({'−' if improvement >= 0 else '+'}{abs(improvement):.4f}) "
          f"on {len(raw_pool)} pooled samples")

    calibrated = {
        w: {name: TemperatureScaler(m, T_global)
            for name, m in models.items()}
        for w, models in models_by_window.items()
    }
    return T_global, calibrated


