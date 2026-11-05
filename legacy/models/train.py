"""
Model training: Logistic Regression for the moneyline ensemble.

XGBoost and LightGBM are excluded from the primary moneyline model:
on ~5,000 games × 130 features, tree models overfit and produce raw
probabilities of 0.75-0.90 for games that actually win ~55% of the time.
No calibration method corrects this without destroying the probability
spread needed for high-confidence picks.

Logistic Regression is well-calibrated by design, doesn't overfit on
tabular data of this size, and gives a 61% hit rate at p≥0.53 in
walk-forward tests — the most reliable signal available.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_all(X_train, y_train, sample_weight=None) -> dict:
    """Train LR only. Returns {name: fitted_pipeline}."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                                       random_state=42)),
    ])
    fit_params = {}
    if sample_weight is not None:
        fit_params["model__sample_weight"] = sample_weight
    pipe.fit(X_train, y_train, **fit_params)
    return {"Logistic Regression": pipe}
