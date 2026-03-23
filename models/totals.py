"""
Totals and spread models for NHL game prediction.

Markets covered
───────────────
• Spread    — P(home covers puck line -1.5) and P(away covers -1.5),
              each trained independently.
• Total goals  — expected combined goals; P(over user-supplied line)

Reliability design
──────────────────
Spread and goals both use the same multi-window, multi-model, calibrated ensemble
approach as the main moneyline model.  Fixed (non-tuned) params are used to keep
training time manageable — the reliability gain comes from ensembling 12 models
(3 classifiers/regressors × 4 windows) rather than from hyperparameter search.

Probability estimation
──────────────────────
• Spread uses isotonic-calibrated classifier probabilities, ensemble-averaged.
• Goals use Poisson distribution: P(over line) = 1 − Poisson.cdf(floor(line), λ)
  where λ = ensemble mean of all regression predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


# ── spread ensemble (binary classification) ──────────────────────────────────

def train_spread_ensemble(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train LR + XGBoost + LightGBM classifiers with fixed good params.
    Returns {name: fitted_pipeline}.
    No RandomizedSearchCV — speed is prioritised; reliability comes from the
    multi-window + calibration + ensemble averaging done in main.py.
    """
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                                       random_state=42)),
    ])
    xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])
    lgbm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1,
        )),
    ])
    models = {
        "Logistic Regression": lr,
        "XGBoost":             xgb,
        "LightGBM":            lgbm,
    }
    for name, m in models.items():
        m.fit(X, y)
    return models


# ── goals regression ensemble ─────────────────────────────────────────────────

def train_regression_all(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train Ridge + XGBRegressor + LGBMRegressor with fixed params.
    Returns {name: fitted_pipeline}.
    """
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=1.0)),
    ])
    xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])
    lgbm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1,
        )),
    ])
    models = {
        "Ridge":    ridge,
        "XGBoost":  xgb,
        "LightGBM": lgbm,
    }
    for name, m in models.items():
        m.fit(X, y)
    return models


# ── probability helpers ───────────────────────────────────────────────────────

def prob_over_goals(predicted: float, line: float) -> float:
    """
    P(total goals > line) using Poisson with lambda = predicted.
    For a half-point line (e.g. 5.5): P(X ≥ 6) = 1 − Poisson.cdf(5, λ).
    """
    lam = max(predicted, 0.01)
    return float(1.0 - poisson.cdf(int(line), lam))


def prob_under(p_over: float) -> float:
    return 1.0 - p_over
