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

def train_spread_ensemble(X: pd.DataFrame, y: pd.Series,
                          sample_weight=None) -> dict:
    """
    LR + shallow XGB + shallow LGBM for spread prediction.
    Covering -1.5 is non-linear (blowout rates depend on matchup gaps, xG
    differentials, PP/PK edges) so tree models with limited depth add signal.
    max_depth=3 prevents overfitting on ~5,000 games.
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
            max_depth=3, n_estimators=200, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=2.0,
            random_state=42, eval_metric="logloss", n_jobs=-1,
        )),
    ])
    lgbm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   LGBMClassifier(
            max_depth=3, num_leaves=15, n_estimators=200, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=2.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )),
    ])
    models = {
        "Logistic Regression": lr,
        "XGBoost":             xgb,
        "LightGBM":            lgbm,
    }
    for name, m in models.items():
        fit_params = {}
        if sample_weight is not None:
            if name == "Logistic Regression":
                fit_params["model__sample_weight"] = sample_weight
            else:
                fit_params["model__sample_weight"] = sample_weight
        m.fit(X, y, **fit_params)
    return models


# ── goals regression ensemble ─────────────────────────────────────────────────

def train_regression_all(X: pd.DataFrame, y: pd.Series,
                         sample_weight=None) -> dict:
    """
    Ridge only for goals regression.
    Backtested on 2024-25 (1,312 games): Ridge alone gives 52.8% totals accuracy
    with near-neutral over bias (51%). Adding XGB/LGBM pushes over bias to 59%
    and drops accuracy — tree models over-learn xGF signals and inflate predictions.
    Shrinkage toward median also hurts accuracy despite reducing MAE.
    Ridge's conservatism is the correct behaviour given hockey's high randomness.
    """
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=1.0)),
    ])
    models = {"Ridge": ridge}
    for name, m in models.items():
        fit_params = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight
        m.fit(X, y, **fit_params)
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
