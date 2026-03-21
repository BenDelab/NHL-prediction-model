"""Model training: Logistic Regression, Random Forest, XGBoost with hyperparameter tuning."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


LR_PARAM_GRID = {
    "model__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [5, 10, 20],
}

XGB_PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


def _tscv(n_splits: int = 3) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    search = RandomizedSearchCV(
        pipe, LR_PARAM_GRID, n_iter=12, cv=_tscv(), scoring="accuracy",
        random_state=42, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    print(f"[LR]  best params: {search.best_params_}  | cv acc: {search.best_score_:.4f}")
    return search.best_estimator_


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf, RF_PARAM_GRID, n_iter=12, cv=_tscv(), scoring="accuracy",
        random_state=42, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    print(f"[RF]  best params: {search.best_params_}  | cv acc: {search.best_score_:.4f}")
    return search.best_estimator_


def train_xgboost(X_train, y_train) -> XGBClassifier:
    xgb = XGBClassifier(
        random_state=42, eval_metric="logloss", use_label_encoder=False, n_jobs=-1
    )
    search = RandomizedSearchCV(
        xgb, XGB_PARAM_GRID, n_iter=12, cv=_tscv(), scoring="accuracy",
        random_state=42, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    print(f"[XGB] best params: {search.best_params_}  | cv acc: {search.best_score_:.4f}")
    return search.best_estimator_


def train_all(X_train, y_train) -> dict:
    """Train all three models. Returns dict keyed by model name."""
    return {
        "Logistic Regression": train_logistic(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }
