"""Model training: LR, XGBoost, LightGBM, and Stacking Ensemble — all with median imputation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


LR_PARAM_GRID = {
    "model__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"],
}

XGB_PARAM_GRID = {
    "model__learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
    "model__max_depth": [3, 4, 5, 6],
    "model__n_estimators": [100, 200, 300, 500],
    "model__subsample": [0.6, 0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "model__reg_alpha": [0, 0.01, 0.1, 1.0],
    "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "model__min_child_weight": [1, 3, 5, 10],
}

LGBM_PARAM_GRID = {
    "model__learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
    "model__num_leaves": [15, 31, 63, 127],
    "model__max_depth": [3, 4, 5, 6, -1],
    "model__n_estimators": [100, 200, 300, 500],
    "model__subsample": [0.6, 0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "model__reg_alpha": [0, 0.01, 0.1, 1.0],
    "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "model__min_child_samples": [10, 20, 30, 50],
}


def _tscv(n_splits: int = 4) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    search = RandomizedSearchCV(
        pipe, LR_PARAM_GRID, n_iter=20, cv=_tscv(), scoring="average_precision",
        random_state=42, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    print(f"  [LR]   best params: {search.best_params_}  | cv acc: {search.best_score_:.4f}")
    return search.best_estimator_



def train_xgboost(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)),
    ])
    search = RandomizedSearchCV(
        pipe, XGB_PARAM_GRID, n_iter=20, cv=_tscv(), scoring="average_precision",
        random_state=42, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    print(f"  [XGB]  best params: {search.best_params_}  | cv acc: {search.best_score_:.4f}")
    return search.best_estimator_


def train_lightgbm(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)),
    ])
    search = RandomizedSearchCV(
        pipe, LGBM_PARAM_GRID, n_iter=20, cv=_tscv(), scoring="average_precision",
        random_state=42, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    print(f"  [LGBM] best params: {search.best_params_}  | cv acc: {search.best_score_:.4f}")
    return search.best_estimator_


def train_stacking(X_train, y_train, base_models: dict) -> StackingClassifier:
    estimators = list(base_models.items())
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        cv=5,  # cross_val_predict requires contiguous folds, not TimeSeriesSplit
        passthrough=False,
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    print(f"  [Stack] stacking ensemble fitted on {len(estimators)} base models")
    return stack


def train_all(X_train, y_train) -> dict:
    """Train LR, XGB, LGBM and a stacking ensemble. Returns {name: fitted_pipeline}."""
    base = {
        "Logistic Regression": train_logistic(X_train, y_train),
        "XGBoost":             train_xgboost(X_train, y_train),
        "LightGBM":            train_lightgbm(X_train, y_train),
    }
    stack = train_stacking(X_train, y_train, base)
    return {**base, "Stacking Ensemble": stack}
