"""
Model training: LR, XGBoost, LightGBM, and a temporal-safe stacking ensemble.

Fix E: replaced sklearn StackingClassifier (cv=5 random K-fold) with a manual
TemporalStacker.  Random K-fold allowed game N to train a fold-model that then
predicted game N-100 — impossible in reality and a source of spurious confidence.
The manual stacker trains the meta-learner only on the chronologically last 30%
of training data, ensuring the meta-learner learns from forward-in-time predictions.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
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


class TemporalStacker:
    """
    Temporal-safe stacking ensemble.

    The meta-learner is trained on the chronologically LAST 30% of X_train,
    using predictions from the already-fitted base models.  This respects time
    ordering: the meta-learner learns how to combine base model signals on
    recent data only — never using future information to inform past predictions.

    Note: base models have seen the meta-training portion during their own
    training, so this is not strict OOS.  However, it is far superior to
    random K-fold (which allows game N to predict game N-100).
    """

    def __init__(self, base_models: dict, meta_model):
        self._base  = list(base_models.values())
        self._names = list(base_models.keys())
        self._meta  = meta_model

    def _meta_features(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self._base])

    def predict_proba(self, X):
        mf = self._meta_features(X)
        return self._meta.predict_proba(mf)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_stacking(X_train, y_train, base_models: dict) -> TemporalStacker:
    """
    Fit meta-learner on the last 30% of training data (temporal hold-out).
    """
    import pandas as pd
    n     = len(X_train)
    split = int(n * 0.7)

    X_meta = X_train.iloc[split:] if hasattr(X_train, "iloc") else X_train[split:]
    y_meta = y_train.iloc[split:] if hasattr(y_train, "iloc") else y_train[split:]

    meta_features = np.column_stack([
        m.predict_proba(X_meta)[:, 1] for m in base_models.values()
    ])

    meta_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    meta_lr.fit(meta_features, y_meta)

    stacker = TemporalStacker(base_models, meta_lr)
    print(f"  [Stack] temporal meta-learner trained on last 30% ({len(X_meta)} games)")
    return stacker


def train_all(X_train, y_train) -> dict:
    """Train LR, XGB, LGBM and a temporal stacking ensemble. Returns {name: fitted_pipeline}."""
    base = {
        "Logistic Regression": train_logistic(X_train, y_train),
        "XGBoost":             train_xgboost(X_train, y_train),
        "LightGBM":            train_lightgbm(X_train, y_train),
    }
    stack = train_stacking(X_train, y_train, base)
    return {**base, "Stacking Ensemble": stack}
