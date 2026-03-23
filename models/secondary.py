"""
Secondary models for OT likelihood and regulation win prediction.

These are lightweight Logistic Regression pipelines — no hyperparameter
tuning needed since they serve as supplementary analysis, not the primary bet signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def train_ot_model(X: pd.DataFrame, y_ot: pd.Series):
    """
    Train a model to predict P(game goes to OT/SO).
    y_ot: binary series where 1 = game went to overtime/shootout.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")),
    ])
    pipe.fit(X, y_ot)
    return pipe


def train_reg_win_model(X: pd.DataFrame, y_reg_win: pd.Series):
    """
    Train a model to predict P(home team wins in regulation).
    y_reg_win: binary series where 1 = home team won in regulation.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")),
    ])
    pipe.fit(X, y_reg_win)
    return pipe
