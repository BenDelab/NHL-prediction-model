"""Evaluation with sliding window validation on the 2024-25 season."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

WINDOWS = [5, 10, 15, 20]


def _rolling_feature_cols(window: int, df: pd.DataFrame) -> list:
    prefix = f"rolling_{window}_"
    return [c for c in df.columns if c.startswith(prefix) or c.startswith(f"opp_rolling_{window}_")]


def get_feature_cols_for_window(window: int, df: pd.DataFrame) -> list:
    """Return feature columns that correspond to a given rolling window N."""
    rolling_cols = _rolling_feature_cols(window, df)
    static_cols = [
        "days_since_last_game", "game_number_in_season",
        "pp_opportunity_rate", "h2h_win_pct",
    ]
    static_cols = [c for c in static_cols if c in df.columns]
    return rolling_cols + static_cols


def evaluate_model_on_window(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "brier": brier_score_loss(y_test, y_prob),
        "log_loss": log_loss(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def sliding_window_evaluation(models: dict, test_df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """
    For each model and each window size N, evaluate on the test season.
    Returns a summary DataFrame with one row per (model, N).
    """
    if windows is None:
        windows = WINDOWS
    y_test_full = test_df["result"].eq("W").astype(int)
    records = []

    for model_name, model in models.items():
        for w in windows:
            feat_cols = get_feature_cols_for_window(w, test_df)
            if not feat_cols:
                print(f"[eval] No feature cols for window {w}, skipping.")
                continue

            X_w = test_df[feat_cols].copy()
            y_w = y_test_full.copy()

            # Drop rows with NaN features
            valid = X_w.notna().all(axis=1)
            X_w = X_w[valid]
            y_w = y_w[valid]

            if len(y_w) == 0:
                continue

            metrics = evaluate_model_on_window(model, X_w, y_w)
            records.append({
                "model": model_name,
                "window": w,
                **metrics,
            })

    return pd.DataFrame(records)


def best_config_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return accuracy pivot table: models as rows, N as columns."""
    pivot = results.pivot(index="model", columns="window", values="accuracy")
    pivot.columns = [f"N={c}" for c in pivot.columns]
    pivot["Best N"] = pivot.idxmax(axis=1)
    return pivot


def predict_game(model, game_row: pd.Series, feature_cols: list) -> dict:
    """Return prediction dict for a single game row."""
    X = game_row[feature_cols].values.reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return {
        "prediction": "WIN" if pred == 1 else "LOSS",
        "confidence": prob if pred == 1 else 1 - prob,
        "raw_prob": prob,
    }
