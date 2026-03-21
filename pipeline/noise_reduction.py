"""Noise reduction: winsorization of rolling features, PDO flag."""

import numpy as np
import pandas as pd


PDO_HIGH = 105.0
PDO_LOW = 95.0


def winsorize_rolling_features(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip rolling average columns to [lower, upper] percentile range."""
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    for col in rolling_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def add_pdo_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary flag for each rolling window where PDO is extreme."""
    for w in [5, 10, 15, 20]:
        col = f"rolling_{w}_pdo"
        if col in df.columns:
            df[f"rolling_{w}_pdo_extreme"] = (
                (df[col] > PDO_HIGH) | (df[col] < PDO_LOW)
            ).astype(int)
    return df


def apply(df: pd.DataFrame) -> pd.DataFrame:
    df = winsorize_rolling_features(df)
    df = add_pdo_flag(df)
    return df
