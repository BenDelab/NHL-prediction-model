"""Noise reduction: winsorization of rolling features, PDO flag."""

import pandas as pd


PDO_HIGH = 105.0
PDO_LOW = 95.0


def winsorize_rolling_features(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip rolling average columns to [lower, upper] percentile range (global)."""
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    for col in rolling_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def winsorize_per_season(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Clip rolling features using percentiles computed within each season.

    Why per-season instead of global:
    - Each NHL season has its own scoring environment (e.g. league GF/game drifts
      over years; rule changes shift Corsi distributions).
    - A global 1st percentile computed across 5 seasons may be dominated by a
      single anomalous season, giving the wrong clipping bound for the others.
    - Per-season percentiles anchor the clipping to the actual distribution of
      the season being clipped — each outlier is extreme relative to its own era.

    Falls back silently to global winsorization if no 'season' column is present.
    """
    if "season" not in df.columns:
        return winsorize_rolling_features(df, lower, upper)

    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    if not rolling_cols:
        return df

    for season, grp_idx in df.groupby("season").groups.items():
        for col in rolling_cols:
            vals = df.loc[grp_idx, col]
            lo = vals.quantile(lower)
            hi = vals.quantile(upper)
            df.loc[grp_idx, col] = vals.clip(lo, hi)

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
    df = winsorize_per_season(df)
    df = add_pdo_flag(df)
    return df
