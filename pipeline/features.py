"""Feature engineering: rolling window features per team, opponent features, and derived features."""

import pandas as pd
import numpy as np

WINDOWS = [5, 10, 15, 20]


def _compute_team_rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling features for a single team's games, in chronological order."""
    df = df.sort_values("date").copy()
    n = window
    prefix = f"rolling_{n}"

    # Shift by 1 so we use only past games (no leakage)
    df[f"{prefix}_win_pct"] = (
        df["result"].eq("W").astype(int).shift(1).rolling(n, min_periods=1).mean()
    )
    df[f"{prefix}_goals_for_avg"] = df["goals_for"].shift(1).rolling(n, min_periods=1).mean()
    df[f"{prefix}_goals_against_avg"] = df["goals_against"].shift(1).rolling(n, min_periods=1).mean()
    df[f"{prefix}_sog_avg"] = df["team_sog"].shift(1).rolling(n, min_periods=1).mean()
    df[f"{prefix}_opp_sog_avg"] = df["opp_sog"].shift(1).rolling(n, min_periods=1).mean()
    df[f"{prefix}_corsi_pct"] = df["corsi_pct"].shift(1).rolling(n, min_periods=1).mean()
    df[f"{prefix}_fenwick_pct"] = df["fenwick_pct"].shift(1).rolling(n, min_periods=1).mean()
    df[f"{prefix}_faceoff_pct"] = df["faceoff_pct"].shift(1).rolling(n, min_periods=1).mean()
    df[f"{prefix}_pdo"] = df["pdo"].shift(1).rolling(n, min_periods=1).mean()

    # Power play efficiency: pp goals / pp opportunities
    pp_goals = df["team_ppg"].shift(1).rolling(n, min_periods=1).sum()
    pp_opp = df["team_ppo"].shift(1).rolling(n, min_periods=1).sum()
    df[f"{prefix}_pp_efficiency"] = pp_goals / pp_opp.replace(0, np.nan)

    # Penalty kill efficiency: 1 - (opp pp goals / opp pp opportunities)
    opp_ppg = df["opp_ppg"].shift(1).rolling(n, min_periods=1).sum()
    opp_ppo = df["opp_ppo"].shift(1).rolling(n, min_periods=1).sum()
    df[f"{prefix}_pk_efficiency"] = 1 - (opp_ppg / opp_ppo.replace(0, np.nan))

    # Derived rolling
    df[f"{prefix}_goal_diff"] = df[f"{prefix}_goals_for_avg"] - df[f"{prefix}_goals_against_avg"]
    df[f"{prefix}_shot_diff"] = df[f"{prefix}_sog_avg"] - df[f"{prefix}_opp_sog_avg"]

    return df


def compute_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features for all teams across all seasons."""
    parts = []
    for (team, season), group in df.groupby(["team", "season"]):
        group = _compute_team_rolling(group.copy(), window=5)
        for w in WINDOWS[1:]:
            group = _compute_team_rolling(group, window=w)
        parts.append(group)
    result = pd.concat(parts).sort_values(["season", "date", "team"]).reset_index(drop=True)
    return result


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-rolling derived features."""
    df = df.sort_values(["team", "season", "date"]).copy()

    # Days since last game (per team across seasons)
    df["days_since_last_game"] = (
        df.groupby(["team", "season"])["date"]
        .diff()
        .dt.days
        .fillna(3)  # default rest for first game
    )

    df["game_number_in_season"] = df["game_num"]

    # Power play opportunity rate (discipline proxy)
    df["pp_opportunity_rate"] = df["team_ppo"] / (df["team_ppo"] + df["opp_ppo"] + 1e-9)

    return df


def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join each game's opponent's most recent rolling features onto the home team's row."""
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]

    # Build lookup of opponent's latest rolling stats before any given date
    opp_df = df[["season", "date", "team"] + rolling_cols].copy()
    opp_df = opp_df.sort_values(["season", "team", "date"])
    rename = {c: f"opp_{c}" for c in rolling_cols}
    opp_df = opp_df.rename(columns=rename | {"team": "opponent_key"})

    df = df.rename(columns={"opponent": "opponent_key"})
    df = df.sort_values("date")

    merged_parts = []
    for season, season_df in df.groupby("season"):
        opp_season = opp_df[opp_df["season"] == season].sort_values("date").copy()
        season_df = season_df.sort_values("date")
        # For each home team row, find opponent's last home game before this date
        result = pd.merge_asof(
            season_df,
            opp_season.drop(columns="season"),
            left_on="date",
            right_on="date",
            left_by="opponent_key",
            right_by="opponent_key",
            direction="backward",
        )
        merged_parts.append(result)

    merged = pd.concat(merged_parts).sort_values(["season", "date", "team"]).reset_index(drop=True)
    merged = merged.rename(columns={"opponent_key": "opponent"})
    return merged


def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head historical win rate of home team vs opponent."""
    df = df.sort_values(["season", "date"]).copy()
    df["win_binary"] = df["result"].eq("W").astype(int)

    h2h_records = []
    cumulative: dict = {}

    for _, row in df.iterrows():
        key = (row["team"], row["opponent"])
        wins, games = cumulative.get(key, (0, 0))
        h2h_win_pct = wins / games if games > 0 else 0.5
        h2h_records.append(h2h_win_pct)
        cumulative[key] = (wins + row["win_binary"], games + 1)

    df["h2h_win_pct"] = h2h_records
    df = df.drop(columns=["win_binary"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = compute_team_features(df)
    df = add_derived_features(df)
    df = add_opponent_features(df)
    df = add_h2h_features(df)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all feature column names (excludes metadata and target)."""
    exclude = {
        "game_num", "date", "team", "home_away", "opponent", "result",
        "goals_for", "goals_against", "overtime_type",
        "team_sog", "opp_sog", "team_pim", "opp_pim",
        "team_ppg", "team_ppo", "team_shg", "opp_ppg", "opp_ppo", "opp_shg",
        "faceoff_wins", "faceoff_losses", "faceoff_pct",
        "corsi_for", "corsi_against", "corsi_pct",
        "fenwick_for", "fenwick_against", "fenwick_pct",
        "off_zone_start_pct", "pdo", "season",
    }
    return [c for c in df.columns if c not in exclude]
