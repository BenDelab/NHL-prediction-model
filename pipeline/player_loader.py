"""
Load NHL player game-by-game stats for player prop modeling.

Sources: player_stats/NHL_player_*.csv  (MoneyPuck, situation='all', non-goalies)

Produces per-player per-game rows with per-player rolling features (shift-1
so the current game's outcome is never leaked into its own features):

  player_roll{w}_sog_mean    — rolling mean shots on goal
  player_roll{w}_sog_std     — rolling std  shots on goal (consistency signal)
  player_roll{w}_ice_mean    — rolling mean icetime (seconds)
  player_roll{w}_xg_mean     — rolling mean individual xGoals
  player_roll{w}_hd_mean     — rolling mean high-danger shots
  player_roll{w}_fenwick_mean— rolling mean unblocked shot attempts (broader than SOG)
  player_roll{w}_ozone_mean  — rolling mean offensive-zone shift-start rate

Opponent context (computed from the same player data):
  opp_roll10_soa             — opponent team's rolling shots-allowed per game (10g)
                               Higher = easier opponent for SOG (team leaks shots)
                               Lower  = tight defensive team (suppresses shooter volume)

Rolling is computed per player across all seasons (recent cross-season form
matters more than a season-boundary reset for projecting the next game).

MIN_ICETIME_SECONDS filters out games where the player appeared only briefly
(e.g. late callup with <5 minutes) which would distort per-player rolling averages.

MIN_SOG_MEAN_FOR_PROP is the minimum roll10_sog_mean required before we offer a
prop recommendation — players below this are 4th-liners who wouldn't have SOG
props offered by sportsbooks.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

PLAYER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "player_stats")

_SEASON_MAP = {
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}
_RENAMES = {"ARI": "UTA", "VGK": "VEG"}

# Games with fewer than this many seconds of icetime are excluded
MIN_ICETIME_SECONDS = 300   # 5 minutes

# Only generate prop picks for players with a rolling SOG average above this
# (filters out 4th-liners for whom sportsbooks don't offer SOG props)
MIN_SOG_MEAN_FOR_PROP = 1.5

ROLLING_WINDOWS = [5, 10]


def load_player_all_seasons(min_icetime: int = MIN_ICETIME_SECONDS) -> pd.DataFrame:
    """
    Load all NHL_player_*.csv files.

    Returns one row per (player, game) for situation='all' non-goalies,
    with icetime >= min_icetime seconds, sorted by (playerId, date).
    Includes extra columns needed for opponent context and role features.
    """
    pattern = os.path.join(PLAYER_DIR, "NHL_player_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("[player_loader] No NHL_player_*.csv files found — prop model disabled.")
        return pd.DataFrame()

    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[player_loader] Could not read {os.path.basename(path)}: {e}")
            continue

        df = df[(df["situation"] == "all") & (df["position"] != "G")].copy()
        if df.empty:
            continue

        df["season"] = df["season"].map(_SEASON_MAP)
        df = df.dropna(subset=["season"])

        df["date"]     = pd.to_datetime(df["gameDate"].astype(str), format="%Y%m%d")
        df["team"]     = df["playerTeam"].replace(_RENAMES)
        df["opp_team"] = df["opposingTeam"].replace(_RENAMES)
        df["icetime"]  = df["icetime"].astype(float)
        df["is_home"]  = (df["home_or_away"] == "HOME").astype(int)

        df = df[df["icetime"] >= min_icetime].copy()

        # Extra columns for role and opponent context
        df["ozone_rate"] = (
            df["I_F_oZoneShiftStarts"] /
            (df["I_F_oZoneShiftStarts"] + df["I_F_dZoneShiftStarts"] + 1e-6)
        )

        keep = [
            "playerId", "name", "date", "season", "team", "opp_team",
            "is_home", "position", "icetime",
            "I_F_shotsOnGoal", "I_F_xGoals", "I_F_highDangerShots",
            "I_F_unblockedShotAttempts", "ozone_rate",
        ]
        keep = [c for c in keep if c in df.columns]
        dfs.append(df[keep])

    if not dfs:
        return pd.DataFrame()

    combined = (
        pd.concat(dfs, ignore_index=True)
          .sort_values(["playerId", "date"])
          .reset_index(drop=True)
    )
    print(f"[player_loader] Loaded {len(combined)} skater-game rows "
          f"({combined['playerId'].nunique()} players, "
          f"{combined['season'].nunique()} seasons).")
    return combined


def build_player_rolling_features(df: pd.DataFrame,
                                   windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add per-player rolling features and opponent defensive context.

    Per-player rolling (shift-1, no leakage):
      player_roll{w}_sog_mean    — shots on goal form
      player_roll{w}_sog_std     — shot volume consistency
      player_roll{w}_ice_mean    — icetime (role stability)
      player_roll{w}_xg_mean     — expected goals (shot quality)
      player_roll{w}_hd_mean     — high-danger shots
      player_roll{w}_fenwick_mean— unblocked shot attempts
      player_roll{w}_ozone_mean  — offensive zone starts rate

    Opponent defensive context (computed from aggregated player data):
      opp_roll10_soa             — rolling-10 shots opponent allows per game
                                   Measures how 'leaky' the opponent defense is.
                                   A player facing DAL (~26 SOA) should get fewer
                                   shots than one facing CBJ (~34 SOA).
    """
    df = df.copy().sort_values(["playerId", "date"]).reset_index(drop=True)

    # ── Per-player rolling ────────────────────────────────────────────────────
    for w in windows:
        min_p = max(1, w // 2)

        def _mean(x):
            return x.shift(1).rolling(w, min_periods=min_p).mean()

        def _std(x):
            return x.shift(1).rolling(w, min_periods=min_p).std()

        for stat, suffix, fn in [
            ("I_F_shotsOnGoal",          "sog_mean",     _mean),
            ("I_F_shotsOnGoal",          "sog_std",      _std),
            ("icetime",                  "ice_mean",     _mean),
            ("I_F_xGoals",               "xg_mean",      _mean),
            ("I_F_highDangerShots",      "hd_mean",      _mean),
            ("I_F_unblockedShotAttempts","fenwick_mean",  _mean),
            ("ozone_rate",               "ozone_mean",   _mean),
        ]:
            if stat not in df.columns:
                continue
            col = f"player_roll{w}_{suffix}"
            df[col] = df.groupby("playerId")[stat].transform(fn)

    # ── Opponent shots-allowed (team-level defensive quality) ─────────────────
    # For each (opp_team, date): sum all players' SOG who played against that team
    # = total shots that opp_team's goalie faced = opp_team's shots-allowed for that game
    opp_soa = (
        df.groupby(["opp_team", "date"])["I_F_shotsOnGoal"]
          .sum()
          .reset_index()
          .rename(columns={"opp_team": "team", "I_F_shotsOnGoal": "_soa"})
          .sort_values(["team", "date"])
    )
    opp_soa["opp_roll10_soa"] = opp_soa.groupby("team")["_soa"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean()
    )
    opp_soa = opp_soa.drop(columns=["_soa"])

    df = df.merge(
        opp_soa,
        left_on=["opp_team", "date"],
        right_on=["team", "date"],
        how="left",
        suffixes=("", "_opp"),
    ).drop(columns=["team_opp"])

    return df


def get_prop_feature_cols(df: pd.DataFrame) -> list[str]:
    """Feature columns used by the prop model."""
    candidates = [
        # Short-window form
        "player_roll5_sog_mean",
        "player_roll5_sog_std",
        "player_roll5_ice_mean",
        "player_roll5_fenwick_mean",
        # Medium-window form
        "player_roll10_sog_mean",
        "player_roll10_sog_std",
        "player_roll10_ice_mean",
        "player_roll10_xg_mean",
        "player_roll10_hd_mean",
        "player_roll10_fenwick_mean",
        "player_roll10_ozone_mean",
        # Opponent defensive context
        "opp_roll10_soa",
        # Game context
        "is_home",
    ]
    return [c for c in candidates if c in df.columns]


def get_current_player_stats(df: pd.DataFrame,
                              team: str,
                              lookback_days: int = 21,
                              opp_team: str | None = None) -> pd.DataFrame:
    """
    Return the most recent rolling stats row for each active player on *team*.

    'Active' = appeared in at least one game in the last lookback_days days
    within the current season.

    If opp_team is given, the opp_roll10_soa column is populated from
    that opponent team's recent defensive record (shots they allow per game).
    This lets us adjust each player's feature vector for the specific
    opponent they're facing tonight — rather than using their last game's opp.
    """
    current_season = df["season"].max()
    cur = df[df["season"] == current_season].copy()

    team_df = cur[cur["team"] == team]
    if team_df.empty:
        return pd.DataFrame()

    cutoff = team_df["date"].max() - pd.Timedelta(days=lookback_days)
    active_players = team_df[team_df["date"] >= cutoff]["playerId"].unique()

    latest = (
        team_df[team_df["playerId"].isin(active_players)]
        .sort_values("date")
        .groupby("playerId")
        .last()
        .reset_index()
    )

    # Inject tonight's opponent context
    if opp_team is not None and "opp_roll10_soa" in df.columns:
        opp_rows = df[df["opp_team"] == opp_team].sort_values("date")
        if not opp_rows.empty:
            latest_opp_soa = opp_rows["opp_roll10_soa"].dropna().iloc[-1] \
                if opp_rows["opp_roll10_soa"].notna().any() else np.nan
            latest["opp_roll10_soa"] = latest_opp_soa

    return latest
