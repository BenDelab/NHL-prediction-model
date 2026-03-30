"""
Feature engineering: rolling window features per team, opponent features, and derived features.

Rolling features are organised into families inside _compute_team_rolling:
  ┌──────────────────┬───────────────────────────────────────────────────────┐
  │ WIN_RATE         │ win%, reg win%, OT win rate, EWM win%, streaks        │
  │ SCORING          │ goals, differentials, rates (shutout, high-scoring…)  │
  │ SHOOTING         │ SOG, shooting%, save%, quality gap                    │
  │ ADVANCED         │ Corsi, Fenwick, PDO, offensive-zone %                 │
  │ SPECIAL_TEAMS    │ PP/PK efficiency, SHG, net PPG                        │
  │ DISCIPLINE       │ PIMs, net discipline                                  │
  │ FACEOFFS         │ faceoff%                                              │
  │ OVERTIME         │ OT rate, conditional OT win rate                      │
  └──────────────────┴───────────────────────────────────────────────────────┘
All raw inputs are shift(1) to prevent data leakage.
"""

import pandas as pd
import numpy as np

WINDOWS = [5, 10, 15, 20]


def _compute_team_rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute all rolling features for one team-season group (chronological order)."""
    df = df.sort_values("date").copy()
    n = window
    p = f"rolling_{n}"

    # ── Raw inputs, all shifted by 1 game to prevent leakage ──────────────────
    wins  = df["result"].eq("W").astype(int).shift(1)
    gf    = df["goals_for"].shift(1)
    ga    = df["goals_against"].shift(1)
    sog   = df["team_sog"].shift(1)
    osog  = df["opp_sog"].shift(1)
    cf    = df["corsi_pct"].shift(1)
    ff    = df["fenwick_pct"].shift(1)
    fo    = df["faceoff_pct"].shift(1)
    pdo   = df["pdo"].shift(1)
    oz    = df["off_zone_start_pct"].shift(1)
    opim  = df["opp_pim"].shift(1)
    tpim  = df["team_pim"].shift(1)
    ot    = df["overtime_type"].ne("REG").astype(int).shift(1)
    regw  = (df["result"].eq("W") & df["overtime_type"].eq("REG")).astype(int).shift(1)
    otw   = (df["result"].eq("W") & df["overtime_type"].ne("REG")).astype(int).shift(1)
    shg   = df["team_shg"].shift(1)
    oshg  = df["opp_shg"].shift(1)
    shoot = (df["goals_for"] / (df["team_sog"] + 0.01)).shift(1)
    save  = (1 - df["goals_against"] / (df["opp_sog"] + 0.01)).shift(1)

    def rm(s): return s.rolling(n, min_periods=1).mean()
    def rs(s): return s.rolling(n, min_periods=2).std().fillna(0)

    # ── FAMILY: WIN_RATE ───────────────────────────────────────────────────────
    # Core win-rate signals in regulation, OT/SO, and exponentially weighted.
    df[f"{p}_win_pct"]     = rm(wins)
    df[f"{p}_reg_win_pct"] = rm(regw)           # regulation wins only
    df[f"{p}_ewm_win_pct"] = wins.ewm(span=n, min_periods=1).mean()
    df[f"{p}_win_pct_std"] = rs(wins)           # consistency of winning

    # OT win rate: wins in OT / games that went to OT (NaN → 0.5 neutral)
    ot_sum  = ot.rolling(n, min_periods=1).sum()
    otw_sum = otw.rolling(n, min_periods=1).sum()
    df[f"{p}_ot_win_rate"] = (otw_sum / ot_sum.replace(0, np.nan)).fillna(0.5)

    # ── FAMILY: SCORING ────────────────────────────────────────────────────────
    # Offensive output, defensive solidity, and game-result profiles.
    df[f"{p}_goals_for_avg"]     = rm(gf)
    df[f"{p}_goals_against_avg"] = rm(ga)
    df[f"{p}_goal_diff"]         = df[f"{p}_goals_for_avg"] - df[f"{p}_goals_against_avg"]
    df[f"{p}_gf_ga_ratio"]       = df[f"{p}_goals_for_avg"] / (df[f"{p}_goals_against_avg"] + 0.01)

    # Game-profile rates (shift already applied to gf/ga)
    shutout    = ga.eq(0).astype(float)           # held opponent scoreless
    high_score = gf.ge(3).astype(float)           # scored 3+
    blowout_w  = (gf - ga).ge(3).astype(float)    # won by 3+
    close_game = (gf - ga).abs().le(1).astype(float)  # 1-goal margin
    df[f"{p}_shutout_rate"]      = rm(shutout)    # P(keep opponent to 0)
    df[f"{p}_high_scoring_rate"] = rm(high_score) # P(score 3+)
    df[f"{p}_blowout_win_rate"]  = rm(blowout_w)  # P(dominant win)
    df[f"{p}_close_game_rate"]   = rm(close_game) # P(1-goal game)

    # EWM goal differential (how recent margin trend compares to rolling average)
    goal_diff_raw = gf - ga
    df[f"{p}_goal_diff_ewm"]     = goal_diff_raw.ewm(span=n, min_periods=1).mean()

    # Std devs
    df[f"{p}_goals_for_std"]     = rs(gf)
    df[f"{p}_goals_against_std"] = rs(ga)
    df[f"{p}_goal_diff_std"]     = rs(goal_diff_raw)

    # ── FAMILY: SHOOTING ───────────────────────────────────────────────────────
    # Shot volume, shooting efficiency, goaltending, and quality gap.
    df[f"{p}_sog_avg"]            = rm(sog)
    df[f"{p}_opp_sog_avg"]        = rm(osog)
    df[f"{p}_shot_diff"]          = df[f"{p}_sog_avg"] - df[f"{p}_opp_sog_avg"]
    df[f"{p}_shooting_pct"]       = rm(shoot)
    df[f"{p}_save_pct"]           = rm(save)
    # shooting_save_sum removed — ≈ PDO/100 which is already a feature (correlation)

    # Shot quality gap: how much better we are at scoring vs how much we allow
    # Positive = attack efficiency > defensive vulnerability → favourable
    df[f"{p}_shot_quality_gap"]   = df[f"{p}_shooting_pct"] - (1 - df[f"{p}_save_pct"])

    # Std devs
    df[f"{p}_sog_std"]            = rs(sog)
    df[f"{p}_opp_sog_std"]        = rs(osog)
    df[f"{p}_shooting_pct_std"]   = rs(shoot)
    df[f"{p}_save_pct_std"]       = rs(save)

    # ── FAMILY: ADVANCED ───────────────────────────────────────────────────────
    # Possession proxies: Corsi (all shot attempts), Fenwick (unblocked), PDO, zone starts.
    df[f"{p}_corsi_pct"]          = rm(cf)
    df[f"{p}_fenwick_pct"]        = rm(ff)
    df[f"{p}_pdo"]                = rm(pdo)
    df[f"{p}_off_zone_avg"]       = rm(oz)

    # Std devs
    df[f"{p}_corsi_pct_std"]      = rs(cf)
    df[f"{p}_fenwick_pct_std"]    = rs(ff)
    df[f"{p}_pdo_std"]            = rs(pdo)
    df[f"{p}_off_zone_std"]       = rs(oz)

    # ── FAMILY: SPECIAL_TEAMS ──────────────────────────────────────────────────
    # Power play / penalty kill efficiency, short-handed goals, net PP goal impact.
    ppg  = df["team_ppg"].shift(1).rolling(n, min_periods=1).sum()
    ppo  = df["team_ppo"].shift(1).rolling(n, min_periods=1).sum()
    oppg = df["opp_ppg"].shift(1).rolling(n, min_periods=1).sum()
    oppo = df["opp_ppo"].shift(1).rolling(n, min_periods=1).sum()
    df[f"{p}_pp_efficiency"]      = ppg  / ppo.replace(0, np.nan)
    df[f"{p}_pk_efficiency"]      = 1 - (oppg / oppo.replace(0, np.nan))
    # net_special_teams removed — derivable from pp_efficiency and pk_efficiency (redundant)
    df[f"{p}_shg_rate"]           = rm(shg)    # short-handed goals scored
    df[f"{p}_opp_shg_rate"]       = rm(oshg)   # opp short-handed goals (PK weakness)

    # Net power-play goals per game (positive = PP scores more than we give up)
    net_ppg_raw = df["team_ppg"].shift(1) - df["opp_ppg"].shift(1)
    df[f"{p}_net_ppg_per_game"]   = rm(net_ppg_raw)

    # ── FAMILY: DISCIPLINE ─────────────────────────────────────────────────────
    # Penalty minutes: our discipline vs opponent's.
    # pim_diff removed — derivable from the two component features (redundant).
    df[f"{p}_team_pim_avg"]       = rm(tpim)
    df[f"{p}_opp_pim_avg"]        = rm(opim)

    # ── FAMILY: FACEOFFS ───────────────────────────────────────────────────────
    # Faceoff win% correlates with zone starts and puck possession.
    df[f"{p}_faceoff_pct"]        = rm(fo)

    # ── FAMILY: OVERTIME ───────────────────────────────────────────────────────
    # How often games go to OT/SO; performance once they do.
    df[f"{p}_ot_rate"]            = rm(ot)     # P(game reaches OT/SO)

    return df


def compute_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features for all teams across all seasons."""
    parts = []
    for (team, season), group in df.groupby(["team", "season"]):
        group = _compute_team_rolling(group.copy(), window=5)
        for w in WINDOWS[1:]:
            group = _compute_team_rolling(group, window=w)
        parts.append(group)
    return pd.concat(parts).sort_values(["season", "date", "team"]).reset_index(drop=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-rolling derived features (schedule context, season-level rates)."""
    df = df.sort_values(["team", "season", "date"]).copy()

    df["days_since_last_game"] = (
        df.groupby(["team", "season"])["date"].diff().dt.days.fillna(3)
    )
    df["game_number_in_season"] = df["game_num"]
    df["pp_opportunity_rate"]   = df["team_ppo"] / (df["team_ppo"] + df["opp_ppo"] + 1e-9)
    df["back_to_back"]          = (df["days_since_last_game"] <= 1).astype(int)

    # Rolling 10-day workload: games played in the 10 calendar days before each game
    def _count_last_10(dates: pd.Series) -> pd.Series:
        arr = dates.values.astype("datetime64[D]")
        result = np.zeros(len(arr), dtype=int)
        for i in range(len(arr)):
            cutoff = arr[i] - np.timedelta64(10, "D")
            result[i] = int(np.sum((arr[:i] >= cutoff) & (arr[:i] < arr[i])))
        return pd.Series(result, index=dates.index)

    df["games_in_last_10"] = (
        df.groupby(["team", "season"])["date"]
        .transform(_count_last_10)
        .fillna(0)
    )

    # Cumulative season win rate (expanding, shift(1) via transform)
    df["season_win_pct"] = (
        df.groupby(["team", "season"])["result"]
        .transform(lambda s: s.eq("W").astype(int).shift(1).expanding(min_periods=1).mean())
        .fillna(0.5)
    )

    return df


def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add home win/loss streak (positive = wins, negative = losses)."""
    df = df.sort_values(["team", "season", "date"]).reset_index(drop=True).copy()

    wins    = df["result"].eq("W").values
    teams   = df["team"].values
    seasons = df["season"].values
    streaks = np.zeros(len(df), dtype=float)
    state: dict = {}

    for i in range(len(df)):
        key = (teams[i], seasons[i])
        s = state.get(key, 0)
        streaks[i] = s  # shift(1): record before updating
        state[key] = (max(s, 0) + 1) if wins[i] else (min(s, 0) - 1)

    df["win_streak"] = streaks
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Short-window win% minus long-window win% — positive = trending up."""
    for ws, wl in [(5, 20), (5, 10), (10, 20), (5, 15), (10, 15)]:
        cs, cl = f"rolling_{ws}_win_pct", f"rolling_{wl}_win_pct"
        if cs in df.columns and cl in df.columns:
            df[f"momentum_{ws}v{wl}"] = df[cs] - df[cl]

    # EWM momentum: fast ewm minus slow ewm
    for ws, wl in [(5, 20), (5, 10)]:
        cs, cl = f"rolling_{ws}_ewm_win_pct", f"rolling_{wl}_ewm_win_pct"
        if cs in df.columns and cl in df.columns:
            df[f"ewm_momentum_{ws}v{wl}"] = df[cs] - df[cl]

    return df


def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join each game's opponent's most recent rolling + season features."""
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    extra_cols   = [c for c in ["season_win_pct", "win_streak"] if c in df.columns]
    team_cols    = rolling_cols + extra_cols

    opp_df = df[["season", "date", "team"] + team_cols].copy()
    opp_df = opp_df.rename(
        columns={c: f"opp_{c}" for c in team_cols} | {"team": "opponent_key"}
    ).sort_values("date")

    df = df.rename(columns={"opponent": "opponent_key"}).sort_values("date")

    parts = []
    for season, s_df in df.groupby("season"):
        opp_s = opp_df[opp_df["season"] == season].sort_values("date").copy()
        result = pd.merge_asof(
            s_df.sort_values("date"),
            opp_s.drop(columns="season"),
            on="date",
            left_by="opponent_key",
            right_by="opponent_key",
            direction="backward",
        )
        parts.append(result)

    merged = pd.concat(parts).sort_values(["season", "date", "team"]).reset_index(drop=True)
    return merged.rename(columns={"opponent_key": "opponent"})


def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head historical win rate of home team vs opponent."""
    df = df.sort_values(["season", "date"]).copy()
    df["win_binary"] = df["result"].eq("W").astype(int)

    cumulative: dict = {}
    h2h = []
    for _, row in df.iterrows():
        key = (row["team"], row["opponent"])
        wins, games = cumulative.get(key, (0, 0))
        h2h.append(wins / games if games > 0 else 0.5)
        cumulative[key] = (wins + row["win_binary"], games + 1)

    df["h2h_win_pct"] = h2h
    return df.drop(columns=["win_binary"])


def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Home team minus opponent for key rolling metrics.
    Captures relative matchup quality directly as a single number.
    """
    diffs = {
        "win_pct":           "win_pct_diff",
        "goals_for_avg":     "gf_diff",
        "fenwick_pct":       "fenwick_diff",
        "pdo":               "pdo_diff",
        "shooting_pct":      "shooting_diff",
        "save_pct":          "save_diff",
        "pp_efficiency":     "pp_eff_diff",
        "pk_efficiency":     "pk_eff_diff",
        "off_zone_avg":      "oz_diff",
        "shutout_rate":      "shutout_rate_diff",
        "high_scoring_rate": "high_scoring_rate_diff",
        "goal_diff_ewm":     "goal_diff_ewm_diff",
        "shot_quality_gap":  "shot_quality_gap_diff",
        "net_ppg_per_game":  "net_ppg_diff",
        # net_special_teams, pim_diff, shooting_save_sum removed (redundant)
    }
    for w in WINDOWS:
        for stat, name in diffs.items():
            home_col = f"rolling_{w}_{stat}"
            opp_col  = f"opp_rolling_{w}_{stat}"
            if home_col in df.columns and opp_col in df.columns:
                df[f"rolling_{w}_{name}"] = df[home_col] - df[opp_col]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = compute_team_features(df)
    df = add_derived_features(df)
    df = add_streak_features(df)
    df = add_opponent_features(df)
    df = add_h2h_features(df)
    df = add_momentum_features(df)
    df = add_differential_features(df)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all engineered feature column names (excludes raw game stats and metadata)."""
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
