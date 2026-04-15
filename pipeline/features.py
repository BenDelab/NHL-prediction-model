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
    # Core win-rate signals in regulation and OT/SO.
    df[f"{p}_win_pct"]     = rm(wins)
    df[f"{p}_reg_win_pct"] = rm(regw)           # regulation wins only
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
    shutout       = ga.eq(0).astype(float)            # held opponent scoreless
    high_score    = gf.ge(3).astype(float)            # scored 3+
    blowout_w     = (gf - ga).ge(3).astype(float)     # won by 3+
    close_game    = (gf - ga).abs().le(1).astype(float)  # 1-goal margin
    win_by_2plus  = (gf - ga).ge(2).astype(float)    # won by 2+ (covers puck line -1.5)
    lose_by_2plus = (ga - gf).ge(2).astype(float)    # lost by 2+ (opp covers -1.5)
    df[f"{p}_shutout_rate"]      = rm(shutout)        # P(keep opponent to 0)
    df[f"{p}_high_scoring_rate"] = rm(high_score)     # P(score 3+)
    df[f"{p}_blowout_win_rate"]  = rm(blowout_w)      # P(dominant win)
    df[f"{p}_close_game_rate"]   = rm(close_game)     # P(1-goal game)
    df[f"{p}_win_by_2plus_rate"] = rm(win_by_2plus)   # P(cover -1.5) — direct spread signal
    df[f"{p}_lose_by_2plus_rate"]= rm(lose_by_2plus)  # P(opp covers -1.5)

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
    # Possession proxies: Corsi (all shot attempts), PDO, zone starts.
    # Fenwick removed — r=0.95 with Corsi, no independent signal.
    df[f"{p}_corsi_pct"]          = rm(cf)
    df[f"{p}_pdo"]                = rm(pdo)
    df[f"{p}_off_zone_avg"]       = rm(oz)

    # Std devs
    df[f"{p}_corsi_pct_std"]      = rs(cf)
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


def _build_full_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct each team's COMPLETE game dates from:
      - rows where team == T  (home appearances)
      - rows where opponent == T  (away appearances)

    The CSV files contain only home-game rows, so computing rest purely from
    home-to-home gaps ignores all away games played in between — a team that
    looks like they had 10 days off may have actually played 3 road games.
    """
    home = df[["team", "season", "date"]].copy()
    away = (df[["opponent", "season", "date"]]
            .rename(columns={"opponent": "team"})
            .copy())
    full = (pd.concat([home, away], ignore_index=True)
              .drop_duplicates(subset=["team", "season", "date"])
              .sort_values(["team", "season", "date"])
              .reset_index(drop=True))
    full["prev_game_date"] = full.groupby(["team", "season"])["date"].shift(1)
    return full


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-rolling derived features (schedule context, season-level rates)."""
    df = df.sort_values(["team", "season", "date"]).copy()

    # ── Full-schedule rest features ───────────────────────────────────────────
    # Use home + away appearances so rest / workload reflect the real calendar.
    full_sched = _build_full_schedule(df)

    # days_since_last_game and back_to_back from true previous game
    df = df.merge(
        full_sched[["team", "season", "date", "prev_game_date"]],
        on=["team", "season", "date"],
        how="left",
    )
    df["days_since_last_game"] = (
        (df["date"] - df["prev_game_date"]).dt.days.fillna(3)
    )
    df["back_to_back"] = (df["days_since_last_game"] <= 1).astype(int)
    df.drop(columns=["prev_game_date"], inplace=True)

    # games_in_last_10: count ALL games (home + away) in the 10 days before each home game
    sched_map: dict = {}
    for (team, season), grp in full_sched.groupby(["team", "season"]):
        sched_map[(team, season)] = grp["date"].values.astype("datetime64[D]")

    g10: list = []
    for _, row in df.iterrows():
        key  = (row["team"], row["season"])
        game_date = np.datetime64(row["date"], "D")
        all_dates = sched_map.get(key, np.array([], dtype="datetime64[D]"))
        cutoff = game_date - np.timedelta64(10, "D")
        count = int(np.sum((all_dates >= cutoff) & (all_dates < game_date)))
        g10.append(count)
    df["games_in_last_10"] = g10

    df["game_number_in_season"] = df["game_num"]
    df["pp_opportunity_rate"]   = df["team_ppo"] / (df["team_ppo"] + df["opp_ppo"] + 1e-9)

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
    """Short-window win% minus long-window win% — positive = trending up.
    Kept to 2 pairs: (5,20) captures near-term surge vs baseline;
    (10,20) captures medium-term trend. Redundant pairs removed."""
    for ws, wl in [(5, 20), (10, 20)]:
        cs, cl = f"rolling_{ws}_win_pct", f"rolling_{wl}_win_pct"
        if cs in df.columns and cl in df.columns:
            df[f"momentum_{ws}v{wl}"] = df[cs] - df[cl]
    return df


def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join each game's opponent's most recent rolling + season features."""
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    extra_cols   = [c for c in [
        "season_win_pct", "win_streak",
        "days_since_last_game", "back_to_back", "games_in_last_10",
    ] if c in df.columns]
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


def add_elo_features(df: pd.DataFrame,
                     k: float = 5.0,
                     home_adv: float = 60.0,
                     season_regress: float = 0.33,
                     start_elo: float = 1500.0) -> pd.DataFrame:
    """
    Compute pre-game Elo ratings for every team across all seasons.

    Parameters
    ----------
    k              : update factor per game (NHL ~5 — high randomness, slow updates)
    home_adv       : Elo points added to home team's expected win probability
    season_regress : fraction of Elo deviation from 1500 regressed to mean each
                     new season (0.33 → a team at 1560 starts next season at 1540)
    start_elo      : initial rating for all teams

    Adds columns
    ------------
    home_elo       : home team Elo entering this game
    away_elo       : away team Elo entering this game (opponent)
    elo_diff       : home_elo - away_elo
    elo_win_prob   : Elo-implied P(home wins) including home ice advantage
    """
    df = df.sort_values(["date", "season"]).copy()
    elos: dict = {}          # team -> current elo
    home_elo_col   = []
    away_elo_col   = []

    prev_season = None

    for _, row in df.iterrows():
        season = row["season"]
        team   = row["team"]
        opp    = row["opponent"]

        # Season-start regression to mean
        if season != prev_season:
            prev_season = season
            for t in list(elos.keys()):
                elos[t] = elos[t] + season_regress * (start_elo - elos[t])

        h_elo = elos.get(team, start_elo)
        a_elo = elos.get(opp,  start_elo)

        home_elo_col.append(h_elo)
        away_elo_col.append(a_elo)

        # Update only for played games
        if pd.notna(row.get("result")):
            expected_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo - home_adv) / 400.0))
            actual_h   = 1.0 if row["result"] == "W" else 0.0
            delta      = k * (actual_h - expected_h)
            elos[team] = h_elo + delta
            elos[opp]  = a_elo - delta

    df["home_elo"]     = home_elo_col
    df["away_elo"]     = away_elo_col
    df["elo_diff"]     = df["home_elo"] - df["away_elo"]
    df["elo_win_prob"] = 1.0 / (1.0 + 10.0 ** (
        (df["away_elo"] - df["home_elo"] - home_adv) / 400.0
    ))
    return df


def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add head-to-head win rate of home team vs opponent within the current season.
    Reset each season so prior-season history doesn't leak into future seasons.
    """
    df = df.sort_values(["season", "date"]).copy()
    df["win_binary"] = df["result"].eq("W").astype(int)

    cumulative: dict = {}
    prev_season = None
    h2h = []
    for _, row in df.iterrows():
        # Reset on new season — prevents cross-season leakage
        if row["season"] != prev_season:
            cumulative = {}
            prev_season = row["season"]
        key = (row["team"], row["opponent"])
        wins, games = cumulative.get(key, (0, 0))
        h2h.append(wins / games if games > 0 else 0.5)
        cumulative[key] = (wins + row["win_binary"], games + 1)

    df["h2h_win_pct"] = h2h
    return df.drop(columns=["win_binary"])


def add_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PDO and efficiency regression-to-mean signals (5-game vs 20-game baseline).
    short_window - long_window: positive = current metric inflated vs baseline → likely to regress.
    Kept to the (5,20) pair only — (5,10) and (10,20) are derivable combinations.
    """
    stats = [("pdo", "pdo_regress"), ("shooting_pct", "shoot_regress"), ("save_pct", "save_regress")]
    for raw_stat, prefix in stats:
        sc = f"rolling_5_{raw_stat}"
        lc = f"rolling_20_{raw_stat}"
        if sc in df.columns and lc in df.columns:
            df[f"{prefix}_5v20"] = df[sc] - df[lc]
    return df


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schedule-context differential features.
    Requires opponent features already joined (opp_days_since_last_game etc. must exist).
    Computes rest / fatigue edge between home team and opponent.
    """
    if "days_since_last_game" in df.columns and "opp_days_since_last_game" in df.columns:
        df["rest_advantage"] = df["days_since_last_game"] - df["opp_days_since_last_game"]
    if "back_to_back" in df.columns and "opp_back_to_back" in df.columns:
        # positive = home team is on a B2B and opponent is not
        df["b2b_disadvantage"] = df["back_to_back"] - df["opp_back_to_back"]
    if "games_in_last_10" in df.columns and "opp_games_in_last_10" in df.columns:
        # positive = opponent is more fatigued
        df["workload_advantage"] = df["opp_games_in_last_10"] - df["games_in_last_10"]
    return df


def add_goalie_features(df: pd.DataFrame, goalie_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-starter rolling goalie features and join to df.

    Rolls stats over the individual goalie's own starts (identified by goalie_id),
    not over team games — this prevents mixing different starters into a single
    rolling average which dilutes the goalie signal.

    Strategy:
      1. For each goalie, compute their rolling N-start averages (shift-1 to
         prevent leakage) across their career in chronological order.
      2. For each team-game in df, look up the goalie who started that game
         via goalie_df and retrieve their per-starter rolling values.

    Features added per window (prefixed rolling_{w}_goalie_):
      gsax          — Goals Saved Above Expected per game
      gsax_flurry   — Flurry-adjusted GSAx
      hd_save_pct   — High-danger save%
    Also adds:
      goalie_starts_rolling_{w} — number of starts in this goalie's rolling window
                                   (proxy for experience/sample reliability)
    """
    if goalie_df is None or goalie_df.empty:
        return df

    if "goalie_id" not in goalie_df.columns:
        # Legacy data without goalie_id — fall back to team-level rolling
        _base_cols = ["gsax", "gsax_flurry", "hd_save_pct"]
        parts = []
        for (team, season), grp in goalie_df.groupby(["team", "season"]):
            grp = grp.sort_values("date").copy()
            new_cols = []
            for col in _base_cols:
                if col not in grp.columns:
                    continue
                shifted = grp[col].shift(1)
                for w in WINDOWS:
                    feat = f"rolling_{w}_goalie_{col}"
                    grp[feat] = shifted.rolling(w, min_periods=1).mean()
                    new_cols.append(feat)
            parts.append(grp[["team", "season", "date"] + new_cols])
        if not parts:
            return df
        rolling_goalie = pd.concat(parts, ignore_index=True).sort_values(["team", "season", "date"])
        merged_parts = []
        for season, s_df in df.groupby("season"):
            g_s = rolling_goalie[rolling_goalie["season"] == season].sort_values("date").drop(columns=["season"])
            result = pd.merge_asof(s_df.sort_values("date"), g_s, on="date", by="team", direction="backward")
            merged_parts.append(result)
        return pd.concat(merged_parts).sort_values(["season", "date", "team"]).reset_index(drop=True)

    goalie_base_cols = ["gsax", "gsax_flurry", "hd_save_pct"]

    # ── Step 1: compute per-goalie rolling stats across ALL seasons ───────────
    # Sort each goalie's starts chronologically and roll over their own history.
    goalie_sorted = goalie_df.sort_values(["goalie_id", "date"]).copy()
    goalie_rolling_parts = []

    for goalie_id, grp in goalie_sorted.groupby("goalie_id"):
        grp = grp.copy()
        new_feat_cols = []
        for col in goalie_base_cols:
            if col not in grp.columns:
                continue
            shifted = grp[col].shift(1)          # shift-1: no leakage
            for w in WINDOWS:
                feat = f"rolling_{w}_goalie_{col}"
                grp[feat] = shifted.rolling(w, min_periods=1).mean()
                new_feat_cols.append(feat)

        # Count of prior starts in this window (sample reliability proxy)
        for w in WINDOWS:
            grp[f"goalie_starts_rolling_{w}"] = (
                grp["gsax"].shift(1).notna().astype(float)
                .rolling(w, min_periods=0).sum()
            )
            new_feat_cols.append(f"goalie_starts_rolling_{w}")

        goalie_rolling_parts.append(
            grp[["goalie_id", "team", "season", "date"] + new_feat_cols]
        )

    if not goalie_rolling_parts:
        return df

    per_goalie_rolling = (
        pd.concat(goalie_rolling_parts, ignore_index=True)
          .sort_values(["team", "date"])
    )

    # ── Step 2: join per-goalie rolling to game rows ──────────────────────────
    # For each home-game row in df, find the goalie who started via goalie_df,
    # then pull their per-starter rolling values.
    # merge_asof backward: for each game row, the most recent start of that
    # goalie on or before that date (= the values entering this game).
    goalie_id_map = (
        goalie_df[["team", "season", "date", "goalie_id"]]
          .sort_values("date")
          .copy()
    )

    feat_cols_to_join = [c for c in per_goalie_rolling.columns
                         if (c.startswith("rolling_") and "goalie" in c)
                         or c.startswith("goalie_starts_")]

    # ── Step 2b: compute team-level rolling as fallback for debut/cold-start goalies ─
    # When a goalie makes their first appearance, per-starter rolling is NaN.
    # Fall back to shift-1 rolling over ALL starters for that team (same as the
    # original team-level approach), so there's always a value.
    team_fallback_parts = []
    for (team, season), grp in goalie_df.groupby(["team", "season"]):
        grp = grp.sort_values("date").copy()
        fb_cols = []
        for col in goalie_base_cols:
            if col not in grp.columns:
                continue
            shifted = grp[col].shift(1)
            for w in WINDOWS:
                feat = f"fb_{w}_goalie_{col}"
                grp[feat] = shifted.rolling(w, min_periods=1).mean()
                fb_cols.append(feat)
        team_fallback_parts.append(grp[["team", "season", "date"] + fb_cols])

    team_fallback = (
        pd.concat(team_fallback_parts, ignore_index=True)
          .sort_values(["team", "date"])
    )

    merged_parts = []
    for season, s_df in df.groupby("season"):
        # First: attach the goalie_id who started each home game this season
        gid_season = (goalie_id_map[goalie_id_map["season"] == season]
                      [["team", "date", "goalie_id"]]
                      .sort_values("date"))

        s_df = pd.merge_asof(
            s_df.sort_values("date"),
            gid_season,
            on="date",
            by="team",
            direction="backward",
        )

        # Second: for each (goalie_id, date), pull their per-starter rolling history
        pgr = per_goalie_rolling.sort_values("date")
        s_df = pd.merge_asof(
            s_df.sort_values("date"),
            pgr[["goalie_id", "date"] + feat_cols_to_join].sort_values("date"),
            on="date",
            by="goalie_id",
            direction="backward",
        )

        # Third: join team-level fallback and fill NaN per-starter values
        fb_s = (team_fallback[team_fallback["season"] == season]
                .drop(columns=["season"])
                .sort_values("date"))
        s_df = pd.merge_asof(
            s_df.sort_values("date"),
            fb_s,
            on="date",
            by="team",
            direction="backward",
        )
        for col in goalie_base_cols:
            for w in WINDOWS:
                feat = f"rolling_{w}_goalie_{col}"
                fb   = f"fb_{w}_goalie_{col}"
                if feat in s_df.columns and fb in s_df.columns:
                    s_df[feat] = s_df[feat].fillna(s_df[fb])
        # Drop fallback columns
        fb_drop = [c for c in s_df.columns if c.startswith("fb_")]
        s_df = s_df.drop(columns=fb_drop)

        merged_parts.append(s_df)

    df = pd.concat(merged_parts).sort_values(["season", "date", "team"]).reset_index(drop=True)
    # Drop the helper goalie_id column from the main df (it's team-level data)
    df = df.drop(columns=["goalie_id"], errors="ignore")
    return df


def add_xg_features(df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling xG features from MoneyPuck game-by-game data and join to df.

    xg_df contains one row per (team, date) for BOTH home and away games, so
    rolling windows reflect true recent form — not just home appearances.
    The rolling values are shift(1) before the window to prevent leakage.

    Features added per window (all prefixed rolling_{w}_):
      xgf_ev, xga_ev, xgp_ev          — even strength quality
      xgf_pp, xga_pp                   — power play offense / PK from opp view
      xgf_pk, xga_pk                   — penalty kill
      xgf_all, xga_all, xgp_all        — all-situations totals
      xg_combined_all                  — xgf_all + xga_all (total expected goals)
    """
    if xg_df is None or xg_df.empty:
        return df

    xg_base_cols = [c for c in xg_df.columns
                    if c.startswith("xg") and c not in ("xg_combined_all",)]
    if "xg_combined_all" in xg_df.columns:
        xg_base_cols.append("xg_combined_all")

    parts = []
    for (team, season), grp in xg_df.groupby(["team", "season"]):
        grp = grp.sort_values("date").copy()
        new_cols = []
        for col in xg_base_cols:
            if col not in grp.columns:
                continue
            shifted = grp[col].shift(1)
            for w in WINDOWS:
                feat = f"rolling_{w}_{col}"
                grp[feat] = shifted.rolling(w, min_periods=1).mean()
                new_cols.append(feat)
        parts.append(grp[["team", "season", "date"] + new_cols])

    if not parts:
        return df

    rolling_xg = pd.concat(parts, ignore_index=True).sort_values(["team", "season", "date"])

    # merge_asof (backward): for each game row in df, pick the most recent xG rolling
    # values on or before that date. This correctly handles upcoming games (no xG entry
    # yet) by carrying forward the last known rolling average.
    merged_parts = []
    for season, s_df in df.groupby("season"):
        xg_s = rolling_xg[rolling_xg["season"] == season].sort_values("date").drop(columns=["season"])
        result = pd.merge_asof(
            s_df.sort_values("date"),
            xg_s,
            on="date",
            by="team",
            direction="backward",
        )
        merged_parts.append(result)

    df = pd.concat(merged_parts).sort_values(["season", "date", "team"]).reset_index(drop=True)
    return df


def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Home team minus opponent for key rolling metrics.
    Captures relative matchup quality directly as a single number.
    """
    diffs = {
        "win_pct":           "win_pct_diff",
        "goals_for_avg":     "gf_diff",
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


def add_standings_features(df: pd.DataFrame,
                           standings_df: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """
    Join playoff motivation features to each game row using merge_asof (backward).

    For each game, finds the most recent standings snapshot before that date and
    joins: points, games_remaining, conf_rank, points_above_cut, clinch_status,
    motivation — for both home team and opponent (via opp_ prefix).

    standings_df: output of standings.build_standings_features(). If None,
                  loads from cached CSVs automatically.
    """
    from pipeline.standings import build_standings_features, load_all_standings

    if standings_df is None:
        raw = load_all_standings()
        if raw.empty:
            return df
        standings_df = build_standings_features(raw)

    if standings_df.empty:
        return df

    feat_cols = ["points", "games_remaining", "conf_rank",
                 "points_above_cut", "clinch_status", "motivation"]
    feat_cols = [c for c in feat_cols if c in standings_df.columns]

    merged_parts = []
    for season, s_df in df.groupby("season"):
        st_s = (standings_df[standings_df["season"] == season]
                .sort_values("date")
                .copy())
        if st_s.empty:
            merged_parts.append(s_df)
            continue

        # Home team standings
        home_st = st_s[["team", "date"] + feat_cols].copy()
        result = pd.merge_asof(
            s_df.sort_values("date"),
            home_st.sort_values("date"),
            on="date",
            by="team",
            direction="backward",
        )

        # Opponent standings (join on opponent key)
        opp_st = st_s[["team", "date"] + feat_cols].rename(
            columns={"team": "opponent"} | {c: f"opp_{c}" for c in feat_cols}
        )
        result = pd.merge_asof(
            result.sort_values("date"),
            opp_st.sort_values("date"),
            on="date",
            by="opponent",
            direction="backward",
        )

        # Derived: motivation differential (home − away)
        if "motivation" in result.columns and "opp_motivation" in result.columns:
            result["motivation_diff"] = result["motivation"] - result["opp_motivation"]
        if "points_above_cut" in result.columns and "opp_points_above_cut" in result.columns:
            result["pts_above_cut_diff"] = result["points_above_cut"] - result["opp_points_above_cut"]

        merged_parts.append(result)

    if not merged_parts:
        return df
    return (pd.concat(merged_parts)
              .sort_values(["season", "date", "team"])
              .reset_index(drop=True))


def build_features(df: pd.DataFrame,
                   xg_df: "pd.DataFrame | None" = None,
                   goalie_df: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    xg_df:     optional MoneyPuck game-by-game xG DataFrame from xg_loader.
    goalie_df: optional per-game starter goalie stats from goalie_loader.
               Both are joined before add_opponent_features so opp_* columns
               are created automatically.
    """
    df = compute_team_features(df)
    df = add_derived_features(df)
    df = add_streak_features(df)
    if xg_df is not None and not xg_df.empty:
        df = add_xg_features(df, xg_df)       # before opponent join → opp_xg_ auto-created
    if goalie_df is not None and not goalie_df.empty:
        df = add_goalie_features(df, goalie_df)  # before opponent join → opp_goalie_ auto-created
    df = add_opponent_features(df)        # opp_* schedule + xG + goalie cols created here
    df = add_standings_features(df)       # motivation / playoff pressure features
    df = add_elo_features(df)             # cumulative power rankings across all seasons
    df = add_h2h_features(df)
    df = add_momentum_features(df)
    df = add_regression_features(df)      # PDO / efficiency regression signals
    df = add_schedule_features(df)        # rest / fatigue diffs vs opponent
    df = add_differential_features(df)
    return df


