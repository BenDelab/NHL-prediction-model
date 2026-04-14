"""
Load NHL goalie game-by-game stats from goalie_stats/NHL_goalie_*.csv.

Returns one row per (team, date) representing the starting goalie's performance
(starter = highest icetime in that game). Features computed:

  gsax          — Goals Saved Above Expected  (xGoals - goals; positive = better than expected)
  gsax_flurry   — Flurry-adjusted GSAx        (flurryAdjustedxGoals - goals)
  hd_save_pct   — High-danger save%           (1 - hdGoals / (hdShots + ε))

These are used by add_goalie_features() in features.py to compute rolling averages
per team over [5, 10, 15, 20] game windows (shift-1 to prevent leakage).

fetch_probable_goalies(date_str) uses the NHL API gamecenter/landing endpoint to
identify the probable starter for each upcoming game on a given date, returning
season stats (GAA, SV%) alongside the goalie name.
"""

from __future__ import annotations

import glob
import os
import time
import requests
import pandas as pd

GOALIE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "goalie_stats")

# MoneyPuck season int → pipeline season string
_SEASON_MAP = {
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}


def load_goalie_all_seasons() -> pd.DataFrame:
    """
    Load all NHL_goalie_*.csv files, deduplicate to one row per (team, date)
    using the starting goalie (max icetime, situation='all'), and return
    a DataFrame with raw per-game goalie metrics.

    Columns returned: team, date, season, gsax, gsax_flurry, hd_save_pct
    """
    pattern = os.path.join(GOALIE_DIR, "NHL_goalie_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("[goalie_loader] No NHL_goalie_*.csv files found — goalie features disabled.")
        return pd.DataFrame()

    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[goalie_loader] Could not read {os.path.basename(path)}: {e}")
            continue

        # Full-game situation only
        df = df[df["situation"] == "all"].copy()
        if df.empty:
            continue

        # Map season int → pipeline string; skip unknown seasons
        df["season"] = df["season"].map(_SEASON_MAP)
        df = df.dropna(subset=["season"])

        # Parse date
        df["date"] = pd.to_datetime(df["gameDate"].astype(str), format="%Y%m%d")

        # Normalise team abbreviation (Arizona → Utah)
        df["team"] = df["playerTeam"].replace("ARI", "UTA")

        # One starter per (team, game): highest icetime
        df = (
            df.sort_values("icetime", ascending=False)
              .drop_duplicates(subset=["team", "date"])
              .copy()
        )

        # Core goalie metrics
        df["gsax"]        = df["xGoals"]               - df["goals"]
        df["gsax_flurry"] = df["flurryAdjustedxGoals"] - df["goals"]
        df["hd_save_pct"] = 1.0 - df["highDangerGoals"] / (df["highDangerShots"] + 1e-9)

        dfs.append(df[["team", "date", "season", "gsax", "gsax_flurry", "hd_save_pct"]])

    if not dfs:
        return pd.DataFrame()

    combined = (
        pd.concat(dfs, ignore_index=True)
          .sort_values(["team", "season", "date"])
          .reset_index(drop=True)
    )
    seasons = combined["season"].nunique()
    teams   = combined["team"].nunique()
    print(f"[goalie_loader] Loaded {len(combined)} team-game rows "
          f"({teams} teams × {seasons} seasons).")
    return combined


# ── NHL API endpoints ─────────────────────────────────────────────────────────
_SCHEDULE_URL  = "https://api-web.nhle.com/v1/schedule/{date}"
_LANDING_URL   = "https://api-web.nhle.com/v1/gamecenter/{game_id}/landing"
_BOXSCORE_URL  = "https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"

# Team abbrev normalisations: NHL API → pipeline convention
_RENAMES = {"ARI": "UTA", "VGK": "VEG"}


def _get(url: str, timeout: int = 10) -> dict:
    """GET a URL, return JSON dict or {} on failure."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _extract_season_stats(goalie_list: list, player_id: int) -> dict:
    """Pull GAA and save% for a specific playerId from goalieSeasonStats.goalies[]."""
    for g in goalie_list:
        if g.get("playerId") == player_id:
            return {
                "gaa":      round(g.get("goalsAgainstAvg", float("nan")), 3),
                "sv_pct":   round(g.get("savePctg",        float("nan")), 4),
                "gp":       g.get("gamesPlayed", 0),
                "record":   f"{g.get('wins',0)}-{g.get('losses',0)}-{g.get('otLosses',0)}",
            }
    return {}


def fetch_probable_goalies(date_str: str) -> dict:
    """
    Fetch the probable starting goalie for every game on date_str.

    Strategy (in order of reliability):
      1. Boxscore playerByGameStats — has 'starter: true' when lineup is confirmed
         (available ~1 hour before puck drop and once the game starts).
      2. Landing page goalieComparison.leaders[0] — team's primary goalie ranked
         by season performance; used as fallback when lineup isn't yet official.

    Returns
    -------
    dict keyed by (home_abbrev, away_abbrev) with values:
      {
        "home_name":   str,   "away_name":   str,
        "home_gaa":    float, "away_gaa":    float,
        "home_sv_pct": float, "away_sv_pct": float,
        "home_gp":     int,   "away_gp":     int,
        "home_record": str,   "away_record": str,
        "source":      str,   # "confirmed" or "probable"
      }
    """
    sched = _get(_SCHEDULE_URL.format(date=date_str))
    games = (sched.get("gameWeek") or [{}])[0].get("games", [])

    result = {}
    for game in games:
        game_id   = game.get("id")
        home_abbr = (game.get("homeTeam") or {}).get("abbrev", "")
        away_abbr = (game.get("awayTeam") or {}).get("abbrev", "")
        if not game_id or not home_abbr or not away_abbr:
            continue

        home_abbr = _RENAMES.get(home_abbr, home_abbr)
        away_abbr = _RENAMES.get(away_abbr, away_abbr)

        # ── Attempt 1: confirmed lineup from boxscore ─────────────────────────
        box = _get(_BOXSCORE_URL.format(game_id=game_id))
        by_game = box.get("playerByGameStats", {})
        home_goalies = by_game.get("homeTeam", {}).get("goalies", [])
        away_goalies = by_game.get("awayTeam", {}).get("goalies", [])

        home_starter = next((g for g in home_goalies if g.get("starter")), None)
        away_starter = next((g for g in away_goalies if g.get("starter")), None)

        if home_starter and away_starter:
            result[(home_abbr, away_abbr)] = {
                "home_name":   home_starter.get("name", {}).get("default", "Unknown"),
                "away_name":   away_starter.get("name", {}).get("default", "Unknown"),
                "home_gaa":    float("nan"),
                "away_gaa":    float("nan"),
                "home_sv_pct": float("nan"),
                "away_sv_pct": float("nan"),
                "home_gp":     0,
                "away_gp":     0,
                "home_record": "",
                "away_record": "",
                "source":      "confirmed",
            }
            time.sleep(0.1)
            continue

        # ── Attempt 2: probable starter from landing page ─────────────────────
        landing = _get(_LANDING_URL.format(game_id=game_id))
        matchup = landing.get("matchup") or {}
        gc      = matchup.get("goalieComparison") or {}
        gs_list = (matchup.get("goalieSeasonStats") or {}).get("goalies", [])

        home_leaders = (gc.get("homeTeam") or {}).get("leaders", [])
        away_leaders = (gc.get("awayTeam") or {}).get("leaders", [])

        if not home_leaders or not away_leaders:
            time.sleep(0.1)
            continue

        h_lead = home_leaders[0]
        a_lead = away_leaders[0]
        h_id   = h_lead.get("playerId")
        a_id   = a_lead.get("playerId")

        h_stats = _extract_season_stats(gs_list, h_id)
        a_stats = _extract_season_stats(gs_list, a_id)

        result[(home_abbr, away_abbr)] = {
            "home_name":   h_lead.get("name", {}).get("default", "Unknown"),
            "away_name":   a_lead.get("name", {}).get("default", "Unknown"),
            "home_gaa":    h_stats.get("gaa",    float("nan")),
            "away_gaa":    a_stats.get("gaa",    float("nan")),
            "home_sv_pct": h_stats.get("sv_pct", float("nan")),
            "away_sv_pct": a_stats.get("sv_pct", float("nan")),
            "home_gp":     h_stats.get("gp",     0),
            "away_gp":     a_stats.get("gp",     0),
            "home_record": h_stats.get("record", ""),
            "away_record": a_stats.get("record", ""),
            "source":      "probable",
        }
        time.sleep(0.1)

    return result
