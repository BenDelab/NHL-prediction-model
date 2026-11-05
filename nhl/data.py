"""Load regular-season games. One row per game, plus one row per team per game.

MoneyPuck's team file is the source of every game stat, and of every result except
shootouts: its goal counts leave out the shootout, so data/shootouts.csv supplies those
winners from the NHL API. Game length tells regulation from overtime.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MONEYPUCK_FILE = ROOT / "data" / "moneypuck_all_teams.csv"
SHOOTOUT_FILE = ROOT / "data" / "shootouts.csv"
FIRST_YEAR = 2008
LAST_YEAR = 2026
LAST_FULL_YEAR = 2025  # the latest season that has finished
TEAM_MAP = {"ARI": "UTA", "VGK": "VEG", "PHX": "UTA", "L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}
REGULAR_SEASON = "02"  # digits 5-6 of an NHL gameId: 01 pre, 02 regular, 03 playoffs
FULL_GAME_SECONDS = 3600
INCOMPLETE_SECONDS = 3500  # a March 2020 game has only 20 minutes on record


def season_name(year: int) -> str:
    return f"{year}-{(year + 1) % 100:02d}"


SEASONS = [season_name(year) for year in range(2021, LAST_FULL_YEAR + 1)]


def _load_moneypuck(first_year: int, last_year: int) -> pd.DataFrame:
    raw = pd.read_csv(MONEYPUCK_FILE, low_memory=False)
    keep = raw["season"].between(first_year, last_year) & (raw["gameId"].astype(str).str[4:6] == REGULAR_SEASON)
    raw = raw[keep].copy()
    raw["season"] = raw["season"].map(season_name)
    raw = _repair_arizona_home_games(raw)
    full = raw[raw["situation"] == "all"]
    short = full.loc[full["iceTime"] < INCOMPLETE_SECONDS, "gameId"].unique()
    return raw[~raw["gameId"].isin(short)]


def _repair_arizona_home_games(raw: pd.DataFrame) -> pd.DataFrame:
    """Arizona's 2011-14 home games list both teams as away, and Arizona as its own opponent."""
    own_opponent = raw["playerTeam"] == raw["opposingTeam"]
    visitors = raw[~own_opponent].drop_duplicates("gameId").set_index("gameId")["playerTeam"]
    raw.loc[own_opponent, "opposingTeam"] = raw.loc[own_opponent, "gameId"].map(visitors)
    raw.loc[own_opponent, "home_or_away"] = "HOME"
    return raw


def _team_game_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the situation rows into one row of raw stats per team-game."""
    keys = ["gameId", "playerTeam"]
    full = raw[raw["situation"] == "all"].set_index(keys)
    even = raw[raw["situation"] == "5on5"].set_index(keys)
    power_play = raw[raw["situation"] == "5on4"].set_index(keys)
    penalty_kill = raw[raw["situation"] == "4on5"].set_index(keys)

    stats = pd.DataFrame({
        "season": full["season"],
        "date": pd.to_datetime(full["gameDate"].astype(str), format="%Y%m%d"),
        "opponent": full["opposingTeam"].replace(TEAM_MAP),
        "is_home": full["home_or_away"].eq("HOME").astype(float),
        "ice_seconds": full["iceTime"],
        "xgf": full["scoreVenueAdjustedxGoalsFor"],
        "xga": full["scoreVenueAdjustedxGoalsAgainst"],
        "xgf_5v5": even["scoreVenueAdjustedxGoalsFor"],
        "xga_5v5": even["scoreVenueAdjustedxGoalsAgainst"],
        "cf_5v5": even["scoreAdjustedShotsAttemptsFor"]
        / (even["scoreAdjustedShotsAttemptsFor"] + even["scoreAdjustedShotsAttemptsAgainst"]),
        "pp_xgf": power_play["xGoalsFor"],
        "pk_xga": penalty_kill["xGoalsAgainst"],
        "gf": full["goalsFor"],
        "ga": full["goalsAgainst"],
        "gsax": full["xGoalsAgainst"] - full["goalsAgainst"],
        "finishing": full["goalsFor"] - full["xGoalsFor"],
        "penalty_diff": full["penaltiesAgainst"] - full["penaltiesFor"],
    }).reset_index().rename(columns={"playerTeam": "team"})
    stats["team"] = stats["team"].replace(TEAM_MAP)
    return stats


def _add_outcomes(games: pd.DataFrame) -> pd.DataFrame:
    shootouts = pd.read_csv(SHOOTOUT_FILE).set_index("gameId")
    tied = games["gf"] == games["ga"]
    missing = set(games.loc[tied, "gameId"]) - set(shootouts.index)
    if missing:
        raise ValueError(f"{len(missing)} tied games have no shootout result; run nhl.fetch_shootouts")
    games["home_win"] = (games["gf"] > games["ga"]).astype(int)
    tied_rows = shootouts.reindex(games.loc[tied, "gameId"])
    games.loc[tied, "home_win"] = (tied_rows["home_score"] > tied_rows["away_score"]).astype(int).to_numpy()
    longer = games["ice_seconds"] > FULL_GAME_SECONDS
    games["overtime_type"] = np.where(tied, "SO", np.where(longer, "OT", "REG"))
    games["extra_time"] = (tied | longer).astype(int)
    return games


def load_games(first_year: int = 2021, last_year: int = LAST_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (games, team_games), checked and sorted by date."""
    team_games = _team_game_stats(_load_moneypuck(first_year, last_year))
    home = team_games[team_games["is_home"] == 1]
    games = home[["gameId", "season", "date", "team", "opponent", "gf", "ga", "ice_seconds"]].rename(
        columns={"team": "home", "opponent": "away"}
    )
    games = _add_outcomes(games).sort_values(["date", "gameId"]).reset_index(drop=True)
    _validate(games, team_games)
    return games, team_games.sort_values(["date", "gameId", "team"]).reset_index(drop=True)


def _validate(games: pd.DataFrame, team_games: pd.DataFrame) -> None:
    if (team_games.groupby("gameId").size() != 2).any():
        raise ValueError("every game needs exactly two team rows")
    if (team_games.groupby("gameId")["is_home"].sum() != 1).any():
        raise ValueError("every game needs exactly one home team")
    if len(games) != team_games["gameId"].nunique():
        raise ValueError("a game was lost while building the games table")
    if team_games.duplicated(["team", "date"]).any():
        raise ValueError("a team has two games on one date")
    if games["home_win"].isna().any():
        raise ValueError("a game has no winner")
