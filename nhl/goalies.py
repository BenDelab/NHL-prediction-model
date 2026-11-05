"""Add the starting goalie's record going into each game.

Who starts is public about an hour before puck drop, so the starter's name counts
as pre-game information. Their stats stop at their previous appearance.
"""
import numpy as np
import pandas as pd

from .data import ROOT, SEASONS

SIDES = ("home", "away")
GOALIE_STATIC = [f"{stat}_{side}" for stat in ("goalie_gsax", "goalie_recent", "goalie_share") for side in SIDES]
SHRINK_GAMES = 10  # pulls goalies with little history toward average
RECENT_STARTS = 10
SHARE_GAMES = 10
GAME_SECONDS = 3600
FIRST_SEASON = int(SEASONS[0][:4])


def load_appearances() -> pd.DataFrame:
    """One row per goalie per regular-season game."""
    frames = []
    for year in range(FIRST_SEASON, FIRST_SEASON + len(SEASONS)):
        frame = pd.read_csv(ROOT / "data" / "goalie_stats" / f"NHL_goalie_{year}.csv",
                            usecols=["playerId", "gameId", "gameDate", "situation", "icetime", "xGoals", "goals"])
        frames.append(frame[(frame["situation"] == "all") & (frame["gameId"].astype(str).str[4:6] == "02")])
    log = pd.concat(frames, ignore_index=True)
    log["gsax"] = log["xGoals"] - log["goals"]
    return log.rename(columns={"playerId": "goalie_id"})


def goalie_form(log: pd.DataFrame) -> pd.DataFrame:
    """Each goalie's record before each appearance. shift(1) excludes the game itself."""
    log = log.sort_values(["goalie_id", "gameDate", "gameId"]).copy()
    by_goalie = log.groupby("goalie_id")
    prior_gsax = by_goalie["gsax"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_games = by_goalie["icetime"].transform(lambda s: s.shift(1).cumsum()).fillna(0) / GAME_SECONDS
    log["goalie_gsax"] = prior_gsax / (prior_games + SHRINK_GAMES)
    recent_sum = by_goalie["gsax"].transform(lambda s: s.shift(1).rolling(RECENT_STARTS, min_periods=1).sum()).fillna(0)
    recent_count = by_goalie["gsax"].transform(lambda s: s.shift(1).rolling(RECENT_STARTS, min_periods=1).count())
    log["goalie_recent"] = recent_sum / (recent_count + SHRINK_GAMES)
    return log[["goalie_id", "gameId", "goalie_gsax", "goalie_recent"]]


def starter_share(starters: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """How many of the team's previous games this goalie started: starter or backup?"""
    table = starters.merge(team_games[["gameId", "team", "date", "season"]], on=["gameId", "team"])
    table = table.sort_values(["season", "team", "date"])
    shares = []
    for _, group in table.groupby(["season", "team"], sort=False):
        ids = group["goalie_id"].to_numpy()
        for i, goalie in enumerate(ids):
            past = ids[max(0, i - SHARE_GAMES):i]
            shares.append(float(np.mean(past == goalie)) if len(past) else 0.5)
    table["goalie_share"] = shares
    return table[["gameId", "team", "goalie_share"]]


def add_goalies(games: pd.DataFrame, team_games: pd.DataFrame, log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add GOALIE_STATIC columns. Games lacking a known starter get average values."""
    starters = pd.read_csv(ROOT / "data" / "starters.csv")
    starters = starters[starters["starters_flagged"] == 1].copy()
    starters["goalie_id"] = starters["goalie_id"].astype(int)
    form = goalie_form(load_appearances() if log is None else log)
    table = starters.merge(form, on=["goalie_id", "gameId"], how="left")
    table = table.merge(starter_share(starters, team_games), on=["gameId", "team"], how="left")

    games = games.copy()
    for side, flag in zip(SIDES, (1, 0)):
        rows = table[table["is_home"] == flag].set_index("gameId")
        for stat in ("goalie_gsax", "goalie_recent", "goalie_share"):
            games[f"{stat}_{side}"] = rows[stat].reindex(games["gameId"]).to_numpy()
    missing = games[GOALIE_STATIC].isna().any(axis=1)
    games[GOALIE_STATIC] = games[GOALIE_STATIC].fillna({c: 0.5 if "share" in c else 0.0 for c in GOALIE_STATIC})
    games.attrs["games_missing_starter"] = int(missing.sum())
    return games
