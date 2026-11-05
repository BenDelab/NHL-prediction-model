"""Turn games into model inputs. This is the only module that reads game history.

The no-leakage rule lives in `build_sequences`: a game's sequence is the team's
earlier games in that season, ending before the game being predicted. Elo and the
schedule context are also pre-game values. tests/test_leakage.py enforces all of it.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

ARENAS = {
    "ANA": (33.81, -117.88), "BOS": (42.37, -71.06), "BUF": (42.87, -78.88),
    "CAR": (35.80, -78.72), "CBJ": (39.97, -83.01), "CGY": (51.04, -114.05),
    "CHI": (41.88, -87.67), "COL": (39.75, -105.01), "DAL": (32.79, -96.81),
    "DET": (42.34, -83.06), "EDM": (53.55, -113.50), "FLA": (26.16, -80.33),
    "LAK": (34.04, -118.27), "MIN": (44.94, -93.10), "MTL": (45.50, -73.57),
    "NJD": (40.73, -74.17), "NSH": (36.16, -86.78), "NYI": (40.72, -73.73),
    "NYR": (40.75, -73.99), "OTT": (45.30, -75.93), "PHI": (39.90, -75.17),
    "PIT": (40.44, -79.99), "SEA": (47.62, -122.35), "SJS": (37.33, -121.90),
    "STL": (38.63, -90.20), "TBL": (27.94, -82.45), "TOR": (43.64, -79.38),
    "UTA": (40.77, -111.90), "VAN": (49.28, -123.11), "VEG": (36.10, -115.18),
    "WPG": (49.89, -97.14), "WSH": (38.90, -77.02),
    "ARI": (33.45, -112.07), "ATL": (33.76, -84.39),
}
UTAH_FIRST_SEASON = "2024-25"  # Arizona seasons share the UTA code but played in Arizona
CHANNELS = [
    "is_home", "won", "xgf", "xga", "xgf_5v5", "xga_5v5", "cf_5v5",
    "pp_xgf", "pk_xga", "gf", "ga", "gsax", "finishing", "penalty_diff",
    "opp_elo", "rest", "travel",
]
ERA_STATS = ["xgf", "xga", "xgf_5v5", "xga_5v5", "pp_xgf", "pk_xga", "gf", "ga", "gsax", "finishing"]
ERA_PRIOR_GAMES = 200  # team-games of last season's average blended in while this season is young
STATIC = [
    "elo_diff", "rest_home", "rest_away", "b2b_home", "b2b_away",
    "travel_home", "travel_away", "load_home", "load_away", "season_progress",
]
MAX_REST = 7
GAMES_PER_SEASON = 1312


def _distance_km(a: str, b: str) -> float:
    (lat1, lon1), (lat2, lon2) = np.radians(ARENAS[a]), np.radians(ARENAS[b])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(h)))


def _location(team: str, season: str) -> str:
    return "ARI" if team == "UTA" and season < UTAH_FIRST_SEASON else team


def era_adjust(team_games: pd.DataFrame) -> pd.DataFrame:
    """Express stats relative to the league's level that season, using earlier dates only.

    Early in a season the level leans on last season's average. Opening night of the
    very first season has no earlier games, so it stays unadjusted.
    """
    frame = team_games.copy()
    frame[ERA_STATS] = frame[ERA_STATS].astype(float)
    original = team_games[ERA_STATS].astype(float)
    previous_mean = None
    for season in sorted(frame["season"].unique()):
        rows = frame["season"] == season
        by_date = original[rows].groupby(frame.loc[rows, "date"]).agg(["sum", "count"]).sort_index()
        for stat in ERA_STATS:
            total = by_date[(stat, "sum")].cumsum().shift(1).fillna(0)
            count = by_date[(stat, "count")].cumsum().shift(1).fillna(0)
            if previous_mean is None:
                level = (total / count.replace(0, np.nan)).fillna(0)
            else:
                level = (total + ERA_PRIOR_GAMES * previous_mean[stat]) / (count + ERA_PRIOR_GAMES)
            frame.loc[rows, stat] = original.loc[rows, stat] - frame.loc[rows, "date"].map(level)
        previous_mean = original[rows].mean()
    return frame


def add_elo(games: pd.DataFrame, k: float = 5.0, home_edge: float = 60.0,
            carry_over: float = 0.67) -> pd.DataFrame:
    """Add each team's Elo going into the game. Ratings update only after it."""
    ratings: dict[str, float] = {}
    elo_home, elo_away = [], []
    season = None
    for game in games.itertuples():
        if game.season != season:
            season = game.season
            ratings = {t: 1500 + carry_over * (r - 1500) for t, r in ratings.items()}
        home, away = ratings.get(game.home, 1500.0), ratings.get(game.away, 1500.0)
        elo_home.append(home)
        elo_away.append(away)
        expected = 1 / (1 + 10 ** ((away - home - home_edge) / 400))
        change = k * (game.home_win - expected)
        ratings[game.home], ratings[game.away] = home + change, away - change
    games = games.copy()
    games["elo_home"], games["elo_away"] = elo_home, elo_away
    games["elo_diff"] = games["elo_home"] - games["elo_away"]
    return games


def add_schedule(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add rest, travel and recent workload, all known before a game starts."""
    frame = team_games.sort_values(["season", "team", "date"]).copy()
    rest, travel, load = [], [], []
    for _, group in frame.groupby(["season", "team"], sort=False):
        location = None
        dates = group["date"].to_numpy()
        for i, row in enumerate(group.itertuples()):
            here = _location(row.team if row.is_home == 1 else row.opponent, row.season)
            rest.append(MAX_REST if i == 0 else min(int((dates[i] - dates[i - 1]) / np.timedelta64(1, "D")), MAX_REST))
            travel.append(0.0 if location is None else _distance_km(location, here))
            load.append(int(np.sum((dates[:i] >= dates[i] - np.timedelta64(7, "D")))))
            location = here
    frame["rest"], frame["travel"], frame["load"] = rest, travel, load
    return frame


def assemble(games: pd.DataFrame, team_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach Elo, schedule and outcome columns to both tables."""
    games = add_elo(games)
    team_games = add_schedule(team_games)

    outcome = games[["gameId", "home_win", "elo_home", "elo_away", "home"]]
    team_games = team_games.merge(outcome, on="gameId", validate="m:1")
    is_home = team_games["team"] == team_games["home"]
    team_games["won"] = np.where(is_home, team_games["home_win"], 1 - team_games["home_win"]).astype(float)
    team_games["opp_elo"] = np.where(is_home, team_games["elo_away"], team_games["elo_home"])

    for side, code in [("home", games["home"]), ("away", games["away"])]:
        lookup = team_games.set_index(["gameId", "team"])
        index = pd.MultiIndex.from_arrays([games["gameId"], code])
        games[f"rest_{side}"] = lookup["rest"].reindex(index).to_numpy()
        games[f"b2b_{side}"] = (games[f"rest_{side}"] == 1).astype(float)
        games[f"travel_{side}"] = lookup["travel"].reindex(index).to_numpy()
        games[f"load_{side}"] = lookup["load"].reindex(index).to_numpy()
    games["season_progress"] = games.groupby("season").cumcount() / GAMES_PER_SEASON
    return games, team_games


@dataclass
class Sequences:
    home: np.ndarray       # (games, length, channels), oldest game first
    away: np.ndarray
    home_mask: np.ndarray  # (games, length), 1 where a real game exists
    away_mask: np.ndarray


def build_sequences(games: pd.DataFrame, team_games: pd.DataFrame, length: int,
                    leak_current_game: bool = False) -> Sequences:
    """Each team's last `length` games before this one, this season only.

    `leak_current_game` exists only so the tests can prove leaks get caught.
    """
    history: dict[tuple[str, str], tuple[np.ndarray, dict]] = {}
    for key, group in team_games.sort_values("date").groupby(["season", "team"]):
        positions = {game_id: i for i, game_id in enumerate(group["gameId"])}
        history[key] = (group[CHANNELS].to_numpy(dtype=np.float32), positions)

    n = len(games)
    arrays = {side: np.zeros((n, length, len(CHANNELS)), np.float32) for side in ("home", "away")}
    masks = {side: np.zeros((n, length), np.float32) for side in ("home", "away")}
    for row, game in enumerate(games.itertuples()):
        for side, team in (("home", game.home), ("away", game.away)):
            values, positions = history[(game.season, team)]
            end = positions[game.gameId] + (1 if leak_current_game else 0)
            past = values[max(0, end - length):end]
            if len(past):
                arrays[side][row, -len(past):] = past
                masks[side][row, -len(past):] = 1
    return Sequences(arrays["home"], arrays["away"], masks["home"], masks["away"])


class Scaler:
    """Standardise inputs using training seasons only."""

    def __init__(self, team_games: pd.DataFrame, games: pd.DataFrame, static: list[str] = STATIC):
        self.columns = static
        self.channel_mean = team_games[CHANNELS].mean().to_numpy(dtype=np.float32)
        self.channel_std = team_games[CHANNELS].std().to_numpy(dtype=np.float32) + 1e-6
        self.static_mean = games[static].mean().to_numpy(dtype=np.float32)
        self.static_std = games[static].std().to_numpy(dtype=np.float32) + 1e-6

    def sequence(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return (values - self.channel_mean) / self.channel_std * mask[..., None]

    def static(self, games: pd.DataFrame) -> np.ndarray:
        return (games[self.columns].to_numpy(dtype=np.float32) - self.static_mean) / self.static_std
