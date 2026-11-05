"""Prove a game's inputs cannot see that game or anything after it."""
import numpy as np
import pandas as pd
import pytest

from nhl.data import load_games
from nhl.features import STATIC, assemble, build_sequences

CUTOFFS = ["2023-12-12", "2024-02-29", "2025-01-14", "2026-03-17"]
LENGTH = 40


@pytest.fixture(scope="module")
def tables():
    return load_games()


def _corrupt_from(games, team_games, cutoff):
    """Scramble every result and stat dated on or after the cutoff."""
    rng = np.random.default_rng(0)
    games, team_games = games.copy(), team_games.copy()
    bad_games = games["date"] >= cutoff
    games.loc[bad_games, "home_win"] = rng.integers(0, 2, bad_games.sum())
    games.loc[bad_games, "overtime_type"] = "SO"
    games.loc[bad_games, "extra_time"] = 1
    bad = team_games["date"] >= cutoff
    for column in ["xgf", "xga", "xgf_5v5", "xga_5v5", "cf_5v5", "pp_xgf", "pk_xga", "gf", "ga",
                   "gsax", "finishing", "penalty_diff"]:
        team_games[column] = team_games[column].astype(float)
        team_games.loc[bad, column] = rng.normal(50, 25, bad.sum())
    return games, team_games


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_inputs_ignore_that_day_and_the_future(tables, cutoff):
    games, team_games = tables
    clean_games, clean_teams = assemble(games, team_games)
    dirty_games, dirty_teams = assemble(*_corrupt_from(games, team_games, cutoff))

    today = np.flatnonzero(clean_games["date"] == cutoff)
    assert len(today) > 0
    np.testing.assert_allclose(clean_games.iloc[today][STATIC], dirty_games.iloc[today][STATIC])
    clean = build_sequences(clean_games, clean_teams, LENGTH)
    dirty = build_sequences(dirty_games, dirty_teams, LENGTH)
    np.testing.assert_array_equal(clean.home[today], dirty.home[today])
    np.testing.assert_array_equal(clean.away[today], dirty.away[today])


def test_check_catches_a_deliberate_leak(tables):
    games, team_games = tables
    cutoff = CUTOFFS[0]
    clean_games, clean_teams = assemble(games, team_games)
    dirty_games, dirty_teams = assemble(*_corrupt_from(games, team_games, cutoff))
    today = np.flatnonzero(clean_games["date"] == cutoff)
    clean = build_sequences(clean_games, clean_teams, LENGTH, leak_current_game=True)
    dirty = build_sequences(dirty_games, dirty_teams, LENGTH, leak_current_game=True)
    assert not np.array_equal(clean.home[today], dirty.home[today])


def test_sequences_only_hold_earlier_games_this_season(tables):
    games, team_games = assemble(*tables)
    sample = games.sample(300, random_state=0)
    seqs = build_sequences(sample, team_games, LENGTH)
    for row, game in enumerate(sample.itertuples()):
        count = int(seqs.home_mask[row].sum())
        prior = team_games[(team_games["team"] == game.home) & (team_games["season"] == game.season)
                           & (team_games["date"] < game.date)]
        assert count == min(len(prior), LENGTH)


def test_only_regular_season_games(tables):
    games, _ = tables
    assert games["gameId"].astype(str).str[4:6].eq("02").all()


def test_outcomes_are_sane(tables):
    games, _ = tables
    assert set(games["overtime_type"]) == {"REG", "OT", "SO"}
    assert games["home_win"].isin([0, 1]).all()
    assert (games["season"] < "2026-27").sum() == 6560


def test_goalie_form_ignores_that_game_and_later():
    from nhl.goalies import goalie_form, load_appearances
    log = load_appearances()
    cutoff = 20250114
    clean = goalie_form(log)
    dirty_log = log.copy()
    later = dirty_log["gameDate"] >= cutoff
    dirty_log.loc[later, "gsax"] = 99.0
    dirty_log.loc[later, "icetime"] = 1.0
    dirty = goalie_form(dirty_log)
    today = log.loc[log["gameDate"] == cutoff, "gameId"]
    assert len(today) > 0
    columns = ["goalie_id", "gameId"]
    a = clean[clean["gameId"].isin(today)].sort_values(columns).reset_index(drop=True)
    b = dirty[dirty["gameId"].isin(today)].sort_values(columns).reset_index(drop=True)
    np.testing.assert_allclose(a[["goalie_gsax", "goalie_recent"]], b[["goalie_gsax", "goalie_recent"]])


def test_goalie_form_check_sees_earlier_changes():
    from nhl.goalies import goalie_form, load_appearances
    log = load_appearances()
    dirty_log = log.copy()
    dirty_log.loc[dirty_log["gameDate"] < 20250114, "gsax"] = 99.0
    today = log.loc[log["gameDate"] == 20250114, "gameId"]
    a = goalie_form(log).query("gameId in @today").sort_values(["goalie_id", "gameId"])
    b = goalie_form(dirty_log).query("gameId in @today").sort_values(["goalie_id", "gameId"])
    assert not np.allclose(a["goalie_gsax"], b["goalie_gsax"])


def test_era_level_ignores_that_day_and_the_future(tables):
    from nhl.features import ERA_STATS, era_adjust
    games, team_games = tables
    cutoff = pd.Timestamp("2025-01-14")
    dirty = team_games.copy()
    dirty[ERA_STATS] = dirty[ERA_STATS].astype(float)
    later = dirty["date"] > cutoff
    dirty.loc[later, ERA_STATS] = 999.0
    clean_adjusted, dirty_adjusted = era_adjust(team_games), era_adjust(dirty)
    earlier = team_games["date"] <= cutoff
    np.testing.assert_allclose(clean_adjusted.loc[earlier, ERA_STATS], dirty_adjusted.loc[earlier, ERA_STATS])


def test_era_level_ignores_other_games_that_day(tables):
    from nhl.features import ERA_STATS, era_adjust
    _, team_games = tables
    cutoff = pd.Timestamp("2025-01-14")
    today = team_games.index[team_games["date"] == cutoff]
    dirty = team_games.copy()
    dirty[ERA_STATS] = dirty[ERA_STATS].astype(float)
    dirty.loc[today[1:], ERA_STATS] = 999.0
    a, b = era_adjust(team_games).loc[today[0], ERA_STATS], era_adjust(dirty).loc[today[0], ERA_STATS]
    np.testing.assert_allclose(a.astype(float), b.astype(float))


def test_arizona_seasons_use_arizona_location():
    from nhl.features import _location
    assert _location("UTA", "2023-24") == "ARI"
    assert _location("UTA", "2024-25") == "UTA"
    assert _location("BOS", "2023-24") == "BOS"


def test_full_history_keeps_every_game():
    games, team_games = load_games(2008)
    assert len(games) == team_games["gameId"].nunique()
    assert team_games.groupby("gameId")["is_home"].sum().eq(1).all()
    assert team_games.groupby("gameId")["team"].nunique().eq(2).all()
    assert games.groupby("season").size()["2011-12"] == 1230
    assert games.groupby("season").size()["2013-14"] == 1230
    assert games["home_win"].isin([0, 1]).all()


def test_each_game_has_matching_sides():
    _, team_games = load_games(2008)
    home = team_games[team_games["is_home"] == 1].set_index("gameId")
    away = team_games[team_games["is_home"] == 0].set_index("gameId").reindex(home.index)
    assert (home["team"] == away["opponent"]).all() and (away["team"] == home["opponent"]).all()
    assert (home["gf"] == away["ga"]).all() and (home["ga"] == away["gf"]).all()
