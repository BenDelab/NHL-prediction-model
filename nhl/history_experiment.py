"""Does training on more past seasons help? Same test games, with training reaching further back.

Run with: python -m nhl.history_experiment
Inputs are the same for every run, so training data is the only difference. Each test
season keeps its chosen architecture.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FIRST_YEAR, load_games, season_name
from .experiment import FOLDS, OUT, summarise, window_table
from .features import STATIC, Scaler, assemble, build_sequences
from .goalie_experiment import CHOSEN
from .train import MAX_EPOCHS, _fit, _predict, _tensors, rescale, temperature

FIRST_YEARS = (2021, 2018, 2013, FIRST_YEAR)
SEEDS = tuple(range(5))
GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)


def seasons_between(first_year: int, last_season: str) -> list[str]:
    last_year = int(last_season[:4])
    return [season_name(y) for y in range(first_year, last_year + 1)]


def run_cnn(games, team_games, seqs, config, train_seasons, val_season, test_season) -> np.ndarray:
    train = np.flatnonzero(games["season"].isin(train_seasons).to_numpy())
    val = np.flatnonzero(games["season"] == val_season)
    test = np.flatnonzero(games["season"] == test_season)

    scaler = Scaler(team_games[team_games["season"].isin(train_seasons)], games.iloc[train])
    train_data, val_data = _tensors(games, seqs, scaler, train), _tensors(games, seqs, scaler, val)
    epochs, val_runs = [], []
    for seed in SEEDS:
        net, best = _fit(train_data, config, seed, MAX_EPOCHS, watch=val_data)
        epochs.append(best)
        val_runs.append(_predict(net, val_data)["home_win"])
    t = temperature(np.mean(val_runs, axis=0), val_data["home_win"].numpy())

    both = np.concatenate([train, val])
    scaler = Scaler(team_games[team_games["season"].isin([*train_seasons, val_season])], games.iloc[both])
    both_data, test_data = _tensors(games, seqs, scaler, both), _tensors(games, seqs, scaler, test)
    runs = [_predict(_fit(both_data, config, seed + 1000, count)[0], test_data)["home_win"]
            for seed, count in zip(SEEDS, epochs)]
    return rescale(np.mean(runs, axis=0), t)


def run_logistic(games, x, train_seasons, val_season, test_season) -> np.ndarray:
    labels = games["home_win"].to_numpy()
    train = games["season"].isin(train_seasons).to_numpy()
    val = (games["season"] == val_season).to_numpy()
    test = (games["season"] == test_season).to_numpy()

    def fit(rows, c):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=c, max_iter=3000))
        return model.fit(x[rows], labels[rows])

    scores = {c: log_loss(labels[val], fit(train, c).predict_proba(x[val])[:, 1]) for c in GRID}
    return fit(train | val, min(scores, key=scores.get)).predict_proba(x[test])[:, 1]


def main() -> None:
    torch.set_num_threads(6)
    games, team_games = assemble(*load_games(FIRST_YEAR))
    print("games", len(games), "by first year",
          {y: int(games["season"].isin(seasons_between(y, "2025-26")).sum()) for y in FIRST_YEARS}, flush=True)
    x = window_table(games, team_games).to_numpy()
    assert x.shape[1] == 72 + len(STATIC)
    sequences = {c.length: build_sequences(games, team_games, c.length) for c in CHOSEN.values()}

    frames = []
    for _, val_season, test_season in FOLDS:
        frame = games.loc[games["season"] == test_season, ["gameId", "season", "date", "home_win"]].copy()
        config = CHOSEN[test_season]
        before_val = [s for s in seasons_between(FIRST_YEARS[-1], val_season) if s != val_season]
        for first_year in FIRST_YEARS:
            train_seasons = [s for s in before_val if int(s[:4]) >= first_year]
            frame[f"lr_{first_year}"] = run_logistic(games, x, train_seasons, val_season, test_season)
            frame[f"cnn_{first_year}"] = run_cnn(games, team_games, sequences[config.length], config,
                                                 train_seasons, val_season, test_season)
            frames_so_far = pd.concat([*frames, frame])
            frames_so_far.to_csv(OUT / "predictions_history.csv", index=False)
            y = frame["home_win"].to_numpy()
            print(test_season, first_year, len(train_seasons), "train seasons",
                  {k: round(v, 4) for k, v in summarise(y, frame[f"cnn_{first_year}"].to_numpy()).items()
                   if k in ("accuracy", "log_loss")},
                  "LR", round(summarise(y, frame[f"lr_{first_year}"].to_numpy())["log_loss"], 4), flush=True)
        frames.append(frame)


if __name__ == "__main__":
    main()
