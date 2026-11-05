"""Does knowing the starting goalie help? Same folds and settings, with and without.

Run with: python -m nhl.goalie_experiment
Uses the architecture each test season already chose on validation data.
"""
import json

import numpy as np
import pandas as pd
import torch

from .data import LAST_FULL_YEAR, load_games
from .experiment import FINAL_SEEDS, FOLDS, OUT, logistic, summarise, window_table
from .features import STATIC, assemble
from .goalies import GOALIE_STATIC, add_goalies
from .model import Config
from .train import fit_predict

CHOSEN = {"2023-24": Config(length=40, kernels=(5,)), "2024-25": Config(length=40, kernels=(9,)),
          "2025-26": Config(length=10, kernels=(3,))}
ALL_STATIC = STATIC + GOALIE_STATIC


def main() -> None:
    torch.set_num_threads(6)
    games, team_games = assemble(*load_games(last_year=LAST_FULL_YEAR))
    games = add_goalies(games, team_games)
    print("games missing a starter:", games.attrs["games_missing_starter"], flush=True)
    base = window_table(games, team_games)
    x_plain = base.to_numpy()
    x_goalie = pd.concat([base, games[GOALIE_STATIC]], axis=1).to_numpy()
    x_elo_goalie = games[["elo_diff"] + GOALIE_STATIC].to_numpy()

    frames = []
    for train_seasons, val_season, test_season in FOLDS:
        frame = fit_predict(games, team_games, CHOSEN[test_season], train_seasons, val_season, test_season,
                            FINAL_SEEDS, static=ALL_STATIC)
        frame["cnn_goalie"] = frame["p_home_win"]
        frame["lr_goalie"], c = logistic(games, x_goalie, train_seasons, val_season, test_season)
        frame["lr_plain"] = logistic(games, x_plain, train_seasons, val_season, test_season)[0]
        frame["elo_goalie"] = logistic(games, x_elo_goalie, train_seasons, val_season, test_season)[0]
        frame["blend_goalie"] = (frame["cnn_goalie"] + frame["lr_goalie"]) / 2
        frames.append(frame)
        print(test_season, "done, LR C", c, "temperature", round(frame.attrs["temperature"], 2), flush=True)
    result = pd.concat(frames)
    result.to_csv(OUT / "predictions_goalie.csv", index=False)
    y = result["home_win"].to_numpy()
    report = {name: {"all": summarise(y, result[name].to_numpy())} for name in
              ["lr_plain", "lr_goalie", "elo_goalie", "cnn_goalie", "blend_goalie"]}
    for name in report:
        for season, group in result.groupby("season"):
            report[name][season] = summarise(group["home_win"].to_numpy(), group[name].to_numpy())
    (OUT / "report_goalie.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
