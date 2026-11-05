"""Walk-forward comparison of the CNN against simple baselines.

Run with: python -m nhl.experiment
Choices are made on validation seasons, and each test season is scored once.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT, SEASONS, load_games
from .features import CHANNELS, STATIC, assemble, build_sequences
from .model import Config
from .train import fit_predict, validate_config

FOLDS = [(SEASONS[:i - 1], SEASONS[i - 1], SEASONS[i]) for i in (2, 3, 4)]
LENGTHS = (10, 20, 40)
KERNELS = ((3,), (5,), (9,), (3, 5), (3, 5, 9))
SEARCH_SEEDS = (0, 1, 2)
FINAL_SEEDS = tuple(range(10))
WINDOWS = (5, 10, 20, 40)
OUT = ROOT / "runs"


def window_table(games, team_games) -> pd.DataFrame:
    """Plain averages over past windows: what the CNN has to beat without convolution."""
    seqs = build_sequences(games, team_games, max(WINDOWS))
    columns = {}
    for w in WINDOWS:
        for i, name in enumerate(CHANNELS):
            def mean(values, mask):
                return (values[:, -w:, i] * mask[:, -w:]).sum(1) / np.maximum(mask[:, -w:].sum(1), 1)
            columns[f"{name}_{w}"] = mean(seqs.home, seqs.home_mask) - mean(seqs.away, seqs.away_mask)
    return pd.concat([pd.DataFrame(columns, index=games.index), games[STATIC]], axis=1)


def logistic(games, x, train_seasons, val_season, test_season):
    """Regularised logistic regression, with strength chosen on the validation season."""
    def fit(rows, c):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=c, max_iter=3000))
        return model.fit(x[rows], games["home_win"][rows])

    train = games["season"].isin(train_seasons).to_numpy()
    val = (games["season"] == val_season).to_numpy()
    test = (games["season"] == test_season).to_numpy()
    grid = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
    scores = {c: log_loss(games["home_win"][val], fit(train, c).predict_proba(x[val])[:, 1]) for c in grid}
    best = min(scores, key=scores.get)
    return fit(train | val, best).predict_proba(x[test])[:, 1], best


def summarise(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "games": len(y),
        "accuracy": accuracy_score(y, p >= 0.5),
        "log_loss": log_loss(y, p),
        "auc": roc_auc_score(y, p) if np.std(p) > 0 else 0.5,
        "picks_away": float(np.mean(p < 0.5)),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    games, team_games = assemble(*load_games())
    x_window = window_table(games, team_games).to_numpy()
    x_elo = games[["elo_diff"]].to_numpy()

    predictions, search = {}, []
    for train_seasons, val_season, test_season in FOLDS:
        started = time.time()
        results = {}
        for length, kernels in itertools.product(LENGTHS, KERNELS):
            config = Config(length=length, kernels=kernels)
            score = validate_config(games, team_games, config, train_seasons, val_season, SEARCH_SEEDS)
            results[config] = score
            search.append({"test_season": test_season, "length": length, "kernels": str(kernels), "val_log_loss": score})
        best = min(results, key=results.get)
        print(f"{test_season}: chose length={best.length} kernels={best.kernels} "
              f"(val {results[best]:.4f}, {time.time() - started:.0f}s)", flush=True)

        frame = fit_predict(games, team_games, best, train_seasons, val_season, test_season, FINAL_SEEDS)
        frame["cnn"] = frame["p_home_win"]
        rows = frame.index.to_numpy()
        frame["home_rate"] = games[games["season"].isin([*train_seasons, val_season])]["home_win"].mean()
        frame["elo"] = logistic(games, x_elo, train_seasons, val_season, test_season)[0]
        frame["window_lr"], c = logistic(games, x_window, train_seasons, val_season, test_season)
        frame["blend"] = (frame["cnn"] + frame["window_lr"]) / 2
        predictions[test_season] = frame
        print(f"  temperature={frame.attrs['temperature']:.2f} epochs={frame.attrs['epochs']} LR C={c}", flush=True)
        assert len(rows) == len(frame)

    # Canaries on one fold. A leak must look too good; shuffled labels must look useless.
    train_seasons, val_season, test_season = FOLDS[-1]
    config = Config()
    leaked = fit_predict(games, team_games, config, train_seasons, val_season, test_season, (0,), leak_current_game=True)
    shuffled_games = games.copy()
    shuffled_games["home_win"] = np.random.default_rng(0).permutation(shuffled_games["home_win"].to_numpy())
    shuffled = fit_predict(shuffled_games, team_games, config, train_seasons, val_season, test_season, (0,))

    everything = pd.concat(predictions.values())
    y = everything["home_win"].to_numpy()
    report = {"models": {}, "canary": {
        "deliberate_leak": summarise(leaked["home_win"].to_numpy(), leaked["p_home_win"].to_numpy()),
        "shuffled_labels": summarise(shuffled["home_win"].to_numpy(), shuffled["p_home_win"].to_numpy()),
    }}
    for name in ["home_rate", "elo", "window_lr", "cnn", "blend"]:
        report["models"][name] = {"all": summarise(y, everything[name].to_numpy())}
        for season, frame in predictions.items():
            report["models"][name][season] = summarise(frame["home_win"].to_numpy(), frame[name].to_numpy())
    everything.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(search).to_csv(OUT / "search.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
