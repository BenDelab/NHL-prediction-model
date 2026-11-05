"""Final comparison: CNN vs logistic regression with era handling and validation-only tuning.

Run with: python -m nhl.tuned_experiment

Every choice comes from the two seasons before the test season: how far back to train,
how fast old seasons fade, era scaling, and the model's settings. Both models then retrain
every RETRAIN_DAYS on games already played. Search results are saved, so it can resume.
"""
import itertools
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FIRST_YEAR, load_games, season_name
from .experiment import OUT, summarise, window_table
from .features import Scaler, assemble, build_sequences, era_adjust
from .model import Config
from .train import MAX_EPOCHS, _fit, _predict, _tensors, rescale, temperature

TESTS = ("2026-27",)  # held back from development and scored once
BASELINE_CONFIG = {"2026-27": Config(length=10, kernels=(3,))}  # the original experiment's latest pick
BASELINE_EPOCHS = 15  # typical of the runs chosen in the original experiment
START_YEARS = (2008, 2013, 2018)
HALF_LIVES = (2.0, 4.0, 8.0, None)
ERA_OPTIONS = (True, False)
C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
TRIALS = 24
SEARCH_SEEDS = (0, 1)
FINAL_SEEDS = tuple(range(5))
RETRAIN_DAYS = 30
WORKERS = 4
CACHE = OUT / "tuned_cache.pkl"
SEARCH_LOG = OUT / "tuned_search.jsonl"

_shared: dict = {}


def order(season: str) -> int:
    return int(season[:4])


def sample_trials() -> list[dict]:
    rng = np.random.default_rng(0)
    trials = [dict(start=2008, half_life=None, era=False, length=20, kernels=(3, 5), filters=12,
                   dropout=0.4, lr=1e-3, weight_decay=0.01)]
    while len(trials) < TRIALS:
        trials.append(dict(
            start=int(rng.choice(START_YEARS)),
            half_life=HALF_LIVES[rng.integers(len(HALF_LIVES))],
            era=bool(rng.integers(2)),
            length=int(rng.choice((10, 20, 40))),
            kernels=[(3,), (5,), (9,), (3, 5)][rng.integers(4)],
            filters=int(rng.choice((8, 12, 24))),
            dropout=float(rng.choice((0.2, 0.4, 0.6))),
            lr=float(rng.choice((3e-4, 1e-3))),
            weight_decay=float(rng.choice((0.01, 0.1))),
        ))
    return trials


def make_config(trial: dict) -> Config:
    return Config(length=trial["length"], kernels=tuple(trial["kernels"]), filters=trial["filters"],
                  dropout=trial["dropout"], lr=trial["lr"], weight_decay=trial["weight_decay"])


def load_shared() -> dict:
    if CACHE.exists():
        with CACHE.open("rb") as handle:
            return pickle.load(handle)
    games, team_games = assemble(*load_games(FIRST_YEAR))
    shared = {"games": games, "teams": {False: team_games, True: era_adjust(team_games)}}
    with CACHE.open("wb") as handle:
        pickle.dump(shared, handle)
    return shared


def init_worker() -> None:
    torch.set_num_threads(2)
    _shared.update(load_shared())
    _shared["sequences"] = {}
    _shared["windows"] = {}


def sequences(length: int, era: bool):
    key = (length, era)
    if key not in _shared["sequences"]:
        _shared["sequences"][key] = build_sequences(_shared["games"], _shared["teams"][era], length)
    return _shared["sequences"][key]


def windows(era: bool) -> np.ndarray:
    if era not in _shared["windows"]:
        _shared["windows"][era] = window_table(_shared["games"], _shared["teams"][era]).to_numpy()
    return _shared["windows"][era]


def recency(seasons: pd.Series, current: str, half_life: float | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(seasons))
    age = order(current) - seasons.map(order).to_numpy()
    return 0.5 ** (age / half_life)


def training_rows(start: int, current: str, cutoff=None) -> np.ndarray:
    """Finished games from the start year up to `current`, ending before `cutoff` if given."""
    games = _shared["games"]
    keep = games["season"].map(order) >= start
    keep &= games["season"].map(order) < order(current) if cutoff is None else games["date"] < cutoff
    return np.flatnonzero(keep.to_numpy())


def scaler_for(rows: np.ndarray, era: bool) -> Scaler:
    games, teams = _shared["games"], _shared["teams"][era]
    subset = games.iloc[rows]
    used = teams["gameId"].isin(subset["gameId"])
    return Scaler(teams[used], subset)


def cnn_fit(trial, config, rows, current, seed, epochs, watch_rows=None):
    games = _shared["games"]
    seqs = sequences(config.length, trial["era"])
    scaler = scaler_for(rows, trial["era"])
    weight = recency(games.iloc[rows]["season"], current, trial["half_life"])
    data = _tensors(games, seqs, scaler, rows, weight)
    watch = None if watch_rows is None else _tensors(games, seqs, scaler, watch_rows)
    net, best = _fit(data, config, seed, epochs, watch=watch)
    return net, best, seqs, scaler


def search_task(task: tuple) -> dict:
    test, index, trial, val, seed = task
    config = make_config(trial)
    games = _shared["games"]
    rows = training_rows(trial["start"], val)
    val_rows = np.flatnonzero(games["season"] == val)
    net, best, seqs, scaler = cnn_fit(trial, config, rows, val, seed, MAX_EPOCHS, val_rows)
    probs = _predict(net, _tensors(games, seqs, scaler, val_rows))["home_win"]
    labels = games.iloc[val_rows]["home_win"].astype(int).to_numpy()
    return {"test": test, "trial": index, "val": val, "seed": seed, "epochs": best,
            "log_loss": float(log_loss(labels, probs.clip(1e-6, 1 - 1e-6))), "probs": probs.tolist()}


def final_task(task: tuple) -> dict:
    """One seed's predictions for one retraining period of the test season."""
    test, trial, config, seed, epochs, begin, end = task
    games = _shared["games"]
    rows = training_rows(trial["start"], test, cutoff=begin)
    period = np.flatnonzero(((games["season"] == test) & (games["date"] >= begin) & (games["date"] < end)).to_numpy())
    net, _, seqs, scaler = cnn_fit(trial, config, rows, test, seed + 1000, epochs)
    return {"test": test, "seed": seed, "begin": str(begin), "rows": period.tolist(),
            "probs": {k: v.tolist() for k, v in _predict(net, _tensors(games, seqs, scaler, period)).items()}}


def lr_predict(settings, rows, current, target) -> np.ndarray:
    start, half_life, era, c = settings
    games = _shared["games"]
    x = windows(era)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=c, max_iter=3000))
    model.fit(x[rows], games.iloc[rows]["home_win"].astype(int),
              logisticregression__sample_weight=recency(games.iloc[rows]["season"], current, half_life))
    return model.predict_proba(x[target])[:, 1]


def lr_search_task(task: tuple) -> dict:
    test, settings, val = task
    games = _shared["games"]
    target = np.flatnonzero(games["season"] == val)
    probs = lr_predict(settings, training_rows(settings[0], val), val, target)
    labels = games.iloc[target]["home_win"].astype(int).to_numpy()
    return {"test": test, "settings": settings, "val": val, "log_loss": float(log_loss(labels, probs)), "probs": probs.tolist()}


def periods(test: str) -> list[tuple]:
    dates = _shared["games"].loc[_shared["games"]["season"] == test, "date"]
    begin, last = dates.min(), dates.max()
    edges = []
    while begin <= last:
        end = begin + pd.Timedelta(days=RETRAIN_DAYS)
        edges.append((begin, end))
        begin = end
    return edges


def val_seasons(test: str) -> tuple[str, str]:
    return season_name(order(test) - 1), season_name(order(test) - 2)


def lr_season(test, settings, val_t, retrain: bool) -> np.ndarray:
    games = _shared["games"]
    out = np.zeros(int((games["season"] == test).sum()))
    positions = games.index[games["season"] == test]
    lookup = pd.Series(np.arange(len(positions)), index=positions)
    edges = periods(test) if retrain else [(games.loc[positions, "date"].min(), games.loc[positions, "date"].max() + pd.Timedelta(days=1))]
    for begin, end in edges:
        target = np.flatnonzero(((games["season"] == test) & (games["date"] >= begin) & (games["date"] < end)).to_numpy())
        if len(target):
            out[lookup.loc[games.index[target]].to_numpy()] = lr_predict(settings, training_rows(settings[0], test, cutoff=begin), test, target)
    return rescale(out, val_t)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    init_worker()
    torch.set_num_threads(4)
    games = _shared["games"]
    trials = sample_trials()
    done = {}
    if SEARCH_LOG.exists():
        for line in SEARCH_LOG.read_text().splitlines():
            row = json.loads(line)
            done[(row["test"], row["trial"], row["val"], row["seed"])] = row
    tasks = [(test, i, trial, val, seed) for test in TESTS for i, trial in enumerate(trials)
             for val in val_seasons(test) for seed in SEARCH_SEEDS
             if (test, i, val, seed) not in done]
    print(f"CNN search: {len(done)} saved, {len(tasks)} to run", flush=True)
    with ProcessPoolExecutor(WORKERS, initializer=init_worker) as pool, SEARCH_LOG.open("a") as log:
        for count, row in enumerate(pool.map(search_task, tasks), 1):
            done[(row["test"], row["trial"], row["val"], row["seed"])] = row
            log.write(json.dumps(row) + "\n")
            log.flush()
            if count % 25 == 0:
                print(f"  {count}/{len(tasks)}", flush=True)

        lr_tasks = [(test, s, val) for test in TESTS
                    for s in itertools.product(START_YEARS, HALF_LIVES, ERA_OPTIONS, C_GRID)
                    for val in val_seasons(test)]
        lr_scores: dict = {}
        for row in pool.map(lr_search_task, lr_tasks, chunksize=8):
            lr_scores.setdefault((row["test"], tuple(row["settings"])), {})[row["val"]] = row
        print("LR search done", flush=True)

        results, chosen = [], {}
        for test in TESTS:
            first_val, second_val = val_seasons(test)
            frame = games.loc[games["season"] == test, ["gameId", "season", "date", "home", "away", "home_win", "extra_time", "overtime_type"]].copy()
            labels = games.loc[games["season"] == first_val, "home_win"].astype(int).to_numpy()

            def score(index, val):
                return float(log_loss(games.loc[games["season"] == val, "home_win"].astype(int),
                                      np.mean([done[(test, index, val, s)]["probs"] for s in SEARCH_SEEDS], axis=0).clip(1e-6, 1 - 1e-6)))
            scores = {i: (score(i, first_val) + score(i, second_val)) / 2 for i in range(len(trials))}
            best_index = min(scores, key=scores.get)
            trial = trials[best_index]
            config = make_config(trial)
            epochs = [max(1, int(np.mean([done[(test, best_index, first_val, s)]["epochs"] for s in SEARCH_SEEDS])))] * len(FINAL_SEEDS)
            val_t = temperature(np.mean([done[(test, best_index, first_val, s)]["probs"] for s in SEARCH_SEEDS], axis=0), labels)

            lr_val = {s: (v[first_val]["log_loss"] + v[second_val]["log_loss"]) / 2 for (t, s), v in lr_scores.items() if t == test}
            lr_best = min(lr_val, key=lr_val.get)
            lr_t = temperature(np.array(lr_scores[(test, lr_best)][first_val]["probs"]), labels)
            chosen[test] = {"cnn": trial, "cnn_val": scores[best_index], "epochs": epochs[0], "cnn_temperature": val_t,
                            "lr": lr_best, "lr_val": lr_val[lr_best], "lr_temperature": lr_t}
            print(test, "chose", chosen[test], flush=True)

            positions = pd.Series(np.arange(len(frame)), index=frame.index)
            def run(edges, label, this_trial, this_config, this_epochs, temp):
                tasks = [(test, this_trial, this_config, seed, this_epochs, begin, end) for begin, end in edges for seed in FINAL_SEEDS]
                total = {}
                for row in pool.map(final_task, tasks):
                    for key, values in row["probs"].items():
                        total.setdefault(key, np.zeros(len(frame)))
                        where = positions.loc[games.index[row["rows"]]].to_numpy()
                        total[key][where] += np.array(values) / len(FINAL_SEEDS)
                for key, values in total.items():
                    frame[f"{label}_{key}"] = values
                frame[label] = rescale(total["home_win"], temp)
                print(test, label, "done", flush=True)

            dates = frame["date"]
            whole = [(dates.min(), dates.max() + pd.Timedelta(days=1))]
            run(periods(test), "cnn_tuned", trial, config, epochs[0], val_t)
            run(whole, "cnn_frozen", trial, config, epochs[0], val_t)
            base_trial = dict(start=2021, half_life=None, era=False)
            run(whole, "cnn_base", base_trial, BASELINE_CONFIG[test], BASELINE_EPOCHS, 1.0)
            frame["lr_tuned"] = lr_season(test, lr_best, lr_t, retrain=True)
            frame["lr_frozen"] = lr_season(test, lr_best, lr_t, retrain=False)
            frame["lr_base"] = lr_season(test, (2021, None, False, 0.003), 1.0, retrain=False)
            frame["blend_tuned"] = (frame["cnn_tuned"] + frame["lr_tuned"]) / 2
            results.append(frame)
            pd.concat(results).to_csv(OUT / "predictions_tuned.csv", index=False)
            (OUT / "tuned_chosen.json").write_text(json.dumps(chosen, indent=2, default=str))
    everything = pd.concat(results)
    report = {"settings": chosen, "models": {}}
    for name in ("cnn_base", "cnn_frozen", "cnn_tuned", "lr_base", "lr_frozen", "lr_tuned", "blend_tuned"):
        report["models"][name] = {}
        for season, group in everything.groupby("season"):
            report["models"][name][season] = summarise(group["home_win"].to_numpy(), group[name].to_numpy())
    (OUT / "report_tuned.json").write_text(json.dumps(report, indent=2, default=str))
    print("finished", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
