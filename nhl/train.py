"""Train on past seasons only, and report on a later season.

For a test season S:
  1. Train on seasons before S-1, watching S-1, to pick the epoch count and calibration.
  2. Retrain on everything before S for that many epochs.
  3. Predict S, which has played no part in any of this.
"""
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from .features import CHANNELS, STATIC, Scaler, Sequences, build_sequences
from .model import Config, GameNet, loss, probabilities

MAX_EPOCHS = 60
PATIENCE = 10
BATCH = 128


def _tensors(games, seqs: Sequences, scaler: Scaler, rows: np.ndarray, weight=None) -> dict[str, torch.Tensor]:
    subset = games.iloc[rows]
    return {
        "weight": torch.tensor(np.ones(len(rows)) if weight is None else weight, dtype=torch.float32),
        "home": torch.tensor(scaler.sequence(seqs.home[rows], seqs.home_mask[rows])),
        "home_mask": torch.tensor(seqs.home_mask[rows]),
        "away": torch.tensor(scaler.sequence(seqs.away[rows], seqs.away_mask[rows])),
        "away_mask": torch.tensor(seqs.away_mask[rows]),
        "static": torch.tensor(scaler.static(subset)),
        "home_win": torch.tensor(subset["home_win"].to_numpy(dtype=np.float32)),
    }


def _predict(net: GameNet, data: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    net.eval()
    with torch.no_grad():
        logits = net(data["home"], data["home_mask"], data["away"], data["away_mask"], data["static"])
        return {k: v.numpy() for k, v in probabilities(logits).items()}


def _fit(data, config: Config, seed: int, epochs: int, watch=None) -> tuple[GameNet, int]:
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    net = GameNet(len(CHANNELS), data["static"].shape[1], config)
    optimiser = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    count = len(data["home_win"])
    best_loss, best_epoch, best_state = np.inf, 0, None
    for epoch in range(1, epochs + 1):
        net.train()
        order = generator.permutation(count)
        for start in range(0, count, BATCH):
            batch = order[start:start + BATCH]
            logits = net(data["home"][batch], data["home_mask"][batch], data["away"][batch],
                         data["away_mask"][batch], data["static"][batch])
            optimiser.zero_grad()
            loss(logits, data["home_win"][batch], data["weight"][batch]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
        if watch is not None:
            current = log_loss(watch["home_win"].numpy(), _predict(net, watch)["home_win"].clip(1e-6, 1 - 1e-6))
            if current < best_loss:
                best_loss, best_epoch = current, epoch
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            elif epoch - best_epoch >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_epoch or epochs


def temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    def score(t):
        return log_loss(labels, expit(logit(probs.clip(1e-6, 1 - 1e-6)) / t).clip(1e-6, 1 - 1e-6))
    return float(minimize_scalar(score, bounds=(0.3, 5), method="bounded").x)


def rescale(probs: np.ndarray, t: float) -> np.ndarray:
    return expit(logit(probs.clip(1e-6, 1 - 1e-6)) / t)


def validate_config(games, team_games, config: Config, train_seasons, val_season, seeds, static=STATIC) -> float:
    """Validation log loss for a config. Never touches the test season."""
    seqs = build_sequences(games, team_games, config.length)
    train = np.flatnonzero(games["season"].isin(train_seasons))
    val = np.flatnonzero(games["season"] == val_season)
    scaler = Scaler(team_games[team_games["season"].isin(train_seasons)], games.iloc[train], static)
    train_data, val_data = _tensors(games, seqs, scaler, train), _tensors(games, seqs, scaler, val)
    runs = []
    for seed in seeds:
        net, _ = _fit(train_data, config, seed, MAX_EPOCHS, watch=val_data)
        runs.append(_predict(net, val_data)["home_win"])
    labels = val_data["home_win"].numpy()
    return float(log_loss(labels, np.mean(runs, axis=0).clip(1e-6, 1 - 1e-6)))


def fit_predict(games, team_games, config: Config, train_seasons, val_season, test_season,
                seeds, leak_current_game: bool = False, static=STATIC) -> pd.DataFrame:
    """Return calibrated predictions for every game of the test season."""
    seqs = build_sequences(games, team_games, config.length, leak_current_game)
    train = np.flatnonzero(games["season"].isin(train_seasons))
    val = np.flatnonzero(games["season"] == val_season)
    test = np.flatnonzero(games["season"] == test_season)

    scaler = Scaler(team_games[team_games["season"].isin(train_seasons)], games.iloc[train], static)
    train_data, val_data = _tensors(games, seqs, scaler, train), _tensors(games, seqs, scaler, val)
    epochs, val_runs = [], []
    for seed in seeds:
        net, best = _fit(train_data, config, seed, MAX_EPOCHS, watch=val_data)
        epochs.append(best)
        val_runs.append(_predict(net, val_data)["home_win"])
    labels = val_data["home_win"].numpy()
    t = temperature(np.mean(val_runs, axis=0), labels)

    both = np.concatenate([train, val])
    seasons = list(train_seasons) + [val_season]
    scaler = Scaler(team_games[team_games["season"].isin(seasons)], games.iloc[both], static)
    both_data, test_data = _tensors(games, seqs, scaler, both), _tensors(games, seqs, scaler, test)
    runs = []
    for seed, count in zip(seeds, epochs):
        net, _ = _fit(both_data, config, seed + 1000, count)
        runs.append(_predict(net, test_data))

    out = games.iloc[test][["gameId", "season", "date", "home", "away", "home_win", "extra_time", "overtime_type"]].copy()
    for key in runs[0]:
        out[f"p_{key}"] = np.mean([r[key] for r in runs], axis=0)
    out["p_home_win"] = rescale(out["p_home_win"].to_numpy(), t)
    out.attrs.update(temperature=t, epochs=epochs)
    return out
