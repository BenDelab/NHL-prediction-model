"""Do hidden layers help on their own? MLP vs logistic regression on the same rolling averages.

Run with: python -m nhl.mlp_experiment
Settings are chosen on validation seasons, and each test season is scored once.
"""
import itertools
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch import nn

from .data import load_games
from .experiment import FINAL_SEEDS, FOLDS, OUT, SEARCH_SEEDS, logistic, summarise, window_table
from .features import assemble
from .train import BATCH, MAX_EPOCHS, PATIENCE, rescale, temperature

HIDDEN = ((8,), (16,), (32,), (16, 8))
DROPOUT = (0.3, 0.5)
WEIGHT_DECAY = (0.01, 0.1)
LR = 1e-3


def build(inputs: int, hidden: tuple[int, ...], dropout: float) -> nn.Module:
    layers: list[nn.Module] = [nn.Dropout(dropout)]
    size = inputs
    for width in hidden:
        layers += [nn.Linear(size, width), nn.GELU(), nn.Dropout(dropout)]
        size = width
    return nn.Sequential(*layers, nn.Linear(size, 1))


def predict(net: nn.Module, x: torch.Tensor) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(x)).squeeze(1).numpy()


def fit(x, y, settings, seed, epochs, watch=None):
    hidden, dropout, weight_decay = settings
    torch.manual_seed(seed)
    order_rng = np.random.default_rng(seed)
    net = build(x.shape[1], hidden, dropout)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=weight_decay)
    best_loss, best_epoch, best_state = np.inf, 0, None
    for epoch in range(1, epochs + 1):
        net.train()
        order = order_rng.permutation(len(y))
        for start in range(0, len(y), BATCH):
            batch = order[start:start + BATCH]
            optimiser.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(x[batch]).squeeze(1), y[batch]).backward()
            optimiser.step()
        if watch is not None:
            current = log_loss(watch[1].numpy(), predict(net, watch[0]).clip(1e-6, 1 - 1e-6))
            if current < best_loss:
                best_loss, best_epoch = current, epoch
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            elif epoch - best_epoch >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_epoch or epochs


def scaled(x: np.ndarray, fit_rows: np.ndarray, rows: np.ndarray) -> torch.Tensor:
    mean, std = x[fit_rows].mean(0), x[fit_rows].std(0) + 1e-6
    return torch.tensor((x[rows] - mean) / std, dtype=torch.float32)


def fit_predict(x, labels, train, val, test, settings, seeds):
    y = lambda rows: torch.tensor(labels[rows], dtype=torch.float32)
    x_train, x_val = scaled(x, train, train), scaled(x, train, val)
    epochs, val_runs = [], []
    for seed in seeds:
        net, best = fit(x_train, y(train), settings, seed, MAX_EPOCHS, watch=(x_val, y(val)))
        epochs.append(best)
        val_runs.append(predict(net, x_val))
    val_mean = np.mean(val_runs, axis=0)
    t = temperature(val_mean, labels[val])
    both = train | val
    x_both, x_test = scaled(x, both, both), scaled(x, both, test)
    runs = [predict(fit(x_both, y(both), settings, seed + 1000, count)[0], x_test)
            for seed, count in zip(seeds, epochs)]
    return rescale(np.mean(runs, axis=0), t), log_loss(labels[val], val_mean.clip(1e-6, 1 - 1e-6))


def main() -> None:
    torch.set_num_threads(4)
    games, team_games = assemble(*load_games())
    x = window_table(games, team_games).to_numpy(dtype=np.float32)
    labels = games["home_win"].to_numpy(dtype=np.float32)
    frames = []
    for train_seasons, val_season, test_season in FOLDS:
        train = games["season"].isin(train_seasons).to_numpy()
        val = (games["season"] == val_season).to_numpy()
        test = (games["season"] == test_season).to_numpy()
        scores = {s: fit_predict(x, labels, train, val, test, s, SEARCH_SEEDS)[1]
                  for s in itertools.product(HIDDEN, DROPOUT, WEIGHT_DECAY)}
        best = min(scores, key=scores.get)
        frame = games.loc[test, ["gameId", "season", "date", "home_win"]].copy()
        frame["mlp"] = fit_predict(x, labels, train, val, test, best, FINAL_SEEDS)[0]
        frame["window_lr"] = logistic(games, x, train_seasons, val_season, test_season)[0]
        frames.append(frame)
        print(test_season, "chose", best, "val", round(scores[best], 4), flush=True)
    result = pd.concat(frames)
    result.to_csv(OUT / "predictions_mlp.csv", index=False)
    report = {name: {"all": summarise(result["home_win"].to_numpy(), result[name].to_numpy())}
              for name in ("window_lr", "mlp")}
    for name in report:
        for season, group in result.groupby("season"):
            report[name][season] = summarise(group["home_win"].to_numpy(), group[name].to_numpy())
    (OUT / "report_mlp.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
