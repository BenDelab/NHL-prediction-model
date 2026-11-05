"""Convolutional network for game winners, ending in a sigmoid probability.

The same encoder reads both teams, so home and away are judged by one set of rules.
Short kernels slide along the sequence of recent games to pick up local patterns.

One sigmoid gives P(home team wins the game), however the game is decided.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    length: int = 20
    kernels: tuple[int, ...] = (3, 5)
    filters: int = 12
    encoding: int = 16
    dropout: float = 0.4
    weight_decay: float = 0.01
    lr: float = 1e-3


class Encoder(nn.Module):
    def __init__(self, channels: int, config: Config):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels + 1, config.filters, k, padding=k // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Conv1d(config.filters, config.filters, k, padding=k // 2),
                nn.GELU(),
            )
            for k in config.kernels
        )
        pooled = 2 * config.filters * len(config.kernels)
        self.project = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(pooled, config.encoding))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, mask.unsqueeze(-1)], dim=-1).transpose(1, 2)
        weight = mask.unsqueeze(1)
        pooled = []
        for branch in self.branches:
            out = branch(inputs)
            average = (out * weight).sum(-1) / weight.sum(-1).clamp(min=1)
            pooled += [average, out[..., -1]]  # the whole sequence, and the latest position
        return self.project(torch.cat(pooled, dim=1))


class GameNet(nn.Module):
    def __init__(self, channels: int, static: int, config: Config):
        super().__init__()
        self.encoder = Encoder(channels, config)
        self.win_head = nn.Sequential(
            nn.Linear(config.encoding + static, 16), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(16, 1)
        )

    def forward(self, home, home_mask, away, away_mask, static) -> torch.Tensor:
        difference = self.encoder(home, home_mask) - self.encoder(away, away_mask)
        return self.win_head(torch.cat([difference, static], dim=1)).squeeze(1)


def probabilities(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"home_win": torch.sigmoid(logits)}


def loss(logits: torch.Tensor, home_win: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`weight` favours recent games."""
    per_game = nn.functional.binary_cross_entropy_with_logits(logits, home_win, reduction="none")
    return (per_game * weight).sum() / weight.sum()
