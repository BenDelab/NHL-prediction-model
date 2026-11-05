"""
Secondary models for OT likelihood and regulation win prediction.

Upgraded from single-LR to the same LR+XGBoost+LightGBM ensemble used by
the totals/spread models.  Returns {name: model} dicts so main.py can
average probabilities across the ensemble — consistent with how every other
market is modelled.
"""

from __future__ import annotations

from models.totals import train_spread_ensemble


def train_ot_model(X, y_ot) -> dict:
    """
    Train an ensemble to predict P(game goes to OT/SO).
    y_ot: binary series where 1 = game went to overtime/shootout.
    Returns {name: fitted_pipeline}.
    """
    return train_spread_ensemble(X, y_ot)


def train_reg_win_model(X, y_reg_win) -> dict:
    """
    Train an ensemble to predict P(home team wins in regulation).
    y_reg_win: binary series where 1 = home team won in regulation.
    Returns {name: fitted_pipeline}.
    """
    return train_spread_ensemble(X, y_reg_win)
