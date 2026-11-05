"""Evaluation with per-window sliding validation on the 2024-25 season."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, log_loss, roc_auc_score,
    precision_recall_curve,
)


class ThresholdedModel:
    """Wraps a fitted model with a custom decision threshold."""
    def __init__(self, model, threshold: float):
        self._model = model
        self.threshold = threshold

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def find_optimal_threshold(model, X_cal, y_cal, beta: float = 0.5) -> float:
    """
    Find the decision threshold that maximises F-beta on the calibration set.
    beta < 1 weights precision more than recall (beta=0.5 means precision 2x more important).
    Returns the optimal threshold.
    """
    probs = model.predict_proba(X_cal)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_cal, probs)

    # F-beta = (1 + beta²) * P * R / (beta² * P + R)
    b2 = beta ** 2
    with np.errstate(invalid="ignore"):
        f_beta = (1 + b2) * precisions * recalls / (b2 * precisions + recalls + 1e-9)

    best_idx = np.argmax(f_beta[:-1])  # last point has no corresponding threshold
    return float(thresholds[best_idx])

WINDOWS = [5, 10, 15, 20]

METRIC_COLS = ["accuracy", "roc_auc", "precision", "recall", "f1", "brier", "log_loss"]


# Rolling stats that add signal on top of Elo/standings for the moneyline model.
# Ablation on 2024-25 season shows these are the only rolling metrics with
# positive incremental AUC. All others (win%, goals, save%, SOG, etc.) add noise
# because Elo already captures wins and the short-window metrics are too noisy.
_MONEYLINE_ROLLING_STATS = [
    # Possession / shot quality (not captured by wins or Elo)
    "corsi_pct",
    # Expected goals (unblocked shot quality — cleaner than raw goals)
    "xgf_ev", "xga_ev", "xgp_ev",
    "xgf_all", "xga_all",
    # Goalie form (per-starter rolling — see goalie_loader)
    "goalie_gsax", "goalie_hd_save_pct",
    # Special teams
    "pp_efficiency", "pk_efficiency",
    # Relative matchup differentials
    "win_pct_diff", "goal_diff_ewm_diff", "shot_quality_gap_diff",
]


def get_feature_cols_for_window(window: int, df: pd.DataFrame) -> list:
    """
    Curated feature columns for the moneyline model at a given rolling window.

    Uses only the rolling stats proven to add incremental signal on top of
    Elo/standings (see _MONEYLINE_ROLLING_STATS). Full rolling columns are
    still available in the DataFrame for spread/totals models via direct access.
    """
    prefix   = f"rolling_{window}_"
    opp_pref = f"opp_rolling_{window}_"
    # Only include signal-bearing rolling columns
    rolling = []
    for stat in _MONEYLINE_ROLLING_STATS:
        for pfx in [prefix, opp_pref]:
            c = pfx + stat
            if c in df.columns:
                rolling.append(c)
    static = _static_features(df)
    return list(dict.fromkeys(rolling + static))


def get_all_rolling_cols_for_window(window: int, df: pd.DataFrame) -> list:
    """Full rolling columns for a window — used by spread/totals models."""
    prefix   = f"rolling_{window}_"
    opp_pref = f"opp_rolling_{window}_"
    rolling  = [c for c in df.columns if c.startswith(prefix) or c.startswith(opp_pref)]
    static   = _static_features(df)
    return rolling + static


def _static_features(df: pd.DataFrame) -> list:
    """Shared static (non-rolling) feature list used by both per-window and unified selectors."""
    candidates = [
        # schedule / fatigue — home team
        "days_since_last_game", "back_to_back", "games_in_last_10",
        "game_number_in_season",
        # schedule / fatigue — opponent (joined via add_opponent_features)
        "opp_days_since_last_game", "opp_back_to_back", "opp_games_in_last_10",
        # schedule differential
        "rest_advantage", "b2b_disadvantage", "workload_advantage",
        # Elo power rankings (cumulative across all seasons)
        "home_elo", "away_elo", "elo_diff", "elo_win_prob",
        # season-level
        "pp_opportunity_rate", "h2h_win_pct", "win_streak",
        "season_win_pct", "opp_season_win_pct", "opp_win_streak",
        # playoff motivation / standings pressure
        "points", "games_remaining", "conf_rank", "points_above_cut",
        "clinch_status",
        "opp_points", "opp_games_remaining", "opp_conf_rank",
        "opp_points_above_cut", "opp_clinch_status",
        "pts_above_cut_diff",
        "is_tanking", "is_must_win", "is_clinched",
        "opp_is_tanking", "opp_is_must_win", "opp_is_clinched",
        # momentum (2 pairs: near-term surge vs baseline, medium-term trend)
        "momentum_5v20", "momentum_10v20",
        # PDO / efficiency regression-to-mean (5v20 only — other pairs are derivable)
        "pdo_regress_5v20", "shoot_regress_5v20", "save_regress_5v20",
    ]
    return [c for c in candidates if c in df.columns]


def get_all_feature_cols(df: pd.DataFrame) -> list:
    """
    Feature columns for the unified model: ALL rolling windows (5/10/15/20)
    for both home and opponent, plus shared static features.
    L2 regularisation (C=0.03) lets the model discover which time horizon
    matters per feature rather than choosing a single window.
    """
    rolling = [c for c in df.columns
               if c.startswith("rolling_") or c.startswith("opp_rolling_")]
    static  = _static_features(df)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in rolling + static:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def evaluate_unified(models_dict: dict, feat_cols: list, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate unified (single-model) dict on test_df.

    models_dict : {model_name: fitted_model}
    feat_cols   : feature columns to use (from get_all_feature_cols)
    Returns a DataFrame with one row per model (window=0 placeholder).
    """
    y_full = test_df["result"].eq("W").astype(int)
    fc = [c for c in feat_cols if c in test_df.columns]
    if not fc:
        return pd.DataFrame()
    X = test_df[fc]
    records = []
    for name, model in models_dict.items():
        try:
            m = _metrics(model, X, y_full)
            records.append({"model": name, "window": 0, **m})
        except Exception as e:
            print(f"  [eval] {name} unified failed: {e}")
    return pd.DataFrame(records)


def _metrics(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy":  accuracy_score(y, y_pred),
        "roc_auc":   roc_auc_score(y, y_prob),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred, zero_division=0),
        "brier":     brier_score_loss(y, y_prob),
        "log_loss":  log_loss(y, y_prob),
    }


def sliding_window_evaluation(models_by_window: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate per-window trained models on the test season.

    models_by_window: {window_int: {model_name: fitted_model}}
    Returns a DataFrame with one row per (model, window).
    """
    y_full = test_df["result"].eq("W").astype(int)
    records = []

    for w, models in models_by_window.items():
        feat_cols = get_feature_cols_for_window(w, test_df)
        feat_cols = [c for c in feat_cols if c in test_df.columns]
        if not feat_cols:
            print(f"  [eval] No feature cols for window {w}, skipping.")
            continue

        X_w = test_df[feat_cols]

        for model_name, model in models.items():
            try:
                m = _metrics(model, X_w, y_full)
                records.append({"model": model_name, "window": w, **m})
            except Exception as e:
                print(f"  [eval] {model_name} N={w} failed: {e}")

    return pd.DataFrame(records)


def print_results(results: pd.DataFrame) -> None:
    """Print a full metrics table grouped by window."""
    if results.empty:
        print("No results to display.")
        return

    for w in sorted(results["window"].unique()):
        df_w = results[results["window"] == w].set_index("model")[METRIC_COLS]
        df_w = df_w.sort_values("accuracy", ascending=False)
        print(f"\n── Window N={w} ──────────────────────────────────────────────────────")
        header = f"  {'Model':<22}" + "".join(f"  {c.upper():<10}" for c in METRIC_COLS)
        print(header)
        print("  " + "─" * (len(header) - 2))
        for model, row in df_w.iterrows():
            vals = "".join(f"  {row[c]:<10.4f}" for c in METRIC_COLS)
            print(f"  {model:<22}{vals}")

    print("\n── Best configuration per metric ─────────────────────────────────────")
    best = results.loc[results["accuracy"].idxmax()]
    print(f"  Best accuracy : {best['model']} | N={int(best['window'])} | "
          f"acc={best['accuracy']:.4f} | AUC={best['roc_auc']:.4f} | "
          f"F1={best['f1']:.4f} | Brier={best['brier']:.4f}")


def predict_game(model, game_row: pd.Series, feature_cols: list) -> dict:
    X = game_row[feature_cols].values.reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return {
        "prediction": "WIN" if pred == 1 else "LOSS",
        "confidence": prob if pred == 1 else 1 - prob,
        "raw_prob": prob,
    }
