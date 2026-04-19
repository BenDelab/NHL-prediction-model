"""
Player prop model: P(player exceeds shots-on-goal line).

Architecture
------------
XGBoost count regressor (objective='count:poisson') predicts expected SOG count μ.

XGBoost is used instead of NB GLM because:
  - Tree-based models capture non-linear interactions (e.g. tired top-liner vs
    weak goalie disproportionately inflates SOG beyond an additive log-linear model)
  - No assumptions about feature scale or linearity
  - Handles missing values natively (optimal split direction learned per feature)

SOG is overdispersed (var/mean ≈ 1.42), so Poisson variance understates tail
probability. We estimate the NB dispersion α via method-of-moments from training
residuals, then apply the NB CDF analytically:

  P(SOG > line) = 1 − NB_CDF(floor(line), n=1/α, p=1/(1+α·μ))

This gives monotonically consistent probabilities across all four lines from
a single μ prediction: P(>1.5) ≥ P(>2.5) ≥ P(>3.5) ≥ P(>4.5) always.

Line selection
--------------
For each player we evaluate all four candidate lines (1.5/2.5/3.5/4.5) and
select the one in the "competitive zone" [0.30, 0.70] where |P - 0.5| is
largest. OVER/UNDER is recommended based on which direction the model favours.
Identical to how the totals model selects between 5.5/6.5/7.5.

Features (from pipeline/player_loader.py):
  player_roll5/10_sog_mean    — recent SOG form (primary μ driver)
  player_roll5/10_sog_std     — SOG consistency
  player_roll5/10_ice_mean    — icetime / role stability
  player_roll10_xg_mean       — shot quality
  player_roll10_hd_mean       — high-danger shots
  player_roll5/10_fenwick_mean— unblocked shot attempts
  player_roll10_ozone_mean    — offensive zone start rate
  opp_roll10_soa              — opponent shots-allowed (team-level defensive context)
  opp_goalie_roll10_gsax      — opponent goalie GSAx rolling mean (goalie quality)
  opp_goalie_roll10_hd_sv     — opponent goalie HD save rate rolling mean
  is_home                     — home/away
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import nbinom
from sklearn.metrics import mean_absolute_error, roc_auc_score

from pipeline.player_loader import (
    MIN_SOG_MEAN_FOR_PROP,
    get_current_player_stats,
    get_prop_feature_cols,
)

LINES           = [1.5, 2.5, 3.5, 4.5]
COMPETITIVE_MIN = 0.30
COMPETITIVE_MAX = 0.70
MIN_CONVICTION  = 0.08
MIN_RECENT_ICETIME = 600   # seconds — confirms active roster

# XGBoost hyperparameters for count regression
_XGB_PARAMS = dict(
    objective        = "count:poisson",
    n_estimators     = 400,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 10,   # prevents overfitting on rare players
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    n_jobs           = -1,
    random_state     = 42,
)


def _nb_p_over(mu: np.ndarray, alpha: float, line: float) -> np.ndarray:
    """
    P(SOG > line) from NB(μ, α) CDF.
    NB2: n = 1/α, p = 1/(1 + α·μ)  ←→  scipy size=n, prob=p
    Half-point line: floor(1.5)=1 → P(X ≥ 2) = 1 − CDF(1)
    """
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return 1.0 - nbinom.cdf(int(line), n, p)


def _estimate_nb_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Method-of-moments estimate of NB2 dispersion α.

    NB2: Var[Y] = μ + α·μ²
    Solving: α = (E[(Y-μ)²] - E[μ]) / E[μ²]
    Clipped to [0.05, 2.0] for numerical stability.
    """
    resid2 = (y - mu) ** 2
    alpha = (resid2.mean() - mu.mean()) / (mu ** 2).mean()
    return float(np.clip(alpha, 0.05, 2.0))


class PropModel:
    """
    Fitted XGBoost count regressor for player SOG prediction.

    Attributes
    ----------
    xgb_model : fitted XGBRegressor (objective='count:poisson')
    alpha     : NB2 dispersion parameter (method-of-moments from training residuals)
    feat_cols : list of feature column names used during training
    """

    def __init__(self, xgb_model, alpha: float, feat_cols: list[str]):
        self.xgb_model = xgb_model
        self.alpha     = alpha
        self.feat_cols = feat_cols

    def predict_mu(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted expected SOG count μ for each row."""
        fc  = [c for c in self.feat_cols if c in X.columns]
        arr = X[fc].values.astype(float)
        return self.xgb_model.predict(arr)

    def p_over(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """P(SOG > line) for each row."""
        return _nb_p_over(self.predict_mu(X), self.alpha, line)

    def p_over_all_lines(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """P(SOG > line) for all candidate lines from a single μ prediction."""
        mu = self.predict_mu(X)
        return {line: _nb_p_over(mu, self.alpha, line) for line in LINES}


def train_prop_model(player_df: pd.DataFrame,
                     train_seasons: set | None = None) -> PropModel | None:
    """
    Fit an XGBoost Poisson count regressor on training data.

    XGBoost handles missing values natively — no imputation applied, which
    allows it to learn optimal split directions for sparse features (e.g. early
    season before rolling windows are populated).

    Dispersion α is estimated post-fit via method-of-moments on training
    residuals, then used with the NB CDF for all probability queries.

    Returns a PropModel, or None if training fails.
    """
    if train_seasons is None:
        all_seasons = sorted(player_df["season"].unique())
        train_seasons = set(all_seasons[:-1])

    train_df  = player_df[player_df["season"].isin(train_seasons)].copy()
    feat_cols = get_prop_feature_cols(train_df)

    if not feat_cols:
        print("[props] No feature columns — prop model disabled.")
        return None

    # Keep rows with at least the core rolling SOG features; NaN in goalie/context
    # features is fine — XGBoost handles them natively.
    core_cols  = [c for c in feat_cols if "sog_mean" in c or "ice_mean" in c]
    valid_mask = train_df[core_cols].notna().any(axis=1)
    X_df       = train_df.loc[valid_mask, feat_cols]
    y_train    = train_df.loc[valid_mask, "I_F_shotsOnGoal"].values.astype(float)

    if len(X_df) < 100:
        print("[props] Insufficient training rows.")
        return None

    X_arr = X_df.values.astype(float)

    model = xgb.XGBRegressor(**_XGB_PARAMS)
    model.fit(X_arr, y_train, verbose=False)

    mu_train = model.predict(X_arr)
    mae      = mean_absolute_error(y_train, mu_train)
    alpha    = _estimate_nb_alpha(y_train, mu_train)

    var_mean = y_train.var() / y_train.mean()
    print(f"  [props] XGB Poisson trained | {len(X_df):,} rows | "
          f"MAE={mae:.3f} | α={alpha:.3f} | var/mean={var_mean:.3f}")
    print(f"  [props] Mean μ={mu_train.mean():.3f}  actual={y_train.mean():.3f}  "
          f"Features: {len(feat_cols)}")

    # Print top feature importances
    imp = model.feature_importances_
    top = sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:5]
    print(f"  [props] Top features: " +
          ", ".join(f"{n}({v:.3f})" for n, v in top))

    return PropModel(model, alpha, feat_cols)


def validate_prop_model(model: PropModel, player_df: pd.DataFrame,
                         val_season: str = "2024-25") -> None:
    """Print walk-forward validation: AUC per line + competitive-zone hit rates."""
    val_df = player_df[player_df["season"] == val_season].copy()
    if val_df.empty:
        print(f"  [props] No data for {val_season}.")
        return

    feat_cols = get_prop_feature_cols(val_df)
    core_cols = [c for c in feat_cols if "sog_mean" in c or "ice_mean" in c]
    valid_mask = val_df[core_cols].notna().any(axis=1)
    val_valid  = val_df[valid_mask].copy().reset_index(drop=True)

    mu_pred = model.predict_mu(val_valid[feat_cols])
    mae     = mean_absolute_error(val_valid["I_F_shotsOnGoal"], mu_pred)

    print(f"\nProp model walk-forward ({val_season}, {valid_mask.sum()} valid rows):")
    print(f"  XGB Poisson — MAE={mae:.3f}  α={model.alpha:.3f}")

    all_probs = model.p_over_all_lines(val_valid[feat_cols])
    print("  Per-line AUC:")
    for line in LINES:
        y_bin = (val_valid["I_F_shotsOnGoal"] > line).astype(int)
        auc   = roc_auc_score(y_bin, all_probs[line])
        print(f"    Over {line}: AUC={auc:.3f}  base={y_bin.mean():.3f}")

    for line in LINES:
        val_valid[f"p_over_{line}"] = all_probs[line]
    picks_meta = val_valid.apply(_select_best_line, axis=1)
    val_valid  = pd.concat([val_valid, picks_meta], axis=1)
    has_pick   = val_valid["best_line"].notna()

    print(f"\n  Competitive-zone picks: {has_pick.sum()} ({has_pick.mean():.1%})  "
          f"[break-even at -110: 52.4%]")
    for min_conv in [0.08, 0.10, 0.12, 0.15]:
        sub = val_valid[has_pick & (val_valid["conviction"] >= min_conv)]
        if len(sub) < 20:
            continue
        c = sub.apply(lambda r: (
            (r["I_F_shotsOnGoal"] > r["best_line"]  and r["direction"] == "OVER") or
            (r["I_F_shotsOnGoal"] <= r["best_line"] and r["direction"] == "UNDER")
        ), axis=1)
        over_sub  = sub[sub["direction"] == "OVER"]
        under_sub = sub[sub["direction"] == "UNDER"]
        o_hit = (over_sub["I_F_shotsOnGoal"] > over_sub["best_line"]).mean() \
                if len(over_sub) > 0 else float("nan")
        u_hit = (under_sub["I_F_shotsOnGoal"] <= under_sub["best_line"]).mean() \
                if len(under_sub) > 0 else float("nan")
        pnl = c.sum() * (100/110) * 10 - (len(sub) - c.sum()) * 10
        roi = pnl / (len(sub) * 10) * 100
        print(f"  conviction≥{min_conv:.2f}: n={len(sub):5d}  hit={c.mean():.3f}  "
              f"ROI={roi:+.1f}%  "
              f"over({len(over_sub)})={o_hit:.3f}  under({len(under_sub)})={u_hit:.3f}")
    print()


def _select_best_line(row: pd.Series) -> pd.Series:
    """Pick the competitive-zone line with highest conviction for one player-game."""
    best_line, best_conviction, best_p = None, 0.0, 0.5
    for line in LINES:
        col = f"p_over_{line}"
        if col not in row.index:
            continue
        p = float(row[col])
        if not (COMPETITIVE_MIN <= p <= COMPETITIVE_MAX):
            continue
        conv = abs(p - 0.5)
        if conv > best_conviction:
            best_conviction = conv
            best_line = line
            best_p    = p

    if best_line is None:
        return pd.Series({"best_line": None, "best_p": 0.5,
                          "conviction": 0.0, "direction": None})
    return pd.Series({
        "best_line":  best_line,
        "best_p":     best_p,
        "conviction": best_conviction,
        "direction":  "OVER" if best_p >= 0.5 else "UNDER",
    })


def predict_props(model: PropModel,
                  player_df: pd.DataFrame,
                  home_team: str,
                  away_team: str,
                  min_conviction: float = MIN_CONVICTION) -> list[dict]:
    """Generate prop picks for active players in tonight's matchup."""
    if model is None:
        return []

    feat_cols = get_prop_feature_cols(player_df)
    if not feat_cols:
        return []

    picks = []
    for team, is_home_flag, opp in [(home_team, 1, away_team), (away_team, 0, home_team)]:
        roster = get_current_player_stats(player_df, team, opp_team=opp)
        if roster.empty:
            continue
        roster = roster[roster["icetime"] >= MIN_RECENT_ICETIME].copy()
        sog_col = "player_roll10_sog_mean"
        if sog_col in roster.columns:
            roster = roster[roster[sog_col] >= MIN_SOG_MEAN_FOR_PROP].copy()
        if roster.empty:
            continue

        fc = [c for c in feat_cols if c in roster.columns]
        if not fc:
            continue

        X = roster[fc].copy()
        X["is_home"] = is_home_flag

        # Require core SOG features; goalie/context NaN is fine for XGBoost
        core = [c for c in fc if "sog_mean" in c]
        valid_mask = X[core].notna().any(axis=1) if core else pd.Series(True, index=X.index)
        if valid_mask.sum() == 0:
            continue

        X_valid    = X[valid_mask]
        valid_rows = roster[valid_mask].reset_index(drop=True)

        all_probs = model.p_over_all_lines(X_valid)
        for line in LINES:
            valid_rows[f"p_over_{line}"] = all_probs[line]

        line_picks = valid_rows.apply(_select_best_line, axis=1)
        valid_rows = pd.concat([valid_rows, line_picks], axis=1)

        for _, row in valid_rows.iterrows():
            if row["best_line"] is None or float(row["conviction"]) < min_conviction:
                continue
            line_probs = {
                line: {"p_over":  round(float(row.get(f"p_over_{line}", 0.5)), 3),
                       "p_under": round(1 - float(row.get(f"p_over_{line}", 0.5)), 3)}
                for line in LINES if f"p_over_{line}" in row.index
            }
            picks.append({
                "name":        row.get("name", "Unknown"),
                "team":        team,
                "position":    row.get("position", "?"),
                "best_line":   float(row["best_line"]),
                "direction":   row["direction"],
                "probability": round(float(row["best_p"]), 3),
                "conviction":  round(float(row["conviction"]), 3),
                "roll10_sog":  round(float(row.get("player_roll10_sog_mean", np.nan)), 2),
                "roll5_sog":   round(float(row.get("player_roll5_sog_mean",  np.nan)), 2),
                "icetime_min": round(float(row.get("player_roll10_ice_mean", 0)) / 60, 1),
                "opp_soa":     round(float(row.get("opp_roll10_soa", np.nan)), 1)
                               if not pd.isna(row.get("opp_roll10_soa")) else None,
                "opp_gsax":    round(float(row.get("opp_goalie_roll10_gsax", np.nan)), 2)
                               if not pd.isna(row.get("opp_goalie_roll10_gsax")) else None,
                "is_home":     bool(is_home_flag),
                "line_probs":  line_probs,
            })

    picks.sort(key=lambda x: -x["conviction"])
    return picks


def print_prop_picks(picks: list[dict], top_n: int = 15,
                     home_team: str = "", away_team: str = "") -> None:
    """Print formatted prop picks with per-line probability breakdown."""
    if not picks:
        print("  No prop picks above conviction threshold.")
        return

    matchup = f"{away_team} @ {home_team}" if home_team else ""
    if matchup:
        print(f"  SOG Props — {matchup}")

    header = (f"  {'Player':<22} {'Pos':<4} {'Team':<5} "
              f"{'Rec':<12} {'P(rec)':<8} "
              f"{'Roll10':<7} {'Roll5':<6} {'Ice':<6} {'OppSOA':<7} {'GSAx':<6} {'H/A'}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    shown = 0
    for p in picks:
        if shown >= top_n:
            break
        ha      = "H" if p["is_home"] else "A"
        opp_soa = f"{p['opp_soa']:.0f}"  if p["opp_soa"]  is not None else "N/A"
        gsax    = f"{p['opp_gsax']:+.2f}" if p["opp_gsax"] is not None else "N/A"
        rec     = f"{p['direction']} {p['best_line']:.1f}"

        print(f"  {p['name']:<22} {p['position']:<4} {p['team']:<5} "
              f"{rec:<12} {p['probability']:<8.3f} "
              f"{p['roll10_sog']:<7.1f} {p['roll5_sog']:<6.1f} "
              f"{p['icetime_min']:<6.0f}m {opp_soa:<7} {gsax:<6} {ha}")

        lp = p.get("line_probs", {})
        if lp:
            parts = []
            for line in sorted(lp.keys()):
                po   = lp[line]["p_over"]
                pu   = lp[line]["p_under"]
                star = " ★" if line == p["best_line"] else "  "
                parts.append(f"o{line:.1f}:{po:.0%}/u:{pu:.0%}{star}")
            print(f"    ({p['conviction']:.2f})  " + "   ".join(parts))

        shown += 1
