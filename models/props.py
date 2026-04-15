"""
Player prop model: P(player exceeds shots-on-goal line).

Architecture
------------
One Logistic Regression model trained per candidate SOG line (1.5 / 2.5 / 3.5 / 4.5).
Each model predicts P(I_F_shotsOnGoal > line) for a player-game.

For each player in an upcoming game we:
  1. Compute P(over X) for all four lines.
  2. Find the line that falls in the "competitive zone" [0.30, 0.70] —
     these are lines where the outcome isn't already obvious from base rates.
  3. Among competitive lines, pick the one where |P - 0.5| is largest
     (maximum model conviction).
  4. Surface both OVER and UNDER as candidates so the model can recommend
     the direction with the edge, not just OVER.

This mirrors the totals model design: instead of "choose UNDER 4.5 because
it's 98% obvious", we find the line where the model's prediction genuinely
diverges from the 50/50 baseline — the same philosophy as picking between
5.5 / 6.5 / 7.5 for the game total.

Walk-forward validation (train 2021-24 → test 2024-25):
  Competitive-zone picks, conviction ≥ 0.08:  n=22,891   hit=65.8%
  Competitive-zone picks, conviction ≥ 0.12:  n=18,797   hit=66.9%
  Competitive-zone picks, conviction ≥ 0.15:  n=12,815   hit=68.1%
    (OVER subset at ≥ 0.15: 65.4%  |  UNDER subset: 68.8%)

Features (from pipeline/player_loader.py):
  player_roll5/10_sog_mean    — recent SOG form (primary driver)
  player_roll5/10_sog_std     — SOG consistency
  player_roll5/10_ice_mean    — icetime / role stability
  player_roll10_xg_mean       — shot quality (xGoals)
  player_roll10_hd_mean       — high-danger shots
  player_roll5/10_fenwick_mean— unblocked shot attempts (broadens beyond SOG)
  player_roll10_ozone_mean    — offensive zone start rate (opportunity signal)
  opp_roll10_soa              — opponent shots-allowed rolling avg
                                 (key: facing DAL ~26 SOA vs CBJ ~34 SOA matters)
  is_home                     — home / away context
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.player_loader import (
    MIN_SOG_MEAN_FOR_PROP,
    get_current_player_stats,
    get_prop_feature_cols,
)

# Lines we model — half-point values so over/under is a clean split with no push
LINES = [1.5, 2.5, 3.5, 4.5]

# "Competitive zone": only consider lines where P(over) is within this range.
# Lines outside this zone are trivially obvious (e.g. UNDER 4.5 for a 4th-liner).
COMPETITIVE_MIN = 0.30
COMPETITIVE_MAX = 0.70

# Minimum |P - 0.5| to surface a pick
MIN_CONVICTION = 0.08

# Minimum icetime in most recent game (confirms active roster status)
MIN_RECENT_ICETIME = 600   # 10 minutes


def _make_target(df: pd.DataFrame, line: float) -> pd.Series:
    return (df["I_F_shotsOnGoal"] > line).astype(int)


def train_prop_model(player_df: pd.DataFrame,
                     train_seasons: set | None = None) -> dict:
    """
    Train one Logistic Regression per SOG line.

    Returns
    -------
    dict: {line: fitted Pipeline}
      e.g. {1.5: pipe_15, 2.5: pipe_25, 3.5: pipe_35, 4.5: pipe_45}
    """
    if train_seasons is None:
        all_seasons = sorted(player_df["season"].unique())
        train_seasons = set(all_seasons[:-1])

    train_df  = player_df[player_df["season"].isin(train_seasons)].copy()
    feat_cols = get_prop_feature_cols(train_df)

    if not feat_cols:
        print("[props] No feature columns found — prop model disabled.")
        return {}

    valid_mask = train_df[feat_cols].notna().all(axis=1)
    X_train    = train_df.loc[valid_mask, feat_cols]

    models = {}
    for line in LINES:
        y_train = _make_target(train_df, line)[valid_mask]
        base    = y_train.mean()

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(C=0.1, max_iter=1000,
                                            solver="lbfgs", random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        models[line] = pipe
        print(f"  [props] Trained over-{line} model | "
              f"{len(X_train)} rows | base rate {base:.1%}")

    return models


def validate_prop_models(models: dict, player_df: pd.DataFrame,
                          val_season: str = "2024-25") -> None:
    """Print walk-forward validation metrics for the competitive-zone approach."""
    val_df = player_df[player_df["season"] == val_season].copy()
    if val_df.empty:
        print(f"  [props] No data for validation season {val_season}.")
        return

    feat_cols  = get_prop_feature_cols(val_df)
    valid_mask = val_df[feat_cols].notna().all(axis=1)
    val_valid  = val_df[valid_mask].copy()
    X_val      = val_valid[feat_cols]

    # Compute all line probabilities
    for line in LINES:
        if line not in models:
            continue
        val_valid[f"p_over_{line}"] = models[line].predict_proba(X_val)[:, 1]

    # Dynamic line selection
    picks = val_valid.apply(_select_best_line, axis=1)
    val_valid = pd.concat([val_valid, picks], axis=1)
    has_pick = val_valid["best_line"].notna()

    correct = val_valid[has_pick].apply(lambda r: (
        (r["I_F_shotsOnGoal"] > r["best_line"]  and r["direction"] == "OVER") or
        (r["I_F_shotsOnGoal"] <= r["best_line"] and r["direction"] == "UNDER")
    ), axis=1)

    print(f"\nProp model walk-forward ({val_season}, {valid_mask.sum()} valid rows):")
    print(f"  Players with a competitive line: {has_pick.sum()} ({has_pick.mean():.1%})")
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
        print(f"  conviction≥{min_conv:.2f}: n={len(sub):5d}  hit={c.mean():.3f}  "
              f"over({len(over_sub)})={o_hit:.3f}  under({len(under_sub)})={u_hit:.3f}")
    print()

    # Per-line AUC summary
    print("  Per-line AUC:")
    for line, model in models.items():
        y = _make_target(val_valid, line)
        p = model.predict_proba(X_val)[:, 1]
        auc  = roc_auc_score(y, p)
        base = y.mean()
        print(f"    Over {line}: AUC={auc:.3f}  base={base:.3f}")


def _select_best_line(row: pd.Series) -> pd.Series:
    """
    For one player-game row (with p_over_{line} columns already computed),
    pick the line in the competitive zone with the highest model conviction.

    Returns: best_line, best_p, conviction, direction
    """
    best_line, best_conviction, best_p = None, 0.0, 0.5
    for line in LINES:
        col = f"p_over_{line}"
        if col not in row.index:
            continue
        p = float(row[col])
        if not (COMPETITIVE_MIN <= p <= COMPETITIVE_MAX):
            continue
        conviction = abs(p - 0.5)
        if conviction > best_conviction:
            best_conviction = conviction
            best_line = line
            best_p = p

    if best_line is None:
        return pd.Series({"best_line": None, "best_p": 0.5,
                          "conviction": 0.0, "direction": None})

    direction = "OVER" if best_p >= 0.5 else "UNDER"
    return pd.Series({"best_line": best_line, "best_p": best_p,
                      "conviction": best_conviction, "direction": direction})


def predict_props(models: dict,
                  player_df: pd.DataFrame,
                  home_team: str,
                  away_team: str,
                  min_conviction: float = MIN_CONVICTION) -> list[dict]:
    """
    Generate prop picks for all active players in home_team and away_team.

    For each player:
      - Compute P(over X) for all four lines using tonight's opponent context
      - Find the line in the competitive zone [0.30, 0.70] with max |P-0.5|
      - Surface as OVER or UNDER depending on which direction has the edge

    Returns a list of dicts sorted by conviction descending.
    """
    if not models:
        return []

    feat_cols = get_prop_feature_cols(player_df)
    if not feat_cols:
        return []

    picks = []
    for team, is_home_flag, opp in [(home_team, 1, away_team), (away_team, 0, home_team)]:
        # Get roster with tonight's opponent context injected
        roster = get_current_player_stats(player_df, team,
                                           opp_team=opp)
        if roster.empty:
            continue

        # Only active players with enough recent icetime
        roster = roster[roster["icetime"] >= MIN_RECENT_ICETIME].copy()
        # Only players whose rolling SOG average clears the minimum threshold
        sog_col = "player_roll10_sog_mean"
        if sog_col in roster.columns:
            roster = roster[roster[sog_col] >= MIN_SOG_MEAN_FOR_PROP].copy()
        if roster.empty:
            continue

        fc_present = [c for c in feat_cols if c in roster.columns]
        if not fc_present:
            continue

        X = roster[fc_present].copy()
        X["is_home"] = is_home_flag   # override with tonight's home/away

        valid_mask = X.notna().all(axis=1)
        if valid_mask.sum() == 0:
            continue

        X_valid    = X[valid_mask]
        valid_rows = roster[valid_mask].reset_index(drop=True)

        # Compute P(over X) for all lines
        for line, model in models.items():
            valid_rows[f"p_over_{line}"] = model.predict_proba(X_valid)[:, 1]

        # Select best line per player
        line_picks = valid_rows.apply(_select_best_line, axis=1)
        valid_rows = pd.concat([valid_rows, line_picks], axis=1)

        for _, row in valid_rows.iterrows():
            if row["best_line"] is None:
                continue
            if float(row["conviction"]) < min_conviction:
                continue

            # Build per-line probability table for display
            line_probs = {
                line: {
                    "p_over":  round(float(row.get(f"p_over_{line}", 0.5)), 3),
                    "p_under": round(1 - float(row.get(f"p_over_{line}", 0.5)), 3),
                }
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
                "is_home":     bool(is_home_flag),
                "line_probs":  line_probs,
            })

    picks.sort(key=lambda x: -x["conviction"])
    return picks


def print_prop_picks(picks: list[dict], top_n: int = 15,
                     home_team: str = "", away_team: str = "") -> None:
    """
    Print a formatted prop picks table.

    For each player shows the recommended line/direction and the full
    probability table across all four lines — same style as the totals table.
    """
    if not picks:
        print("  No prop picks above conviction threshold.")
        return

    matchup = f"{away_team} @ {home_team}" if home_team else ""
    if matchup:
        print(f"  SOG Props — {matchup}")

    header = (f"  {'Player':<22} {'Pos':<4} {'Team':<5} "
              f"{'Rec':<12} {'P(rec)':<8} "
              f"{'Roll10':<7} {'Roll5':<6} {'Ice':<6} {'OppSOA':<8} {'H/A'}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    shown = 0
    for p in picks:
        if shown >= top_n:
            break
        ha       = "H" if p["is_home"] else "A"
        r10      = f"{p['roll10_sog']:.1f}"
        r5       = f"{p['roll5_sog']:.1f}"
        ice      = f"{p['icetime_min']:.0f}m"
        opp_soa  = f"{p['opp_soa']:.0f}" if p["opp_soa"] is not None else "N/A"
        rec      = f"{p['direction']} {p['best_line']:.1f}"
        prob_str = f"{p['probability']:.3f}"
        conv_str = f"({p['conviction']:.2f})"

        print(f"  {p['name']:<22} {p['position']:<4} {p['team']:<5} "
              f"{rec:<12} {prob_str:<8} "
              f"{r10:<7} {r5:<6} {ice:<6} {opp_soa:<8} {ha}")

        # Per-line probability breakdown (like totals table)
        lp = p.get("line_probs", {})
        if lp:
            parts = []
            for line in sorted(lp.keys()):
                po = lp[line]["p_over"]
                pu = lp[line]["p_under"]
                best_marker = " ★" if line == p["best_line"] else "  "
                parts.append(f"o{line:.1f}:{po:.0%}/u:{pu:.0%}{best_marker}")
            print(f"    {conv_str}  " + "   ".join(parts))

        shown += 1
