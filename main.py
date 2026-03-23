"""
NHL Home Game Outcome Predictor — end-to-end pipeline.

Usage
─────
  python main.py                          # train + live validation only
  python main.py --predict                # train + predict next upcoming games
  python main.py --predict --bankroll 500 # with custom bankroll (default $1000)
  python main.py --predict --date 2026-03-25  # predict for a specific date
"""

import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pipeline import data_loader, features, noise_reduction, calibration, betting
from models import train, evaluate, secondary, totals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NHL Home Game Predictor")
    p.add_argument("--predict",   action="store_true", help="Run prediction mode for upcoming games")
    p.add_argument("--bankroll",  type=float, default=1000.0, help="Bankroll in dollars (default: 1000)")
    p.add_argument("--date",      type=str,   default=None,   help="Predict for date YYYY-MM-DD (default: next available)")
    p.add_argument("--team",      type=str,   default=None,   help="Single-game: team abbreviation")
    p.add_argument("--game",      type=int,   default=None,   help="Single-game: game number in season")
    return p


def _spread_ensemble_prob(models_by_window: dict, game_row: pd.Series,
                          df_ref: pd.DataFrame) -> float:
    """Average calibrated spread probability across all windows and models."""
    probs = []
    for w, models in models_by_window.items():
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_ref)
                     if c in df_ref.columns]
        X = game_row[feat_cols].to_frame().T
        for model in models.values():
            try:
                probs.append(float(model.predict_proba(X)[0][1]))
            except Exception:
                pass
    return float(np.mean(probs)) if probs else 0.5


def _goals_ensemble_pred(models_by_window: dict, game_row: pd.Series,
                         df_ref: pd.DataFrame) -> float:
    """Average regression prediction across all windows and models."""
    preds = []
    for w, models in models_by_window.items():
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_ref)
                     if c in df_ref.columns]
        X = game_row[feat_cols].to_frame().T
        for model in models.values():
            try:
                preds.append(float(model.predict(X)[0]))
            except Exception:
                pass
    return float(np.mean(preds)) if preds else 5.5


def _ensemble_probs(calibrated: dict, game_row: pd.Series, df_ref: pd.DataFrame) -> dict:
    """
    Run every (window, model) combination on one game row.
    Returns {label: probability} and the ensemble mean.
    """
    all_probs: dict[str, float] = {}
    for w, models in calibrated.items():
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_ref) if c in df_ref.columns]
        X = game_row[feat_cols].to_frame().T
        for name, model in models.items():
            try:
                p = float(model.predict_proba(X)[0][1])
                all_probs[f"{name}  N={w}"] = p
            except Exception:
                pass
    return all_probs


def main():
    args = build_parser().parse_args()

    # ── 1. Load ALL seasons (2021-22 through 2025-26) ────────────────────────
    print("Loading data...")
    df_all = data_loader.load_all_seasons()

    # ── 2. Feature engineering on the full dataset ───────────────────────────
    #   Rolling features use shift(1), so upcoming games (NaN result) get
    #   correct feature vectors computed from all prior played games.
    print("Engineering features...")
    df_all_fe = features.build_features(df_all)

    # ── 3. Noise reduction — fit on all played rows, apply everywhere ─────────
    print("Applying noise reduction...")
    played_mask = df_all_fe["result"].notna()
    df_all_fe.loc[played_mask] = noise_reduction.apply(df_all_fe[played_mask].copy())

    # Split
    df_train_fe   = df_all_fe[df_all_fe["season"].isin(data_loader.TRAIN_SEASONS)].copy()
    df_predict_fe = df_all_fe[df_all_fe["season"] == data_loader.PREDICT_SEASON].copy()

    y_train = df_train_fe["result"].eq("W").astype(int)

    # ── 4. Train one set of models per window ────────────────────────────────
    print(f"\nTraining on {len(df_train_fe)} games across {data_loader.TRAIN_SEASONS}")
    nan_rows = df_train_fe[[c for c in df_train_fe.columns if c.startswith("rolling_")]].isna().any(axis=1).sum()
    print(f"  ({nan_rows} rows with NaN — handled by imputer in each pipeline)\n")

    all_models: dict = {}
    for w in evaluate.WINDOWS:
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]
        print(f"── Window N={w} | {len(feat_cols)} features ──────────────────────────")
        all_models[w] = train.train_all(df_train_fe[feat_cols], y_train)

    # ── 5. Calibrate per window ───────────────────────────────────────────────
    print("\nCalibrating models...")
    cal_split = int(len(df_train_fe) * 0.8)
    calibrated: dict = {}

    for w, models in all_models.items():
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]
        X_cal = df_train_fe[feat_cols].iloc[cal_split:]
        y_cal = y_train.iloc[cal_split:]
        print(f"  N={w}:")
        calibrated[w] = {name: calibration.calibrate(m, X_cal, y_cal) for name, m in models.items()}

    # ── 6. Threshold optimisation ─────────────────────────────────────────────
    print("\nOptimising decision thresholds (F-0.5)...")
    for w, models in calibrated.items():
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]
        X_cal = df_train_fe[feat_cols].iloc[cal_split:]
        y_cal = y_train.iloc[cal_split:]
        for name, model in list(models.items()):
            t = evaluate.find_optimal_threshold(model, X_cal, y_cal, beta=0.5)
            calibrated[w][name] = evaluate.ThresholdedModel(model, t)
            print(f"  N={w} {name}: threshold={t:.3f}")

    # ── 7. Live validation on played 2025-26 games ───────────────────────────
    df_live = df_predict_fe[df_predict_fe["result"].notna()].copy()
    if not df_live.empty:
        print(f"\nLive validation on {len(df_live)} played 2025-26 games...")
        results = evaluate.sliding_window_evaluation(calibrated, df_live)
        evaluate.print_results(results)
    else:
        results = pd.DataFrame()
        print("\nNo played 2025-26 games found for live validation.")

    # ── 8. Predict upcoming games ─────────────────────────────────────────────
    if not args.predict:
        return

    upcoming = df_predict_fe[df_predict_fe["result"].isna()].sort_values("date")
    if upcoming.empty:
        print("\nNo upcoming games found in 2025-26 data.")
        return

    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        target_date = upcoming["date"].min()

    next_games = upcoming[upcoming["date"] == target_date]
    if next_games.empty:
        print(f"\nNo upcoming games on {target_date.date()}.")
        return

    # Determine best window from live validation (highest accuracy)
    if not results.empty:
        best_row    = results.loc[results["accuracy"].idxmax()]
        best_window = int(best_row["window"])
        best_model  = best_row["model"]
        print(f"\nBest validated config: {best_model} | N={best_window} "
              f"| acc={best_row['accuracy']:.3f} | AUC={best_row['roc_auc']:.3f}")
    else:
        best_window = 20
        best_model  = "XGBoost"

    # ── 8b. Train secondary models (OT likelihood + regulation win) ───────────
    print("\nTraining secondary models (OT / regulation win)...")
    sec_feat_cols = [c for c in evaluate.get_feature_cols_for_window(best_window, df_train_fe)
                     if c in df_train_fe.columns]
    X_sec = df_train_fe[sec_feat_cols]

    played_train = df_train_fe["result"].notna() & df_train_fe["overtime_type"].notna()
    X_sec_played = X_sec[played_train]
    y_ot      = df_train_fe.loc[played_train, "overtime_type"].ne("REG").astype(int)
    y_reg_win = (df_train_fe.loc[played_train, "result"].eq("W") &
                 df_train_fe.loc[played_train, "overtime_type"].eq("REG")).astype(int)

    ot_model      = secondary.train_ot_model(X_sec_played, y_ot)
    reg_win_model = secondary.train_reg_win_model(X_sec_played, y_reg_win)
    print(f"  OT base rate (train): {y_ot.mean():.1%}  |  "
          f"Reg-win base rate: {y_reg_win.mean():.1%}")

    # ── 8c. Train totals / spread models (multi-window, calibrated ensemble) ──
    print("\nTraining spread / totals models (4 windows × 3 models each)...")
    played_all = (df_train_fe["result"].notna() &
                  df_train_fe["goals_for"].notna() &
                  df_train_fe["team_sog"].notna())
    df_played = df_train_fe[played_all]

    y_home_spread = (df_played["goals_for"] - df_played["goals_against"] >= 2).astype(int)
    y_away_spread = (df_played["goals_against"] - df_played["goals_for"] >= 2).astype(int)
    y_total_goals = df_played["goals_for"] + df_played["goals_against"]
    print(f"  Home -1.5 base rate: {y_home_spread.mean():.1%}  |  "
          f"Away -1.5 base rate: {y_away_spread.mean():.1%}  |  "
          f"Avg total goals: {y_total_goals.mean():.1f}")

    home_spread_by_window: dict = {}
    away_spread_by_window: dict = {}
    total_goals_by_window: dict = {}

    for w in evaluate.WINDOWS:
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe)
                     if c in df_train_fe.columns]
        X_w = df_played[feat_cols]
        print(f"  ── Window N={w} ─────────────────────────────────────────")
        print(f"    [spread home]")
        home_spread_by_window[w] = totals.train_spread_ensemble(X_w, y_home_spread)
        print(f"    [spread away]")
        away_spread_by_window[w] = totals.train_spread_ensemble(X_w, y_away_spread)
        print(f"    [goals regression]")
        total_goals_by_window[w] = totals.train_regression_all(X_w, y_total_goals)

    # Calibrate spread models (goals regression doesn't output probabilities)
    print("\n  Calibrating spread models...")
    for w in evaluate.WINDOWS:
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe)
                     if c in df_train_fe.columns]
        X_w       = df_played[feat_cols]
        cal_split = int(len(X_w) * 0.8)
        X_cal     = X_w.iloc[cal_split:]
        y_home_cal = y_home_spread.iloc[cal_split:]
        y_away_cal = y_away_spread.iloc[cal_split:]
        home_spread_by_window[w] = {
            name: calibration.calibrate(m, X_cal, y_home_cal)
            for name, m in home_spread_by_window[w].items()
        }
        away_spread_by_window[w] = {
            name: calibration.calibrate(m, X_cal, y_away_cal)
            for name, m in away_spread_by_window[w].items()
        }
        print(f"    N={w}: spread calibrated")

    print(f"\n{'═'*68}")
    print(f"  NHL PREDICTIONS — {target_date.date()}  "
          f"| Bankroll: ${args.bankroll:,.0f}")
    print(f"{'═'*68}")

    bet_summary = []

    for _, game in next_games.iterrows():
        # Ensemble probabilities across ALL windows and models
        all_probs = _ensemble_probs(calibrated, game, df_all_fe)
        if not all_probs:
            print(f"\n  {game['team']} vs {game['opponent']}: features unavailable, skipping.")
            continue

        ensemble_p   = float(np.mean(list(all_probs.values())))
        n_win        = sum(1 for p in all_probs.values() if p >= 0.5)
        consensus    = f"{n_win}/{len(all_probs)} models predict WIN"
        prediction   = "WIN" if ensemble_p >= 0.5 else "LOSS"
        conf_flag    = "▲ HIGH" if abs(ensemble_p - 0.5) > 0.08 else ("▶ MED" if abs(ensemble_p - 0.5) > 0.04 else "▼ LOW")

        # Per-window average (cleaner breakdown than 16 rows)
        window_avgs = {}
        for w in evaluate.WINDOWS:
            w_probs = {k: v for k, v in all_probs.items() if f"N={w}" in k}
            if w_probs:
                window_avgs[f"N={w} avg"] = float(np.mean(list(w_probs.values())))

        print(f"\n  {'─'*64}")
        game_num_str = f"Game #{int(game['game_num'])}" if pd.notna(game['game_num']) else ""
        print(f"  {game['team']:4s} (home) vs {game['opponent']:4s}  "
              f"{game_num_str}  | Confidence: {conf_flag}")
        print(f"  {'─'*64}")
        print(f"  Ensemble probability : {ensemble_p:>6.1%}  →  predicted {prediction}")
        print(f"  Model consensus      : {consensus}")
        print(f"  Window breakdown     : " +
              "  ".join(f"{k}: {v:.1%}" for k, v in window_avgs.items()))

        # ── OT / Regulation analysis ──────────────────────────────────────────
        X_game = game[sec_feat_cols].to_frame().T
        p_ot      = float(ot_model.predict_proba(X_game)[0][1])
        p_reg_win = float(reg_win_model.predict_proba(X_game)[0][1])

        # Pull per-team OT profile from the best-window rolling features
        bw = best_window
        home_ot_rate     = game.get(f"rolling_{bw}_ot_rate",     np.nan)
        home_ot_win_rate = game.get(f"rolling_{bw}_ot_win_rate", np.nan)
        away_ot_rate     = game.get(f"opp_rolling_{bw}_ot_rate",     np.nan)
        away_ot_win_rate = game.get(f"opp_rolling_{bw}_ot_win_rate", np.nan)

        ot_flag = ("▲ ELEVATED" if p_ot >= 0.33
                   else ("▶ POSSIBLE" if p_ot >= 0.22 else "▼ UNLIKELY"))

        print(f"\n  ── OT / Regulation Analysis ─────────────────────────────")
        print(f"  P(OT/SO)             : {p_ot:>6.1%}  {ot_flag}")

        if p_ot < 0.30:
            print(f"  → Low OT risk — regulation win probability: {p_reg_win:.1%}")
        else:
            # Show per-team OT profiles
            def _fmt(v): return f"{v:.1%}" if pd.notna(v) and not np.isnan(float(v)) else "N/A"
            print(f"  → OT possible — team OT profiles (last {bw} games):")
            print(f"    {game['team']:4s}  OT rate: {_fmt(home_ot_rate)}"
                  f"   OT win rate: {_fmt(home_ot_win_rate)}")
            print(f"    {game['opponent']:4s}  OT rate: {_fmt(away_ot_rate)}"
                  f"   OT win rate: {_fmt(away_ot_win_rate)}")

            # Net OT edge: if home team wins OT more often, note it
            if (pd.notna(home_ot_win_rate) and pd.notna(away_ot_win_rate)
                    and not np.isnan(float(home_ot_win_rate))
                    and not np.isnan(float(away_ot_win_rate))):
                net_ot = float(home_ot_win_rate) - float(away_ot_win_rate)
                if abs(net_ot) >= 0.10:
                    leader = game['team'] if net_ot > 0 else game['opponent']
                    print(f"    OT edge: {leader} ({net_ot:+.1%} net OT win rate)")

            print(f"  Regulation win prob  : {p_reg_win:.1%}"
                  f"  (if game stays in regulation)")

        # ── Totals / Spread analysis ──────────────────────────────────────────
        p_home_spread    = _spread_ensemble_prob(home_spread_by_window, game, df_all_fe)
        p_away_spread    = _spread_ensemble_prob(away_spread_by_window, game, df_all_fe)
        pred_total_goals = _goals_ensemble_pred(total_goals_by_window, game, df_all_fe)

        print(f"\n  ── Totals / Spread ──────────────────────────────────────")
        print(f"  P({game['team']:4s} covers -1.5)       : {p_home_spread:.1%}")
        print(f"  P({game['opponent']:4s} covers -1.5)       : {p_away_spread:.1%}")
        print(f"  Predicted total goals       : {pred_total_goals:.1f}")

        def _parse_odds(prompt):
            raw = input(prompt).strip()
            if not raw:
                return None
            try:
                val = float(raw)
                if val <= 1.0:
                    raise ValueError
                return val
            except ValueError:
                print("  Invalid odds — must be a decimal > 1.0. Skipping.")
                return None

        def _parse_line(prompt):
            raw = input(prompt).strip()
            if not raw:
                return None
            try:
                return float(raw)
            except ValueError:
                print("  Invalid line — must be a number. Skipping.")
                return None

        # ── Moneyline odds ────────────────────────────────────────────────────
        home_odds = _parse_odds(
            f"\n  [Moneyline] {game['team']} (home) decimal odds   (Enter to skip): ")
        away_odds = _parse_odds(
            f"  [Moneyline] {game['opponent']} (away) decimal odds  (Enter to skip): ")

        # ── Spread (puck line -1.5) odds — each side independent ─────────────
        home_spread_odds = _parse_odds(
            f"  [Spread]    {game['team']} -1.5 decimal odds        (Enter to skip): ")
        away_spread_odds = _parse_odds(
            f"  [Spread]    {game['opponent']} -1.5 decimal odds     (Enter to skip): ")

        # ── Total goals line + over/under odds ────────────────────────────────
        goals_line      = _parse_line(
            f"  [Goals O/U] Book's total-goals line  (e.g. 5.5, Enter to skip): ")
        goals_over_odds = goals_under_odds = None
        if goals_line is not None:
            goals_over_odds  = _parse_odds(f"    OVER  {goals_line} decimal odds: ")
            goals_under_odds = _parse_odds(f"    UNDER {goals_line} decimal odds: ")

        all_odds_skipped = all(x is None for x in [
            home_odds, away_odds, home_spread_odds, away_spread_odds,
            goals_over_odds, goals_under_odds,
        ])
        if all_odds_skipped:
            print("  [skipped]")
            continue

        def _print_bet(label: str, p: float, odds: float):
            """Print EV/Kelly for one bet side and append to bet_summary if +EV."""
            imp        = betting.implied_prob(odds)
            ev         = betting.expected_value(p, odds)
            kf         = betting.kelly_fraction(p, odds)
            edg        = betting.edge(p, odds)
            stake_full = kf * args.bankroll
            stake_half = kf * 0.5 * args.bankroll
            bet_flag   = "✓ BET" if ev > 0 else "✗ SKIP"

            print(f"  ── {label} ──────────────────────────────────────")
            print(f"  Model est. prob      : {p:.1%}")
            print(f"  Decimal odds         : {odds:.2f}")
            print(f"  Book implied prob    : {imp:.1%}")
            print(f"  Your edge            : {edg:+.1%}")
            print(f"  Expected value       : {ev:+.1%}   {bet_flag}")
            if ev > 0:
                print(f"  Full Kelly stake     : {kf:.1%}   (${stake_full:,.2f})")
                print(f"  Half Kelly stake     : {kf*0.5:.1%}   (${stake_half:,.2f})"
                      f"  ← recommended")
                bet_summary.append({
                    "matchup":    label,
                    "prob":       p,
                    "odds":       odds,
                    "ev":         ev,
                    "full_kelly": stake_full,
                    "half_kelly": stake_half,
                })
            print()

        print()

        # Moneyline
        p_home = ensemble_p
        p_away = 1.0 - ensemble_p
        if home_odds is not None:
            _print_bet(f"{game['team']} ML (home)", p_home, home_odds)
        if away_odds is not None:
            _print_bet(f"{game['opponent']} ML (away)", p_away, away_odds)

        # Spread — each side uses its own independent model
        if home_spread_odds is not None:
            _print_bet(f"{game['team']} -1.5 (spread)", p_home_spread, home_spread_odds)
        if away_spread_odds is not None:
            _print_bet(f"{game['opponent']} -1.5 (spread)", p_away_spread, away_spread_odds)

        # Total goals
        if goals_line is not None:
            p_over_goals  = totals.prob_over_goals(pred_total_goals, goals_line)
            p_under_goals = totals.prob_under(p_over_goals)
            if goals_over_odds is not None:
                _print_bet(f"Goals OVER  {goals_line}", p_over_goals, goals_over_odds)
            if goals_under_odds is not None:
                _print_bet(f"Goals UNDER {goals_line}", p_under_goals, goals_under_odds)

    # ── Betting summary ───────────────────────────────────────────────────────
    if bet_summary:
        total_half = sum(s["half_kelly"] for s in bet_summary)
        total_full = sum(s["full_kelly"] for s in bet_summary)
        print(f"\n{'═'*68}")
        print(f"  BETTING SUMMARY — {target_date.date()}")
        print(f"{'═'*68}")
        print(f"  Positive-EV bets   : {len(bet_summary)}")
        print(f"  Total half-Kelly   : ${total_half:,.2f}  "
              f"({total_half/args.bankroll:.1%} of bankroll)")
        print(f"  Total full-Kelly   : ${total_full:,.2f}  "
              f"({total_full/args.bankroll:.1%} of bankroll)")
        print(f"\n  {'Bet':<30} {'Est.P':>7} {'Odds':>6} {'EV%':>7}"
              f" {'½Kelly%':>8} {'Stake':>10}")
        print(f"  {'─'*74}")
        for s in sorted(bet_summary, key=lambda x: x["ev"], reverse=True):
            print(f"  {s['matchup']:<30} {s['prob']:>7.1%} {s['odds']:>6.2f}"
                  f" {s['ev']:>+7.1%} {s['half_kelly']/args.bankroll:>8.1%}"
                  f" ${s['half_kelly']:>9.2f}")
    elif args.predict:
        print(f"\n  No positive-EV bets found for {target_date.date()}.")


if __name__ == "__main__":
    main()
