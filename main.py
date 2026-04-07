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


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Combine recency (season) weights with early-season game-number ramp.

    Season weights: older seasons count less so recent form drives training.
    Early-season ramp: first ~15 games of a season are volatile/incomplete —
    ramp from 0 → 1 over games 1-15 so they don't anchor the model.
    """
    season_w = {"2021-22": 1.0, "2022-23": 1.2, "2023-24": 1.5, "2024-25": 2.0}
    sw = df["season"].map(season_w).fillna(1.0).values
    game_nums = df["game_num"].fillna(15).clip(lower=0).values
    early = np.minimum(game_nums / 15.0, 1.0)
    return (sw * early).astype(float)


def _log_feature_importances(all_models: dict, df_train_fe: pd.DataFrame,
                              n_top: int = 15) -> None:
    """Print top and bottom N features averaged across XGB and LGBM over all windows."""
    importance_sum: dict = {}
    count = 0
    for w, models in all_models.items():
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe)
                     if c in df_train_fe.columns]
        for name, model in models.items():
            if name not in ("XGBoost", "LightGBM"):
                continue
            pipe = getattr(model, "_model", model)  # unwrap ThresholdedModel if needed
            estimator = getattr(pipe, "named_steps", {}).get("model", None)
            if estimator is None or not hasattr(estimator, "feature_importances_"):
                continue
            imps = estimator.feature_importances_
            for feat, imp in zip(feat_cols, imps[:len(feat_cols)]):
                importance_sum[feat] = importance_sum.get(feat, 0.0) + float(imp)
            count += 1

    if count == 0 or not importance_sum:
        return

    avg = {k: v / count for k, v in importance_sum.items()}
    ranked = sorted(avg.items(), key=lambda x: -x[1])
    print(f"\n  Feature importances (avg across {count} XGB+LGBM models):")
    print(f"  Top {n_top}:")
    for feat, imp in ranked[:n_top]:
        print(f"    {feat:<52} {imp:.4f}")
    print(f"  Bottom {n_top}:")
    for feat, imp in ranked[-n_top:]:
        print(f"    {feat:<52} {imp:.4f}")


def _walk_forward_validation(df_train_fe: pd.DataFrame,
                              y_train: pd.Series) -> None:
    """
    Train on 2021-22 → 2023-24, test on 2024-25 using a fast LR pipeline.
    Gives a realistic out-of-time accuracy estimate independent of live-season data.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    train_seasons = {"2021-22", "2022-23", "2023-24"}
    train_mask = df_train_fe["season"].isin(train_seasons)
    test_mask  = df_train_fe["season"] == "2024-25"

    if train_mask.sum() < 100 or test_mask.sum() < 20:
        print("  [walk-forward] insufficient data, skipping.")
        return

    print("\nWalk-forward validation  (train 2021-24 → test 2024-25, LR-only):")
    for w in evaluate.WINDOWS:
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe)
                     if c in df_train_fe.columns]
        X_wf_tr = df_train_fe.loc[train_mask, feat_cols]
        y_wf_tr = y_train[train_mask]
        X_wf_te = df_train_fe.loc[test_mask, feat_cols]
        y_wf_te = y_train[test_mask]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ])
        pipe.fit(X_wf_tr, y_wf_tr)
        acc = accuracy_score(y_wf_te, pipe.predict(X_wf_te))
        auc = roc_auc_score(y_wf_te, pipe.predict_proba(X_wf_te)[:, 1])
        print(f"  N={w:2d}  acc={acc:.3f}  AUC={auc:.3f}  ({len(y_wf_te)} test games)")


def _secondary_prob(models: dict, X: pd.DataFrame) -> float:
    """Average predict_proba across a secondary ensemble dict."""
    probs = []
    for m in models.values():
        try:
            probs.append(float(m.predict_proba(X)[0][1]))
        except Exception:
            pass
    return float(np.mean(probs)) if probs else 0.5


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


def _ensemble_probs(calibrated: dict, game_row: pd.Series, df_ref: pd.DataFrame,
                    weights: dict | None = None) -> dict:
    """
    Run every (window, model) combination on one game row.
    Returns {label: probability}.  Weighted mean is computed at the call site.
    weights: optional {label: weight} from live accuracy softmax (Fix F).
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


def _explain_bet(s: dict) -> str:
    """
    Generate a 1-2 sentence plain-English rationale for a top-pick bet.
    Uses context stored in the bet_summary entry.
    """
    bet_type = s.get("bet_type", "ml")
    prob     = s["prob"]
    edg      = s["edg"]
    ev       = s["ev"]

    if bet_type == "ml":
        n_con   = s.get("n_consensus", 0)
        n_tot   = s.get("total_models", 16)
        wavgs   = s.get("window_avgs", {})
        side    = s.get("side", "home")
        team    = s.get("home_team" if side == "home" else "away_team", "the team")

        # Consensus sentence
        pct_agree = n_con / n_tot if n_tot else 0
        if pct_agree >= 0.80:
            consensus_str = f"{n_con}/{n_tot} models back {team}"
        elif pct_agree >= 0.65:
            consensus_str = f"A solid {n_con}/{n_tot} models favour {team}"
        else:
            consensus_str = f"{n_con}/{n_tot} models lean toward {team}"

        # Window consistency sentence
        if wavgs:
            vals = list(wavgs.values())
            spread_w = max(vals) - min(vals)
            short_key = min(wavgs, key=lambda k: int(k.split("=")[1].split(" ")[0]))
            long_key  = max(wavgs, key=lambda k: int(k.split("=")[1].split(" ")[0]))
            short_p, long_p = wavgs[short_key], wavgs[long_key]
            if spread_w < 0.05:
                form_str = "All rolling windows are tightly aligned — both recent and long-term form agree."
            elif short_p > long_p + 0.04:
                form_str = (f"Recent form ({short_key}: {short_p:.0%}) outpaces the long-term baseline "
                            f"({long_key}: {long_p:.0%}), suggesting positive momentum.")
            elif long_p > short_p + 0.04:
                form_str = (f"Long-term indicators ({long_key}: {long_p:.0%}) are stronger than recent "
                            f"form ({short_key}: {short_p:.0%}) — sustained quality over the season.")
            else:
                form_str = "Short and long-term windows are broadly consistent across the board."
        else:
            form_str = f"Edge of {edg:+.1%} after de-juicing the book's margin."

        return f"{consensus_str} at {prob:.0%}. {form_str}"

    elif bet_type in ("spread_fav", "spread_dog"):
        fav  = s.get("fav_team", "the favourite")
        dog  = s.get("dog_team", "the underdog")
        line = "-1.5" if bet_type == "spread_fav" else "+1.5"
        team = fav if bet_type == "spread_fav" else dog
        opp  = dog if bet_type == "spread_fav" else fav
        return (f"Model prices {team} {line} at {prob:.0%} — a {edg:+.1%} true edge "
                f"after removing the vig. Goal differential and recent dominant-win "
                f"rate both support a multi-goal margin over {opp}.")

    elif bet_type in ("goals_over", "goals_under"):
        pred  = s.get("pred_goals", 0.0)
        line  = s.get("goals_line", 0.0)
        diff  = abs(pred - line)
        direction = "over" if bet_type == "goals_over" else "under"
        side_word = "above" if bet_type == "goals_over" else "below"
        driver = ("offensive output and high-scoring rates for both sides"
                  if bet_type == "goals_over"
                  else "defensive profiles and lower expected outputs from both teams")
        return (f"Predicted total of {pred:.1f} goals sits {diff:.1f} {side_word} "
                f"the book's line of {line}. {driver.capitalize()} support the {direction}.")

    return f"Model estimates {prob:.0%} probability with {edg:+.1%} de-juiced edge and {ev:+.1%} EV."


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

    # ── 4. Walk-forward validation (out-of-time signal check) ────────────────
    _walk_forward_validation(df_train_fe, y_train)

    # ── 5. Compute sample weights (recency + early-season ramp) ──────────────
    sample_weights = compute_sample_weights(df_train_fe)
    print(f"\nSample weights: min={sample_weights.min():.2f}  "
          f"max={sample_weights.max():.2f}  mean={sample_weights.mean():.2f}")

    # ── 6. Train / calibrate split ────────────────────────────────────────────
    # True hold-out calibration: models must NEVER see calibration data during
    # training, otherwise the calibrators adjust probabilities based on
    # memorised patterns and cannot fix systematic overconfidence on new games.
    #
    #   0  …  74%   →  model training
    #   75 … 89%   →  calibrator fitting  (cal_fit  — unseen by base models)
    #   90 … 100%  →  method selection + threshold tuning  (cal_eval)
    n_train      = len(df_train_fe)
    model_cutoff = int(n_train * 0.75)
    cal_start    = model_cutoff
    eval_start   = int(n_train * 0.90)

    df_model_fe  = df_train_fe.iloc[:model_cutoff]
    y_model      = y_train.iloc[:model_cutoff]
    sw_model     = sample_weights[:model_cutoff]

    print(f"\nTraining on {len(df_model_fe)} games (first 75% of training data)")
    nan_rows = df_model_fe[[c for c in df_model_fe.columns if c.startswith("rolling_")]].isna().any(axis=1).sum()
    print(f"  ({nan_rows} rows with NaN — handled by imputer in each pipeline)\n")
    print(f"  Cal-fit : rows {cal_start}–{eval_start-1}  ({eval_start - cal_start} games, unseen by models)")
    print(f"  Cal-eval: rows {eval_start}–{n_train-1}   ({n_train - eval_start} games)\n")

    all_models: dict = {}
    for w in evaluate.WINDOWS:
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_model_fe) if c in df_model_fe.columns]
        print(f"── Window N={w} | {len(feat_cols)} features ──────────────────────────")
        all_models[w] = train.train_all(df_model_fe[feat_cols], y_model,
                                        sample_weight=sw_model)

    # ── Log feature importances (XGB + LGBM, averaged across windows) ────────
    _log_feature_importances(all_models, df_model_fe)

    # ── 7. Calibrate per window (true hold-out: 75–90% unseen by base models) ──
    print("\nCalibrating models (true hold-out data)...")
    calibrated: dict = {}

    for w, models in all_models.items():
        feat_cols  = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]
        X_cal_fit  = df_train_fe[feat_cols].iloc[cal_start:eval_start]
        y_cal_fit  = y_train.iloc[cal_start:eval_start]
        X_cal_eval = df_train_fe[feat_cols].iloc[eval_start:]
        y_cal_eval = y_train.iloc[eval_start:]
        print(f"  N={w}  (fit on {len(X_cal_fit)} games, eval on {len(X_cal_eval)} games):")
        calibrated[w] = {
            name: calibration.calibrate(m, X_cal_fit, y_cal_fit, X_cal_eval, y_cal_eval)
            for name, m in models.items()
        }

    # ── 8. Threshold optimisation (held-out eval set) ────────────────────────
    print("\nOptimising decision thresholds (F-0.5, held-out eval set)...")
    for w, models in calibrated.items():
        feat_cols  = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]
        X_cal_eval = df_train_fe[feat_cols].iloc[eval_start:]
        y_cal_eval = y_train.iloc[eval_start:]
        for name, model in list(models.items()):
            t = evaluate.find_optimal_threshold(model, X_cal_eval, y_cal_eval, beta=0.5)
            calibrated[w][name] = evaluate.ThresholdedModel(model, t)
            print(f"  N={w} {name}: threshold={t:.3f}")

    # ── 9. Live validation + build ensemble weights (Fix F) ──────────────────
    df_live = df_predict_fe[df_predict_fe["result"].notna()].copy()
    model_weights: dict | None = None

    if not df_live.empty:
        print(f"\nLive validation on {len(df_live)} played 2025-26 games...")
        results = evaluate.sliding_window_evaluation(calibrated, df_live)
        evaluate.print_results(results)

        # Weight each (window, model) by its live accuracy using softmax.
        # Temperature=8 keeps weights meaningful without over-concentrating on one model.
        if not results.empty:
            acc_map = {
                f"{row['model']}  N={int(row['window'])}": row["accuracy"]
                for _, row in results.iterrows()
            }
            accs   = np.array(list(acc_map.values()))
            exp_a  = np.exp(accs * 8.0)
            weights = exp_a / exp_a.sum()
            model_weights = dict(zip(acc_map.keys(), weights))
            print("\n  Ensemble weights (by live accuracy):")
            for k, w in sorted(model_weights.items(), key=lambda x: -x[1]):
                print(f"    {k:<40} {w:.3f}  (acc={acc_map[k]:.3f})")
    else:
        results = pd.DataFrame()
        print("\nNo played 2025-26 games found for live validation.")

    # ── 10. Predict upcoming games ───────────────────────────────────────────
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

    ot_models      = secondary.train_ot_model(X_sec_played, y_ot)
    reg_win_models = secondary.train_reg_win_model(X_sec_played, y_reg_win)
    print(f"  OT base rate (train): {y_ot.mean():.1%}  |  "
          f"Reg-win base rate: {y_reg_win.mean():.1%}  |  "
          f"ensemble: {len(ot_models)} models each")

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
        # Ensemble probabilities across ALL windows and models (Fix F: weighted)
        all_probs = _ensemble_probs(calibrated, game, df_all_fe, weights=model_weights)
        if not all_probs:
            print(f"\n  {game['team']} vs {game['opponent']}: features unavailable, skipping.")
            continue

        if model_weights:
            probs_arr   = np.array([all_probs[k] for k in all_probs])
            wts_arr     = np.array([model_weights.get(k, 1.0) for k in all_probs])
            wts_arr    /= wts_arr.sum()
            ensemble_p  = float(np.dot(probs_arr, wts_arr))
        else:
            ensemble_p  = float(np.mean(list(all_probs.values())))
        n_win        = sum(1 for p in all_probs.values() if p >= 0.5)
        consensus    = f"{n_win}/{len(all_probs)} models predict WIN"
        prediction   = "WIN" if ensemble_p >= 0.5 else "LOSS"
        conf_flag    = "▲ HIGH" if ensemble_p > 0.63 else ("▶ MED" if ensemble_p > 0.57 else "▼ LOW")

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
        X_game    = game[sec_feat_cols].to_frame().T
        p_ot      = _secondary_prob(ot_models, X_game)
        p_reg_win = _secondary_prob(reg_win_models, X_game)

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

        # ── Spread (puck line) — favourite gets -1.5, underdog gets +1.5 ────────
        # Book favourite = lower decimal odds; fall back to model probability.
        if home_odds is not None and away_odds is not None:
            home_is_fav = home_odds <= away_odds
        elif home_odds is not None:
            home_is_fav = home_odds < 2.0
        else:
            home_is_fav = ensemble_p >= 0.5

        fav_team  = game["team"]      if home_is_fav else game["opponent"]
        dog_team  = game["opponent"]  if home_is_fav else game["team"]
        p_fav_m15 = p_home_spread     if home_is_fav else p_away_spread
        p_dog_p15 = 1.0 - p_fav_m15  # +1.5 is complement of opponent's -1.5

        fav_spread_odds = _parse_odds(
            f"  [Spread]    {fav_team} -1.5 decimal odds           (Enter to skip): ")
        dog_spread_odds = _parse_odds(
            f"  [Spread]    {dog_team} +1.5 decimal odds           (Enter to skip): ")

        # ── Total goals line + over/under odds ────────────────────────────────
        goals_line      = _parse_line(
            f"  [Goals O/U] Book's total-goals line  (e.g. 5.5, Enter to skip): ")
        goals_over_odds = goals_under_odds = None
        if goals_line is not None:
            goals_over_odds  = _parse_odds(f"    OVER  {goals_line} decimal odds: ")
            goals_under_odds = _parse_odds(f"    UNDER {goals_line} decimal odds: ")

        all_odds_skipped = all(x is None for x in [
            home_odds, away_odds, fav_spread_odds, dog_spread_odds,
            goals_over_odds, goals_under_odds,
        ])
        if all_odds_skipped:
            print("  [skipped]")
            continue

        # Fix C: de-juice implied prob when both ML odds are available
        dj_home_imp = (betting.dejuiced_implied_prob(home_odds, away_odds)
                       if home_odds and away_odds else None)
        dj_away_imp = (betting.dejuiced_implied_prob(away_odds, home_odds)
                       if home_odds and away_odds else None)

        MIN_EDGE = 0.07  # minimum de-juiced edge to flag a bet

        def _print_bet(label: str, p: float, odds: float,
                       other_odds: float | None = None, context: dict | None = None):
            """
            Print EV/Kelly for one bet side.
            context: extra fields stored in bet_summary for explanation generation.
            """
            imp        = betting.implied_prob(odds)
            ev         = betting.expected_value(p, odds)
            kf         = betting.kelly_fraction(p, odds)
            stake_full = kf * args.bankroll
            stake_half = kf * 0.5 * args.bankroll
            # Confidence reflects how likely THIS bet is to hit, not just model conviction.
            # p < 0.5 should never be HIGH — you're not confident the bet wins.
            conf = "▲ HIGH" if p > 0.63 else ("▶ MED" if p > 0.57 else "▼ LOW")

            if other_odds is not None:
                edg      = betting.edge_dejuiced(p, odds, other_odds)
                edge_lbl = f"{edg:+.1%} (de-juiced)"
            else:
                edg      = betting.edge(p, odds)
                edge_lbl = f"{edg:+.1%}"

            is_value = (ev > 0) and (edg >= MIN_EDGE) and (conf != "▼ LOW")
            bet_flag = "✓ BET" if is_value else "✗ SKIP"

            print(f"  ── {label} ──────────────────────────────────────")
            print(f"  Model est. prob      : {p:.1%}   {conf}")
            print(f"  Decimal odds         : {odds:.2f}")
            print(f"  Book implied prob    : {imp:.1%}")
            print(f"  Your edge            : {edge_lbl}")
            print(f"  Expected value       : {ev:+.1%}   {bet_flag}")
            if is_value:
                print(f"  Full Kelly stake     : {kf:.1%}   (${stake_full:,.2f})")
                print(f"  Half Kelly stake     : {kf*0.5:.1%}   (${stake_half:,.2f})"
                      f"  ← recommended")
                entry = {
                    "game_key":   f"{game['team']} vs {game['opponent']}",
                    "matchup":    label,
                    "prob":       p,
                    "odds":       odds,
                    "ev":         ev,
                    "edg":        edg,
                    "kf":         kf,          # raw Kelly fraction (before portfolio scaling)
                    "full_kelly": stake_full,
                    "half_kelly": stake_half,
                    "conf":       conf,
                }
                if context:
                    entry.update(context)
                bet_summary.append(entry)
            print()

        print()

        # Moneyline
        p_home = ensemble_p
        p_away = 1.0 - ensemble_p
        ml_ctx = {
            "bet_type":      "ml",
            "n_consensus":   n_win,
            "total_models":  len(all_probs),
            "window_avgs":   window_avgs,
            "home_team":     game["team"],
            "away_team":     game["opponent"],
        }
        if home_odds is not None:
            _print_bet(f"{game['team']} ML (home)", p_home, home_odds,
                       other_odds=away_odds,
                       context={**ml_ctx, "side": "home"})
        if away_odds is not None:
            _print_bet(f"{game['opponent']} ML (away)", p_away, away_odds,
                       other_odds=home_odds,
                       context={**ml_ctx, "side": "away"})

        # Spread
        spread_ctx_fav = {"bet_type": "spread_fav", "fav_team": fav_team, "dog_team": dog_team}
        spread_ctx_dog = {"bet_type": "spread_dog", "fav_team": fav_team, "dog_team": dog_team}
        if fav_spread_odds is not None:
            _print_bet(f"{fav_team} -1.5 (spread)", p_fav_m15, fav_spread_odds,
                       other_odds=dog_spread_odds, context=spread_ctx_fav)
        if dog_spread_odds is not None:
            _print_bet(f"{dog_team} +1.5 (spread)", p_dog_p15, dog_spread_odds,
                       other_odds=fav_spread_odds, context=spread_ctx_dog)

        # Total goals
        if goals_line is not None:
            p_over_goals  = totals.prob_over_goals(pred_total_goals, goals_line)
            p_under_goals = totals.prob_under(p_over_goals)
            goals_ctx = {"pred_goals": pred_total_goals, "goals_line": goals_line}
            if goals_over_odds is not None:
                _print_bet(f"Goals OVER  {goals_line}", p_over_goals, goals_over_odds,
                           other_odds=goals_under_odds,
                           context={**goals_ctx, "bet_type": "goals_over"})
            if goals_under_odds is not None:
                _print_bet(f"Goals UNDER {goals_line}", p_under_goals, goals_under_odds,
                           other_odds=goals_over_odds,
                           context={**goals_ctx, "bet_type": "goals_under"})

    # ── Betting summary ───────────────────────────────────────────────────────
    if bet_summary:
        # ── Portfolio-level stake scaling ─────────────────────────────────────
        # Standard Kelly is derived assuming one bet at a time. When placing N
        # simultaneous bets the individual fractions must be scaled so total
        # exposure stays within a safe ceiling (25 % of bankroll).
        # Scale factor = min(1, MAX_EXPOSURE / raw_total_half_frac).
        MAX_EXPOSURE = 0.25  # never risk more than 25 % of bankroll in one session
        raw_half_frac = sum(s["kf"] * 0.5 for s in bet_summary)
        scale = min(1.0, MAX_EXPOSURE / raw_half_frac) if raw_half_frac > 0 else 1.0
        scaled = scale < 1.0

        for s in bet_summary:
            s["adj_half_kelly"] = s["kf"] * 0.5 * scale * args.bankroll
            s["adj_full_kelly"] = s["kf"]       * scale * args.bankroll

        total_half = sum(s["adj_half_kelly"] for s in bet_summary)
        total_full = sum(s["adj_full_kelly"] for s in bet_summary)

        print(f"\n{'═'*68}")
        print(f"  BETTING SUMMARY — {target_date.date()}")
        print(f"{'═'*68}")
        print(f"  Positive-EV bets   : {len(bet_summary)}")
        if scaled:
            print(f"  Portfolio scale    : {scale:.2f}×  "
                  f"(raw total would be {raw_half_frac/1:.1%} → capped at {MAX_EXPOSURE:.0%})")
        print(f"  Total half-Kelly   : ${total_half:,.2f}  "
              f"({total_half/args.bankroll:.1%} of bankroll)")
        print(f"  Total full-Kelly   : ${total_full:,.2f}  "
              f"({total_full/args.bankroll:.1%} of bankroll)")
        ranked = sorted(bet_summary, key=lambda x: x["ev"], reverse=True)
        print(f"\n  {'#':<3} {'Game':<24} {'Bet':<24} {'Conf':<8}"
              f" {'Est.P':>7} {'Odds':>6} {'EV%':>7} {'½Kelly%':>8} {'Stake':>10}")
        print(f"  {'─'*98}")
        for rank, s in enumerate(ranked, 1):
            print(f"  {rank:<3} {s['game_key']:<24} {s['matchup']:<24} {s['conf']:<8}"
                  f" {s['prob']:>7.1%} {s['odds']:>6.2f} {s['ev']:>+7.1%}"
                  f" {s['adj_half_kelly']/args.bankroll:>8.1%} ${s['adj_half_kelly']:>9.2f}")

        # ── Top picks with reasoning ──────────────────────────────────────────
        # Composite score weights all three signals equally.
        # score = EV + de-juiced edge + excess probability above 0.5
        # All three in the same unit (fraction), so no normalisation needed.
        def _top_score(s):
            return s["ev"] + s["edg"] + (s["prob"] - 0.5)

        top_picks = sorted(
            [s for s in bet_summary
             if s["conf"] == "▲ HIGH"
             and s["ev"]  >= 0.10
             and s["edg"] >= 0.10],
            key=_top_score,
            reverse=True,
        )

        if top_picks:
            print(f"\n{'═'*68}")
            print(f"  ★  TOP PICKS — highest conviction bets")
            print(f"{'═'*68}")
            for i, s in enumerate(top_picks, 1):
                print(f"\n  {i}. {s['matchup']}  [{s['game_key']}]")
                print(f"     {s['prob']:.0%} | EV {s['ev']:+.1%} | Edge {s['edg']:+.1%} | {s['conf']}")
                print(f"     Recommended stake: ${s['adj_half_kelly']:,.2f}  "
                      f"({s['adj_half_kelly']/args.bankroll:.1%} of bankroll)")
                print(f"     {_explain_bet(s)}")
    elif args.predict:
        print(f"\n  No positive-EV bets found for {target_date.date()}.")


if __name__ == "__main__":
    main()
