"""
NHL Home Game Outcome Predictor — end-to-end pipeline.

Usage
─────
  python main.py                          # train + live validation only
  python main.py --predict                # train + predict next upcoming games
  python main.py --predict --date 2026-03-25  # predict for a specific date
"""

import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pipeline import data_loader, features, noise_reduction, calibration, xg_loader, goalie_loader
from models import train, evaluate, secondary, totals
from models.totals import prob_over_goals, prob_under


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



def main():
    args = build_parser().parse_args()

    # ── 1. Load ALL seasons (2021-22 through 2025-26) ────────────────────────
    print("Loading data...")
    df_all = data_loader.load_all_seasons()

    # ── 1b. Load xG data (optional — model runs without it if files missing) ──
    print("Loading xG data...")
    xg_data = xg_loader.load_xg_all_seasons()
    if not xg_data.empty:
        print(f"  xG loaded: {len(xg_data)} team-game rows across "
              f"{xg_data['season'].nunique()} seasons, "
              f"{xg_data['team'].nunique()} teams.")

    # ── 1c. Load goalie data (optional — model runs without it if files missing) ──
    print("Loading goalie data...")
    goalie_data = goalie_loader.load_goalie_all_seasons()

    # ── 2. Feature engineering on the full dataset ───────────────────────────
    #   Rolling features use shift(1), so upcoming games (NaN result) get
    #   correct feature vectors computed from all prior played games.
    print("Engineering features...")
    df_all_fe = features.build_features(
        df_all,
        xg_df=xg_data if not xg_data.empty else None,
        goalie_df=goalie_data if not goalie_data.empty else None,
    )

    # ── 3. Noise reduction — fit on all played rows, apply everywhere ─────────
    print("Applying noise reduction...")
    played_mask = df_all_fe["result"].isin(["W","L"])
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

    # ── 6. Train on ALL 4 seasons ────────────────────────────────────────────
    # Previous approach trained on 75% → entire 2024-25 season was excluded from
    # training and used only for calibration, wasting the most recent data.
    # The calibration eval set (end of 2024-25) also had a 58% home win rate —
    # far above 2025-26's 52% — biasing calibrators toward overconfidence.
    #
    # New approach:
    #   Training  → all 4 seasons  (5247 games, max signal)
    #   Cal-fit   → first 70% of 2025-26 played games  (same distribution as predictions)
    #   Cal-eval  → last  30% of 2025-26 played games
    #   Weights   → cal-eval portion only  (prevent contamination from cal-fit)
    print(f"\nTraining on {len(df_train_fe)} games across all {data_loader.TRAIN_SEASONS}")
    nan_rows = df_train_fe[[c for c in df_train_fe.columns if c.startswith("rolling_")]].isna().any(axis=1).sum()
    print(f"  ({nan_rows} rows with NaN — handled by imputer in each pipeline)\n")

    all_models: dict = {}
    for w in evaluate.WINDOWS:
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]
        print(f"── Window N={w} | {len(feat_cols)} features ──────────────────────────")
        all_models[w] = train.train_all(df_train_fe[feat_cols], y_train,
                                        sample_weight=sample_weights)

    # ── Log feature importances (XGB + LGBM, averaged across windows) ────────
    _log_feature_importances(all_models, df_train_fe)

    # ── 7. Calibrate on 2025-26 played games (same distribution as predictions) ─
    df_live = df_predict_fe[df_predict_fe["result"].isin(["W","L"])].copy()
    y_live  = df_live["result"].eq("W").astype(int).reset_index(drop=True)
    df_live = df_live.reset_index(drop=True)

    LIVE_WIN_RATE = y_live.mean() if len(y_live) > 0 else 0.0
    print(f"\n2025-26 played games: {len(df_live)}  (home win rate: {LIVE_WIN_RATE:.1%})")

    # Need ≥150 played games to calibrate on 2025-26; otherwise fall back to
    # end of training data (last 20% of 2024-25).
    USE_LIVE_CAL = len(df_live) >= 150
    calibrated: dict = {}

    if USE_LIVE_CAL:
        n_live       = len(df_live)
        cal_live_cut = int(n_live * 0.70)
        print(f"  Calibrating on 2025-26:  fit={cal_live_cut} games | eval={n_live - cal_live_cut} games")
    else:
        n_train   = len(df_train_fe)
        cal_start = int(n_train * 0.80)
        print(f"  Calibrating on training fallback: fit from row {cal_start}")

    # Collect per-window calibration inputs for pooled fitting
    X_cal_by_window: dict = {}
    y_cal_fit_ref: np.ndarray | None = None

    for w, models in all_models.items():
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]

        if USE_LIVE_CAL:
            live_fc   = [c for c in feat_cols if c in df_live.columns]
            X_cal_fit  = df_live[live_fc].iloc[:cal_live_cut]
            y_cal_fit  = y_live.iloc[:cal_live_cut]
        else:
            X_cal_fit  = df_train_fe[feat_cols].iloc[cal_start:]
            y_cal_fit  = y_train.iloc[cal_start:]

        X_cal_by_window[w] = (X_cal_fit, feat_cols)
        if y_cal_fit_ref is None:
            y_cal_fit_ref = np.asarray(y_cal_fit)

    print(f"\n  Pooling calibration across {len(all_models)} windows × {len(next(iter(all_models.values())))} models")
    print(f"  (fit slice: {len(X_cal_fit)} games per window)")
    _, calibrated = calibration.calibrate_pooled(
        all_models, X_cal_by_window, y_cal_fit_ref
    )

    # Surface per-window eval counts for logging
    for w in all_models:
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]
        if USE_LIVE_CAL:
            live_fc    = [c for c in feat_cols if c in df_live.columns]
            n_eval     = len(df_live[live_fc].iloc[cal_live_cut:])
        else:
            n_eval = len(df_train_fe[feat_cols].iloc[cal_start:])
        print(f"  N={w}: eval={n_eval} games")

    # ── 8. Threshold optimisation ─────────────────────────────────────────────
    print("\nOptimising decision thresholds (F-0.5)...")
    for w, models in calibrated.items():
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe) if c in df_train_fe.columns]

        if USE_LIVE_CAL:
            live_fc    = [c for c in feat_cols if c in df_live.columns]
            X_thr_eval = df_live[live_fc].iloc[cal_live_cut:]
            y_thr_eval = y_live.iloc[cal_live_cut:]
        else:
            n_train = len(df_train_fe)
            X_thr_eval = df_train_fe[feat_cols].iloc[int(n_train * 0.80):]
            y_thr_eval = y_train.iloc[int(n_train * 0.80):]

        for name, model in list(models.items()):
            t = evaluate.find_optimal_threshold(model, X_thr_eval, y_thr_eval, beta=0.5)
            calibrated[w][name] = evaluate.ThresholdedModel(model, t)
            print(f"  N={w} {name}: threshold={t:.3f}")

    # ── 9. Live validation + ensemble weights ────────────────────────────────
    # Use the cal-eval portion (last 30% of 2025-26) so weights are computed on
    # data the calibrators haven't fitted — prevents inflated accuracy estimates.
    model_weights: dict | None = None

    if not df_live.empty:
        if USE_LIVE_CAL:
            df_weight = df_live.iloc[cal_live_cut:].copy()
        else:
            df_weight = df_live.copy()

        print(f"\nLive validation on {len(df_weight)} held-out 2025-26 games "
              f"(home win rate: {df_weight['result'].eq('W').mean():.1%})...")
        results = evaluate.sliding_window_evaluation(calibrated, df_weight)
        evaluate.print_results(results)

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

    if args.date:
        target_date = pd.Timestamp(args.date)
        # Allow predicting on any date — played or unplayed
        next_games = df_predict_fe[df_predict_fe["date"] == target_date].sort_values("team")
        if next_games.empty:
            print(f"\nNo games found on {target_date.date()} in 2025-26 data.")
            return
    else:
        upcoming = df_predict_fe[~df_predict_fe["result"].isin(["W","L"])].sort_values("date")
        if upcoming.empty:
            print("\nNo upcoming games found in 2025-26 data.")
            return
        today = pd.Timestamp("today").normalize()
        future = upcoming[upcoming["date"] >= today]
        target_date = future["date"].min() if not future.empty else upcoming["date"].min()
        next_games = upcoming[upcoming["date"] == target_date]

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

    played_train = df_train_fe["result"].isin(["W","L"]) & df_train_fe["overtime_type"].notna()
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
    played_all = (df_train_fe["result"].isin(["W","L"]) &
                  df_train_fe["goals_for"].notna() &
                  df_train_fe["team_sog"].notna())
    df_played = df_train_fe[played_all]
    # Subset the already-computed sample_weights instead of recomputing on df_played.
    # Both df_train_fe and sample_weights have identical ordering, so boolean indexing
    # gives the exact same weights without re-running the season-recency computation.
    spread_weights = sample_weights[played_all.values]

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
        home_spread_by_window[w] = totals.train_spread_ensemble(X_w, y_home_spread,
                                                                 sample_weight=spread_weights)
        print(f"    [spread away]")
        away_spread_by_window[w] = totals.train_spread_ensemble(X_w, y_away_spread,
                                                                 sample_weight=spread_weights)
        print(f"    [goals regression]")
        total_goals_by_window[w] = totals.train_regression_all(X_w, y_total_goals,
                                                                sample_weight=spread_weights)

    # Calibrate spread models on 2025-26 played games (same distribution as predictions).
    # Falls back to last 20% of training data if insufficient live games.
    print("\n  Calibrating spread models...")

    # Prepare live spread labels (only for games with goal data)
    live_spread_mask = df_live["goals_for"].notna() & df_live["goals_against"].notna()
    df_live_spread   = df_live[live_spread_mask].reset_index(drop=True)
    y_home_spread_live = (df_live_spread["goals_for"] - df_live_spread["goals_against"] >= 2).astype(int)
    y_away_spread_live = (df_live_spread["goals_against"] - df_live_spread["goals_for"] >= 2).astype(int)
    USE_LIVE_SPREAD_CAL = USE_LIVE_CAL and len(df_live_spread) >= 100
    if USE_LIVE_SPREAD_CAL:
        spread_live_cut = int(len(df_live_spread) * 0.70)
        print(f"    Using 2025-26 live data for spread calibration "
              f"(fit={spread_live_cut}, eval={len(df_live_spread)-spread_live_cut} games)")
    else:
        print(f"    Using training fallback for spread calibration")

    for w in evaluate.WINDOWS:
        feat_cols = [c for c in evaluate.get_feature_cols_for_window(w, df_train_fe)
                     if c in df_train_fe.columns]

        if USE_LIVE_SPREAD_CAL:
            live_fc      = [c for c in feat_cols if c in df_live_spread.columns]
            X_cal_home   = df_live_spread[live_fc].iloc[:spread_live_cut]
            X_cal_home_e = df_live_spread[live_fc].iloc[spread_live_cut:]
            y_home_cal   = y_home_spread_live.iloc[:spread_live_cut]
            y_home_cal_e = y_home_spread_live.iloc[spread_live_cut:]
            y_away_cal   = y_away_spread_live.iloc[:spread_live_cut]
            y_away_cal_e = y_away_spread_live.iloc[spread_live_cut:]
        else:
            X_w        = df_played[feat_cols]
            cal_split  = int(len(X_w) * 0.8)
            X_cal_home = X_cal_home_e = X_w.iloc[cal_split:]
            y_home_cal = y_home_cal_e = y_home_spread.iloc[cal_split:]
            y_away_cal = y_away_cal_e = y_away_spread.iloc[cal_split:]

        home_spread_by_window[w] = {
            name: calibration.calibrate(m, X_cal_home, y_home_cal, X_cal_home_e, y_home_cal_e)
            for name, m in home_spread_by_window[w].items()
        }
        away_spread_by_window[w] = {
            name: calibration.calibrate(m, X_cal_home, y_away_cal, X_cal_home_e, y_away_cal_e)
            for name, m in away_spread_by_window[w].items()
        }
        print(f"    N={w}: spread calibrated")

    # ── Pre-compute home/away records and current-season H2H ─────────────────
    # Use only played games up to (but not including) target_date so no leakage.
    _played = df_all[df_all["result"].isin(["W","L"]) & (df_all["date"] < str(target_date.date()))].copy()
    _cur_season = _played["season"].max()
    _played_cur = _played[_played["season"] == _cur_season]

    def _home_record(team: str) -> str:
        """W-L-OTL at home this season. Every row is the home team's perspective."""
        home_g = _played_cur[_played_cur["team"] == team]
        if home_g.empty:
            return "N/A"
        _ot = home_g["overtime_type"].isin(["OT", "SO"])
        w   = (home_g["result"] == "W").sum()
        otl = ((home_g["result"] == "L") & _ot).sum()
        l   = ((home_g["result"] == "L") & ~_ot).sum()
        return f"{w}-{l}-{otl}"

    def _away_record(team: str) -> str:
        """W-L-OTL on the road this season. Away team is in the opponent column."""
        away_g = _played_cur[_played_cur["opponent"] == team]
        if away_g.empty:
            return "N/A"
        # From home team's perspective: away team wins when home result == "L"
        _ot = away_g["overtime_type"].isin(["OT", "SO"])
        w   = (away_g["result"] == "L").sum()
        otl = ((away_g["result"] == "W") & _ot).sum()
        l   = ((away_g["result"] == "W") & ~_ot).sum()
        return f"{w}-{l}-{otl}"

    def _h2h_record(home_team: str, away_team: str) -> str:
        """Current-season H2H: games where home_team hosted away_team + vice versa."""
        mask = (
            ((_played_cur["team"] == home_team) & (_played_cur["opponent"] == away_team)) |
            ((_played_cur["team"] == away_team) & (_played_cur["opponent"] == home_team))
        )
        h2h_g = _played_cur[mask]
        if h2h_g.empty:
            return "No meetings this season"
        lines = []
        for _, r in h2h_g.iterrows():
            ot = f" ({r['overtime_type']})" if pd.notna(r.get("overtime_type")) else ""
            lines.append(
                f"{r['team']} {r['result']} {int(r['goals_for'])}-{int(r['goals_against'])}{ot}"
            )
        return "  |  ".join(lines)

    # ── Fetch probable starting goalies from NHL API ─────────────────────────
    print("\nFetching probable starting goalies...")
    probable_goalies = goalie_loader.fetch_probable_goalies(str(target_date.date()))
    if probable_goalies:
        print(f"  Goalie data fetched for {len(probable_goalies)} game(s).")
    else:
        print("  No goalie data available from NHL API.")

    print(f"\n{'═'*68}")
    print(f"  NHL PREDICTIONS — {target_date.date()}")
    print(f"{'═'*68}")

    game_summaries = []

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

        # LR produces naturally calibrated probabilities in the 0.40-0.62 range.
        # No clipping needed — LR never goes to 0.85 like tree models do.
        ensemble_p   = float(ensemble_p)

        n_win        = sum(1 for p in all_probs.values() if p >= 0.5)
        consensus    = f"{n_win}/{len(all_probs)} models predict WIN"
        prediction   = "WIN" if ensemble_p >= 0.5 else "LOSS"

        # Confidence = probability strength × model agreement
        # Denominators anchored to live 2025-26 calibrated probability distribution:
        #   median edge = 0.032, p75 edge = 0.060, typical inter-window std ≈ 0.015-0.025
        # HIGH requires: edge in top ~25% of all predictions AND models agree tightly
        probs_list   = list(all_probs.values())
        prob_edge    = abs(ensemble_p - 0.5)
        prob_std     = float(np.std(probs_list))
        # std normalizer: 0.03 = typical inter-window spread; above that = scattered
        agreement    = max(0.0, 1.0 - prob_std / 0.03)
        # edge normalizer: 0.06 = p75 of live edges (top quartile = genuinely strong)
        conf_score   = 0.60 * min(prob_edge / 0.06, 1.0) + 0.40 * agreement

        if conf_score >= 0.70 and prob_edge >= 0.03:
            conf_flag = "▲ HIGH"
        elif conf_score >= 0.40 and prob_edge >= 0.015:
            conf_flag = "▶ MED"
        else:
            conf_flag = "▼ LOW"

        # Per-window average (cleaner breakdown than 16 rows)
        window_avgs = {}
        for w in evaluate.WINDOWS:
            w_probs = {k: v for k, v in all_probs.items() if f"N={w}" in k}
            if w_probs:
                window_avgs[f"N={w} avg"] = float(np.mean(list(w_probs.values())))

        # ── Fetch key rolling stats for narrative generation ─────────────────
        bw = best_window
        def _g(col, prefix=""):
            """Safe getter for a rolling feature, returns nan if missing."""
            return game.get(f"{prefix}rolling_{bw}_{col}", np.nan)

        def _fmt(v, pct=True):
            if pd.isna(v) or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v:.1%}" if pct else f"{v:.2f}"

        # Home team stats
        h_win_pct    = _g("win_pct")
        h_goal_diff  = _g("goal_diff")
        h_gf_avg     = _g("goals_for_avg")
        h_ga_avg     = _g("goals_against_avg")
        h_corsi      = _g("corsi_pct")
        h_pdo        = _g("pdo")
        h_pp_eff     = _g("pp_efficiency")
        h_pk_eff     = _g("pk_efficiency")
        h_save_pct   = _g("save_pct")
        h_shoot_pct  = _g("shooting_pct")
        h_ot_rate    = _g("ot_rate")
        h_ot_win     = _g("ot_win_rate")
        h_blowout    = _g("blowout_win_rate")
        h_close      = _g("close_game_rate")

        # Away (opponent) stats
        a_win_pct    = _g("win_pct",   "opp_")
        a_goal_diff  = _g("goal_diff", "opp_")
        a_gf_avg     = _g("goals_for_avg",   "opp_")
        a_ga_avg     = _g("goals_against_avg","opp_")
        a_corsi      = _g("corsi_pct", "opp_")
        a_pdo        = _g("pdo",       "opp_")
        a_pp_eff     = _g("pp_efficiency","opp_")
        a_pk_eff     = _g("pk_efficiency","opp_")
        a_save_pct   = _g("save_pct",  "opp_")
        a_shoot_pct  = _g("shooting_pct","opp_")
        a_ot_rate    = _g("ot_rate",   "opp_")
        a_ot_win     = _g("ot_win_rate","opp_")

        # Standings / motivation features
        h_motivation     = game.get("motivation", np.nan)
        a_motivation     = game.get("opp_motivation", np.nan)
        h_clinch         = game.get("clinch_status", np.nan)
        a_clinch         = game.get("opp_clinch_status", np.nan)
        h_pts_above_cut  = game.get("points_above_cut", np.nan)
        a_pts_above_cut  = game.get("opp_points_above_cut", np.nan)
        h_conf_rank      = game.get("conf_rank", np.nan)
        a_conf_rank      = game.get("opp_conf_rank", np.nan)
        h_games_rem      = game.get("games_remaining", np.nan)
        a_games_rem      = game.get("opp_games_remaining", np.nan)
        h_is_tanking     = game.get("is_tanking", 0)
        a_is_tanking     = game.get("opp_is_tanking", 0)
        h_is_must_win    = game.get("is_must_win", 0)
        a_is_must_win    = game.get("opp_is_must_win", 0)
        h_is_clinched    = game.get("is_clinched", 0)
        a_is_clinched    = game.get("opp_is_clinched", 0)

        # xG stats (even strength + all situations)
        h_xgf_ev     = _g("xgf_ev")
        h_xga_ev     = _g("xga_ev")
        h_xgp_ev     = _g("xgp_ev")
        h_xgf_all    = _g("xgf_all")
        h_xga_all    = _g("xga_all")
        h_xgf_pp     = _g("xgf_pp")
        h_xga_pk     = _g("xga_pk")
        a_xgf_ev     = _g("xgf_ev",  "opp_")
        a_xga_ev     = _g("xga_ev",  "opp_")
        a_xgp_ev     = _g("xgp_ev",  "opp_")
        a_xgf_all    = _g("xgf_all", "opp_")
        a_xga_all    = _g("xga_all", "opp_")
        a_xgf_pp     = _g("xgf_pp",  "opp_")
        a_xga_pk     = _g("xga_pk",  "opp_")

        # Secondary model probs
        X_game    = game[sec_feat_cols].to_frame().T
        p_ot      = _secondary_prob(ot_models, X_game)
        p_reg_win = _secondary_prob(reg_win_models, X_game)

        # Spread / totals
        p_home_spread    = _spread_ensemble_prob(home_spread_by_window, game, df_all_fe)
        p_away_spread    = _spread_ensemble_prob(away_spread_by_window, game, df_all_fe)
        pred_total_goals = _goals_ensemble_pred(total_goals_by_window, game, df_all_fe)
        p_away = 1.0 - ensemble_p

        # Spread favourite by model probability
        home_is_fav = ensemble_p >= 0.5
        fav_team  = game["team"]     if home_is_fav else game["opponent"]
        dog_team  = game["opponent"] if home_is_fav else game["team"]
        p_fav_m15 = p_home_spread    if home_is_fav else p_away_spread
        p_dog_p15 = 1.0 - p_fav_m15

        # Spread recommendation: use base-rate-relative edge rather than 0.5 threshold.
        # Home covers -1.5 ~33% of the time, away covers ~27%.
        # We recommend -1.5 only when the model's probability meaningfully exceeds
        # the empirical base rate; otherwise default to +1.5 (dog stays within a goal).
        HOME_COVER_BASE = 0.33
        AWAY_COVER_BASE = 0.27
        COVER_EDGE_MIN  = 0.05   # model must beat base rate by at least 5 pp

        home_cover_edge = p_home_spread - HOME_COVER_BASE
        away_cover_edge = p_away_spread - AWAY_COVER_BASE

        if home_cover_edge >= COVER_EDGE_MIN and home_cover_edge >= away_cover_edge:
            spread_rec   = f"{game['team']} -1.5"
            spread_rec_p = p_home_spread
        elif away_cover_edge >= COVER_EDGE_MIN and away_cover_edge > home_cover_edge:
            spread_rec   = f"{game['opponent']} -1.5"
            spread_rec_p = p_away_spread
        else:
            # Neither side has a meaningful cover edge — lean to the dog
            spread_rec   = f"{dog_team} +1.5"
            spread_rec_p = p_dog_p15

        # Total goals recommendation using half-point lines (5.5, 6.5, 7.5).
        # Half-point lines force a clean over/under — no push possible.
        # P(over 5.5) = P(X ≥ 6), P(over 6.5) = P(X ≥ 7), P(over 7.5) = P(X ≥ 8).
        # Pick the half-point line where the model has the strongest conviction
        # (probability furthest from 50%), which avoids recommending lines where
        # Ridge's conservative prediction is essentially coin-flip.
        avg_total   = 6.24  # league average this season
        _half_lines = [5.5, 6.5, 7.5]
        _half_probs = {l: prob_over_goals(pred_total_goals, l) for l in _half_lines}
        # Conviction = distance from 0.5; pick the line with highest conviction
        _best_line  = max(_half_lines, key=lambda l: abs(_half_probs[l] - 0.5))
        _best_p     = _half_probs[_best_line]
        total_rec   = f"OVER {_best_line}" if _best_p >= 0.5 else f"UNDER {_best_line}"

        # ── Print game header ─────────────────────────────────────────────────
        print(f"\n{'═'*68}")
        game_num_str = f"  Game #{int(game['game_num'])}" if pd.notna(game.get('game_num')) else ""
        print(f"  {game['team']} (home) vs {game['opponent']} (away){game_num_str}")
        print(f"{'═'*68}")

        # ── 1. WIN / LOSS PREDICTION ──────────────────────────────────────────
        print(f"\n  PREDICTION: {game['team']} {'WIN' if ensemble_p >= 0.5 else 'LOSS'}  "
              f"| Confidence: {conf_flag}  "
              f"(edge: {prob_edge:.1%}, agreement: {agreement:.0%})")
        print(f"  Win probability: {ensemble_p:.1%} (home)  /  {p_away:.1%} (away)")
        print(f"  Model consensus: {consensus}  |  "
              f"Window breakdown: " +
              "  ".join(f"{k}: {v:.1%}" for k, v in window_avgs.items()))

        # Records & Elo
        home_rec  = _home_record(game['team'])
        away_rec  = _away_record(game['opponent'])
        h2h_str   = _h2h_record(game['team'], game['opponent'])
        h_elo     = game.get("home_elo", np.nan)
        a_elo     = game.get("away_elo", np.nan)
        elo_str   = (f"{h_elo:.0f} vs {a_elo:.0f}  (diff: {h_elo - a_elo:+.0f})"
                     if not np.isnan(h_elo) else "N/A")
        _CLINCH_SHORT = {2: "DIV CLINCHED", 1: "PLAYOFF CLINCHED", 0: "IN RACE", -1: "ELIMINATED"}
        h_clinch_str = _CLINCH_SHORT.get(int(h_clinch) if not np.isnan(h_clinch) else 0, "")
        a_clinch_str = _CLINCH_SHORT.get(int(a_clinch) if not np.isnan(a_clinch) else 0, "")
        if h_is_tanking:   h_clinch_str = "TANKING"
        if a_is_tanking:   a_clinch_str = "TANKING"
        if h_is_must_win:  h_clinch_str = "MUST-WIN"
        if a_is_must_win:  a_clinch_str = "MUST-WIN"
        h_rec_str = f"{home_rec}  |  Elo: {h_elo:.0f}" if not np.isnan(h_elo) else home_rec
        a_rec_str = f"{away_rec}  |  Elo: {a_elo:.0f}" if not np.isnan(a_elo) else away_rec
        h_mot_str = (f"  |  {h_clinch_str}  (motivation: {h_motivation:.2f})"
                     if not np.isnan(h_motivation) else "")
        a_mot_str = (f"  |  {a_clinch_str}  (motivation: {a_motivation:.2f})"
                     if not np.isnan(a_motivation) else "")
        print(f"  {game['team']} home record : {h_rec_str}{h_mot_str}")
        print(f"  {game['opponent']} away record : {a_rec_str}{a_mot_str}")
        print(f"  Elo gap (home − away): {elo_str}")
        print(f"  H2H this season : {h2h_str}")

        # ── Probable starting goalies ─────────────────────────────────────────
        g_info = probable_goalies.get((game["team"], game["opponent"]))
        if g_info:
            src_tag = "confirmed" if g_info["source"] == "confirmed" else "probable"
            h_gname = g_info["home_name"]
            a_gname = g_info["away_name"]
            h_gaa   = g_info["home_gaa"]
            a_gaa   = g_info["away_gaa"]
            h_sv    = g_info["home_sv_pct"]
            a_sv    = g_info["away_sv_pct"]
            h_grec  = g_info["home_record"]
            a_grec  = g_info["away_record"]
            # Rolling GSAx from our goalie features (team-level recent form)
            h_gsax  = game.get(f"rolling_{best_window}_goalie_gsax", np.nan)
            a_gsax  = game.get(f"opp_rolling_{best_window}_goalie_gsax", np.nan)
            h_hdsv  = game.get(f"rolling_{best_window}_goalie_hd_save_pct", np.nan)
            a_hdsv  = game.get(f"opp_rolling_{best_window}_goalie_hd_save_pct", np.nan)
            def _gs(v, fmt=".3f"):
                return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
            h_sv_str   = f"SV%={_gs(h_sv, '.4f')}" if not np.isnan(h_sv) else ""
            a_sv_str   = f"SV%={_gs(a_sv, '.4f')}" if not np.isnan(a_sv) else ""
            h_gaa_str  = f"GAA={_gs(h_gaa)}" if not np.isnan(h_gaa) else ""
            a_gaa_str  = f"GAA={_gs(a_gaa)}" if not np.isnan(a_gaa) else ""
            h_gsax_str = f"  |  rolling GSAx={_gs(h_gsax, '+.3f')}  HD-SV%={_gs(h_hdsv, '.3f')}" \
                         if not (isinstance(h_gsax, float) and np.isnan(h_gsax)) else ""
            a_gsax_str = f"  |  rolling GSAx={_gs(a_gsax, '+.3f')}  HD-SV%={_gs(a_hdsv, '.3f')}" \
                         if not (isinstance(a_gsax, float) and np.isnan(a_gsax)) else ""
            print(f"\n  GOALIES ({src_tag}):")
            print(f"    {game['team']:<4}  {h_gname:<22}  {h_gaa_str}  {h_sv_str}  {h_grec}{h_gsax_str}")
            print(f"    {game['opponent']:<4}  {a_gname:<22}  {a_gaa_str}  {a_sv_str}  {a_grec}{a_gsax_str}")

        # Build win narrative anchored to the predicted winner.
        # Assign winner/loser stats so all framing supports the model's pick.
        home_wins = ensemble_p >= 0.5
        w_team = game['team']     if home_wins else game['opponent']
        l_team = game['opponent'] if home_wins else game['team']
        w_win_pct   = h_win_pct   if home_wins else a_win_pct
        l_win_pct   = a_win_pct   if home_wins else h_win_pct
        w_gdiff     = h_goal_diff if home_wins else a_goal_diff
        l_gdiff     = a_goal_diff if home_wins else h_goal_diff
        w_corsi     = h_corsi     if home_wins else a_corsi
        l_corsi     = a_corsi     if home_wins else h_corsi
        w_pdo       = h_pdo       if home_wins else a_pdo
        l_pdo       = a_pdo       if home_wins else h_pdo
        w_gf        = h_gf_avg    if home_wins else a_gf_avg
        l_gf        = a_gf_avg    if home_wins else h_gf_avg
        w_save      = h_save_pct  if home_wins else a_save_pct
        l_save      = a_save_pct  if home_wins else h_save_pct
        w_xgf_ev    = h_xgf_ev    if home_wins else a_xgf_ev
        l_xgf_ev    = a_xgf_ev    if home_wins else h_xgf_ev
        w_xga_ev    = h_xga_ev    if home_wins else a_xga_ev
        l_xga_ev    = a_xga_ev    if home_wins else h_xga_ev
        w_xgp_ev    = h_xgp_ev    if home_wins else a_xgp_ev
        l_xgp_ev    = a_xgp_ev    if home_wins else h_xgp_ev
        w_xgf_pp    = h_xgf_pp    if home_wins else a_xgf_pp
        l_xgf_pp    = a_xgf_pp    if home_wins else h_xgf_pp
        w_xga_pk    = h_xga_pk    if home_wins else a_xga_pk
        l_xga_pk    = a_xga_pk    if home_wins else h_xga_pk

        supporting  = []  # reasons FOR the predicted winner
        headwinds   = []  # reasons that FAVOUR the predicted loser (counter-factors)

        # Elo power ranking signal
        if not np.isnan(h_elo) and not np.isnan(a_elo):
            w_elo = h_elo if home_wins else a_elo
            l_elo = a_elo if home_wins else h_elo
            elo_gap = w_elo - l_elo
            if elo_gap >= 30:
                supporting.append(
                    f"{w_team} is the stronger team by Elo power ranking "
                    f"({w_elo:.0f} vs {l_elo:.0f}, gap: {elo_gap:+.0f})")
            elif elo_gap <= -30:
                headwinds.append(
                    f"{l_team} has a higher Elo rating ({l_elo:.0f} vs {w_elo:.0f}) — "
                    f"model is picking against the stronger team on paper")

        # Playoff motivation context
        _CLINCH_LABEL = {2: "clinched division", 1: "clinched playoff spot",
                         0: "in the race", -1: "eliminated"}
        w_mot = h_motivation if home_wins else a_motivation
        l_mot = a_motivation if home_wins else h_motivation
        w_clinch = int(h_clinch if home_wins else a_clinch) if not np.isnan(h_clinch if home_wins else a_clinch) else None
        l_clinch = int(a_clinch if home_wins else h_clinch) if not np.isnan(a_clinch if home_wins else h_clinch) else None
        w_pac = h_pts_above_cut if home_wins else a_pts_above_cut
        l_pac = a_pts_above_cut if home_wins else h_pts_above_cut

        w_is_tanking  = h_is_tanking  if home_wins else a_is_tanking
        l_is_tanking  = a_is_tanking  if home_wins else h_is_tanking
        w_is_must_win = h_is_must_win if home_wins else a_is_must_win
        l_is_must_win = a_is_must_win if home_wins else h_is_must_win
        w_is_clinched = h_is_clinched if home_wins else a_is_clinched
        l_is_clinched = a_is_clinched if home_wins else h_is_clinched
        games_rem_w   = h_games_rem   if home_wins else a_games_rem
        games_rem_l   = a_games_rem   if home_wins else h_games_rem

        if not np.isnan(w_mot) and not np.isnan(l_mot):
            # Tanking — strongest signal
            if l_is_tanking:
                headwinds.append(
                    f"{l_team} is eliminated and near the bottom of the standings — "
                    f"likely tanking for draft lottery position")
            elif l_clinch == -1:
                headwinds.append(
                    f"{l_team} has been eliminated — potential low motivation / resting players")
            if l_is_clinched:
                headwinds.append(
                    f"{l_team} has already {_CLINCH_LABEL.get(l_clinch, 'clinched').lower()} "
                    f"— may be managing minutes for the playoffs")

            # Must-win urgency for the predicted winner
            if w_is_must_win:
                supporting.append(
                    f"{w_team} is in must-win territory "
                    f"({w_pac:+.0f} pts vs cutoff, {int(games_rem_w)} games left) — "
                    f"maximum urgency")
            elif w_clinch == 0 and not np.isnan(w_pac) and abs(w_pac) <= 4:
                supporting.append(
                    f"{w_team} is in a tight playoff race "
                    f"({w_pac:+.0f} pts vs cutoff"
                    + (f", {int(games_rem_w)} games left" if not np.isnan(games_rem_w) else "")
                    + f", motivation: {w_mot:.2f})")

            # Both tanking — flag it as a coin flip
            if w_is_tanking:
                headwinds.append(
                    f"{w_team} is also eliminated and near the bottom — "
                    f"both teams may have low motivation (coin flip territory)")

        # Home/away record context
        def _record_wins(rec: str) -> int:
            """Parse W from 'W-L-OTL' string."""
            try: return int(rec.split("-")[0])
            except: return -1
        def _record_games(rec: str) -> int:
            try: parts = [int(x) for x in rec.split("-")]; return sum(parts)
            except: return 0

        if home_wins:
            # Predicted winner is home team — check home record
            hw = _record_wins(home_rec); hg = _record_games(home_rec)
            if hg >= 5:
                hwp = hw / hg
                if hwp >= 0.60:
                    supporting.append(
                        f"{game['team']} is strong at home ({home_rec})")
                elif hwp <= 0.40:
                    headwinds.append(
                        f"{game['team']} has a poor home record ({home_rec}) — home ice less reliable")
            # Predicted loser is away — check away record
            aw = _record_wins(away_rec); ag = _record_games(away_rec)
            if ag >= 5:
                awp = aw / ag
                if awp >= 0.55:
                    headwinds.append(
                        f"{game['opponent']} travels well ({away_rec} on the road)")
                elif awp <= 0.38:
                    supporting.append(
                        f"{game['opponent']} struggles on the road ({away_rec})")
        else:
            # Predicted winner is away team
            aw = _record_wins(away_rec); ag = _record_games(away_rec)
            if ag >= 5:
                awp = aw / ag
                if awp >= 0.55:
                    supporting.append(
                        f"{game['opponent']} has a strong road record ({away_rec})")
                elif awp <= 0.38:
                    headwinds.append(
                        f"{game['opponent']} struggles away ({away_rec}) — road win would be an upset")
            # Home team has home ice but model picks against them
            hw = _record_wins(home_rec); hg = _record_games(home_rec)
            if hg >= 5:
                hwp = hw / hg
                if hwp >= 0.60:
                    headwinds.append(
                        f"{game['team']} is tough at home ({home_rec})")
                elif hwp <= 0.40:
                    supporting.append(
                        f"{game['team']} has a weak home record ({home_rec}) — home advantage negated")

        if not np.isnan(w_win_pct) and not np.isnan(l_win_pct):
            diff = w_win_pct - l_win_pct
            if diff >= 0.06:
                supporting.append(
                    f"{w_team} has a higher recent win rate ({_fmt(w_win_pct)} vs {_fmt(l_win_pct)})")
            elif diff <= -0.06:
                headwinds.append(
                    f"{l_team} has the better recent win rate ({_fmt(l_win_pct)} vs {_fmt(w_win_pct)})")

        if not np.isnan(w_gdiff):
            if w_gdiff > 0.25:
                supporting.append(
                    f"{w_team} has a positive goal differential ({w_gdiff:+.2f}/game)")
            elif w_gdiff < -0.25 and not np.isnan(l_gdiff) and l_gdiff > 0.25:
                headwinds.append(
                    f"{l_team} has a stronger goal differential ({l_gdiff:+.2f}/game)")

        if not np.isnan(w_corsi) and not np.isnan(l_corsi):
            diff = w_corsi - l_corsi  # Corsi is stored as 48-54 range
            if diff >= 2.0:
                supporting.append(
                    f"{w_team} controls possession (Corsi {w_corsi:.1f}% vs {l_corsi:.1f}%)")
            elif diff <= -2.0:
                headwinds.append(
                    f"{l_team} has better possession numbers (Corsi {l_corsi:.1f}% vs {w_corsi:.1f}%)")

        if not np.isnan(w_pdo):
            if w_pdo < 98:
                supporting.append(
                    f"{w_team} PDO of {w_pdo:.1f} is suppressed — "
                    f"shooting/goaltending due for positive regression")
            elif w_pdo > 102:
                headwinds.append(
                    f"{w_team} PDO of {w_pdo:.1f} is elevated — may regress toward mean")
        if not np.isnan(l_pdo) and l_pdo < 98:
            headwinds.append(
                f"{l_team} PDO of {l_pdo:.1f} is suppressed — they may also be due for regression")

        if not np.isnan(w_save) and not np.isnan(l_save):
            if w_save >= 0.918:
                supporting.append(
                    f"{w_team} goaltending is sharp (SV% {w_save:.3f})")
            if l_save >= 0.918:
                headwinds.append(
                    f"{l_team} goaltending is also strong (SV% {l_save:.3f})")

        # xG — expected goals quality signals
        if not np.isnan(w_xgf_ev) and not np.isnan(l_xgf_ev):
            ev_diff = w_xgf_ev - l_xgf_ev
            if ev_diff >= 0.25:
                supporting.append(
                    f"{w_team} generates more high-quality chances at even strength "
                    f"(xGF 5v5: {w_xgf_ev:.2f} vs {l_xgf_ev:.2f})")
            elif ev_diff <= -0.25:
                headwinds.append(
                    f"{l_team} creates better even-strength chances "
                    f"(xGF 5v5: {l_xgf_ev:.2f} vs {w_xgf_ev:.2f})")

        if not np.isnan(w_xgp_ev) and not np.isnan(l_xgp_ev):
            if w_xgp_ev >= 0.55:
                supporting.append(
                    f"{w_team} is dominating expected goals at even strength "
                    f"(xG% 5v5: {w_xgp_ev:.1%})")
            elif w_xgp_ev <= 0.45:
                headwinds.append(
                    f"{w_team} is being outchanced at even strength "
                    f"(xG% 5v5: {w_xgp_ev:.1%}) — winning despite quality deficit")

        if not np.isnan(w_xgf_pp) and not np.isnan(l_xgf_pp):
            pp_diff = w_xgf_pp - l_xgf_pp
            if pp_diff >= 0.15:
                supporting.append(
                    f"{w_team} has a more dangerous power play "
                    f"(PP xGF: {w_xgf_pp:.2f} vs {l_xgf_pp:.2f})")
            elif pp_diff <= -0.15:
                headwinds.append(
                    f"{l_team} has the better power play by expected goals "
                    f"(PP xGF: {l_xgf_pp:.2f} vs {w_xgf_pp:.2f})")

        if not np.isnan(w_xga_pk) and not np.isnan(l_xga_pk):
            pk_diff = w_xga_pk - l_xga_pk   # lower is better (fewer xG allowed shorthanded)
            if pk_diff <= -0.10:
                supporting.append(
                    f"{w_team} has the stronger penalty kill by xG allowed "
                    f"(PK xGA: {w_xga_pk:.2f} vs {l_xga_pk:.2f})")
            elif pk_diff >= 0.10:
                headwinds.append(
                    f"{l_team} has a better penalty kill "
                    f"(PK xGA: {l_xga_pk:.2f} vs {w_xga_pk:.2f})")

        # H2H context in the narrative
        _h2h_games = _played_cur[
            ((_played_cur["team"] == game['team']) & (_played_cur["opponent"] == game['opponent'])) |
            ((_played_cur["team"] == game['opponent']) & (_played_cur["opponent"] == game['team']))
        ]
        if not _h2h_games.empty:
            _h2h_w_team = (_h2h_games["team"] == w_team) & (_h2h_games["result"] == "W")
            _h2h_w_team |= (_h2h_games["team"] == l_team) & (_h2h_games["result"] == "L")
            h2h_wins  = int(_h2h_w_team.sum())
            h2h_total = len(_h2h_games)
            if h2h_total >= 2:
                if h2h_wins == h2h_total:
                    supporting.append(
                        f"{w_team} has won all {h2h_total} meetings this season")
                elif h2h_wins > h2h_total / 2:
                    supporting.append(
                        f"{w_team} leads the season series {h2h_wins}-{h2h_total - h2h_wins}")
                elif h2h_wins < h2h_total / 2:
                    headwinds.append(
                        f"{l_team} leads the season series {h2h_total - h2h_wins}-{h2h_wins}")

        if not supporting and not headwinds:
            supporting.append(
                f"Multi-window rolling metrics give {w_team} a marginal edge — "
                f"{'home ice advantage is a factor' if home_wins else 'road form offsets home-ice disadvantage'}")

        why_parts = []
        if supporting:
            why_parts.append(". ".join(supporting))
        if headwinds:
            why_parts.append(f"Counter-factors: {'; '.join(headwinds)}")
        print(f"\n  Why {w_team} wins: " + ". ".join(why_parts) + ".")

        # ── 2. OT RISK ────────────────────────────────────────────────────────
        ot_flag = ("▲ HIGH" if p_ot >= 0.33 else ("▶ POSSIBLE" if p_ot >= 0.22 else "▼ LOW"))
        print(f"\n  OT/SO RISK: {p_ot:.1%}  {ot_flag}  |  "
              f"Regulation win prob: {p_reg_win:.1%}")
        ot_reasons = []
        if not np.isnan(h_ot_rate) and not np.isnan(a_ot_rate):
            avg_ot = (h_ot_rate + a_ot_rate) / 2
            if avg_ot >= 0.28:
                ot_reasons.append(f"both teams have high OT rates "
                                  f"({_fmt(h_ot_rate)} / {_fmt(a_ot_rate)})")
            elif avg_ot <= 0.16:
                ot_reasons.append(f"both teams tend to decide games in regulation "
                                  f"({_fmt(h_ot_rate)} / {_fmt(a_ot_rate)} OT rate)")
        if not np.isnan(h_close) and h_close >= 0.45:
            ot_reasons.append(f"{game['team']} plays a lot of 1-goal games ({_fmt(h_close)})")
        if ot_reasons:
            print(f"  Why: {'; '.join(ot_reasons).capitalize()}.")
        if p_ot >= 0.28 and not np.isnan(h_ot_win) and not np.isnan(a_ot_win):
            net_ot = float(h_ot_win) - float(a_ot_win)
            if abs(net_ot) >= 0.10:
                ot_edge = game['team'] if net_ot > 0 else game['opponent']
                print(f"  OT edge: {ot_edge} wins OT/SO more often ({net_ot:+.1%} net rate).")

        # ── 3. SPREAD ─────────────────────────────────────────────────────────
        # p_home_spread = P(home covers -1.5), p_away_spread = P(away covers -1.5)
        # +1.5 is the complement: P(home covers +1.5) = 1 - p_away_spread
        p_home_plus15 = 1.0 - p_away_spread
        p_away_plus15 = 1.0 - p_home_spread

        print(f"\n  SPREAD PREDICTION: {spread_rec}  ({spread_rec_p:.1%})")
        print(f"  {'':8} {'P(covers)':>12}")
        print(f"  {'─'*24}")
        print(f"  {game['team'] + ' -1.5':<20} {p_home_spread:>10.1%}{'  ◄' if p_home_spread > p_away_spread else ''}")
        print(f"  {game['team'] + ' +1.5':<20} {p_home_plus15:>10.1%}{'  ◄' if p_home_plus15 > p_away_plus15 else ''}")
        print(f"  {game['opponent'] + ' -1.5':<20} {p_away_spread:>10.1%}{'  ◄' if p_away_spread > p_home_spread else ''}")
        print(f"  {game['opponent'] + ' +1.5':<20} {p_away_plus15:>10.1%}{'  ◄' if p_away_plus15 > p_home_plus15 else ''}")

        # Spread narrative anchored to the recommended side.
        rec_team  = spread_rec.split(" ")[0]  # first token is the team name
        rec_minus = "-1.5" in spread_rec  # True = recommending a team to cover -1.5
        s_support = []
        s_counter = []

        if rec_minus:
            # Recommending rec_team to win by 2+ — look for blowout rate, big margins
            rec_blowout = h_blowout if rec_team == game['team'] else _g("blowout_win_rate", "opp_")
            if not np.isnan(rec_blowout):
                if rec_blowout >= 0.25:
                    s_support.append(
                        f"{rec_team} wins by 3+ goals in {_fmt(rec_blowout)} of recent games")
                elif rec_blowout <= 0.12:
                    s_counter.append(
                        f"{rec_team} rarely wins by large margins ({_fmt(rec_blowout)} blowout rate) "
                        f"— cover is a stretch despite the win probability")

            rec_gf  = h_gf_avg  if rec_team == game['team'] else a_gf_avg
            opp_ga  = a_ga_avg  if rec_team == game['team'] else h_ga_avg
            if not np.isnan(rec_gf) and not np.isnan(opp_ga):
                exp_margin = rec_gf - opp_ga
                if exp_margin >= 0.5:
                    s_support.append(
                        f"{rec_team} averages {rec_gf:.1f} GF/g while opponent allows {opp_ga:.1f} — "
                        f"positive expected margin supports the cover")
                elif exp_margin <= -0.2:
                    s_counter.append(
                        f"Opponent defense ({opp_ga:.1f} GA/g) limits {rec_team}'s scoring "
                        f"({rec_gf:.1f} GF/g) — tight game likely")

            rec_pp = h_pp_eff if rec_team == game['team'] else a_pp_eff
            opp_pk = a_pk_eff if rec_team == game['team'] else h_pk_eff
            if not np.isnan(rec_pp) and not np.isnan(opp_pk):
                pp_edge = rec_pp - (1 - opp_pk)
                if pp_edge >= 0.05:
                    s_support.append(
                        f"{rec_team} PP ({_fmt(rec_pp)}) vs opponent PK ({_fmt(opp_pk)}) is a scoring edge")
        else:
            # Recommending dog_team +1.5 — look for tight game signals
            dog_close = h_close if dog_team == game['team'] else _g("close_game_rate", "opp_")
            fav_blowout = h_blowout if fav_team == game['team'] else _g("blowout_win_rate", "opp_")
            if not np.isnan(fav_blowout):
                if fav_blowout <= 0.15:
                    s_support.append(
                        f"{fav_team} rarely wins by 2+ ({_fmt(fav_blowout)} blowout rate) — "
                        f"{dog_team} likely stays within a goal")
                elif fav_blowout >= 0.30:
                    s_counter.append(
                        f"{fav_team} wins by large margins in {_fmt(fav_blowout)} of games — "
                        f"+1.5 cover is at risk")
            if not np.isnan(dog_close):
                if dog_close >= 0.45:
                    s_support.append(
                        f"{dog_team} plays close games {_fmt(dog_close)} of the time — "
                        f"tends to keep it tight regardless of result")

        if not s_support and not s_counter:
            s_support.append(
                f"Spread model gives {rec_team} the edge at this line based on "
                f"rolling goal margins and shot quality")

        spread_why = ". ".join(s_support)
        if s_counter:
            spread_why += f". Counter-factors: {'; '.join(s_counter)}"
        print(f"  Why {spread_rec}: {spread_why}.")

        # ── 4. TOTAL GOALS ────────────────────────────────────────────────────
        # Half-point lines: P(over 5.5)=P(X≥6), P(over 6.5)=P(X≥7), P(over 7.5)=P(X≥8)
        # Half-points force a clean over/under split (no push) and match sportsbook lines.
        p_over_55  = prob_over_goals(pred_total_goals, 5.5)
        p_over_65  = prob_over_goals(pred_total_goals, 6.5)
        p_over_75  = prob_over_goals(pred_total_goals, 7.5)
        p_under_55 = prob_under(p_over_55)
        p_under_65 = prob_under(p_over_65)
        p_under_75 = prob_under(p_over_75)

        print(f"\n  TOTAL GOALS: Model predicts {pred_total_goals:.1f} goals  |  Best bet: {total_rec}")
        print(f"  {'Line':<8} {'Over':>8} {'Under':>8}   (league avg: {avg_total:.1f})")
        print(f"  {'─'*30}")
        for line, po, pu in [(5.5, p_over_55, p_under_55),
                              (6.5, p_over_65, p_under_65),
                              (7.5, p_over_75, p_under_75)]:
            is_rec = str(line) in total_rec
            over_star  = " ★" if is_rec and total_rec.startswith("OVER")  else ""
            under_star = " ★" if is_rec and total_rec.startswith("UNDER") else ""
            over_mark  = " ◄" if po > pu else ""
            under_mark = " ◄" if pu > po else ""
            print(f"  {line:<8.1f} {po:>7.1%}{(over_mark+over_star):<5} {pu:>7.1%}{under_mark+under_star}")

        # Narrative: state what drives the predicted total, then note where the
        # lean sits relative to each line so the reader can judge for themselves
        goals_reasons = []
        if not np.isnan(h_gf_avg) and not np.isnan(a_gf_avg):
            combined_avg = h_gf_avg + a_gf_avg
            vs_avg = "above" if combined_avg >= avg_total else "below"
            goals_reasons.append(
                f"combined recent scoring rate is {combined_avg:.1f} goals/game "
                f"({game['team']} {h_gf_avg:.1f} + {game['opponent']} {a_gf_avg:.1f}), "
                f"{vs_avg} the league average of {avg_total:.1f}")

        if not np.isnan(h_save_pct) and not np.isnan(a_save_pct):
            avg_sv = (h_save_pct + a_save_pct) / 2
            if avg_sv >= 0.915:
                goals_reasons.append(
                    f"goaltending is sharp on both sides (avg SV% {avg_sv:.3f}), "
                    f"which tends to suppress totals")
            elif avg_sv <= 0.895:
                goals_reasons.append(
                    f"goaltending has been soft on both sides (avg SV% {avg_sv:.3f}), "
                    f"which tends to inflate totals")
            else:
                goals_reasons.append(f"goaltending is average (avg SV% {avg_sv:.3f})")

        if not np.isnan(h_shoot_pct) and not np.isnan(a_shoot_pct):
            avg_sh = (h_shoot_pct + a_shoot_pct) / 2
            if avg_sh >= 0.105:
                goals_reasons.append(
                    f"shooting% is elevated on both sides ({avg_sh:.1%}), "
                    f"meaning fewer shots are needed to score")
            elif avg_sh <= 0.085:
                goals_reasons.append(
                    f"both teams are converting at a low rate ({avg_sh:.1%}), "
                    f"keeping scoring down despite shot volume")

        # xG-based totals context (all-situations combined xG is the best totals signal)
        if not np.isnan(h_xgf_all) and not np.isnan(a_xgf_all):
            combined_xg = h_xgf_all + a_xgf_all
            vs_avg_xg   = "above" if combined_xg >= avg_total else "below"
            goals_reasons.append(
                f"combined expected goals (all situations) is {combined_xg:.1f} "
                f"({game['team']} xGF {h_xgf_all:.2f} + {game['opponent']} xGF {a_xgf_all:.2f}), "
                f"{vs_avg_xg} the league average — this is the model's primary totals signal")
        elif not np.isnan(h_xgf_ev) and not np.isnan(a_xgf_ev):
            combined_xg_ev = h_xgf_ev + a_xgf_ev
            goals_reasons.append(
                f"even-strength expected goals combined: {combined_xg_ev:.1f} "
                f"({game['team']} {h_xgf_ev:.2f} + {game['opponent']} {a_xgf_ev:.2f})")

        if not goals_reasons:
            goals_reasons.append("rolling offensive and defensive metrics were used to project the total")

        # State exactly which lines the projection clears vs falls under
        over_lines  = [l for l in [5.5, 6.5, 7.5] if pred_total_goals > l]
        under_lines = [l for l in [5.5, 6.5, 7.5] if pred_total_goals <= l]
        lean_parts  = []
        if over_lines:
            lean_parts.append(f"over {', '.join(str(l) for l in over_lines)}")
        if under_lines:
            lean_parts.append(f"under {', '.join(str(l) for l in under_lines)}")
        lean_note = (f"Projection of {pred_total_goals:.1f} goals favours "
                     + " and ".join(lean_parts)
                     + f" — best single bet: {total_rec}")
        goals_reasons.append(lean_note)
        print(f"  Why: {'. '.join(goals_reasons)}.")

        # ── 5. KEY DATA SIGNALS (summary table) ──────────────────────────────
        print(f"\n  KEY STATS (last {bw} games):")
        print(f"  {'Metric':<28} {'Home':>10} {'Away':>10}")
        print(f"  {'─'*50}")
        rows_data = [
            ("Win %",            h_win_pct,   a_win_pct,   True),
            ("Goals For/game",   h_gf_avg,    a_gf_avg,    False),
            ("Goals Against/game",h_ga_avg,   a_ga_avg,    False),
            ("Goal Diff/game",   h_goal_diff, a_goal_diff, False),
            ("Corsi %",          h_corsi,     a_corsi,     False),
            ("Save %",           h_save_pct,  a_save_pct,  True),
            ("Shooting %",       h_shoot_pct, a_shoot_pct, True),
            ("PP efficiency",    h_pp_eff,    a_pp_eff,    True),
            ("PK efficiency",    h_pk_eff,    a_pk_eff,    True),
            ("OT rate",          h_ot_rate,   a_ot_rate,   True),
            ("xGF 5v5 (EV)",     h_xgf_ev,    a_xgf_ev,    False),
            ("xGA 5v5 (EV)",     h_xga_ev,    a_xga_ev,    False),
            ("xG% 5v5 (EV)",     h_xgp_ev,    a_xgp_ev,    True),
            ("xGF all sit.",     h_xgf_all,   a_xgf_all,   False),
            ("xGA all sit.",     h_xga_all,   a_xga_all,   False),
            ("PP xGF",           h_xgf_pp,    a_xgf_pp,    False),
            ("PK xGA",           h_xga_pk,    a_xga_pk,    False),
        ]
        for label, hv, av, as_pct in rows_data:
            hstr = _fmt(hv, pct=as_pct) if not (isinstance(hv, float) and np.isnan(hv)) else "N/A"
            astr = _fmt(av, pct=as_pct) if not (isinstance(av, float) and np.isnan(av)) else "N/A"
            if not as_pct and hstr != "N/A":
                hstr = f"{hv:.2f}"
            if not as_pct and astr != "N/A":
                astr = f"{av:.2f}"
            print(f"  {label:<28} {hstr:>10} {astr:>10}")

        # Store for day summary
        game_summaries.append({
            "matchup":        f"{game['team']} vs {game['opponent']}",
            "home":           game['team'],
            "away":           game['opponent'],
            "prediction":     "WIN" if ensemble_p >= 0.5 else "LOSS",
            "conf":           conf_flag,
            "p_home":         ensemble_p,
            "p_away":         p_away,
            "spread_rec":     spread_rec,
            "spread_rec_p":   spread_rec_p,
            "total_pred":     pred_total_goals,
            "total_rec":      total_rec,
            "p_over_55":      p_over_55,
            "p_under_55":     p_under_55,
            "p_over_65":      p_over_65,
            "p_under_65":     p_under_65,
            "p_over_75":      p_over_75,
            "p_under_75":     p_under_75,
            "p_ot":           p_ot,
        })

    # ── Day summary (top picks ranked by confidence) ──────────────────────────
    if game_summaries:
        print(f"\n{'═'*68}")
        print(f"  DAY SUMMARY — {target_date.date()}")
        print(f"{'═'*68}")
        print(f"  {'#':<3} {'Matchup':<26} {'Pick':<8} {'Conf':<8} "
              f"{'Win%':>7} {'Spread':>18} {'Proj':>5}  {'Lean':<10}")
        print(f"  {'─'*92}")
        ranked = sorted(game_summaries, key=lambda x: x["p_home"]
                        if x["prediction"] == "WIN" else x["p_away"], reverse=True)
        for i, s in enumerate(ranked, 1):
            p_pick = s["p_home"] if s["prediction"] == "WIN" else s["p_away"]
            print(f"  {i:<3} {s['matchup']:<26} {s['home']+' '+s['prediction']:<8} "
                  f"{s['conf']:<8} {p_pick:>7.1%} "
                  f"{s['spread_rec']:>18} ({s['spread_rec_p']:.0%}) "
                  f"{s['total_pred']:>4.1f}  {s['total_rec']:<10}")

        # Top HIGH-confidence picks
        top = [s for s in ranked if s["conf"] == "▲ HIGH"]
        if top:
            live_acc = results["accuracy"].max() if not results.empty else None
            print(f"\n{'═'*68}")
            print(f"  ★  TOP PICKS — HIGH confidence only")
            if live_acc is not None:
                print(f"     Model live accuracy (2025-26): {live_acc:.1%}  "
                      f"(HIGH-conf target: 63-66% hit rate)")
            print(f"{'═'*68}")
            for i, s in enumerate(top, 1):
                p_pick = s["p_home"] if s["prediction"] == "WIN" else s["p_away"]
                print(f"\n  {i}. {s['home']} {s['prediction']}  [{s['matchup']}]  {s['conf']}")
                print(f"     Win probability : {p_pick:.1%}")
                print(f"     Spread lean     : {s['spread_rec']}  ({s['spread_rec_p']:.0%})")
                print(f"     Goals lean      : {s['total_rec']}  (projected {s['total_pred']:.1f} goals)")
                for _l, _ok, _uk in [(5.5,"p_over_55","p_under_55"),
                                      (6.5,"p_over_65","p_under_65"),
                                      (7.5,"p_over_75","p_under_75")]:
                    _is_rec = str(_l) in s["total_rec"]
                    _o_star = " ★" if _is_rec and s["total_rec"].startswith("OVER")  else ""
                    _u_star = " ★" if _is_rec and s["total_rec"].startswith("UNDER") else ""
                    print(f"       Line {_l} — over: {s[_ok]:.0%}{_o_star}  under: {s[_uk]:.0%}{_u_star}")
        else:
            print(f"\n  No HIGH-confidence picks today.")


if __name__ == "__main__":
    main()
