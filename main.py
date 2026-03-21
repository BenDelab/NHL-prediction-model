"""
NHL Home Game Outcome Predictor — end-to-end pipeline.

Usage:
    python main.py
    python main.py --team VEG --game 5
"""

import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pipeline import data_loader, features, noise_reduction, calibration
from models import train, evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NHL Home Game Predictor")
    p.add_argument("--team", type=str, default=None, help="Team abbreviation (e.g. VEG)")
    p.add_argument("--game", type=int, default=None, help="Game number in season")
    return p


def main():
    args = build_parser().parse_args()

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("Loading data...")
    df_all = data_loader.load_all_seasons()
    df_train_raw = df_all[df_all["season"].isin(data_loader.TRAIN_SEASONS)].copy()
    df_test_raw = df_all[df_all["season"] == data_loader.TEST_SEASON].copy()

    if df_test_raw.empty:
        print("WARNING: 2024-25 season data not found. Evaluation will be skipped.")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("Engineering features...")
    df_all_fe = features.build_features(df_all)
    df_train_fe = df_all_fe[df_all_fe["season"].isin(data_loader.TRAIN_SEASONS)].copy()
    df_test_fe = df_all_fe[df_all_fe["season"] == data_loader.TEST_SEASON].copy()

    # ── 3. Noise reduction ───────────────────────────────────────────────────
    print("Applying noise reduction...")
    df_train_fe = noise_reduction.apply(df_train_fe)
    if not df_test_fe.empty:
        df_test_fe = noise_reduction.apply(df_test_fe)

    # ── 4. Prepare training data ─────────────────────────────────────────────
    feat_cols = features.get_feature_cols(df_train_fe)
    # Use the largest window's features for training
    train_feat_cols = evaluate.get_feature_cols_for_window(20, df_train_fe)
    train_feat_cols = [c for c in train_feat_cols if c in df_train_fe.columns]

    y_train = df_train_fe["result"].eq("W").astype(int)
    X_train = df_train_fe[train_feat_cols].copy()

    valid_idx = X_train.notna().all(axis=1)
    X_train = X_train[valid_idx]
    y_train = y_train[valid_idx]

    print(f"Training on {len(X_train)} games across {data_loader.TRAIN_SEASONS}...")

    # ── 5. Train models ──────────────────────────────────────────────────────
    print("\nTraining models...")
    models = train.train_all(X_train, y_train)

    # ── 6. Calibrate (use last 20% of training data as calibration set) ──────
    print("\nCalibrating models...")
    cal_split = int(len(X_train) * 0.8)
    X_cal = X_train.iloc[cal_split:]
    y_cal = y_train.iloc[cal_split:]
    X_fit = X_train.iloc[:cal_split]
    y_fit = y_train.iloc[:cal_split]

    calibrated_models = {}
    for name, model in models.items():
        try:
            cal_model = calibration.calibrate(model, X_cal, y_cal, method="sigmoid")
            calibrated_models[name] = cal_model
        except Exception as e:
            print(f"  [{name}] calibration failed ({e}), using raw model.")
            calibrated_models[name] = model

    # ── 7. Sliding window evaluation ─────────────────────────────────────────
    if not df_test_fe.empty:
        print("\nRunning sliding window evaluation on 2024-25 season...")
        results = evaluate.sliding_window_evaluation(calibrated_models, df_test_fe, windows=[20])

        if not results.empty:
            print("\n── Accuracy by Model & Window ──")
            pivot = evaluate.best_config_table(results)
            print(pivot.to_string())

            best_row = results.loc[results["accuracy"].idxmax()]
            best_model_name = best_row["model"]
            best_window = int(best_row["window"])
            print(
                f"\nBest config: {best_model_name} | N={best_window} | "
                f"Accuracy={best_row['accuracy']:.4f} | ROC-AUC={best_row['roc_auc']:.4f}"
            )
        else:
            print("No evaluation results (likely missing test features).")
    else:
        results = pd.DataFrame()
        best_model_name = "XGBoost"
        best_window = 10

    # ── 8. Single-game prediction ────────────────────────────────────────────
    if args.team and args.game is not None and not df_test_fe.empty:
        team_games = df_test_fe[
            (df_test_fe["team"] == args.team) & (df_test_fe["game_num"] == args.game)
        ]
        if team_games.empty:
            print(f"\nNo game found for team={args.team}, game={args.game} in 2024-25.")
        else:
            game_row = team_games.iloc[0]
            model = calibrated_models.get(best_model_name, list(calibrated_models.values())[0])
            feat_cols_w = evaluate.get_feature_cols_for_window(best_window, df_test_fe)
            feat_cols_w = [c for c in feat_cols_w if c in df_test_fe.columns]

            result = evaluate.predict_game(model, game_row, feat_cols_w)
            actual = "W" if game_row["result"] == "W" else "L"
            correct = "✓" if (result["prediction"] == "WIN") == (actual == "W") else "✗"

            print(f"\nPREDICTION — Game {args.game}: {args.team} vs {game_row['opponent']} ({game_row['date'].date()})")
            print(f"  Model: {best_model_name} | Window: N={best_window}")
            print(f"  Prediction: {args.team} {result['prediction']} (home) | Confidence: {result['confidence']*100:.1f}%")
            print(f"  Actual Result: {actual} {correct}")


if __name__ == "__main__":
    main()
