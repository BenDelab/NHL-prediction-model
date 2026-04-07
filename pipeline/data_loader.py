"""Load and concatenate season CSVs from the data/ directory."""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

SEASON_FILES = {
    "2021-22": "NHL_2021-22_cleaned.csv",
    "2022-23": "NHL_2022-23_cleaned.csv",
    "2023-24": "NHL_2023-24_cleaned.csv",
    "2024-25": "NHL_2024-25_cleaned.csv",
    "2025-26": "NHL_2025-26_cleaned.csv",
}

TRAIN_SEASONS  = ["2021-22", "2022-23", "2023-24", "2024-25"]
PREDICT_SEASON = "2025-26"


def load_season(season: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, SEASON_FILES[season])
    df = pd.read_csv(path, parse_dates=["date"])
    df["season"] = season
    return df


def load_all_seasons() -> pd.DataFrame:
    dfs = []
    for season in SEASON_FILES:
        path = os.path.join(DATA_DIR, SEASON_FILES[season])
        if os.path.exists(path):
            dfs.append(load_season(season))
        else:
            print(f"[data_loader] WARNING: {SEASON_FILES[season]} not found — skipping.")
    if not dfs:
        raise FileNotFoundError("No season CSV files found in data/")
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["season", "date", "team"]).reset_index(drop=True)
    return df


def load_train() -> pd.DataFrame:
    dfs = []
    for season in TRAIN_SEASONS:
        path = os.path.join(DATA_DIR, SEASON_FILES[season])
        if os.path.exists(path):
            dfs.append(load_season(season))
    if not dfs:
        raise FileNotFoundError("No training season CSVs found.")
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["season", "date", "team"]).reset_index(drop=True)
    return df


