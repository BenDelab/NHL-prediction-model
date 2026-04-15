"""
Load MoneyPuck game-by-game xG data from per-team CSV files.

Expected file naming: data/{TEAM}_{SEASON}_xG.csv
  e.g. data/CAR_2024-25_xG.csv, data/BOS_2024-25_xG.csv, ...

Each file has the standard MoneyPuck team-game format with one row per
(team, game, situation). Situations mapped to short names:
  5on5 → ev   (even strength)
  5on4 → pp   (power play — team has extra skater)
  4on5 → pk   (penalty kill — team is shorthanded)
  all  → all  (all situations combined)

Returns one row per (team, date) with xGF / xGA / xGP for each situation,
plus xg_combined_all = xgf_all + xga_all (total expected goals).
"""

from __future__ import annotations

import glob
import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
XG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "xG")

# situation code → short name used in column names
SITUATIONS: dict[str, str] = {
    "5on5": "ev",
    "5on4": "pp",
    "4on5": "pk",
    "all":  "all",
}

# Filename pattern: {TEAM}_{SEASON}_xG.csv  e.g. CAR_2024-25_xG.csv
_FNAME_RE = re.compile(r"^([A-Z]{2,3})_(\d{4}-\d{2})_xG\.csv$")


def _parse_team_file(path: str) -> pd.DataFrame:
    """Parse one team xG file. Returns empty DataFrame on any problem."""
    fname = os.path.basename(path)
    m = _FNAME_RE.match(fname)
    if not m:
        return pd.DataFrame()

    season = m.group(2)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[xg_loader] Could not read {fname}: {e}")
        return pd.DataFrame()

    # Keep Team Level rows only (not individual player rows)
    if "position" in df.columns:
        df = df[df["position"] == "Team Level"].copy()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["gameDate"].astype(str), format="%Y%m%d")
    df["team"] = df["playerTeam"].replace({"ARI": "UTA", "VGK": "VEG"})

    # Pivot situations into separate columns, one row per team+game
    pivoted_parts = []
    for sit_code, sit_name in SITUATIONS.items():
        sub = df[df["situation"] == sit_code][
            ["team", "date", "xGoalsFor", "xGoalsAgainst", "xGoalsPercentage"]
        ].copy()
        if sub.empty:
            continue
        sub = sub.rename(columns={
            "xGoalsFor":        f"xgf_{sit_name}",
            "xGoalsAgainst":    f"xga_{sit_name}",
            "xGoalsPercentage": f"xgp_{sit_name}",
        })
        pivoted_parts.append(sub)

    if not pivoted_parts:
        return pd.DataFrame()

    result = pivoted_parts[0]
    for part in pivoted_parts[1:]:
        result = result.merge(part, on=["team", "date"], how="outer")

    # Combined expected goals (both teams' xG per game — useful for totals model)
    if "xgf_all" in result.columns and "xga_all" in result.columns:
        result["xg_combined_all"] = result["xgf_all"] + result["xga_all"]

    result["season"] = season
    return result.sort_values(["team", "date"]).reset_index(drop=True)


def load_xg_all_seasons() -> pd.DataFrame:
    """
    Discover and load all {TEAM}_{SEASON}_xG.csv files in data/.
    Returns one concatenated DataFrame sorted by team + date.
    """
    pattern = os.path.join(XG_DIR, "*", "*_xG.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("[xg_loader] No *_xG.csv files found in data/ — model will run without xG.")
        return pd.DataFrame()

    dfs = []
    for path in files:
        df = _parse_team_file(path)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = (
        pd.concat(dfs, ignore_index=True)
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )
    seasons = combined["season"].nunique()
    teams   = combined["team"].nunique()
    print(f"[xg_loader] Loaded {len(combined)} team-game rows "
          f"({teams} teams × {seasons} seasons).")
    return combined
