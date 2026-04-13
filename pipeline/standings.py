"""
Playoff motivation features from NHL standings.

Scrapes daily standings from the NHL API for all training + current seasons
and computes per-team per-game motivation scores:

  clinch_status    : 2=clinched division, 1=clinched playoff spot, 0=in race,
                     -1=eliminated
  points           : team's current points
  games_played     : games played entering this game
  games_remaining  : games left in 82-game season
  conf_rank        : conference standing (1=1st, 16=last)
  points_above_cut : pts above/below the 8th seed in conference
                     (positive = safely in, negative = on the bubble / out)
  motivation       : composite [0,1] score
                       - clinched → low (~0.3)
                       - eliminated → low (~0.2)
                       - tight race (|pts_above_cut| ≤ 5, ≥5 games left) → high (~0.9)
                       - comfortable cushion → medium (~0.6)

Usage
-----
  from pipeline.standings import build_standings_features, scrape_and_save

  # One-time scrape (saves CSVs to data/standings/):
  scrape_and_save()

  # In the feature pipeline:
  standings_df = build_standings_features()
  # Then add_standings_features(df, standings_df) joins it to game rows.
"""

from __future__ import annotations

import os
import time
import requests
import pandas as pd
import numpy as np

DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")

# NHL API: standings as of a given date
_API = "https://api-web.nhle.com/v1/standings/{date}"

# One standings snapshot per week is enough for training seasons.
# For the current season we want daily (scraped on demand).
SEASON_DATE_RANGES = {
    "2021-22": ("2021-10-12", "2022-04-29"),
    "2022-23": ("2022-10-07", "2023-04-14"),
    "2023-24": ("2023-10-10", "2024-04-18"),
    "2024-25": ("2024-10-04", "2025-04-18"),
    "2025-26": ("2025-10-07", "2026-04-20"),
}

# ARI → UTA rename (same as xg_loader)
_RENAMES = {"ARI": "UTA", "VGK": "VEG"}


def _fetch_standings(date_str: str) -> pd.DataFrame:
    """Fetch standings for a single date from the NHL API."""
    url = _API.format(date=date_str)
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [standings] failed to fetch {date_str}: {e}")
        return pd.DataFrame()

    rows = []
    for t in data.get("standings", []):
        abbrev = t["teamAbbrev"]["default"] if isinstance(t["teamAbbrev"], dict) else t["teamAbbrev"]
        abbrev = _RENAMES.get(abbrev, abbrev)
        clinch = t.get("clinchIndicator", "")
        rows.append({
            "date":           date_str,
            "team":           abbrev,
            "points":         t.get("points", 0),
            "games_played":   t.get("gamesPlayed", 0),
            "wins":           t.get("wins", 0),
            "losses":         t.get("losses", 0),
            "ot_losses":      t.get("otLosses", 0),
            "conf_rank":      t.get("conferenceSequence", 16),
            "wildcard_rank":  t.get("wildcardSequence", 0),
            "clinch":         clinch,   # 'z','x','e',''
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])

    # Compute points_above_cut: difference between this team's pts and the 8th-seed pts
    # Group by conference (using conf_rank ≤ 8 as proxy for "in").
    # Split into East (WPG,WSH,TOR,TBL,CAR,OTT,FLA,NJD,MTL,BUF,BOS,DET,NYI,NYR,PHI,PIT,CBJ,NSH)
    # and West (everything else) — easier: conf_rank determines this already.
    east = {"BOS","BUF","CAR","CBJ","DET","FLA","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","TBL","TOR","WSH","MNT"}
    west = {"ANA","CGY","CHI","COL","DAL","EDM","LAK","MIN","SEA","SJS","STL","UTA","VAN","VEG","WPG","ARI","ARI"}

    # Use a simpler approach: rank all teams, 8th-place conference pts = cutoff
    # Identify conference by checking known east teams
    EAST_TEAMS = {"BOS","BUF","CAR","CBJ","DET","FLA","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","TBL","TOR","WSH"}
    df["conference"] = df["team"].apply(lambda t: "E" if t in EAST_TEAMS else "W")

    pts_above = []
    for _, row in df.iterrows():
        conf_df = df[df["conference"] == row["conference"]].sort_values("points", ascending=False)
        cutoff_pts = conf_df.iloc[7]["points"] if len(conf_df) >= 8 else 0
        pts_above.append(row["points"] - cutoff_pts)
    df["points_above_cut"] = pts_above

    return df


def _date_range_weekly(start: str, end: str) -> list[str]:
    """Every 7 days from start to end (inclusive of last available)."""
    dates = pd.date_range(start, end, freq="7D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def scrape_season(season: str, force: bool = False) -> pd.DataFrame:
    """
    Scrape standings for one season and cache to data/standings/{season}.csv.
    If the file already exists, load from cache unless force=True.
    """
    os.makedirs(STANDINGS_DIR, exist_ok=True)
    path = os.path.join(STANDINGS_DIR, f"standings_{season}.csv")

    if os.path.exists(path) and not force:
        df = pd.read_csv(path, parse_dates=["date"])
        print(f"  [standings] {season}: loaded {len(df)} rows from cache.")
        return df

    start, end = SEASON_DATE_RANGES[season]
    dates = _date_range_weekly(start, end)
    print(f"  [standings] Scraping {season}: {len(dates)} snapshots ({start} → {end})")

    dfs = []
    for d in dates:
        snap = _fetch_standings(d)
        if not snap.empty:
            dfs.append(snap)
        time.sleep(0.3)   # be polite to the API

    if not dfs:
        print(f"  [standings] WARNING: no data scraped for {season}")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True).sort_values(["team", "date"])
    combined.to_csv(path, index=False)
    print(f"  [standings] {season}: saved {len(combined)} rows to {path}")
    return combined


def scrape_and_save(seasons: list[str] | None = None, force: bool = False) -> None:
    """Scrape and cache standings for all (or specified) seasons."""
    if seasons is None:
        seasons = list(SEASON_DATE_RANGES.keys())
    for season in seasons:
        scrape_season(season, force=force)


def load_all_standings() -> pd.DataFrame:
    """Load all cached standings CSVs. Returns empty DataFrame if none found."""
    os.makedirs(STANDINGS_DIR, exist_ok=True)
    dfs = []
    for season in SEASON_DATE_RANGES:
        path = os.path.join(STANDINGS_DIR, f"standings_{season}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            df["season"] = season
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)


def _motivation_score(row: pd.Series) -> float:
    """
    Non-linear playoff motivation score in [0, 1].

    Captures three distinct regimes:

    1. TANKING (eliminated + near-worst record):
       Eliminated teams near the bottom of the league standings may be
       incentivised to lose for draft lottery position. Score → 0.05.

    2. LOW MOTIVATION (eliminated or seeding locked):
       Already out or seeding irrelevant — likely resting starters.
       Score → 0.10–0.35 depending on how locked in seeding is.

    3. MUST-WIN URGENCY (the non-linear spike):
       A team that needs a win to stay alive gets a sharp urgency boost.
       The key threshold: pts_needed_to_clinch ≈ games_remaining × 2 − pts_above_cut.
       When points_above_cut ≤ 0 AND games_remaining ≤ 6 → near-must-win → 0.95–1.0.
       When points_above_cut ∈ [1,4] AND games_remaining ≤ 8 → high urgency → 0.85–0.95.

    4. COMFORTABLE RACE (safely in or safely out):
       Normal competitive motivation → 0.55–0.70.
    """
    clinch        = row.get("clinch", "")
    pts_above_cut = row.get("points_above_cut", 0)
    games_rem     = max(int(row.get("games_remaining", 10)), 0)
    conf_rank     = row.get("conf_rank", 8)

    # ── Eliminated teams ──────────────────────────────────────────────────────
    if clinch == "e":
        # Tanking signal: eliminated AND near the bottom of the league (conf_rank ≥ 14)
        if conf_rank >= 14:
            return 0.05   # likely tanking for draft lottery
        return 0.15       # just eliminated, may still play hard

    # ── Clinched — seeding concerns only ──────────────────────────────────────
    if clinch == "z":
        # Division clinched — only president's trophy or seeding matters
        return 0.30
    if clinch == "x":
        # Playoff spot clinched — seeding still has value but no urgency
        # If locked into a seed (conf_rank gap large), rest starters
        return 0.40

    # ── Still in the race ────────────────────────────────────────────────────
    # points_above_cut: positive = above 8th seed, negative = below

    # Magic number: how many pts needed to clinch given games remaining
    # Roughly: if pts_above_cut < 0 and games_rem * 2 < abs(pts_above_cut) → mathematically out
    magic_pts_possible = games_rem * 2   # max pts a team can earn

    if pts_above_cut < 0 and magic_pts_possible < abs(pts_above_cut) - 2:
        # Mathematically eliminated but API hasn't flagged it yet
        return 0.15

    # Non-linear urgency: the closer to elimination, the sharper the spike
    if pts_above_cut <= 0 and games_rem <= 6:
        # On the bubble or below it with very few games left → must-win territory
        urgency = 1.0 - (max(pts_above_cut, -4) + 4) / 4.0 * 0.05   # tiny softening
        return min(1.0, 0.90 + urgency * 0.10)

    if 0 < pts_above_cut <= 3 and games_rem <= 8:
        # Barely above the cutoff, few games left → very high urgency
        return 0.85 + (3 - pts_above_cut) / 3.0 * 0.10

    if pts_above_cut <= 0 and games_rem <= 15:
        # Below cutoff but still time to recover
        recovery_ratio = min(1.0, magic_pts_possible / max(abs(pts_above_cut) + 2, 1))
        return 0.65 + 0.20 * recovery_ratio

    if 0 < pts_above_cut <= 6:
        # Safely above but not locked in — comfortable but motivated
        return 0.65 - pts_above_cut * 0.02   # 0.65 at +0 → 0.53 at +6

    # Comfortably in (gap > 6) — playing for seeding only
    return max(0.45, 0.55 - (pts_above_cut - 6) * 0.01)


def build_standings_features(standings_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute per-team per-snapshot motivation features from standings data.
    Returns a DataFrame with columns:
      team, date, season, points, games_played, games_remaining,
      conf_rank, points_above_cut, clinch_status, motivation
    """
    if standings_df is None:
        standings_df = load_all_standings()
    if standings_df.empty:
        return pd.DataFrame()

    df = standings_df.copy()
    df["games_remaining"] = 82 - df["games_played"]

    # Numeric clinch status
    clinch_map = {"z": 2, "x": 1, "": 0, "e": -1}
    df["clinch_status"] = df["clinch"].map(clinch_map).fillna(0).astype(int)

    df["motivation"] = df.apply(_motivation_score, axis=1)

    # Binary flags for extreme states
    df["is_tanking"]   = ((df["clinch"] == "e") & (df["conf_rank"] >= 14)).astype(int)
    df["is_must_win"]  = (
        (df["clinch"] == "") &
        (df["points_above_cut"] <= 0) &
        (df["games_remaining"] <= 6)
    ).astype(int)
    df["is_clinched"]  = df["clinch"].isin(["x", "z"]).astype(int)

    cols = ["team", "date", "season", "points", "games_played", "games_remaining",
            "conf_rank", "points_above_cut", "clinch_status", "motivation",
            "is_tanking", "is_must_win", "is_clinched"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)
