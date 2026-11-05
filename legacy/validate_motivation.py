"""
Motivation score validation: does the motivation feature add real signal
beyond what rolling stats already predict?

Method
──────
1. For each played game in training seasons, compute the ensemble's base
   win probability using ONLY the rolling + schedule features (no motivation).
2. Group games by motivation bucket (tanking, eliminated, in-race, must-win,
   clinched, etc.)
3. Compare: base model predicted win% vs actual win rate per bucket.
4. If motivation adds signal: must-win teams should win MORE than the base
   model predicts; tanking teams should win LESS.

The gap (actual − predicted) tells us whether the current motivation score
thresholds capture a real effect.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pipeline import data_loader, features, noise_reduction, xg_loader
from pipeline.standings import build_standings_features, load_all_standings
from models import evaluate

TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
WINDOWS       = [10, 20]   # two windows for speed; enough for validation


# ── 1. Build feature matrix ───────────────────────────────────────────────────
print("Loading data...")
df_raw = data_loader.load_all_seasons()
df_raw = df_raw[df_raw["season"].isin(TRAIN_SEASONS)] if "season" in df_raw.columns else df_raw
xg_df  = xg_loader.load_xg_all_seasons()
df_fe  = features.build_features(df_raw, xg_df=xg_df if not xg_df.empty else None)

played_mask = df_fe["result"].isin(["W", "L"])
df_fe.loc[played_mask] = noise_reduction.apply(df_fe[played_mask].copy())

# ── 2. Build standings features ───────────────────────────────────────────────
print("Loading standings...")
standings_df = build_standings_features()
if standings_df.empty:
    print("ERROR: No standings data found. Run pipeline.standings.scrape_and_save() first.")
    exit(1)

# ── 3. Train a base model WITHOUT motivation features ─────────────────────────
# Use only rolling + schedule features — deliberately exclude motivation columns
MOTIVATION_COLS = {
    "motivation", "clinch_status", "points_above_cut", "points",
    "games_remaining", "conf_rank",
    "opp_motivation", "opp_clinch_status", "opp_points_above_cut",
    "opp_points", "opp_games_remaining", "opp_conf_rank",
    "motivation_diff", "pts_above_cut_diff",
    "is_tanking", "is_must_win", "is_clinched",
    "opp_is_tanking", "opp_is_must_win", "opp_is_clinched",
    "h_motivation", "h_clinch", "h_pts_above_cut",
}

print("\nTraining base models (no motivation features)...")
df_played = df_fe[played_mask].copy()
y_all     = df_played["result"].eq("W").astype(int)

base_preds = np.zeros(len(df_played))
counts     = np.zeros(len(df_played))

for w in WINDOWS:
    feat_cols = evaluate.get_feature_cols_for_window(w, df_played)
    feat_cols = [c for c in feat_cols if c in df_played.columns
                 and c not in MOTIVATION_COLS]

    lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ])

    # Walk-forward: train on first 60%, predict on rest
    n = len(df_played)
    split = int(n * 0.60)
    lr.fit(df_played[feat_cols].iloc[:split], y_all.iloc[:split])
    probs = lr.predict_proba(df_played[feat_cols])[:, 1]
    base_preds += probs
    counts     += 1

base_preds /= counts


# ── 4. Join motivation buckets ────────────────────────────────────────────────
# Merge standings onto each played game (backward merge: last snapshot before game)
print("Joining motivation buckets...")
standings_df["date"] = pd.to_datetime(standings_df["date"])
df_played = df_played.copy()
df_played["date"] = pd.to_datetime(df_played["date"])

# merge_asof backward per team
parts = []
for team, grp in df_played.groupby("team"):
    st = standings_df[standings_df["team"] == team].sort_values("date")
    if st.empty:
        grp = grp.copy()
        grp["st_motivation"]       = np.nan
        grp["st_clinch_status"]    = np.nan
        grp["st_points_above_cut"] = np.nan
        grp["st_games_remaining"]  = np.nan
        grp["st_conf_rank"]        = np.nan
        grp["st_is_tanking"]       = 0
        grp["st_is_must_win"]      = 0
        grp["st_is_clinched"]      = 0
        parts.append(grp)
        continue
    st_cols = st[["date","motivation","clinch_status","points_above_cut",
                  "games_remaining","conf_rank","is_tanking","is_must_win","is_clinched"]].copy()
    # Rename to avoid collision with existing feature columns
    st_cols = st_cols.rename(columns={
        "motivation":       "st_motivation",
        "clinch_status":    "st_clinch_status",
        "points_above_cut": "st_points_above_cut",
        "games_remaining":  "st_games_remaining",
        "conf_rank":        "st_conf_rank",
        "is_tanking":       "st_is_tanking",
        "is_must_win":      "st_is_must_win",
        "is_clinched":      "st_is_clinched",
    })
    merged = pd.merge_asof(
        grp.sort_values("date"), st_cols,
        on="date", direction="backward",
    )
    parts.append(merged)

df_val = pd.concat(parts).sort_index()
df_val["base_pred"] = base_preds
df_val["actual"]    = y_all.values


# ── 5. Bucket by motivation state ─────────────────────────────────────────────
def motivation_bucket(row):
    cs  = row.get("st_clinch_status", 0)
    mot = row.get("st_motivation", 0.5)
    itk = row.get("st_is_tanking", 0)
    imw = row.get("st_is_must_win", 0)
    icl = row.get("st_is_clinched", 0)

    if itk == 1:                          return "TANKING"
    if cs == -1:                          return "ELIMINATED"
    if imw == 1:                          return "MUST-WIN"
    if icl == 1:                          return "CLINCHED"
    if isinstance(mot, float):
        if mot >= 0.80:                   return "HIGH URGENCY (0.80+)"
        if mot >= 0.65:                   return "IN RACE (0.65-0.80)"
        if mot >= 0.50:                   return "COMFORTABLE (0.50-0.65)"
    return "OTHER"

df_val["bucket"] = df_val.apply(motivation_bucket, axis=1)


# ── 6. Analysis per bucket ────────────────────────────────────────────────────
print("\n" + "="*72)
print("MOTIVATION SCORE VALIDATION")
print("="*72)
print(f"\n{'Bucket':<26} {'N':>6}  {'Base pred':>10}  {'Actual':>8}  {'Lift':>8}  {'Signal?'}")
print("-"*72)

bucket_order = ["TANKING","ELIMINATED","CLINCHED","COMFORTABLE (0.50-0.65)",
                "IN RACE (0.65-0.80)","HIGH URGENCY (0.80+)","MUST-WIN","OTHER"]

for bucket in bucket_order:
    grp = df_val[df_val["bucket"] == bucket]
    if len(grp) < 20:
        continue
    base_mean   = grp["base_pred"].mean()
    actual_mean = grp["actual"].mean()
    lift        = actual_mean - base_mean
    n           = len(grp)

    # Simple binomial test: is the lift significant?
    from scipy.stats import binomtest
    pval = binomtest(int(grp["actual"].sum()), n, base_mean, alternative="two-sided").pvalue
    sig  = "**" if pval < 0.05 else ("*" if pval < 0.10 else "")

    print(f"  {bucket:<24} {n:>6}  {base_mean:>10.1%}  {actual_mean:>8.1%}  {lift:>+8.1%}  {sig}")

print()
print("** p<0.05, * p<0.10 (binomial test: actual vs base predicted rate)")


# ── 7. Threshold sensitivity: MUST-WIN by games_remaining ─────────────────────
print("\n" + "="*72)
print("MUST-WIN THRESHOLD SENSITIVITY  (pts_above_cut ≤ 0, games_rem varies)")
print("="*72)
print(f"\n{'games_rem ≤':<14} {'N':>6}  {'Base pred':>10}  {'Actual':>8}  {'Lift':>8}")
print("-"*50)

bubble = df_val[(df_val["st_clinch_status"] == 0) &
                (df_val["st_points_above_cut"] <= 0)]
for threshold in [4, 6, 8, 10, 15]:
    grp = bubble[bubble["st_games_remaining"] <= threshold]
    if len(grp) < 10:
        continue
    base_mean   = grp["base_pred"].mean()
    actual_mean = grp["actual"].mean()
    lift        = actual_mean - base_mean
    print(f"  ≤ {threshold:<12} {len(grp):>6}  {base_mean:>10.1%}  {actual_mean:>8.1%}  {lift:>+8.1%}")


# ── 8. Threshold sensitivity: TANKING by conf_rank ────────────────────────────
print("\n" + "="*72)
print("TANKING THRESHOLD SENSITIVITY  (eliminated, conf_rank varies)")
print("="*72)
print(f"\n{'conf_rank ≥':<14} {'N':>6}  {'Base pred':>10}  {'Actual':>8}  {'Lift':>8}")
print("-"*50)

elim = df_val[df_val["st_clinch_status"] == -1]
for threshold in [10, 12, 14, 16]:
    grp = elim[elim["st_conf_rank"] >= threshold]
    if len(grp) < 10:
        continue
    base_mean   = grp["base_pred"].mean()
    actual_mean = grp["actual"].mean()
    lift        = actual_mean - base_mean
    print(f"  ≥ {threshold:<12} {len(grp):>6}  {base_mean:>10.1%}  {actual_mean:>8.1%}  {lift:>+8.1%}")


# ── 9. Motivation score correlation with residuals ───────────────────────────
print("\n" + "="*72)
print("CORRELATION: motivation score vs actual outcome residual")
print("="*72)
df_corr = df_val.dropna(subset=["st_motivation"])
residual = df_corr["actual"] - df_corr["base_pred"]
r = np.corrcoef(df_corr["st_motivation"], residual)[0, 1]
print(f"\n  Pearson r(motivation, actual - base_pred) = {r:.4f}")
print(f"  {'Positive r → higher motivation helps teams outperform their stats' if r > 0 else 'Negative or near-zero → motivation score direction may be inverted/noisy'}")

# Quintile breakdown
df_corr = df_corr.copy()
df_corr["mot_quintile"] = pd.qcut(df_corr["st_motivation"], 5, labels=False)
print(f"\n  {'Motivation quintile':<22}  {'N':>6}  {'Mean motivation':>16}  {'Actual win%':>12}  {'Lift':>8}")
print("  " + "-"*68)
for q, grp in df_corr.groupby("mot_quintile"):
    base  = grp["base_pred"].mean()
    act   = grp["actual"].mean()
    print(f"  Q{q+1} (motivation {grp['st_motivation'].min():.2f}-{grp['st_motivation'].max():.2f})  "
          f"{len(grp):>6}  {grp['st_motivation'].mean():>16.3f}  {act:>12.1%}  {act-base:>+8.1%}")

print()
