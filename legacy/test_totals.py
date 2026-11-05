"""
Totals regression comparison: Ridge (current) vs 4 alternatives.

Options tested
──────────────
0. Ridge (current baseline)
1. Quantile regression (median) — more robust central estimate
2. Split home/away regressors — predict home goals + away goals separately
3. Pace-proxy features — add combined SOG, avg SV%, combined PP opp to Ridge
4. H2H blend — blend Ridge prediction with historical team-pair scoring rate

All tested with walk-forward: train on 2021-22/2022-23, test 2023-24
                               train on 2021-22/2022-23/2023-24, test 2024-25

Metrics
───────
• MAE (predicted total vs actual total)
• Accuracy at 5.5, 6.5, 7.5 (half-point lines, conviction pick)
• Over-prediction bias (% games where pred > actual)
• OVER rate at 6.5 (should be ~40-60%, not 8% or 89%)
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

from pipeline import data_loader, features, noise_reduction, xg_loader
from models import evaluate
from models.totals import prob_over_goals

BACKTEST_ROUNDS = [
    (["2021-22", "2022-23"],            "2023-24"),
    (["2021-22", "2022-23", "2023-24"], "2024-25"),
]
HALF_LINES = [5.5, 6.5, 7.5]


# ── helpers ────────────────────────────────────────────────────────────────────

def make_ridge():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   Ridge(alpha=1.0)),
    ])

def make_quantile():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   QuantileRegressor(quantile=0.5, alpha=0.1)),
    ])

def ensemble_predict(models_by_w, df_tr, X_row):
    """Average regression prediction across all windows."""
    ps = []
    for w, ms in models_by_w.items():
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr)
              if c in X_row.index]
        X = X_row[fc].to_frame().T
        for m in ms.values():
            try: ps.append(float(m.predict(X)[0]))
            except: pass
    return float(np.mean(ps)) if ps else 6.24

def evaluate_predictions(predicted, actual, label):
    """Compute all metrics for a set of predictions."""
    mae       = mean_absolute_error(actual, predicted)
    over_bias = (predicted > actual).mean()

    results = {"label": label, "mae": mae, "over_bias": over_bias}

    for line in HALF_LINES:
        y_over = (actual > line).astype(int)
        p_over = np.array([prob_over_goals(p, line) for p in predicted])
        picks  = (p_over >= 0.5).astype(int)
        acc    = accuracy_score(y_over, picks)
        results[f"acc_{line}"] = acc
        results[f"over_rate_{line}"] = picks.mean()

    # Best-conviction pick
    all_p = np.column_stack([
        np.array([prob_over_goals(p, l) for p in predicted]) for l in HALF_LINES
    ])
    conv      = np.abs(all_p - 0.5)
    best_idx  = conv.argmax(axis=1)
    best_p    = all_p[np.arange(len(predicted)), best_idx]
    best_line = np.array(HALF_LINES)[best_idx]
    best_pick = (best_p >= 0.5).astype(int)
    best_act  = np.array([(actual[i] > best_line[i]).astype(int)
                           for i in range(len(predicted))])
    results["conv_acc"]      = (best_pick == best_act).mean()
    results["conv_over_rate"] = best_pick.mean()
    return results


# ── pace-proxy features ────────────────────────────────────────────────────────

def add_pace_features(df_tr, df_te):
    """
    Add combined-game-pace features derived from existing rolling columns.
    For each window: combined_sog, avg_sv_pct, combined_pp_opp, h_gf + a_ga avg.
    These aggregate home + away signals into explicit game-level pace proxies.
    """
    df_tr = df_tr.copy()
    df_te = df_te.copy()
    for df in [df_tr, df_te]:
        for w in [5, 10, 15, 20]:
            p  = f"rolling_{w}"
            op = f"opp_rolling_{w}"
            # combined shot volume (home SOG + away SOG)
            h_sog = f"{p}_sog_avg"
            a_sog = f"{op}_sog_avg"
            if h_sog in df.columns and a_sog in df.columns:
                df[f"{p}_combined_sog"] = df[h_sog] + df[a_sog]
            # average save% (weaker goaltending = more goals)
            h_sv = f"{p}_save_pct"
            a_sv = f"{op}_save_pct"
            if h_sv in df.columns and a_sv in df.columns:
                df[f"{p}_avg_sv_pct"] = (df[h_sv] + df[a_sv]) / 2
            # combined PP opportunities (more penalties = more PP goals)
            h_pp = f"{p}_pp_efficiency"
            a_pp = f"{op}_pp_efficiency"
            if h_pp in df.columns and a_pp in df.columns:
                df[f"{p}_combined_pp_eff"] = (df[h_pp].fillna(0.2) +
                                               df[a_pp].fillna(0.2)) / 2
            # expected total from scoring averages (home GF + away GF)
            h_gf = f"{p}_goals_for_avg"
            a_gf = f"{op}_goals_for_avg"
            if h_gf in df.columns and a_gf in df.columns:
                df[f"{p}_expected_total"] = df[h_gf] + df[a_gf]
    return df_tr, df_te

def get_pace_feature_cols(w, df):
    base = evaluate.get_feature_cols_for_window(w, df)
    base = [c for c in base if c in df.columns]
    pace_extras = [f"rolling_{w}_combined_sog", f"rolling_{w}_avg_sv_pct",
                   f"rolling_{w}_combined_pp_eff", f"rolling_{w}_expected_total"]
    return base + [c for c in pace_extras if c in df.columns]


# ── H2H blend helper ───────────────────────────────────────────────────────────

def build_h2h_totals(df_tr):
    """
    Build a lookup: (home_team, away_team) → historical mean total goals.
    Falls back to the season mean when H2H sample < 3 games.
    """
    played = df_tr[df_tr["goals_for"].notna() & df_tr["goals_against"].notna()].copy()
    played["total"] = played["goals_for"] + played["goals_against"]
    league_mean = played["total"].mean()

    h2h = {}
    for (ht, at), grp in played.groupby(["team", "opponent"]):
        if len(grp) >= 3:
            h2h[(ht, at)] = grp["total"].mean()
    return h2h, league_mean

def h2h_blend(ridge_pred, team, opponent, h2h, league_mean, alpha=0.7):
    """
    Blend Ridge prediction with H2H mean.
    alpha = weight on Ridge (higher = more Ridge, less H2H).
    When H2H exists use it; otherwise fall back to league mean.
    """
    h2h_mean = h2h.get((team, opponent), h2h.get((opponent, team), league_mean))
    return alpha * ridge_pred + (1 - alpha) * h2h_mean


# ── Main comparison ────────────────────────────────────────────────────────────

print("Loading features...")
df_all  = data_loader.load_all_seasons()
xg_data = xg_loader.load_xg_all_seasons()
df_fe   = features.build_features(df_all, xg_df=xg_data if not xg_data.empty else None)
played  = df_fe["result"].isin(["W", "L"])
df_fe.loc[played] = noise_reduction.apply(df_fe[played].copy())

all_results = []

for train_seasons, test_season in BACKTEST_ROUNDS:
    print(f"\n{'─'*72}")
    print(f"  Train {train_seasons}  →  Test {test_season}")
    print(f"{'─'*72}")

    df_tr_raw = df_fe[df_fe["season"].isin(train_seasons)].copy()
    df_te_raw = (df_fe[(df_fe["season"] == test_season) & played]
                 .sort_values("date").reset_index(drop=True))

    df_tr_g = df_tr_raw[df_tr_raw["goals_for"].notna() &
                         df_tr_raw["goals_against"].notna()].copy()
    df_te_g = df_te_raw[df_te_raw["goals_for"].notna() &
                         df_te_raw["goals_against"].notna()].copy()

    y_tr_total = df_tr_g["goals_for"] + df_tr_g["goals_against"]
    actual     = (df_te_g["goals_for"] + df_te_g["goals_against"]).values.astype(float)

    # ── 0. Baseline: Ridge on total goals ─────────────────────────────────────
    print("  [0] Ridge (baseline)...")
    ridge_models = {}
    for w in evaluate.WINDOWS:
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr_raw)
              if c in df_tr_g.columns]
        m = make_ridge()
        m.fit(df_tr_g[fc], y_tr_total)
        ridge_models[w] = {"Ridge": m}

    pred0 = np.array([ensemble_predict(ridge_models, df_tr_raw, df_te_g.iloc[i])
                       for i in range(len(df_te_g))])
    r0 = evaluate_predictions(pred0, actual, "0_Ridge_baseline")
    all_results.append({**r0, "season": test_season})

    # ── 1. Quantile regression ────────────────────────────────────────────────
    print("  [1] Quantile regression (median)...")
    q_models = {}
    for w in evaluate.WINDOWS:
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr_raw)
              if c in df_tr_g.columns]
        m = make_quantile()
        m.fit(df_tr_g[fc], y_tr_total)
        q_models[w] = {"Quantile": m}

    pred1 = np.array([ensemble_predict(q_models, df_tr_raw, df_te_g.iloc[i])
                       for i in range(len(df_te_g))])
    r1 = evaluate_predictions(pred1, actual, "1_Quantile_median")
    all_results.append({**r1, "season": test_season})

    # ── 2. Split home + away regressors ───────────────────────────────────────
    print("  [2] Split home/away regressors...")
    home_models = {}; away_models = {}
    for w in evaluate.WINDOWS:
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr_raw)
              if c in df_tr_g.columns]
        mh = make_ridge(); ma = make_ridge()
        mh.fit(df_tr_g[fc], df_tr_g["goals_for"])
        ma.fit(df_tr_g[fc], df_tr_g["goals_against"])
        home_models[w] = {"Ridge_home": mh}
        away_models[w] = {"Ridge_away": ma}

    pred_home = np.array([ensemble_predict(home_models, df_tr_raw, df_te_g.iloc[i])
                           for i in range(len(df_te_g))])
    pred_away = np.array([ensemble_predict(away_models, df_tr_raw, df_te_g.iloc[i])
                           for i in range(len(df_te_g))])
    pred2 = pred_home + pred_away
    r2 = evaluate_predictions(pred2, actual, "2_Split_home_away")
    all_results.append({**r2, "season": test_season})

    # ── 3. Pace-proxy features ────────────────────────────────────────────────
    print("  [3] Pace-proxy features...")
    df_tr_pace, df_te_pace = add_pace_features(df_tr_raw, df_te_raw)
    df_tr_pg = df_tr_pace[df_tr_pace["goals_for"].notna() &
                           df_tr_pace["goals_against"].notna()].copy()
    df_te_pg = df_te_pace[df_te_pace["goals_for"].notna() &
                           df_te_pace["goals_against"].notna()].copy()
    y_tr_pg  = df_tr_pg["goals_for"] + df_tr_pg["goals_against"]

    pace_models = {}
    for w in evaluate.WINDOWS:
        fc = [c for c in get_pace_feature_cols(w, df_tr_pace) if c in df_tr_pg.columns]
        m = make_ridge()
        m.fit(df_tr_pg[fc], y_tr_pg)
        pace_models[w] = {"Ridge_pace": m}

    pred3 = np.array([ensemble_predict(pace_models, df_tr_pace, df_te_pg.iloc[i])
                       for i in range(len(df_te_pg))])
    r3 = evaluate_predictions(pred3, (df_te_pg["goals_for"] +
                                       df_te_pg["goals_against"]).values.astype(float),
                               "3_Pace_features")
    all_results.append({**r3, "season": test_season})

    # ── 4. H2H blend ──────────────────────────────────────────────────────────
    print("  [4] H2H blend (Ridge + historical team-pair mean)...")
    h2h_lookup, league_mean = build_h2h_totals(df_tr_g)
    print(f"      H2H pairs with ≥3 games: {len(h2h_lookup)}  |  League mean: {league_mean:.2f}")

    # Best alpha: try 0.6, 0.7, 0.8 (weight on Ridge)
    best_alpha, best_mae = 0.7, 999
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        blended = np.array([
            h2h_blend(pred0[i], df_te_g.iloc[i]["team"],
                      df_te_g.iloc[i]["opponent"], h2h_lookup, league_mean, alpha)
            for i in range(len(df_te_g))
        ])
        mae = mean_absolute_error(actual, blended)
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha

    pred4 = np.array([
        h2h_blend(pred0[i], df_te_g.iloc[i]["team"],
                  df_te_g.iloc[i]["opponent"], h2h_lookup, league_mean, best_alpha)
        for i in range(len(df_te_g))
    ])
    print(f"      Best alpha (Ridge weight): {best_alpha}")
    r4 = evaluate_predictions(pred4, actual, f"4_H2H_blend_a{best_alpha}")
    all_results.append({**r4, "season": test_season})


# ── Print comparison table ────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  TOTALS COMPARISON RESULTS")
print(f"{'='*80}")

for season in ["2023-24", "2024-25"]:
    print(f"\n  Season: {season}")
    rows = [r for r in all_results if r["season"] == season]
    print(f"  {'Method':<30} {'MAE':>6}  {'OvBias':>7}  "
          f"{'Acc5.5':>7}  {'Acc6.5':>7}  {'Acc7.5':>7}  "
          f"{'ConvAcc':>8}  {'OvRate6.5':>10}")
    print("  " + "─"*78)
    for r in rows:
        print(f"  {r['label']:<30} {r['mae']:>6.3f}  {r['over_bias']:>7.1%}  "
              f"{r['acc_5.5']:>7.3f}  {r['acc_6.5']:>7.3f}  {r['acc_7.5']:>7.3f}  "
              f"{r['conv_acc']:>8.3f}  {r['over_rate_6.5']:>10.1%}")

print(f"\n  Key metrics:")
print(f"  • MAE: lower is better (closer to actual total)")
print(f"  • OvBias: ideally ~50% (not always predicting over or under)")
print(f"  • Acc6.5: most important line — where actual 50/50 split lives")
print(f"  • OvRate6.5: ideally 40-60% (not all OVER or all UNDER)")
print(f"  • ConvAcc: best-conviction pick accuracy (picks highest-conviction line per game)")
