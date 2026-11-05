"""
Totals: direct binary classifier vs Ridge regression.

Instead of predicting a goal total then converting to over/under via Poisson,
train a classifier directly on: "will this game go over 6.5?"

Tests three approaches:
  A. Ridge (current baseline)
  B. Direct classifier: LR + XGB + LGBM ensemble on over/under 6.5
  C. Direct classifier at multiple lines: 5.5, 6.5, 7.5 independently

Evaluation uses the same base-rate-relative threshold as the spread model:
recommend OVER only when P(over) meaningfully exceeds the empirical base rate.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from pipeline import data_loader, features, noise_reduction, xg_loader
from models import evaluate
from models.totals import prob_over_goals

BACKTEST_ROUNDS = [
    (["2021-22", "2022-23"],            "2023-24"),
    (["2021-22", "2022-23", "2023-24"], "2024-25"),
]
HALF_LINES = [5.5, 6.5, 7.5]


# ── model builders ────────────────────────────────────────────────────────────

def train_over_classifier(X, y, sample_weight=None):
    """LR + shallow XGB + LGBM — same structure as spread classifier."""
    lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", random_state=42)),
    ])
    xgb = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m",   XGBClassifier(
            max_depth=3, n_estimators=200, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=2.0,
            random_state=42, eval_metric="logloss", n_jobs=-1,
        )),
    ])
    lgbm = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m",   LGBMClassifier(
            max_depth=3, num_leaves=15, n_estimators=200, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=2.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )),
    ])
    models = {"LR": lr, "XGB": xgb, "LGBM": lgbm}
    for name, m in models.items():
        fp = {}
        if sample_weight is not None:
            fp["m__sample_weight"] = sample_weight
        m.fit(X, y, **fp)
    return models


def train_ridge(X, y, sample_weight=None):
    m = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("m",   Ridge(alpha=1.0)),
    ])
    fp = {}
    if sample_weight is not None:
        fp["m__sample_weight"] = sample_weight
    m.fit(X, y, **fp)
    return {"Ridge": m}


# ── ensemble helpers ──────────────────────────────────────────────────────────

def clf_ensemble_prob(models_by_w, df_tr, X_row):
    """Average P(over) from all window × model combinations."""
    ps = []
    for w, ms in models_by_w.items():
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr)
              if c in X_row.index]
        X = X_row[fc].to_frame().T
        for m in ms.values():
            try: ps.append(float(m.predict_proba(X)[0][1]))
            except: pass
    return float(np.mean(ps)) if ps else 0.5


def reg_ensemble_pred(models_by_w, df_tr, X_row):
    """Average regression prediction."""
    ps = []
    for w, ms in models_by_w.items():
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr)
              if c in X_row.index]
        X = X_row[fc].to_frame().T
        for m in ms.values():
            try: ps.append(float(m.predict(X)[0]))
            except: pass
    return float(np.mean(ps)) if ps else 6.24


# ── evaluation ────────────────────────────────────────────────────────────────

def print_line_results(label, probs, actual_over, base_rate, line):
    """Print accuracy both at 0.5 threshold and base-rate-relative threshold."""
    # 0.5 threshold
    picks_50  = (probs >= 0.50).astype(int)
    acc_50    = accuracy_score(actual_over, picks_50)
    over_50   = picks_50.mean()

    # Base-rate-relative: only pick OVER when p > base_rate + 0.05
    over_edge_min = 0.05
    picks_br  = np.where(probs >= base_rate + over_edge_min, 1,
                np.where(probs <= base_rate - over_edge_min, 0, -1))
    mask_dec  = picks_br != -1   # games where model has an opinion
    acc_br    = accuracy_score(actual_over[mask_dec], picks_br[mask_dec]) if mask_dec.sum() > 0 else np.nan
    n_over_br = (picks_br == 1).sum()
    n_under_br= (picks_br == 0).sum()
    n_pass_br = (picks_br == -1).sum()

    # AUC (ranking quality)
    auc = roc_auc_score(actual_over, probs) if len(np.unique(actual_over)) > 1 else np.nan

    print(f"  {label:<30} line={line}  base={base_rate:.1%}  AUC={auc:.3f}")
    print(f"    p≥0.50 threshold:  acc={acc_50:.3f}  OVER={over_50:.1%}  UNDER={1-over_50:.1%}")
    if mask_dec.sum() > 0:
        print(f"    base+5pp threshold: acc={acc_br:.3f}  n_over={n_over_br}  n_under={n_under_br}  n_skip={n_pass_br}")
    return {"label": label, "line": line, "auc": auc,
            "acc_50": acc_50, "over_50": over_50,
            "acc_br": acc_br if mask_dec.sum() > 0 else np.nan,
            "n_over_br": n_over_br, "n_under_br": n_under_br}


# ── main ─────────────────────────────────────────────────────────────────────

print("Loading features...")
df_all  = data_loader.load_all_seasons()
xg_data = xg_loader.load_xg_all_seasons()
df_fe   = features.build_features(df_all, xg_df=xg_data if not xg_data.empty else None)
played  = df_fe["result"].isin(["W","L"])
df_fe.loc[played] = noise_reduction.apply(df_fe[played].copy())

summary = []

for train_seasons, test_season in BACKTEST_ROUNDS:
    print(f"\n{'═'*72}")
    print(f"  Train {train_seasons}  →  Test {test_season}")
    print(f"{'═'*72}")

    df_tr = df_fe[df_fe["season"].isin(train_seasons)].copy()
    df_te = (df_fe[(df_fe["season"]==test_season) & played]
             .sort_values("date").reset_index(drop=True))

    df_tr_g = df_tr[df_tr["goals_for"].notna() & df_tr["goals_against"].notna()].copy()
    df_te_g = df_te[df_te["goals_for"].notna() & df_te["goals_against"].notna()].copy()

    y_tr_total = df_tr_g["goals_for"] + df_tr_g["goals_against"]
    actual_total = (df_te_g["goals_for"] + df_te_g["goals_against"]).values.astype(float)

    # ── A. Ridge baseline ─────────────────────────────────────────────────────
    print("\n  [A] Ridge (baseline)...")
    ridge_by_w = {}
    for w in evaluate.WINDOWS:
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr) if c in df_tr_g.columns]
        ridge_by_w[w] = train_ridge(df_tr_g[fc], y_tr_total)

    pred_total = np.array([reg_ensemble_pred(ridge_by_w, df_tr, df_te_g.iloc[i])
                            for i in range(len(df_te_g))])

    for line in HALF_LINES:
        y_over   = (actual_total > line).astype(int)
        p_over   = np.array([prob_over_goals(p, line) for p in pred_total])
        base     = float(y_over.mean())
        r = print_line_results("Ridge→Poisson", p_over, y_over, base, line)
        summary.append({**r, "season": test_season, "method": "A_Ridge"})

    # ── B. Direct classifier at 6.5 only ─────────────────────────────────────
    print("\n  [B] Direct classifier at each line...")
    for line in HALF_LINES:
        y_over_tr = (y_tr_total > line).astype(int)
        y_over_te = (actual_total > line).astype(int)
        base      = float(y_over_te.mean())
        base_tr   = float(y_over_tr.mean())
        print(f"\n    Line {line}: train over%={base_tr:.1%}  test over%={base:.1%}  N={len(y_over_te)}")

        clf_by_w = {}
        for w in evaluate.WINDOWS:
            fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr) if c in df_tr_g.columns]
            clf_by_w[w] = train_over_classifier(df_tr_g[fc], y_over_tr)

        p_over = np.array([clf_ensemble_prob(clf_by_w, df_tr, df_te_g.iloc[i])
                            for i in range(len(df_te_g))])

        r = print_line_results("Classifier", p_over, y_over_te, base, line)
        summary.append({**r, "season": test_season, "method": "B_Classifier"})

    # ── C. Calibrated classifier (use last 30% of training as cal set) ────────
    print("\n  [C] Calibrated classifier at 6.5 (temperature scaling)...")
    from pipeline import calibration as cal_module

    line = 6.5
    y_over_tr = (y_tr_total > line).astype(int)
    y_over_te = (actual_total > line).astype(int)
    base      = float(y_over_te.mean())
    n_tr      = len(df_tr_g)
    cal_split = int(n_tr * 0.80)

    clf_by_w_raw = {}
    for w in evaluate.WINDOWS:
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr) if c in df_tr_g.columns]
        clf_by_w_raw[w] = train_over_classifier(df_tr_g[fc], y_over_tr)

    # Pool calibration
    X_cal_by_w = {}
    for w in evaluate.WINDOWS:
        fc = [c for c in evaluate.get_feature_cols_for_window(w, df_tr) if c in df_tr_g.columns]
        X_cal_by_w[w] = (df_tr_g[fc].iloc[cal_split:], fc)
    y_cal = np.asarray(y_over_tr.iloc[cal_split:])

    _, clf_by_w_cal = cal_module.calibrate_pooled(clf_by_w_raw, X_cal_by_w, y_cal)

    p_over_cal = np.array([clf_ensemble_prob(clf_by_w_cal, df_tr, df_te_g.iloc[i])
                            for i in range(len(df_te_g))])

    r = print_line_results("Calibrated Classifier", p_over_cal, y_over_te, base, line)
    summary.append({**r, "season": test_season, "method": "C_CalibratedClf"})


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n\n{'='*72}")
print("  FINAL COMPARISON: Acc@6.5 (the only line that matters)")
print(f"{'='*72}")
print(f"\n  {'Method':<25} {'Season':<10} {'AUC':>6}  {'Acc(0.5)':>9}  {'OVER%':>7}  {'Acc(edge)':>10}  {'N_over':>7}  {'N_under':>8}")
print("  " + "─"*80)

for r in summary:
    if r["line"] == 6.5:
        acc_br_str = f"{r['acc_br']:.3f}" if not np.isnan(r.get('acc_br', np.nan)) else "  —  "
        print(f"  {r['method']:<25} {r['season']:<10} {r['auc']:>6.3f}  "
              f"{r['acc_50']:>9.3f}  {r['over_50']:>7.1%}  {acc_br_str:>10}  "
              f"{r['n_over_br']:>7}  {r['n_under_br']:>8}")

print(f"\n  Notes:")
print(f"  • AUC measures ranking quality (does p(over) rank games correctly?)")
print(f"  • Acc(0.5): accuracy when picking OVER if p>0.5, UNDER otherwise")
print(f"  • Acc(edge): accuracy on games where model exceeds base rate by 5pp")
print(f"  • N_over / N_under: how many bets each way (balance check)")
