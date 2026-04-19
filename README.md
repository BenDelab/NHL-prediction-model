# NHL Prediction System

A full-stack NHL prediction pipeline covering **moneyline outcomes**, **game totals (over/under goals)**, and **player shot-on-goal props**. The system is built for practical edge identification against sportsbook lines — not academic accuracy metrics.

Realistic performance ceiling: ~55–58% on moneylines (same range as Vegas models), 65–68% hit rate on player props at conviction ≥ 0.12.

---

## What It Does

### 1. Moneyline Prediction
Predicts whether the home team wins each NHL game. Uses an ensemble of XGBoost, LightGBM, and Logistic Regression models trained on rolling team performance features. Outputs calibrated win probabilities to compare against implied market odds and identify positive-EV bets.

### 2. Totals (Over/Under Goals)
Predicts the combined goal total for each game using a Poisson-based approach with xGoals data. Evaluates lines at 5.5, 6.5, and 7.5 goals — selecting the line in the most contested probability range rather than always picking the same line.

### 3. Player Props (Shots on Goal)
Predicts each skater's expected shots on goal using an **XGBoost count regressor** (`objective='count:poisson'`). Converts predicted μ to over/under probabilities via a **Negative Binomial CDF** (handles the overdispersion SOG data exhibits — var/mean ≈ 1.42). Evaluates lines at 1.5 / 2.5 / 3.5 / 4.5 SOG, selecting the competitive-zone line where the model has the most edge.

---

## Architecture

```
scrape/
│
├── data/                          # Cleaned season CSVs (2021-22 → 2025-26)
├── xG/                            # Per-team xGoals CSVs by season
├── player_stats/                  # Per-player game logs (MoneyPuck)
├── goalie_stats/                  # Per-goalie game logs (MoneyPuck)
│
├── pipeline/
│   ├── data_loader.py             # Load & merge season CSVs
│   ├── features.py                # Rolling team features (win%, GF/GA, Corsi, Fenwick…)
│   ├── noise_reduction.py         # PDO regression flags, winsorization
│   ├── calibration.py             # Platt/isotonic calibration of ensemble probabilities
│   ├── xg_loader.py               # Load xGoals data per team
│   ├── goalie_loader.py           # Load goalie stats for team-level save quality
│   ├── player_loader.py           # Load skater game logs + rolling features + goalie context
│   ├── standings.py               # NHL standings scraper → playoff motivation features
│   └── betting.py                 # Edge calculation, de-juicing, Kelly sizing
│
├── models/
│   ├── train.py                   # Moneyline ensemble training (XGB + LGBM + LR)
│   ├── evaluate.py                # Sliding-window validation, feature column logic
│   ├── secondary.py               # Secondary model layer (temporal stacking)
│   ├── totals.py                  # Poisson GLM for total goals, line selection
│   └── props.py                   # XGBoost count regressor + NB CDF for SOG props
│
├── main.py                        # End-to-end: train → validate → predict tonight's games
├── backtest.py                    # Walk-forward backtest with ROI simulation
└── requirements.txt
```

---

## Models in Detail

### Moneyline — Ensemble Classifier

**Features** (rolling windows of 5 / 10 / 15 / 20 games, computed per team with no leakage):
- Win %, goals for/against, shot differentials (Corsi, Fenwick, SOG)
- Power play / penalty kill efficiency, faceoff rate
- PDO (shooting% + save% — a mean-reversion indicator)
- xGoals for/against (from MoneyPuck)
- Goalie rolling save quality (GSAx, HD-SV%)
- Opponent rolling stats (both sides of every matchup)
- Playoff motivation score (from standings — clinch status, points above cut, games remaining)
- Rest days, season game number, home/away

**Training:** Seasons 2021-22 → 2024-25 with recency weighting (newer seasons weighted up to 2×). Early-season games ramped from 0 → 1 weight over first 15 games.

**Ensemble:** XGBoost + LightGBM + Logistic Regression per rolling window; secondary stacking layer combines window-model pairs. Calibration via isotonic regression.

**Backtest results (walk-forward, held-out seasons):**
- Overall accuracy: 55–58%
- High-confidence bets (calibrated p ≥ 0.55, edge ≥ 0.06): 63–67% hit rate
- Half-Kelly staking on high-confidence picks produces positive ROI at realistic moneyline odds

### Totals — Poisson GLM

Predicts combined goals per game using a log-linear Poisson regression on team-level xGoals, defensive quality, and pace features. Evaluates P(over 5.5), P(over 6.5), P(over 7.5) and selects the line where the model has the most edge away from 50/50.

### Player Props — XGBoost + Negative Binomial CDF

**Why XGBoost over Negative Binomial GLM:**
NB GLM is log-linear — it can't capture interactions like "high-fenwick forward vs weak goalie → disproportionately more SOG." XGBoost learns these non-linear combinations directly.

**Why NB CDF over Poisson CDF:**
SOG is overdispersed (var/mean ≈ 1.42). Poisson assumes var = mean, systematically underestimating tail probabilities. After XGBoost predicts μ, the NB dispersion α is estimated from training residuals (method of moments), and P(SOG > line) = 1 − NB_CDF(floor(line), n=1/α, p=1/(1+α·μ)) is applied analytically. This guarantees P(>1.5) ≥ P(>2.5) ≥ P(>3.5) ≥ P(>4.5) always.

**Features:**
- Per-player rolling SOG mean/std, icetime, xGoals, high-danger shots, Fenwick, ozone start rate (windows 5 + 10)
- Opponent team shots-allowed per game (rolling 10)
- Opponent goalie rolling GSAx and HD save % (rolling 10) — *the key addition that enables per-goalie adjustment beyond team-level defense*
- Home/away flag

**Competitive-zone line selection:** Only considers lines where the model's P(over) ∈ [0.30, 0.70]. Picks the line with the highest |P − 0.50|. Identical logic to how the totals model picks 5.5 vs 6.5 vs 7.5.

**Backtest results (trained 2021-24, tested 2024-25, walk-forward):**
- AUC per line: 0.707 (o1.5) → 0.737 (o2.5) → 0.769 (o3.5) → 0.803 (o4.5)
- Hit rate at conviction ≥ 0.12: **67.1%** (break-even at −110 odds = 52.4%)
- ROI at conviction ≥ 0.15: **+30.6%** at flat $10 stake

---

## Usage

### Predict tonight's games

```bash
python main.py --predict
```

Trains on all available seasons, then outputs moneyline predictions, totals, and player props for the next scheduled games. Includes calibrated probabilities, implied edge vs market, and Kelly-sized bet recommendations.

```bash
python main.py --predict --date 2026-03-25
```

Predict for a specific date.

```bash
python main.py --predict --props
```

Include player SOG prop predictions alongside moneyline/totals.

### Backtest historical performance

```bash
python backtest.py
```

Runs two walk-forward rounds:
- Train 2021-23 → test 2023-24
- Train 2021-24 → test 2024-25

Outputs per-model accuracy, high-confidence hit rate, cumulative ROI by month, and player props backtest results.

```bash
python backtest.py --quick          # Logistic Regression only (fast)
python backtest.py --no-props       # Skip player props backtest
```

---

## Data Sources

| Data | Source | Location |
|------|--------|----------|
| Team game logs | MoneyPuck (`moneypuck.com`) | `data/` |
| Per-team xGoals | MoneyPuck | `xG/` |
| Skater game logs | MoneyPuck | `player_stats/` |
| Goalie game logs | MoneyPuck | `goalie_stats/` |
| NHL standings | NHL API (`api-web.nhle.com`) | `data/standings/` |

Team abbreviations follow MoneyPuck conventions: Arizona → `UTA` (Utah), Vegas → `VEG`.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
statsmodels
scipy
requests
matplotlib
seaborn
jupyter
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Key Design Decisions

**Rolling features, not single-game stats.** NHL is the highest-variance major sport (games decided by 1–2 lucky bounces). Single-game stats are noise. Rolling windows of 5–20 games extract the underlying signal.

**Competitive-zone line selection.** For totals and props, the system doesn't pick the "easiest" line (e.g. always recommend UNDER 4.5 SOG at 95% confidence — that's not an edge, that's a tautology). It finds the line closest to 50/50 where the model disagrees with the market.

**Goalie quality is opponent context, not team context.** `opp_roll10_soa` captures team-level shot suppression; `opp_goalie_roll10_gsax` captures the *goalie's* quality independently. A leaky defensive team with an elite goalie (e.g. early-season Carolina) would mislead a model that only uses team SOA.

**Dispersion matters for count data.** SOG follows a Negative Binomial distribution, not Poisson. Using a Poisson CDF underestimates the probability of extreme outcomes (0 SOG or 6+ SOG), which is where prop betting edge lives.

**Calibration before edge calculation.** Raw model probabilities are not reliable for EV calculations. The pipeline applies isotonic calibration on held-out data before comparing any predicted probability to implied market odds.

**Walk-forward validation only.** No cross-validation with shuffling — that would leak future information into past predictions. Every validation result in this project uses training data that precedes the test period in calendar time.
