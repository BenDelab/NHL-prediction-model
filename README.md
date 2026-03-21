# NHL Home Game Outcome Predictor

A machine learning pipeline that predicts whether the **home team will win or lose** an NHL game, using rolling team performance statistics and a sliding-window validation strategy.

The goal is practical: given what we know about both teams heading into a game, can we predict the outcome well enough to identify when a prediction is confident and correct?

---

## Data

### Source Files

Four cleaned CSV files, one per season, all sharing the same schema:

| File | Seasons | Role |
|------|---------|------|
| `NHL_2021-22_cleaned.csv` | 2021-22 | Training |
| `NHL_2022-23_cleaned.csv` | 2022-23 | Training |
| `NHL_2023-24_cleaned.csv` | 2023-24 | Training |
| `NHL_2024-25_cleaned.csv` | 2024-25 | Validation / Test |

Each row represents a **single home game** for one team. Every row contains that game's stats from the home team's perspective.

### Raw Columns (30 total)

| Column | Description |
|--------|-------------|
| `game_num` | Team's game number in the season (1–82) |
| `date` | Game date (YYYY-MM-DD) |
| `team` | Home team abbreviation (e.g., TOR, VEG) |
| `home_away` | Always "home" in this dataset |
| `opponent` | Away team abbreviation |
| `result` | **Target variable** — W (win) or L (loss) |
| `goals_for` / `goals_against` | Goals scored by home / away team |
| `overtime_type` | REG, OT, or SO |
| `team_sog` / `opp_sog` | Shots on goal |
| `team_pim` / `opp_pim` | Penalty minutes |
| `team_ppg` / `team_ppo` | Power play goals / opportunities |
| `team_shg` | Short-handed goals |
| `opp_ppg` / `opp_ppo` / `opp_shg` | Opponent's PP/SH stats |
| `faceoff_wins` / `faceoff_losses` / `faceoff_pct` | Faceoff stats |
| `corsi_for` / `corsi_against` / `corsi_pct` | Corsi (shot attempt) metrics |
| `fenwick_for` / `fenwick_against` / `fenwick_pct` | Fenwick (unblocked shot attempts) |
| `off_zone_start_pct` | Offensive zone faceoff start percentage |
| `pdo` | PDO (shooting% + save%) — a luck/regression indicator |

### Baseline

Home team win rate is ~53.7%, so a "always predict home win" baseline gives ~54% accuracy. The model must beat this convincingly.

---

## Project Structure

```
nhl-predictor/
│
├── data/
│   ├── NHL_2021-22_cleaned.csv
│   ├── NHL_2022-23_cleaned.csv
│   ├── NHL_2023-24_cleaned.csv
│   └── NHL_2024-25_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploration & understanding
│   ├── 02_feature_engineering.ipynb  # Build all features
│   ├── 03_model_training.ipynb       # Train & compare models
│   └── 04_evaluation.ipynb           # Sliding window validation
│
├── pipeline/
│   ├── __init__.py
│   ├── data_loader.py                # Load & concatenate season CSVs
│   ├── features.py                   # Feature engineering functions
│   ├── noise_reduction.py            # Smoothing & noise filtering
│   └── calibration.py                # Probability calibration
│
├── models/
│   ├── __init__.py
│   ├── train.py                      # Model training logic
│   └── evaluate.py                   # Evaluation & sliding window
│
├── main.py                           # End-to-end: load → predict → summarize
├── requirements.txt
└── README.md
```

### Design Philosophy

Each notebook is a **development workspace** — explore, iterate, visualize. Once logic is solid, the clean version of each function lives in `pipeline/` or `models/`. `main.py` imports from those modules and runs the full pipeline end-to-end with no notebook dependency.

---

## Pipeline Detail

### Phase 1 — EDA (`01_eda.ipynb`)

Understand the data before touching it.

- Distribution of each feature (histograms, box plots)
- Win/loss distributions: are certain stats strongly correlated with wins?
- Per-team trends across the season (do teams improve/decline?)
- Correlation matrix across all numeric features
- Check for missing values, outliers, and data quality issues
- PDO distribution (high PDO = lucky, expect regression — this is important context)

### Phase 2 — Feature Engineering (`02_feature_engineering.ipynb` → `pipeline/features.py`)

The raw data gives us single-game snapshots. The model needs **rolling context** — what has this team been doing recently? Features are computed per-team using only past games (no data leakage).

**Rolling Window Features (computed for each of N = 5, 10, 15, 20 recent games):**

- `rolling_win_pct` — win percentage over last N games
- `rolling_goals_for_avg` / `rolling_goals_against_avg` — scoring trends
- `rolling_sog_avg` / `rolling_opp_sog_avg` — shot volume trends
- `rolling_corsi_pct` / `rolling_fenwick_pct` — possession quality
- `rolling_pp_efficiency` — power play goals / opportunities over N games
- `rolling_pk_efficiency` — 1 - (opp power play goals / opp opportunities)
- `rolling_faceoff_pct` — faceoff win rate
- `rolling_pdo` — PDO trend (luck indicator: values far from 100 tend to regress)

**Derived Features:**

- `goal_diff_rolling` — rolling goal differential (goals for - goals against)
- `shot_diff_rolling` — rolling shot differential
- `pp_opportunity_rate` — how often the team gets PP chances (discipline proxy)
- `days_since_last_game` — rest advantage (computed from date column)
- `game_number_in_season` — proxy for fatigue / sample size at that point in the year

**Opponent Features:**

For each game, also compute the **opponent's** rolling stats (using the opponent's recent games). This gives the model both sides of the matchup.

**Head-to-Head Features (optional / stretch):**

- `h2h_win_pct` — historical win rate of home team vs this specific opponent over the dataset

**Important:** All rolling features use **only past data** for that team. Game 1 of a season will have NaN rolling features (handled via drop or imputation from the prior season's tail).

### Phase 3 — Noise Reduction (`pipeline/noise_reduction.py`)

NHL is a high-variance sport. Single-game stats are noisy. Noise reduction is applied **gently** — the goal is smoothing, not aggressive filtering.

- **Rolling averages are the primary noise reduction** — already handled in feature engineering. Using multiple window sizes (5, 10, 15, 20) captures both short-term form and longer trends.
- **PDO regression flag** — flag games where PDO is extreme (>105 or <95 over a window). This doesn't remove data, but adds a feature the model can learn from.
- **Outlier softening** — for features with extreme single-game values (e.g., 12-goal games, 90%+ faceoff pct), apply winsorization at the 1st/99th percentile on the rolling averages. This prevents one fluky game from distorting the window.
- **No rows are deleted** — noise reduction produces features or transforms values, it does not discard games.

### Phase 4 — Model Training (`03_model_training.ipynb` → `models/train.py`)

**Target Variable:** `result` encoded as binary (1 = home win, 0 = home loss).

**Models to train and compare:**

| Model | Why |
|-------|-----|
| **Logistic Regression** | Interpretable baseline; good at showing which features matter linearly |
| **Random Forest** | Handles non-linear interactions, robust to noise, built-in feature importance |
| **XGBoost** | Gradient boosting typically gives best performance on tabular data, handles class imbalance well |

**Training Data:** 2021-22 through 2023-24 seasons (after feature engineering).

**Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` with time-aware cross-validation (no shuffling — respect chronological order). Key hyperparameters to tune per model:

- Logistic Regression: regularization strength (C), penalty type
- Random Forest: n_estimators, max_depth, min_samples_leaf
- XGBoost: learning_rate, max_depth, n_estimators, subsample, colsample_bytree

### Phase 5 — Calibration (`pipeline/calibration.py`)

Raw model probabilities are often poorly calibrated (e.g., when the model says "70% chance of a home win," it should actually win ~70% of the time). This matters for betting applications.

- **Platt scaling** (sigmoid) — fits a logistic regression on top of model outputs
- **Isotonic regression** — non-parametric calibration, more flexible
- Compare calibration curves (reliability diagrams) before and after calibration
- Calibrate on a held-out portion of the training data (not the test set)

### Phase 6 — Evaluation & Sliding Window (`04_evaluation.ipynb` → `models/evaluate.py`)

This is where the project validates whether the model actually works in a realistic scenario.

**Sliding Window Validation:**

The model is tested on the **2024-25 season** using a sliding window approach. For a given game, the model only sees the last N games of rolling features to make its prediction.

```
Example: Predicting Game 21 for Toronto

  N=5:   Uses rolling features from games 16-20
  N=10:  Uses rolling features from games 11-20
  N=15:  Uses rolling features from games 6-20
  N=20:  Uses rolling features from games 1-20
```

For **each value of N** (5, 10, 15, 20):
- Walk through every eligible game in the 2024-25 season
- Generate the rolling features using only the last N games
- Predict win/loss using each trained model
- Record whether the prediction was correct

**Evaluation Metrics (per model, per N):**

- **Accuracy** — overall correct prediction rate (must beat 54% baseline)
- **Precision / Recall / F1** — especially important if predictions are asymmetric
- **Brier Score** — measures probability calibration quality
- **Log Loss** — penalizes confident wrong predictions heavily
- **ROC-AUC** — overall discriminative ability

**Output: Best N Comparison Table**

```
| Model              | N=5   | N=10  | N=15  | N=20  | Best N |
|--------------------|-------|-------|-------|-------|--------|
| Logistic Regression| 0.58  | 0.61  | 0.59  | 0.57  | 10     |
| Random Forest      | 0.60  | 0.63  | 0.62  | 0.60  | 10     |
| XGBoost            | 0.61  | 0.64  | 0.63  | 0.61  | 10     |
```
*(Example values — actual results will vary)*

---

## main.py

The final script ties everything together. It takes no notebook dependency — it imports from `pipeline/` and `models/` and runs the full flow.

**What it does:**

1. Loads all season CSVs via `pipeline.data_loader`
2. Runs feature engineering via `pipeline.features`
3. Applies noise reduction via `pipeline.noise_reduction`
4. Trains models on 2021-22 to 2023-24 via `models.train`
5. Calibrates probabilities via `pipeline.calibration`
6. Runs sliding window evaluation on 2024-25 via `models.evaluate`
7. Prints a summary: per-model, per-N accuracy table + best configuration
8. For a specific game, prints a prediction like:

```
PREDICTION — Game 5: VEG vs SEA (2024-10-18)
  Model: XGBoost | Window: N=10
  Prediction: VEG WIN (home) | Confidence: 64.2%
  Actual Result: W ✓
```

**Usage:**

```bash
python main.py
```

Or to predict a specific matchup at a specific game number:

```bash
python main.py --team VEG --game 5
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyter
```

---

## Key Design Decisions

1. **Home-only perspective.** The dataset only contains home games, so every prediction is "will the home team win?" This simplifies the problem to a clean binary classification.

2. **Rolling windows as the core feature strategy.** Single-game NHL stats are extremely noisy. Rolling averages over recent games are the most natural and effective way to extract signal.

3. **Multiple window sizes tested, not assumed.** Instead of guessing whether 5 or 20 games is the right lookback, the sliding window evaluation empirically determines the best N for each model.

4. **Noise reduction as feature transformation, not data removal.** No games are thrown away. Noise is handled through smoothing (rolling averages) and softening (winsorization), preserving sample size.

5. **Calibration as a separate step.** Because the end goal involves interpreting probabilities (for betting value), the model's raw outputs are calibrated so that predicted probabilities are meaningful.

6. **Notebooks for development, Python files for production.** Notebooks let you explore and visualize. The clean logic lives in importable modules. `main.py` proves the pipeline works end-to-end without Jupyter.
