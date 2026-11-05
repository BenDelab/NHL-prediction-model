# Predicting NHL Game Winners Without Data Leakage

This project asks a simple question: **once leakage is removed, can a convolutional neural network find patterns in a team's recent games that plain averages miss?**

**Not reliably.** On the untouched 2026-27 season so far (246 games), the CNN reached 61.0% accuracy and logistic regression on rolling averages reached 58.9%. Logistic regression had the slightly better log loss (0.6538 to 0.6554), and neither difference can be told apart from noise. Both beat always picking the home team (56.1%, log loss 0.6863).

The project is a from-scratch rebuild of an earlier system that had grown hard to trust. Its main contribution is the method: a pipeline that tests for leakage directly, scores every model on later seasons only, and reports ideas that didn't work.

**Status:** 2026-27 was held back during development and scored once, through November 3, 2026. With only 246 games, accuracy can be off by about 6 points either way, and the season is still being played.

## What changed from the old system

The old system covered moneylines, totals, spreads and player props, with a large set of hand-built features. An audit found these problems:

| Problem in the old system | Evidence | What the rebuild does |
|---|---|---|
| **Standings leaked the result.** A snapshot dated D already counted games played on D, and games on D could read it. | On those days (14.7% of games), AUC was 0.648 with the leak and 0.608 without it | Standings are gone. Elo and schedule context are computed strictly before each game |
| **A same-game stat was used as an input.** `pp_opportunity_rate` came from the game being predicted. | Small effect, but a true leak | Inputs come from one module, and tests check it |
| **Full-season percentiles were used to clip outliers.** | Negligible effect | Scaling is fitted on training seasons only |
| **Live inputs differed from training inputs.** Upcoming games lost the most recent game from their rolling xG and goalie stats. | Changed nearly all of the 104 games checked | Sequences are built the same way for every game |
| **"Rolling 5" meant the last 5 home games.** The source files only had home rows. | Every row in the season CSVs was a home game | MoneyPuck team logs cover home and away games |
| **Evaluation flattered the model.** Thresholds were tuned and scored on the same games, and the default backtest ran a different model from production. | Calibration on training games found T of about 1.0-1.1 when about 1.2-1.4 was needed | One walk-forward process produces every reported number |

Fixing the first three moved the old logistic regression from 57.4% to 56.9% accuracy, and from 0.6747 to 0.6771 log loss. The leak inflated results a little rather than creating them.

The rebuild also narrows the scope to **regular-season game winners**, however the game is decided. Player props, totals, spreads and separate overtime predictions were removed. The old code remains available in the git history.

## Data

| Source | Used for | Coverage |
|---|---|---|
| [MoneyPuck](https://moneypuck.com/data.htm) team game logs | All game stats and results | 21,853 regular-season games, 2008-09 to November 3, 2026 |
| [NHL API](https://api-web.nhle.com/v1/gamecenter/2023020100/boxscore) boxscores | Winners of tied games (`data/shootouts.csv`) | All 2,158 tied games |
| [NHL API](https://api-web.nhle.com/v1/gamecenter/2023020100/boxscore) boxscores | Official starting goalie (`data/starters.csv`) | 6,560 games, 2021-22 to 2025-26 |
| [MoneyPuck](https://moneypuck.com/data.htm) goalie game logs | Starting goalie's earlier record | 2021-22 to 2025-26 |

Game data is provided by MoneyPuck.com, whose terms allow free non-commercial use with credit.

What the audit found:

- **Regular season only:** digits 5-6 of each `gameId` identify the game type, and the loader keeps only regular-season games.
- **Shootouts:** MoneyPuck goal counts leave out the shootout, so its tied games have no winner. Those winners come from the NHL API. Shootouts aren't predicted separately; they count as ordinary wins and losses. One tied record from March 2020 isn't a real shootout and is excluded.
- **Mislabeled Arizona games:** MoneyPuck lists 106 Arizona home games from 2011 to 2014 with Arizona as its own opponent. The loader restores the real opponent, and its checks stop if any game is lost or has no winner.
- **Recording drift:** giveaways went from 7.7 to 15.0 per game and takeaways from 7.0 to 4.8 between 2023-24 and 2024-25. That looks like a recording change, so both are excluded.
- **Starters:** the "most ice time" goalie matched the official starter 96.9% of the time on the 4,954 games compared. It can name a reliever after a starter is pulled, which depends on how the game went, so official starters are used instead.
- **Era changes:** league scoring ranged from about 2.65 to 3.1 goals per team per game, and shootouts fell from about 13-15% of games to about 7-8% once 3-on-3 overtime arrived in 2015-16.
- **Not included:** injuries, lineups and betting odds.

## How the model works

```
home team's past games ──> shared CNN encoder ──> home encoding ─┐
                                                                 ├─> difference + game context ─> sigmoid
away team's past games ──> shared CNN encoder ──> away encoding ─┘
```

**Inputs.** Each team's sequence holds its games so far this season, oldest first. Each game has 17 stats: home or away, win, score-and-venue-adjusted xG for and against (all situations and 5-on-5), Corsi, power-play and penalty-kill xG, goals for and against, goalie saves above expected, finishing, penalties, opponent Elo, rest and travel. Ten pre-game context values are added, including Elo difference, rest, back-to-backs, travel and recent workload. Era scaling is an option that validation can choose: it expresses stats relative to the league's level that season, using earlier dates only. All inputs to both models are standardized (z-scores) using training games only.

**Network.** One encoder reads both teams, so both are judged by the same rules. Short kernels slide along the sequence, and two convolution layers feed average and latest-position pooling. The hidden layers use nonlinear GELU activations, with a sigmoid only at the output, so this is a genuinely different model from logistic regression rather than a more complicated version of it. Each reported prediction averages 5 training runs with different random seeds.

**Output.** One sigmoid gives the probability that the home team wins the game, whether it ends in regulation, overtime or a shootout.

An earlier version used three sigmoids: whether the game would go past regulation, the winner in regulation, and the winner past regulation. It couldn't tell which games would go past regulation (AUC 0.513, with log loss slightly worse than a flat rate), and it had no shown skill in shootouts (49.3% over 278 games). The extra outputs weren't earning their place, so the problem became a single win-or-lose prediction.

## How it was tested

**Leakage tests** (`tests/test_leakage.py`, all passing):

- Every result and stat from a chosen date onward is scrambled, and inputs for that date's games must not change.
- A deliberately leaky version must fail that check.
- Goalie stats and era scaling get the same treatment, and sequences may only hold earlier games from the same season.

**Sanity checks on the final one-output model:**

| Check | Accuracy on 2025-26 | Meaning |
|---|---|---|
| Deliberately feeding each game its own data | 100% | A leak would be obvious |
| Shuffling the results, four times | 51.1-54.1% (AUC averaging 0.50) | With nothing to learn, it has no real skill |

**Walk-forward testing.** Each season is scored on models that only saw earlier games. Settings are chosen using only the two seasons before the test season, which are averaged because one season is noisy:

| Choice | CNN | Logistic regression |
|---|---|---|
| Training start | 2008, 2013 or 2018 | Same |
| How fast old seasons fade (half-life) | 2, 4 or 8 seasons, or no fading | Same |
| Era scaling | On or off | Same |
| Model settings | 24 random setups of sequence length, kernels, filters, dropout, learning rate and weight decay | `C` |

For 2026-27, that means settings came from 2024-25 and 2025-26. Calibration comes from the season just before the test season. Both models then retrain every 30 days during the test season, using only games already played.

Differences between models use a 95% bootstrap range over game dates. Log loss is the main measure because it rewards honest probabilities, and a coin flip scores 0.6931.

## Results

### Final test: 2026-27

| Model (246 games, through November 3, 2026) | Accuracy | Log loss | AUC |
|---|---|---|---|
| Always pick home | 56.1% | 0.6863 | n/a |
| Logistic regression, original setup | 60.6% | 0.6463 | 0.653 |
| CNN, original setup | 61.4% | 0.6565 | 0.637 |
| **Logistic regression, tuned** | 58.9% | 0.6538 | 0.644 |
| **CNN, tuned** | 61.0% | 0.6554 | 0.646 |
| **Tuned models averaged** | 60.2% | 0.6540 | 0.647 |

These are the project's final results. The 95% log loss ranges are 0.635 to 0.676 for logistic regression, 0.639 to 0.675 for the CNN, and 0.637 to 0.674 for the average.

| Comparison | Log loss change | 95% range | Reading |
|---|---|---|---|
| CNN vs logistic regression, both tuned | +0.0016 | -0.0061 to +0.0095 | **No clear winner** |
| Tuned vs original, CNN | -0.0010 | -0.0111 to +0.0094 | No clear change |
| Tuned vs original, logistic regression | +0.0075 | -0.0029 to +0.0162 | No clear change |

The accuracy gap is five games: the CNN got 150 right and logistic regression got 145. These numbers are higher than in the development seasons, but they cover only the first weeks of one season, so that shouldn't be read as the model improving.

Validation chose a CNN starting in 2008 with no fading or era scaling, 20-game sequences and kernels of 3 + 5. Logistic regression chose a 2008 start with no fading, era scaling on and `C` of 0.0003. The CNN's training length came from its 2025-26 validation runs, which did best after one pass through the data, so the final CNN models each trained for only one pass. A later check on the validation seasons showed this cost very little (see below).

### Development seasons

These three seasons were examined many times while the project was built, so they may be slightly optimistic. They were scored with the three-output version described above, not the final one-output model.

| Model (3,936 games) | Accuracy | Log loss | AUC |
|---|---|---|---|
| Always pick home | 54.2% | 0.6897 | n/a |
| Logistic regression, original setup | 57.4% | 0.6703 | 0.608 |
| CNN, original setup | 57.7% | 0.6702 | 0.607 |
| **Logistic regression, tuned** | 57.6% | 0.6681 | 0.613 |
| **CNN, tuned** | 57.9% | 0.6680 | 0.613 |
| **Tuned models averaged** | 57.7% | 0.6675 | 0.614 |

The original setup trains on 2021-22 onward, with no era handling and no retraining. The always-home log loss uses the home win rate from earlier seasons.

| Comparison | Log loss change | 95% range | Reading |
|---|---|---|---|
| CNN vs logistic regression, both tuned | -0.0001 | -0.0022 to +0.0021 | **No clear winner** |
| Tuned vs original, CNN | -0.0022 | -0.0043 to -0.0001 | Clear improvement |
| Tuned vs original, logistic regression | -0.0022 | -0.0041 to -0.0003 | Clear improvement |
| Retrained vs frozen for the season, CNN | +0.0000 | -0.0005 to +0.0006 | No clear change |
| Retrained vs frozen for the season, logistic regression | -0.0004 | -0.0009 to +0.0001 | No clear change |
| Average vs tuned logistic regression | -0.0006 | -0.0017 to +0.0005 | No clear change |
| Average vs tuned CNN | -0.0005 | -0.0016 to +0.0005 | No clear change |

Tuning helped both models equally, so it didn't separate them.

#### What validation chose

| For test season | CNN start | CNN fading | CNN era scaling | CNN sequence and kernels | Logistic start | Logistic fading | Logistic era scaling |
|---|---|---|---|---|---|---|---|
| 2023-24 | 2008 | None | No | 20 games, 3 + 5 | 2013 | 4 seasons | Yes |
| 2024-25 | 2013 | 4 seasons | No | 40 games, 9 | 2008 | 4 seasons | No |
| 2025-26 | 2008 | None | No | 10 games, 9 | 2008 | None | No |

Long history was chosen every time: 2008 in four picks and 2013 in the other two. Era scaling was chosen once in six, so there's little evidence it helps here. The picks change from season to season, which suggests many setups are close to tied.

#### By season

| Test season | Always home | Logistic regression, tuned | CNN, tuned | Average |
|---|---|---|---|---|
| 2023-24 | 54.1% | 59.9% | 60.5% | 60.4% |
| 2024-25 | 56.3% | 58.2% | 58.6% | 58.6% |
| 2025-26 | 52.2% | 54.6% | 54.6% | 54.2% |

The CNN has the better accuracy in the first two seasons and ties in the third, and those gaps are within noise. On log loss, the CNN leads in 2023-24 (0.6561 to 0.6573), they match in 2024-25 (0.6629 to 0.6628), and logistic regression leads in 2025-26 (0.6841 to 0.6851). Every model is well below its earlier level in 2025-26, which is a reminder that a single season can mislead.

#### Confidence

Predicted probabilities roughly match reality, with a shortfall of about 3 points in two of the bands (averaged model):

| Favourite's predicted chance | Average prediction | Games | Actual win rate |
|---|---|---|---|
| 50-55% | 52.5% | 1,267 | 49.1% |
| 55-60% | 57.4% | 1,069 | 56.5% |
| 60-65% | 62.3% | 743 | 59.2% |
| 65%+ | 70.2% | 857 | 70.8% |
| **All picks at 60%+** | | **1,600** | **65.4%** |

### Comparison with Vegas

The target was to get close to sportsbooks, whose favourites are often said to win about 60% of the time. Published figures include [58.6% from 2013-14 to 2019-20](https://www.actionnetwork.com/nhl/nhl-favorites-2021-22-season-record) and [61.2% for home favourites midway through 2024-25](https://www.dailyfaceoff.com/news/2024-25-nhl-betting-trends). At about 58% in the development seasons and 59-61% on the small 2026-27 sample, the models may be within a few points, but **this is only a rough comparison** because those figures cover different games. Betting odds were deliberately left out of this project, so no direct comparison was made.


## Other questions tested

These separate tests ran on the development seasons with the earlier three-output CNN, before the final data corrections. Older shootout games weren't training examples in these runs.

**Are hockey trends short?** Not in this data. Longer windows predicted the next game better every time:

| Past games averaged (xG differential) | 1 | 3 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| Correlation with winning the next game | 0.078 | 0.100 | 0.115 | 0.130 | 0.152 | 0.160 |

Recent form still adds a little beyond the season average (a partial correlation of about 0.05 for the last 10 games), though that could be extra evidence of team quality rather than a true trend. Kernel size barely mattered in an initial search, where all sizes landed within about 0.003 in validation log loss. The final search chose kernels of 3 + 5 once and 9 twice across the three test seasons. Short kernels work, but they weren't shown to find anything averages miss.

**Would more past data help?** Yes, modestly, for both models. In this test, settings stayed fixed and training reached back as far as 2008, with the same test games each time.

| Training starts | Games available, through 2025-26 | Logistic regression | CNN | Average |
|---|---|---|---|---|
| 2021-22 | 6,554 | 57.3%, 0.6703 | 57.5%, 0.6704 | 57.5%, 0.6696 |
| 2018-19 | 9,774 | 57.3%, 0.6688 | 58.0%, 0.6693 | 57.5%, 0.6684 |
| 2013-14 | 15,924 | 57.6%, 0.6689 | 58.2%, 0.6681 | 57.8%, 0.6678 |
| 2008-09 | 21,494 | 58.1%, 0.6678 | 57.8%, 0.6680 | 57.9%, **0.6672** |

Each cell shows accuracy and log loss. These runs used 5 seeds instead of 10 and Elo carried over from 2008, so the 2021-22 row differs slightly from the main results.

Going from 2021-22 to 2008-09 improved log loss by about 0.0025 for all three models. The improvement is clear for the CNN (-0.0048 to -0.0002) and the average (-0.0048 to -0.0001), and just misses for logistic regression (-0.0054 to +0.0004). Extra data didn't separate the models: with everything back to 2008-09, the CNN and logistic regression are still tied (+0.0002, range -0.0021 to +0.0025). The tuned results above chose the training start on validation seasons, and always chose 2008 or 2013.

**Would more training passes help?** No. This check came after the final test and used only the 2024-25 and 2025-26 validation seasons, with the setup chosen for 2026-27 and three seeds. The final results weren't rerun.

| Training passes | 1 | 2 | 5 | 10 | 25 |
|---|---|---|---|---|---|
| Average validation log loss | 0.6722 | 0.6716 | 0.6724 | 0.6732 | 0.6747 |

At a slower learning rate (0.0003), the best point moved to 6 passes with about the same score (0.6714). Extra passes reuse the same games rather than adding new information, and past the first few the model starts fitting noise.

**Does knowing the starting goalie help?** Not overall. This test used the 2021-22 onward setup, without era handling or retraining. Adding each official starter's record from earlier games only made no reliable difference:

| Model | Accuracy without → with | Log loss without → with | Log loss change [95% range] |
|---|---|---|---|
| Logistic regression | 57.6% → 57.7% | 0.6709 → 0.6711 | +0.0002 [-0.0009, +0.0012] |
| CNN | 57.9% → 57.4% | 0.6700 → 0.6704 | +0.0004 [-0.0013, +0.0021] |
| Average | 57.7% → 57.8% | 0.6694 → 0.6698 | +0.0004 [-0.0009, +0.0016] |

There is one hint worth a later test. In the 1,109 games where a team started a goalie with under 30% of its last 10 starts, logistic regression improved by -0.0019 [-0.0037, -0.0000]. It got slightly worse in other games. This subset was picked after seeing the overall result and the effect is small, so it isn't evidence yet. Goalies are not part of the current best model.

## Limitations

- **Small final test.** 2026-27 is untouched but has only 246 games so far, and development seasons have been examined repeatedly.
- **Model changed before the final test.** The switch to one output was based on development results, and those seasons weren't rescored with it.
- **Limited search.** Only 24 CNN setups were tried, and each was scored on two validation seasons with two seeds, so a better setup could exist.
- **Calibration mismatch.** The calibration temperature comes from a model trained without the season just before the test season, then gets applied to models trained with it.
- **Possible leakage from outside the pipeline.** MoneyPuck's xG model may have been fitted using later seasons. This couldn't be verified or ruled out from the data.
- **Starter timing.** Starters are public about an hour before puck drop, so a goalie model could only predict after that point.
- **Missing information.** Injuries and lineups aren't included, and they may explain some of what the model can't.
- **Small margins.** With this many games, accuracy can be off by about 1.5 points either way, so most differences here are within noise.

## Running it

`data/` isn't in the repository, so download these from [MoneyPuck's data page](https://moneypuck.com/data.htm) first:

- The full team game-level file, saved as `data/moneypuck_all_teams.csv`
- The goalie game logs, saved as `data/goalie_stats/NHL_goalie_2021.csv` through `NHL_goalie_2025.csv` (only the goalie test uses these)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m nhl.fetch_shootouts         # shootout winners; only fetches missing games
.venv/bin/python -m nhl.fetch_starters          # official starters; only fetches missing games

.venv/bin/python -m pytest tests                # leakage tests
.venv/bin/python -m nhl.experiment              # CNN and baselines (about 30 minutes)
.venv/bin/python -m nhl.goalie_experiment       # with and without goalies (about 10 minutes)
.venv/bin/python -m nhl.history_experiment      # more past seasons (over an hour)
.venv/bin/python -m nhl.tuned_experiment        # final test on 2026-27 (about 45 minutes)
```

Results are written to `runs/`. `report_tuned.json` holds the final 2026-27 results, and `report_tuned_development.json` holds the three-output development results. The other experiments cover the finished seasons only.

## Project layout

```
nhl/
  data.py               load and check games and outcomes
  features.py           Elo, schedule and sequences; the only code that reads game history
  goalies.py            starting goalie's record before each game
  model.py              CNN and sigmoid outputs
  train.py              training, calibration and walk-forward prediction
  experiment.py         CNN vs baselines
  goalie_experiment.py  with and without goalies
  history_experiment.py  training on seasons back to 2008
  tuned_experiment.py   final comparison, tuned on validation seasons only
  fetch_starters.py     download official starters
  fetch_shootouts.py    download winners of tied games
tests/test_leakage.py   proves inputs can't see the current game or later ones
data/                   source data (not in the repository)
```
