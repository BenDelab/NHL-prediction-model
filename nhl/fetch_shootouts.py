"""Save the winner of every shootout game to data/shootouts.csv.

MoneyPuck's goal counts leave out the shootout, so its tied games get their result
from the NHL API. Run with: python -m nhl.fetch_shootouts
Safe to re-run: games already saved are skipped.
"""
import csv
import time

import pandas as pd
import requests

from .data import LAST_YEAR, MONEYPUCK_FILE, REGULAR_SEASON, ROOT

URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
OUT = ROOT / "data" / "shootouts.csv"
FIELDS = ["gameId", "home_score", "away_score", "last_period"]
PAUSE_SECONDS = 0.1
RETRIES = 4


def fetch(game_id: int) -> dict:
    for attempt in range(RETRIES):
        try:
            response = requests.get(URL.format(game_id=game_id), timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def tied_games() -> list[int]:
    raw = pd.read_csv(MONEYPUCK_FILE, usecols=["season", "gameId", "situation", "goalsFor", "goalsAgainst"])
    keep = ((raw["situation"] == "all") & (raw["season"] <= LAST_YEAR)
            & (raw["gameId"].astype(str).str[4:6] == REGULAR_SEASON) & (raw["goalsFor"] == raw["goalsAgainst"]))
    return sorted(raw.loc[keep, "gameId"].unique().tolist())


def main() -> None:
    done = set()
    if OUT.exists():
        with OUT.open() as handle:
            done = {int(row["gameId"]) for row in csv.DictReader(handle)}
    todo = [g for g in tied_games() if g not in done]
    print(f"{len(done)} saved, {len(todo)} to fetch", flush=True)
    is_new = not OUT.exists()
    with OUT.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        if is_new:
            writer.writeheader()
        for count, game_id in enumerate(todo, 1):
            box = fetch(game_id)
            writer.writerow({"gameId": game_id, "home_score": box["homeTeam"]["score"],
                             "away_score": box["awayTeam"]["score"],
                             "last_period": box["gameOutcome"]["lastPeriodType"]})
            handle.flush()
            if count % 250 == 0:
                print(f"{count}/{len(todo)}", flush=True)
            time.sleep(PAUSE_SECONDS)
    print("done", flush=True)


if __name__ == "__main__":
    main()
