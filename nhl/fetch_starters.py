"""Save the official starting goalie for every game to data/starters.csv.

Run with: python -m nhl.fetch_starters
Safe to re-run: games already saved are skipped.
"""
import csv
import time

import requests

from .data import ROOT, TEAM_MAP, load_games

URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
OUT = ROOT / "data" / "starters.csv"
FIELDS = ["gameId", "team", "is_home", "goalie_id", "goalie_name", "goalies_listed", "starters_flagged"]
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


def rows_for(game_id: int, box: dict) -> list[dict]:
    rows = []
    for key, is_home in (("homeTeam", 1), ("awayTeam", 0)):
        goalies = box["playerByGameStats"][key]["goalies"]
        flagged = [g for g in goalies if g.get("starter")]
        starter = flagged[0] if len(flagged) == 1 else None
        abbrev = box[key]["abbrev"]
        rows.append({
            "gameId": game_id,
            "team": TEAM_MAP.get(abbrev, abbrev),
            "is_home": is_home,
            "goalie_id": starter["playerId"] if starter else "",
            "goalie_name": starter["name"]["default"] if starter else "",
            "goalies_listed": len(goalies),
            "starters_flagged": len(flagged),
        })
    return rows


def main() -> None:
    games, _ = load_games()
    done = set()
    if OUT.exists():
        with OUT.open() as handle:
            done = {int(row["gameId"]) for row in csv.DictReader(handle)}
    todo = [g for g in games["gameId"].tolist() if g not in done]
    print(f"{len(done)} saved, {len(todo)} to fetch", flush=True)

    is_new = not OUT.exists()
    with OUT.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        if is_new:
            writer.writeheader()
        for count, game_id in enumerate(todo, 1):
            writer.writerows(rows_for(game_id, fetch(game_id)))
            handle.flush()
            if count % 500 == 0:
                print(f"{count}/{len(todo)}", flush=True)
            time.sleep(PAUSE_SECONDS)
    print("done", flush=True)


if __name__ == "__main__":
    main()
