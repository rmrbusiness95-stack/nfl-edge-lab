"""NFL Edge Lab: fetch current NFL odds from The Odds API.

An optional, standalone data-collection module. It doesn't modify model
predictions or approve picks. Configure ODDS_API_KEY as a GitHub Actions secret
before running it; never commit the key into source control.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

ENDPOINT = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
OUTPUT = Path("data/nfl_odds_latest.json")


def main():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("No ODDS_API_KEY configured; skipped without network request.")
        return

    response = requests.get(
        ENDPOINT,
        params={
            "apiKey": key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
        timeout=45,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Odds provider returned HTTP {response.status_code}")
    events = response.json()
    if not isinstance(events, list):
        raise ValueError("Invalid odds response: expected event list")

    output = []
    for event in events:
        books = []
        for book in event.get("bookmakers", []):
            markets = []
            for market in book.get("markets", []):
                if market.get("key") not in ("h2h", "spreads", "totals"):
                    continue
                markets.append({
                    "key": market.get("key"),
                    "last_update": market.get("last_update"),
                    "outcomes": [{
                        "name": outcome.get("name"),
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                    } for outcome in market.get("outcomes", [])],
                })
            if markets:
                books.append({
                    "key": book.get("key"),
                    "title": book.get("title"),
                    "last_update": book.get("last_update"),
                    "markets": markets,
                })
        output.append({
            "id": event.get("id"),
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "bookmakers": books,
        })

    payload = {
        "source": "The Odds API v4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "games": output,
        "approved_picks": [],
        "note": "Market odds snapshot only; no validated edge or sharp-money claims.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)
    print(f"Saved odds for {len(output)} NFL games")
    print("Provider requests remaining:", response.headers.get("x-requests-remaining", "unknown"))


if __name__ == "__main__":
    main()
