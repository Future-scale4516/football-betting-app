"""
Odds coverage check — run this once to see, per league:
  - how many bookmakers are quoting live events
  - which markets (1X2/totals/handicap) actually return data

Usage:
    python odds_coverage_check.py YOUR_API_KEY
"""

import sys
import requests

API_KEY = sys.argv[1] if len(sys.argv) > 1 else "PASTE_YOUR_KEY_HERE"
BASE_URL = "https://api.the-odds-api.com/v4"

# Candidate league keys — confirm exact spelling against /v4/sports first,
# these are the odds-api's typical naming but can drift.
LEAGUES = [
    "soccer_epl",
    "soccer_efl_champ",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_spain_la_liga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
]

MARKETS = "h2h,totals,spreads"  # 1X2, over/under, handicap


def check_league(sport_key: str) -> None:
    url = f"{BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": API_KEY,
        "regions": "uk,eu",
        "markets": MARKETS,
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=15)

    if resp.status_code != 200:
        print(f"{sport_key:30s} FAILED  ({resp.status_code}: {resp.text[:80]})")
        return

    events = resp.json()
    if not events:
        print(f"{sport_key:30s} 0 events returned (out of season / no data)")
        return

    # Use the first event as a representative sample
    sample = events[0]
    bookmakers = sample.get("bookmakers", [])
    markets_seen = set()
    for bk in bookmakers:
        for m in bk.get("markets", []):
            markets_seen.add(m["key"])

    print(f"{sport_key:30s} {len(events):3d} events | "
          f"{len(bookmakers):2d} bookmakers on sample event | "
          f"markets: {', '.join(sorted(markets_seen)) or 'none'}")

    remaining = resp.headers.get("x-requests-remaining")
    if remaining:
        print(f"{'':30s} (requests remaining this period: {remaining})")


if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("Pass your API key as an argument: python odds_coverage_check.py YOUR_KEY")
        sys.exit(1)

    print(f"{'LEAGUE':30s} COVERAGE\n" + "-" * 70)
    for league in LEAGUES:
        check_league(league)
