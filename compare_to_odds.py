"""
Compare model probability vs de-vigged market probability — the actual
edge calculation this whole app exists for.

Requires dixon_coles_sketch.py in the same folder.

Usage:
    python3 compare_to_odds.py YOUR_ODDS_API_KEY
"""

import sys
import requests
import pandas as pd

from dixon_coles_sketch import fit_league, score_matrix, derive_markets

ODDS_API_KEY = sys.argv[1] if len(sys.argv) > 1 else None
ODDS_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
TRAINING_DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

# Odds API team name -> football-data.co.uk team name.
# Add to this as mismatches show up — the script will print any it can't match.
TEAM_NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Hull City": "Hull",
    "Coventry City": "Coventry",
    # add more here if the script flags an unmatched fixture below
}

PROMOTED_NO_HISTORY = {"Hull", "Coventry", "Ipswich"}


def normalise_name(odds_api_name: str) -> str:
    return TEAM_NAME_MAP.get(odds_api_name, odds_api_name)


def fit_model():
    df = pd.read_csv(TRAINING_DATA_URL)
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    fixtures = list(df.itertuples(index=False, name=None))
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    return fit_league(fixtures, teams)


def fetch_odds():
    if not ODDS_API_KEY:
        print("Pass your Odds API key: python3 compare_to_odds.py YOUR_KEY")
        sys.exit(1)

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    resp = requests.get(ODDS_URL, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"Odds API error {resp.status_code}: {resp.text[:200]}")
        sys.exit(1)
    return resp.json()


def devig(odds_home, odds_draw, odds_away):
    """Remove the bookmaker's margin to get true implied probability."""
    imp_home, imp_draw, imp_away = 1 / odds_home, 1 / odds_draw, 1 / odds_away
    overround = imp_home + imp_draw + imp_away
    return imp_home / overround, imp_draw / overround, imp_away / overround


def average_market_odds(event):
    """Average decimal odds across every bookmaker quoting this event."""
    home_odds, draw_odds, away_odds = [], [], []
    home_team, away_team = event["home_team"], event["away_team"]

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market["key"] != "h2h":
                continue
            for outcome in market["outcomes"]:
                if outcome["name"] == home_team:
                    home_odds.append(outcome["price"])
                elif outcome["name"] == away_team:
                    away_odds.append(outcome["price"])
                elif outcome["name"] == "Draw":
                    draw_odds.append(outcome["price"])

    if not (home_odds and draw_odds and away_odds):
        return None
    return (sum(home_odds) / len(home_odds),
            sum(draw_odds) / len(draw_odds),
            sum(away_odds) / len(away_odds),
            len(home_odds))  # bookmaker count, for confidence


def main():
    print("Fitting model on last season's results...")
    model = fit_model()

    print("Fetching live odds...\n")
    events = fetch_odds()

    print(f"{'Fixture':40s} {'Model H/D/A':22s} {'Market H/D/A':22s} {'Edge (H)':>9s}")
    print("-" * 100)

    for event in events:
        home_raw, away_raw = event["home_team"], event["away_team"]
        home, away = normalise_name(home_raw), normalise_name(away_raw)

        if home not in model["attack"] or away not in model["attack"]:
            missing = [t for t in (home, away) if t not in model["attack"]]
            reason = "promoted, no rating yet" if any(
                t in PROMOTED_NO_HISTORY for t in missing) else "name mismatch — check TEAM_NAME_MAP"
            print(f"{home_raw} vs {away_raw:30s} SKIPPED ({reason}: {missing})")
            continue

        market = average_market_odds(event)
        if market is None:
            print(f"{home_raw} vs {away_raw:30s} SKIPPED (no complete odds set)")
            continue
        odds_h, odds_d, odds_a, n_books = market
        mkt_h, mkt_d, mkt_a = devig(odds_h, odds_d, odds_a)

        grid = score_matrix(home, away, model)
        m = derive_markets(grid)["1X2"]

        edge_h = m["home"] - mkt_h
        fixture_label = f"{home} vs {away}"
        model_str = f"{m['home']:.0%}/{m['draw']:.0%}/{m['away']:.0%}"
        market_str = f"{mkt_h:.0%}/{mkt_d:.0%}/{mkt_a:.0%} ({n_books}bk)"
        print(f"{fixture_label:40s} {model_str:22s} {market_str:22s} {edge_h:+8.1%}")


if __name__ == "__main__":
    main()
