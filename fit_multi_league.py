"""
Fit Dixon-Coles across multiple English divisions.

Data source: football-data.co.uk (free, no API key needed).
Requires dixon_coles_sketch.py in the same folder.

IMPORTANT — League One and League Two have no odds coverage from
The Odds API, so these are FORECAST-ONLY: you get team ratings and
match probabilities, but no edge calculation, no traffic-light colour,
no CLV tracking. That layer only exists where there's a market price
to compare against. Premier League is the only one of the three
that's launch-ready for the full betting-edge pipeline right now.

Usage:
    python3 fit_multi_league.py
"""

import pandas as pd
from dixon_coles_sketch import fit_league, score_matrix, derive_markets

# football-data.co.uk division codes
LEAGUES = {
    "Premier League": {"code": "E0", "has_odds": True},
    "League One":     {"code": "E2", "has_odds": False},
    "League Two":     {"code": "E3", "has_odds": False},
}

SEASON = "2526"  # last completed season (2025/26)


def load_fixtures(division_code: str):
    url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{division_code}.csv"
    df = pd.read_csv(url)
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    fixtures = list(df.itertuples(index=False, name=None))
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    return fixtures, teams


def report_top_bottom(model, teams, league_name, has_odds, n=5):
    ranked = sorted(teams, key=lambda t: -model["attack"][t])
    tag = "" if has_odds else "  [FORECAST ONLY — no odds coverage, no edge/CLV layer]"
    print(f"\n=== {league_name}{tag} ===")
    print(f"Home advantage: {model['home_adv']:.3f} | Rho: {model['rho']:.3f}")
    print(f"Top {n} attack ratings: {', '.join(ranked[:n])}")
    print(f"Bottom {n} attack ratings: {', '.join(ranked[-n:])}")


def fit_all():
    fitted = {}
    for name, info in LEAGUES.items():
        print(f"Fetching and fitting {name}...")
        fixtures, teams = load_fixtures(info["code"])
        model = fit_league(fixtures, teams)
        fitted[name] = {"model": model, "teams": teams, "has_odds": info["has_odds"]}
        report_top_bottom(model, teams, name, info["has_odds"])
    return fitted


if __name__ == "__main__":
    results = fit_all()

    print("\n" + "=" * 60)
    print("Summary: only Premier League feeds the betting-edge pipeline.")
    print("League One/Two ratings above are for forecast reference —")
    print("useful for spotting form/strength trends, not for staking,")
    print("since there's no market price to calculate edge against.")
