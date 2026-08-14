"""
Fit Dixon-Coles on real Premier League data.

Data source: football-data.co.uk (free, no API key needed).
Requires dixon_coles_sketch.py to be in the same folder.

Usage:
    python3 fit_premier_league.py
"""

import pandas as pd
import numpy as np

from dixon_coles_sketch import fit_league, score_matrix, derive_markets

# 2025/26 season (last completed season) — E0 = Premier League
DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

# This season's promoted trio — no PL history yet, model can't rate them
# until fixture data exists. Flagged here rather than silently dropped.
PROMOTED_NO_HISTORY = {"Hull", "Coventry", "Ipswich"}


def load_fixtures():
    df = pd.read_csv(DATA_URL)
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    fixtures = list(df.itertuples(index=False, name=None))
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    return fixtures, teams


def report_ratings(model, teams):
    print("\nTeam ratings (higher attack = better, lower defence = better):")
    print(f"{'Team':20s} {'Attack':>8s} {'Defence':>8s}")
    for t in sorted(teams, key=lambda x: -model["attack"][x]):
        print(f"{t:20s} {model['attack'][t]:8.3f} {model['defence'][t]:8.3f}")

    print(f"\nHome advantage: {model['home_adv']:.3f}")
    print(f"Rho (low-score correlation): {model['rho']:.3f}"
          f"  {'-- check this against Dixon-Coles literature (~-0.10 to -0.15)' if not (-0.20 < model['rho'] < -0.05) else '(looks in expected range)'}")


def forecast_fixture(model, home, away):
    if home not in model["attack"] or away not in model["attack"]:
        missing = [t for t in (home, away) if t not in model["attack"]]
        print(f"\nCan't forecast {home} vs {away} — no fitted rating for: {missing}")
        if any(t in PROMOTED_NO_HISTORY for t in missing):
            print("(expected — this team is newly promoted with no top-flight "
                  "history yet; needs Championship-seeded ratings for now)")
        return

    grid = score_matrix(home, away, model)
    markets = derive_markets(grid)

    print(f"\n{home} vs {away}:")
    m = markets["1X2"]
    print(f"  1X2 — {home} win {m['home']:.1%} | Draw {m['draw']:.1%} | {away} win {m['away']:.1%}")
    ou = markets["O/U 2.5"]
    print(f"  O/U 2.5 — Over {ou['over']:.1%} | Under {ou['under']:.1%}")
    b = markets["BTTS"]
    print(f"  BTTS — Yes {b['yes']:.1%} | No {b['no']:.1%}")

    top_scores = np.dstack(np.unravel_index(
        np.argsort(-grid.ravel())[:5], grid.shape))[0]
    print("  Most likely scorelines:")
    for hg, ag in top_scores:
        print(f"    {hg}-{ag}: {grid[hg, ag]:.1%}")


if __name__ == "__main__":
    print("Fetching last season's results...")
    fixtures, teams = load_fixtures()
    print(f"Loaded {len(fixtures)} fixtures across {len(teams)} teams.")

    print("Fitting Dixon-Coles model...")
    model = fit_league(fixtures, teams)

    report_ratings(model, teams)

    # Sanity-check forecasts — swap these for real upcoming fixtures once
    # the new season's schedule is out
    forecast_fixture(model, "Arsenal", "Man City")
    forecast_fixture(model, "Liverpool", "Chelsea")
    forecast_fixture(model, "Arsenal", "Hull")  # will flag as unratable
