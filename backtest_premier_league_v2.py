"""
Rolling backtest v2 — using time-decay weighted fitting.

Same rolling logic as backtest_premier_league.py, but each training
fit now down-weights older matches instead of treating all matches
in the window equally. Compare this output directly against the
v1 backtest results to see whether recency weighting fixes the
calibration problem.

Requires dixon_coles_sketch.py and dixon_coles_weighted.py in the
same folder.

Usage:
    python3 backtest_premier_league_v2.py
"""

import numpy as np
import pandas as pd
from dixon_coles_sketch import score_matrix, derive_markets
from dixon_coles_weighted import fit_league_weighted

DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
ROUND_SIZE = 10
MIN_TRAINING_ROUNDS = 10
XI = 0.0018  # recency decay rate — see dixon_coles_weighted.py for what this means


def load_and_sort():
    df = pd.read_csv(DATA_URL, parse_dates=["Date"], dayfirst=True)
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    return df.sort_values("Date").reset_index(drop=True)


def outcome_label(hg, ag):
    if hg > ag:
        return "H"
    elif hg < ag:
        return "A"
    return "D"


def run_backtest():
    df = load_and_sort()
    rounds = [df.iloc[i:i + ROUND_SIZE] for i in range(0, len(df), ROUND_SIZE)]
    records = []

    for r_idx in range(MIN_TRAINING_ROUNDS, len(rounds)):
        train_df = pd.concat(rounds[:r_idx])
        test_round = rounds[r_idx]

        cutoff_date = test_round["Date"].min()
        teams = sorted(set(train_df["HomeTeam"]) | set(train_df["AwayTeam"]))

        fixtures_with_days_ago = [
            (row.HomeTeam, row.AwayTeam, row.FTHG, row.FTAG,
             (cutoff_date - row.Date).days)
            for row in train_df.itertuples()
        ]
        model = fit_league_weighted(fixtures_with_days_ago, teams, xi=XI)

        for _, row in test_round.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            if home not in model["attack"] or away not in model["attack"]:
                continue

            grid = score_matrix(home, away, model)
            m = derive_markets(grid)["1X2"]
            records.append({
                "home": home, "away": away,
                "p_home": m["home"], "p_draw": m["draw"], "p_away": m["away"],
                "actual": outcome_label(row["FTHG"], row["FTAG"]),
            })

        print(f"Round {r_idx}/{len(rounds)} backtested "
              f"({len(records)} predictions scored so far)")

    return pd.DataFrame(records)


def score_results(results: pd.DataFrame):
    print(f"\n{'='*60}\nBACKTEST v2 (time-weighted) — {len(results)} predictions\n{'='*60}")

    outcome_map = {"H": "p_home", "D": "p_draw", "A": "p_away"}
    log_losses = [
        -np.log(max(row[outcome_map[row["actual"]]], 1e-10))
        for _, row in results.iterrows()
    ]
    print(f"Log loss: {np.mean(log_losses):.4f}  "
          f"(random-guess baseline is 1.0986 — compare against v1's result)")

    def predicted_label(row):
        probs = {"H": row["p_home"], "D": row["p_draw"], "A": row["p_away"]}
        return max(probs, key=probs.get)

    results["predicted"] = results.apply(predicted_label, axis=1)
    accuracy = (results["predicted"] == results["actual"]).mean()
    print(f"Accuracy (top pick correct): {accuracy:.1%}")

    print("\nCalibration check (home win probability):")
    print(f"{'Predicted range':20s} {'N':>5s} {'Actual home-win rate':>22s}")
    results["p_bucket"] = pd.cut(results["p_home"], bins=np.arange(0, 1.1, 0.1))
    for bucket, group in results.groupby("p_bucket", observed=True):
        if len(group) == 0:
            continue
        actual_rate = (group["actual"] == "H").mean()
        print(f"{str(bucket):20s} {len(group):5d} {actual_rate:21.1%}")


if __name__ == "__main__":
    print("Running time-weighted rolling backtest — may take a couple of minutes...\n")
    results = run_backtest()
    score_results(results)
