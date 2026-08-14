"""
Rolling backtest for the Dixon-Coles Premier League model.

Logic: fit on everything BEFORE a given round, predict that round,
record the result, roll forward. Never lets the model see a match
it's being scored on — same discipline as the MLB backtesting.

Requires dixon_coles_sketch.py in the same folder.

Usage:
    python3 backtest_premier_league.py
"""

import numpy as np
import pandas as pd
from dixon_coles_sketch import fit_league, score_matrix, derive_markets

DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
ROUND_SIZE = 10          # ~1 gameweek per round
MIN_TRAINING_ROUNDS = 10  # don't test until the model has ~100 games to learn from


def load_and_sort():
    df = pd.read_csv(DATA_URL, parse_dates=["Date"], dayfirst=True)
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def outcome_label(hg, ag):
    if hg > ag:
        return "H"
    elif hg < ag:
        return "A"
    return "D"


def run_backtest():
    df = load_and_sort()
    rounds = [df.iloc[i:i + ROUND_SIZE] for i in range(0, len(df), ROUND_SIZE)]

    records = []  # each: predicted probs for H/D/A + actual outcome

    for r_idx in range(MIN_TRAINING_ROUNDS, len(rounds)):
        train_df = pd.concat(rounds[:r_idx])
        test_round = rounds[r_idx]

        teams = sorted(set(train_df["HomeTeam"]) | set(train_df["AwayTeam"]))
        fixtures = list(train_df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
                         .itertuples(index=False, name=None))
        model = fit_league(fixtures, teams)

        for _, row in test_round.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            if home not in model["attack"] or away not in model["attack"]:
                continue  # promoted team with no training history yet — skip, don't fabricate

            grid = score_matrix(home, away, model)
            m = derive_markets(grid)["1X2"]
            actual = outcome_label(row["FTHG"], row["FTAG"])

            records.append({
                "home": home, "away": away,
                "p_home": m["home"], "p_draw": m["draw"], "p_away": m["away"],
                "actual": actual,
            })

        print(f"Round {r_idx}/{len(rounds)} backtested "
              f"({len(records)} predictions scored so far)")

    return pd.DataFrame(records)


def score_results(results: pd.DataFrame):
    print(f"\n{'='*60}\nBACKTEST RESULTS — {len(results)} predictions\n{'='*60}")

    # Log loss — the real calibration metric. Lower is better;
    # ~1.0-1.1 is roughly "no better than guessing" for a 3-way market.
    outcome_map = {"H": "p_home", "D": "p_draw", "A": "p_away"}
    log_losses = [
        -np.log(max(row[outcome_map[row["actual"]]], 1e-10))
        for _, row in results.iterrows()
    ]
    print(f"Log loss: {np.mean(log_losses):.4f}  "
          f"(rough benchmark: under ~1.0 is meaningfully better than a coin flip)")

    # Accuracy — did the highest-probability outcome actually happen?
    def predicted_label(row):
        probs = {"H": row["p_home"], "D": row["p_draw"], "A": row["p_away"]}
        return max(probs, key=probs.get)

    results["predicted"] = results.apply(predicted_label, axis=1)
    accuracy = (results["predicted"] == results["actual"]).mean()
    print(f"Accuracy (top pick correct): {accuracy:.1%}")

    # Calibration — bucket predicted home-win probability into deciles,
    # compare against how often home actually won in that bucket.
    print("\nCalibration check (home win probability):")
    print(f"{'Predicted range':20s} {'N':>5s} {'Actual home-win rate':>22s}")
    results["p_bucket"] = pd.cut(results["p_home"], bins=np.arange(0, 1.1, 0.1))
    for bucket, group in results.groupby("p_bucket", observed=True):
        if len(group) == 0:
            continue
        actual_rate = (group["actual"] == "H").mean()
        print(f"{str(bucket):20s} {len(group):5d} {actual_rate:21.1%}")
    print("\n(A well-calibrated model has 'actual home-win rate' close to the")
    print(" midpoint of each predicted range — e.g. the 0.5-0.6 bucket should")
    print(" see home teams actually win roughly 50-60% of the time. Large,")
    print(" consistent gaps mean the model is over- or under-confident.)")


if __name__ == "__main__":
    print("Running rolling backtest — this refits the model many times, "
          "may take a couple of minutes...\n")
    results = run_backtest()
    score_results(results)
