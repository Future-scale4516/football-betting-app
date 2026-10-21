"""
Logs the bookmaker prices the app is showing right now, so they can later be
compared with the closing prices (closing-line value, CLV).

Run LOCALLY (the cloud app's disk is wiped on restart), then commit
data/odds_log.csv with GitHub Desktop like the other data files:

    python3 log_odds.py            # today + next 2 days
    python3 log_odds.py 4          # today + next 4 days

Best habit: run it the morning of the day you'd place bets, once or twice a
week. Each run costs Odds API credits (one odds call per league per day).
Appends; the same price logged twice for one run time is not duplicated.
"""
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

LOG = Path(__file__).parent / "data" / "odds_log.csv"
COLS = ["logged_at", "match_date", "league", "fixture", "market",
        "selection", "logged_odds", "market_prob", "model_prob", "kickoff"]


def collect(run_all, days):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    out = []
    for i in range(days):
        d = date.today() + timedelta(days=i)
        rows, _ = run_all(d, model_only=False)
        if rows is None or rows.empty:
            continue
        r = rows[rows["odds"].notna() & (rows["prob_source"] == "model")]
        for x in r.itertuples():
            out.append([now, d.isoformat(), x.league, x.fixture, x.market,
                        x.selection, x.odds, x.market_prob, x.model_prob,
                        str(x.kickoff) if x.kickoff is not None else ""])
    return pd.DataFrame(out, columns=COLS)


def append(new, path=LOG):
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(
            ["logged_at", "match_date", "fixture", "market", "selection"])
    new.to_csv(path, index=False)
    return len(new)


if __name__ == "__main__":
    from football_core import run_all
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    df = collect(run_all, days)
    if df.empty:
        print("No priced selections found — nothing logged.")
    else:
        total = append(df)
        print(f"Logged {len(df)} prices across {df.fixture.nunique()} fixtures "
              f"({total} rows in {LOG}). Now commit data/odds_log.csv.")
