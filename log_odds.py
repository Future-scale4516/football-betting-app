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
        "selection", "logged_odds", "market_prob", "model_prob", "kickoff",
        "benchmark", "book"]


def preflight():
    """Free call (costs no credits): is the key accepted and how many credits
    are left? Prints the answer so a failed run says WHY. Returns True to go on."""
    import requests
    from env_config import ODDS_API_KEY
    if not ODDS_API_KEY:
        print("✗ No Odds API key found. Put ODDS_API_KEY=... in the .env file in this folder.")
        return False
    try:
        r = requests.get("https://api.the-odds-api.com/v4/sports/",
                         params={"apiKey": ODDS_API_KEY}, timeout=20)
    except Exception as e:
        print(f"✗ Couldn't reach the Odds API: {e}")
        return False
    left, used = r.headers.get("x-requests-remaining"), r.headers.get("x-requests-used")
    if r.status_code == 401:
        print("✗ The Odds API rejected the key (401). Check .env — is it the NEW key?")
        return False
    if r.status_code == 429 or (left is not None and float(left) <= 0):
        print("✗ Out of Odds API credits (usage limit reached). Credits come back when "
              "your plan renews; or set ODDS_REGIONS = \"uk\" in football_core.py to use half as many.")
        return False
    if r.status_code != 200:
        print(f"✗ Odds API returned {r.status_code}: {r.text[:150]}")
        return False
    print(f"✓ Odds API key OK — {left} credits left ({used} used this period).")
    if left is not None and float(left) < 150:
        print("  ⚠ Fewer than 150 left: one 3-day run can use that many. Try: python3 log_odds.py 1")
    return True


def collect(run_all, days):
    seen = set()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    out = []
    for i in range(days):
        d = date.today() + timedelta(days=i)
        rows, notes = run_all(d, model_only=False)
        for n in notes or []:
            if n not in seen:
                seen.add(n)
                print(f"  note ({d}): {n}")
        if rows is None or rows.empty:
            continue
        r = rows[rows["odds"].notna() & (rows["prob_source"] == "model")]
        for x in r.itertuples():
            out.append([now, d.isoformat(), x.league, x.fixture, x.market,
                        x.selection, x.odds, x.market_prob, x.model_prob,
                        str(x.kickoff) if x.kickoff is not None else "",
                        getattr(x, "benchmark", None), getattr(x, "book", None)])
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
    if not preflight():
        raise SystemExit(1)
    from football_core import run_all
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    df = collect(run_all, days)
    if df.empty:
        print("No priced selections found — nothing logged. Read the 'note' lines above for why "
              "(no games that day, an API error, or no team ratings).")
    else:
        total = append(df)
        print(f"Logged {len(df)} prices across {df.fixture.nunique()} fixtures "
              f"({total} rows in {LOG}). Now commit data/odds_log.csv.")
