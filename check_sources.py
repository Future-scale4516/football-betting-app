"""
Checks the things I couldn't verify from outside your machine.

    source venv/bin/activate
    python3 check_sources.py

It answers four questions:
  1. Which football sport keys does your Odds API account actually have —
     in particular League One, League Two, and anything for the National
     League?
  2. Does each events_key in league_config.py return games?
  3. Do the free events calls really cost no quota?
  4. What's in the fixtures.csv snapshot (why the app said "No fixtures
     found")?

Paste the output back and I'll adjust league_config.py to match.
"""

import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from env_config import require_api_key
from league_config import LEAGUES
from fixtures_source import fetch_events, parse_events

UK = ZoneInfo("Europe/London")
API = "https://api.the-odds-api.com/v4/sports"


def quota(resp):
    return (resp.headers.get("x-requests-used", "?"),
            resp.headers.get("x-requests-remaining", "?"))


def summarise_snapshot(path):
    """Returns printable lines describing a fixtures.csv snapshot."""
    p = Path(path)
    if not p.exists():
        return ["  data/fixtures.csv not found — run refresh_data.py first"]
    df = pd.read_csv(p)
    if "Div" not in df.columns:
        return [f"  no 'Div' column — found {list(df.columns)[:6]}"]
    df["_d"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    lines = [f"  {len(df)} rows in total"]
    for div, g in df.groupby("Div"):
        days = sorted({d for d in g["_d"].dropna()})
        span = f"{days[0]} -> {days[-1]}" if days else "no valid dates"
        lines.append(f"  {div:4s} {len(g):3d} fixtures   {span}")
    wanted = {cfg["data_code"]: name for name, cfg in LEAGUES.items()
              if cfg.get("data_code")}
    missing = [f"{c} ({n})" for c, n in wanted.items() if c not in set(df["Div"])]
    lines.append("  leagues with NO fixtures in the file: "
                 + (", ".join(missing) if missing else "none"))
    return lines


def main():
    key = require_api_key()

    print("=" * 66)
    print("1) FOOTBALL SPORT KEYS ON YOUR ODDS API ACCOUNT")
    print("=" * 66)
    r = requests.get(API, params={"apiKey": key, "all": "true"}, timeout=20)
    used0, rem0 = quota(r)
    print(f"status {r.status_code} | quota used {used0}, remaining {rem0}\n")
    if r.status_code != 200:
        if r.status_code == 401:
            print("The Odds API rejected this key (401). If it says DEACTIVATED_KEY, "
                  "the plan was cancelled or a payment failed.\nPut the new key in "
                  ".env (and in Streamlit Cloud -> Settings -> Secrets), then re-run.")
        else:
            print("Couldn't list sports:", r.text[:200])
        print("\nSkipping the Odds API sections — section 4 below doesn't need a key.\n")
        print("=" * 66)
        print("4) THE FIXTURES.CSV SNAPSHOT")
        print("=" * 66)
        for line in summarise_snapshot("data/fixtures.csv"):
            print(line)
        return

    soccer = [s for s in r.json() if s["key"].startswith("soccer")]
    hints = ("england", "efl", "epl", "league", "national", "conference", "fa_cup")
    interesting = [s for s in soccer
                   if any(h in (s["key"] + " " + s["title"]).lower() for h in hints)]
    for s in sorted(interesting, key=lambda s: s["key"]):
        print(f"  {s['key']:34s} {s['title']:30s} active={s.get('active')}")
    print(f"\n  ({len(soccer)} soccer keys in total)")

    have = {s["key"] for s in soccer}
    print("\n  Configured events/odds keys:")
    for name, cfg in LEAGUES.items():
        k = cfg.get("events_key") or cfg.get("odds_key")
        if k is None:
            print(f"    {name:18s} no key configured")
        else:
            print(f"    {name:18s} {k:34s} {'EXISTS' if k in have else 'NOT FOUND'}")

    print("\n" + "=" * 66)
    print("2) WHAT EACH EVENTS KEY RETURNS")
    print("=" * 66)
    today = datetime.now(UK).date()
    for name, cfg in LEAGUES.items():
        k = cfg.get("events_key") or cfg.get("odds_key")
        if not k:
            continue
        events, err = fetch_events(k, key)
        if err:
            print(f"  {name:18s} {err}")
            continue
        todays = parse_events(events, today)
        print(f"  {name:18s} {len(events):3d} upcoming | {len(todays):2d} today ({today})")
        for h, a, ko, started in todays[:3]:
            print(f"      {ko} {h} v {a}{'  [started]' if started else ''}")

    print("\n" + "=" * 66)
    print("3) DO THE EVENTS CALLS COST QUOTA?")
    print("=" * 66)
    r2 = requests.get(API, params={"apiKey": key}, timeout=20)
    used1, rem1 = quota(r2)
    print(f"  before the events calls: used {used0}, remaining {rem0}")
    print(f"  after  the events calls: used {used1}, remaining {rem1}")
    try:
        delta = int(used1) - int(used0)
        print("  -> events calls were FREE" if delta == 0
              else f"  -> {delta} credits were used — tell me and I'll cut the calls back")
    except ValueError:
        print("  (couldn't read the quota headers)")

    print("\n" + "=" * 66)
    print("4) THE FIXTURES.CSV SNAPSHOT")
    print("=" * 66)
    for line in summarise_snapshot("data/fixtures.csv"):
        print(line)


if __name__ == "__main__":
    main()
