"""
Fetch every league's data and save it into data/ for the app to read.

WHY THIS EXISTS
---------------
football-data.co.uk's anti-bot protection blocks requests coming from
Streamlit Cloud's server IPs, so the deployed app can't fetch directly —
it fails regardless of which network you're personally on. Your own
machine isn't blocked, so this script does the fetching locally and the
results get committed to the repo. The app then reads the committed
snapshot, and only tries a live fetch if the snapshot is missing.

USAGE
-----
    source venv/bin/activate
    python3 refresh_data.py

Then commit the data/ folder via GitHub Desktop and push. The deployed
app picks it up on the next redeploy.

Run this once or twice a week during the season — the snapshot is only
as current as the last time you ran it, and the app tells you how old it
is so stale data is visible rather than silent.
"""

import io
import json
import time
from datetime import date, datetime
from pathlib import Path

import requests
import pandas as pd

from league_config import LEAGUES, SEASON

DATA_DIR = Path("data")
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}


def current_season_code(today=None):
    """football-data labels 2026/27 as '2627'. Seasons start in July."""
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{str(start)[-2:]}{str(start + 1)[-2:]}"


def fetch(url, retries=3):
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code in (502, 503, 504) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.content.decode("utf-8-sig", errors="replace")
        except requests.exceptions.RequestException as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last


def save_csv(text, path, label):
    """Validates before saving — writing a blocked-page HTML response over
    good data would be worse than failing."""
    if text.lstrip()[:1] == "<":
        print(f"  ✗ {label}: got a webpage, not a CSV — skipped")
        return False
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        df = pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip")

    if "HomeTeam" not in df.columns:
        print(f"  ✗ {label}: no HomeTeam column, got {list(df.columns)[:5]} — skipped")
        return False

    path.write_text(text, encoding="utf-8")
    print(f"  ✓ {label}: {len(df)} rows")
    return True


def main():
    DATA_DIR.mkdir(exist_ok=True)
    cur_season = current_season_code()
    print(f"Season codes — previous: {SEASON}, current: {cur_season}\n")

    saved = {"previous": [], "current": [], "fixtures": False}

    print("Previous season (model training base):")
    for name, cfg in LEAGUES.items():
        code = cfg["data_code"]
        if not code:
            continue          # market-only competition — nothing to download
        try:
            text = fetch(BASE.format(season=SEASON, code=code))
            if save_csv(text, DATA_DIR / f"{SEASON}_{code}.csv", name):
                saved["previous"].append(name)
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}")

    print("\nCurrent season (in-season form):")
    for name, cfg in LEAGUES.items():
        code = cfg["data_code"]
        if not code:
            continue          # market-only competition — nothing to download
        try:
            text = fetch(BASE.format(season=cur_season, code=code))
            if save_csv(text, DATA_DIR / f"{cur_season}_{code}.csv", name):
                saved["current"].append(name)
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}")

    print("\nUpcoming fixtures:")
    try:
        text = fetch(FIXTURES_URL)
        lines = text.splitlines()
        idx = next((i for i, l in enumerate(lines) if l.startswith("Div,Date")), None)
        if idx is not None:
            clean = "\n".join(lines[idx:])
            saved["fixtures"] = save_csv(clean, DATA_DIR / "fixtures.csv", "fixtures")
        else:
            print("  ✗ fixtures: no 'Div,Date' header found")
    except Exception as e:
        print(f"  ✗ fixtures: {type(e).__name__}")

    manifest = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "previous_season": SEASON,
        "current_season": cur_season,
        "leagues_previous": saved["previous"],
        "leagues_current": saved["current"],
        "fixtures": saved["fixtures"],
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. {len(saved['previous'])} previous-season and "
          f"{len(saved['current'])} current-season files saved to data/.")
    print("Commit the data/ folder and push to update the deployed app.")


if __name__ == "__main__":
    main()
