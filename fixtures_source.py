"""
Fixture sources that don't depend on football-data.co.uk's fixtures.csv.

WHY: fixtures.csv is thin and irregularly updated — your Thursday refresh
held just 12 rows — and it only reaches the app via the committed
snapshot, so it can never be fresher than the last time you ran
refresh_data.py. On a matchday that means "No fixtures found" even when
games are on.

The Odds API's /events endpoint lists upcoming and in-play matches with
kickoff times. It should not count against the quota (check the usage
counter after the first run to confirm), and it's reachable from
Streamlit Cloud — the odds calls already work from there, unlike
football-data.co.uk.

Nothing here imports Streamlit, so it can be tested on its own.
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

UK_TZ = ZoneInfo("Europe/London")
EVENTS_URL = "https://api.the-odds-api.com/v4/sports/{key}/events"


def fetch_events(sport_key: str, api_key: str, timeout: int = 20):
    """Returns (events_list_or_None, error_message_or_None)."""
    try:
        resp = requests.get(
            EVENTS_URL.format(key=sport_key),
            params={"apiKey": api_key, "dateFormat": "iso"},
            timeout=timeout)
    except requests.exceptions.RequestException as e:
        return None, f"events request failed ({type(e).__name__})"
    if resp.status_code == 401:
        return None, "Odds API key rejected (401) — check ODDS_API_KEY and billing"
    if resp.status_code == 429:
        return None, "Odds API quota used up (429)"
    if resp.status_code != 200:
        return None, f"Odds API {resp.status_code} for '{sport_key}'"
    try:
        return resp.json(), None
    except ValueError:
        return None, "events response wasn't valid JSON"


def parse_events(events, target_date, now=None):
    """Events -> [(home, away, 'HH:MM' UK time, already_started)] for one
    date, in kickoff order. Dates are judged in UK time, not UTC or the
    server's local time."""
    now = now or datetime.now(UK_TZ)
    out = []
    for e in events or []:
        ts = e.get("commence_time")
        if not ts or not e.get("home_team") or not e.get("away_team"):
            continue
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UK_TZ)
        if dt.date() != target_date:
            continue
        out.append((e["home_team"], e["away_team"], dt.strftime("%H:%M"), dt <= now))
    out.sort(key=lambda r: r[2])
    return out


# "12:30 Chesterfield v Tranmere", "Chesterfield vs Tranmere",
# "Chesterfield - Tranmere". The separator must have spaces around it so
# hyphenated team names aren't split down the middle.
_LINE = re.compile(
    r"^\s*(?:(\d{1,2}[:.]\d{2})\s+)?(.+?)\s+(?:v|vs\.?|-|–|—)\s+(.+?)\s*$",
    re.IGNORECASE)


def parse_fixture_lines(text: str):
    """Typed/pasted fixtures -> ([(home, away, 'HH:MM' or None)], [bad_lines]).
    Blank lines are ignored; anything unparseable is reported, not dropped
    silently."""
    good, bad = [], []
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        if not m:
            bad.append(raw.strip())
            continue
        ko, home, away = m.group(1), m.group(2).strip(), m.group(3).strip()
        if ko:
            ko = ko.replace(".", ":").zfill(5)
        good.append((home, away, ko))
    return good, bad
