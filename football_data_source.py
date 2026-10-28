"""
Robust loader for football-data.co.uk CSVs.

This source has caused repeated grief: bare pandas requests get an HTTP
300, and a wrong season code returns an HTML error page that pandas then
tries to parse as a one-column CSV ("Expected 1 fields in line 7, saw 2").

So this module does three things the old code didn't:
  1. Sends a normal User-Agent (fixes the 300).
  2. Tries several season codes, because "which season is current" is
     genuinely ambiguous in August and a guess shouldn't break the page.
  3. VALIDATES that what came back is really a results CSV (has the
     columns we need) instead of trusting a 200 response.

Every attempt is recorded so a failure produces a precise diagnostic
rather than another cryptic parser error.
"""

import io
import time
from pathlib import Path
import urllib.parse
import requests
import pandas as pd

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}

# Stopgap only — see _attempt() below. Not a permanent fix; replace with a
# proper non-cloud-IP data pipeline when there's time (see the conversation
# this was added in — Streamlit Cloud's own IP started getting blocked by
# football-data.co.uk's anti-bot protection, unrelated to any user network).
_PROXY_PREFIX = "https://api.allorigins.win/raw?url="

REQUIRED_COLS = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def season_candidates(target_date):
    """Season codes to try, most likely first.

    football-data labels 2026/27 as '2627'. A match in Aug-Dec belongs to
    the season starting that year; Jan-Jul belongs to the one that started
    the previous year. Both neighbours are tried as a fallback because
    early-season files sometimes lag.
    """
    y = target_date.year
    start = y if target_date.month >= 7 else y - 1
    codes = []
    for s in (start, start - 1, start + 1):
        codes.append(f"{str(s)[-2:]}{str(s + 1)[-2:]}")
    return codes


def _attempt(url):
    """Returns (dataframe_or_None, description_of_what_happened).

    Retries a couple of times on a transient 502/503/504 — football-
    data.co.uk's own server briefly overloading isn't worth giving up on
    immediately for. If direct requests keep failing, falls back to a
    public proxy (a different, unrelated problem: Streamlit Cloud's own
    IP being blocked — see the _PROXY_PREFIX note above)."""
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=25)
        except Exception as e:
            if attempt == 2:
                resp = None
                break
            time.sleep(1.5 * (attempt + 1))
            continue

        if resp.status_code in (502, 503, 504) and attempt < 2:
            time.sleep(1.5 * (attempt + 1))
            continue
        break

    if resp is None or resp.status_code != 200:
        direct_failure = (f"HTTP {resp.status_code}" if resp is not None
                           else "request failed")
        try:
            proxied_url = _PROXY_PREFIX + urllib.parse.quote(url, safe="")
            resp = requests.get(proxied_url, timeout=35)
            if resp.status_code != 200:
                return None, f"{direct_failure}; proxy also HTTP {resp.status_code}"
        except Exception as e:
            return None, f"{direct_failure}; proxy also failed ({type(e).__name__})"

    text = resp.content.decode("utf-8-sig", errors="replace")
    if not text.strip():
        return None, "empty response"
    if text.lstrip()[:1] == "<":
        return None, "got a webpage, not a CSV (file likely doesn't exist)"

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(io.StringIO(text), engine="python",
                              on_bad_lines="skip")
        except Exception as e:
            return None, f"unparseable CSV ({type(e).__name__})"
    except Exception as e:
        return None, f"read error ({type(e).__name__})"

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return None, (f"CSV missing expected columns {sorted(missing)} — "
                      f"found {list(df.columns)[:6]}")

    return df, f"OK ({len(df)} rows)"


DATA_DIR = Path(__file__).parent / "data"


def _snapshot_df(season, code):
    """The committed data/ copy (see refresh_data.py), or None."""
    path = DATA_DIR / f"{season}_{code}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None


def _with_dates(df):
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True,
                                  errors="coerce").dt.date
    return df


def _newest(df):
    d = df["_date"].dropna()
    return f"{d.max():%d %b}" if len(d) else "unknown"


def load_results(code: str, target_date):
    """Tries each candidate season until one yields a file containing the
    target date. For each season: the committed snapshot first (the live
    site blocks Streamlit Cloud, and the public proxy fallback can serve an
    old cached copy), then the live fetch.

    Returns (dataframe_or_None, list_of_attempt_descriptions). The log is
    shown in the UI, including how recent each file's newest result is, so
    "not published yet" and "stale copy" are distinguishable.
    """
    log = []
    for season in season_candidates(target_date):
        snap = _snapshot_df(season, code)
        if snap is not None and not (REQUIRED_COLS - set(snap.columns)):
            snap = _with_dates(snap)
            if (snap["_date"] == target_date).any():
                log.append(f"{season}/{code}.csv → snapshot OK "
                           f"({len(snap)} rows, newest result {_newest(snap)})")
                return snap, log
            log.append(f"{season}/{code}.csv → snapshot has no games on "
                       f"{target_date} (newest result {_newest(snap)})")

        url = BASE.format(season=season, code=code)
        df, what = _attempt(url)
        log.append(f"{season}/{code}.csv → live: {what}")
        if df is not None:
            df = _with_dates(df)
            if (df["_date"] == target_date).any():
                return df, log
            log.append(f"   (live copy has no games on {target_date}; its "
                       f"newest result is {_newest(df)})")
    return None, log


def parse_uploaded(file) -> pd.DataFrame:
    """Manual fallback — user downloads the CSV themselves and uploads it."""
    df = pd.read_csv(file)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"That file is missing {sorted(missing)}")
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    return df
