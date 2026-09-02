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
import requests
import pandas as pd

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}

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
    """Returns (dataframe_or_None, description_of_what_happened)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
    except Exception as e:
        return None, f"request failed ({type(e).__name__})"

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"

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


def load_results(code: str, target_date):
    """Tries each candidate season until one yields a valid results CSV.

    Returns (dataframe_or_None, list_of_attempt_descriptions). The
    attempt log is surfaced in the UI so a failure is diagnosable
    instead of mysterious.
    """
    log = []
    for season in season_candidates(target_date):
        url = BASE.format(season=season, code=code)
        df, what = _attempt(url)
        log.append(f"{season}/{code}.csv → {what}")
        if df is not None:
            df["_date"] = pd.to_datetime(df["Date"], dayfirst=True,
                                          errors="coerce").dt.date
            # Only accept a season that actually contains the target date;
            # otherwise keep trying, since an old season file will parse
            # fine but contain nothing useful.
            if (df["_date"] == target_date).any():
                return df, log
            log.append(f"   (parsed, but has no fixtures on {target_date})")
    return None, log


def parse_uploaded(file) -> pd.DataFrame:
    """Manual fallback — user downloads the CSV themselves and uploads it."""
    df = pd.read_csv(file)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"That file is missing {sorted(missing)}")
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    return df
