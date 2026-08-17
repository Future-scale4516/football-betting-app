"""
Auto-settlement from football-data.co.uk results.

Nothing in the app knew whether a match had finished — picks sat "open"
forever until settled by hand. This grades them from the same source we
already use for model fitting, and pulls CLOSING odds out of the same
file so CLV actually gets populated.

Important: league_config.SEASON is last completed season (what the model
fits on). Grading needs the CURRENT season's file, hence CURRENT_SEASON
below. Bump this each August.
"""

import pandas as pd
from league_config import LEAGUES

CURRENT_SEASON = "2627"   # 2026/27 — update when the season rolls over
RESULTS_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# Closing-odds columns, best first. football-data's naming varies by file,
# so each is tried in turn and skipped if absent rather than assumed.
CLOSING_COLS = {
    ("1X2", "Home"):      ["AvgCH", "B365CH", "AvgH", "B365H"],
    ("1X2", "Draw"):      ["AvgCD", "B365CD", "AvgD", "B365D"],
    ("1X2", "Away"):      ["AvgCA", "B365CA", "AvgA", "B365A"],
    ("O/U 2.5", "Over"):  ["AvgC>2.5", "B365C>2.5", "Avg>2.5", "B365>2.5"],
    ("O/U 2.5", "Under"): ["AvgC<2.5", "B365C<2.5", "Avg<2.5", "B365<2.5"],
}


def _fetch(code: str):
    url = RESULTS_URL.format(season=CURRENT_SEASON, code=code)
    df = pd.read_csv(url)
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    return df


def load_results_for(leagues):
    """Returns {league_name: dataframe} plus any error notes."""
    out, notes = {}, []
    for lg in leagues:
        cfg = LEAGUES.get(lg)
        if not cfg:
            continue
        try:
            out[lg] = _fetch(cfg["data_code"])
        except Exception as e:
            notes.append(f"{lg}: couldn't load {CURRENT_SEASON} results ({e})")
    return out, notes


def grade(market: str, selection: str, hg: int, ag: int):
    """Did this selection win? Returns True/False, or None if we can't
    grade that market rather than guessing."""
    total = hg + ag

    if market == "1X2":
        if selection == "Home":
            return hg > ag
        if selection == "Draw":
            return hg == ag
        if selection == "Away":
            return ag > hg

    if market.startswith("O/U"):
        try:
            line = float(market.split()[-1])
        except ValueError:
            return None
        if selection.startswith("Over"):
            return total > line
        if selection.startswith("Under"):
            return total < line

    if market == "BTTS":
        both = hg > 0 and ag > 0
        return both if selection == "Yes" else (not both)

    return None


def find_closing_odds(row, market: str, selection: str):
    side = selection.split()[0] if selection.split()[0] in ("Over", "Under") else selection
    for col in CLOSING_COLS.get((market, side), []):
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                continue
    return None


def settle_open_picks(open_picks: pd.DataFrame):
    """Grades every open pick it can find a finished match for.

    Returns (list of (index, won, closing_odds), notes). Anything without
    a matching finished fixture is left alone — a missing result usually
    means the match hasn't been played or the file hasn't updated yet,
    and guessing would corrupt the log.
    """
    leagues = [l for l in open_picks["league"].dropna().unique()]
    results, notes = load_results_for(leagues)

    settlements = []
    unmatched = 0

    for idx, pick in open_picks.iterrows():
        lg = pick.get("league")
        if lg not in results:
            continue

        fixture = str(pick["fixture"])
        if " vs " not in fixture:
            continue
        home, away = fixture.split(" vs ", 1)

        df = results[lg]
        match = df[(df["HomeTeam"] == home) & (df["AwayTeam"] == away)]

        # Same fixture can recur across a season — prefer the one on the
        # logged date, fall back to a unique match only.
        if len(match) > 1:
            try:
                target = pd.to_datetime(pick["fixture_date"]).date()
                match = match[match["_date"] == target]
            except Exception:
                pass
        if len(match) != 1:
            unmatched += 1
            continue

        row = match.iloc[0]
        if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
            unmatched += 1
            continue

        won = grade(str(pick["market"]), str(pick["selection"]),
                     int(row["FTHG"]), int(row["FTAG"]))
        if won is None:
            continue

        settlements.append((idx, won, find_closing_odds(
            row, str(pick["market"]), str(pick["selection"]))))

    if unmatched:
        notes.append(
            f"{unmatched} pick(s) had no finished result yet — either the match "
            "hasn't been played or football-data hasn't published it. Left open."
        )

    return settlements, notes
