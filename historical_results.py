"""
Recomputes what the model would have predicted for a past date — using
only data available BEFORE that date — then grades it against the real
result. No pre-logged picks required.

This mirrors the MLB app's Results page: pick a date, click Load, done.
The model is refit each time using last season in full plus whatever
games from the current season happened earlier than the selected date,
so there's no lookahead into the result being graded.
"""

import pandas as pd
import streamlit as st

from dixon_coles_sketch import fit_league, score_matrix, derive_markets
from league_config import LEAGUES, data_url
from auto_settle import CURRENT_SEASON, RESULTS_URL, grade, find_closing_odds

MARKET_SELECTIONS = {
    "1X2": [("Home", "home"), ("Draw", "draw"), ("Away", "away")],
    "O/U 2.5": [("Over 2.5", "over"), ("Under 2.5", "under")],
    "BTTS": [("Yes", "yes"), ("No", "no")],
}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_current_season(league_name: str):
    cfg = LEAGUES[league_name]
    url = RESULTS_URL.format(season=CURRENT_SEASON, code=cfg["data_code"])
    df = pd.read_csv(url)
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _load_last_season_fixtures(league_name: str):
    df = pd.read_csv(data_url(league_name))
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    return list(df.itertuples(index=False, name=None))


def _fit_as_of(league_name: str, target_date):
    """Last season in full + this season's completed games strictly
    before target_date. Early in a new season this is just last
    season's model, same as everywhere else in the app — it's only
    once enough current-season games exist that this starts to diverge
    and actually reflect in-season form.

    Returns (model_or_None, current_season_df_or_None, error_or_None).
    """
    base = _load_last_season_fixtures(league_name)
    try:
        cur = _load_current_season(league_name)
    except Exception as e:
        return None, None, f"couldn't load current-season file ({e})"

    prior = cur[(cur["_date"] < target_date)
                & cur["FTHG"].notna() & cur["FTAG"].notna()]
    prior_fixtures = list(
        prior[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
        .itertuples(index=False, name=None))

    fixtures = base + prior_fixtures
    teams = sorted({t for fx in fixtures for t in (fx[0], fx[1])})
    if len(teams) < 4 or len(fixtures) < 20:
        return None, cur, "not enough prior data to fit yet"
    return fit_league(fixtures, teams), cur, None


def build_historical_results(target_date, leagues, markets=("1X2", "O/U 2.5", "BTTS"),
                              pick_mode="most_likely"):
    """
    pick_mode: 'most_likely' grades only the model's top selection per
    market per fixture (mirrors the MLB app's one-pick-per-market view).
    'all' grades every selection.
    Returns (rows_df, notes).
    """
    rows, notes = [], []

    for league in leagues:
        model, cur, err = _fit_as_of(league, target_date)
        if model is None:
            notes.append(f"{league}: skipped — {err}")
            continue

        day_games = cur[(cur["_date"] == target_date)
                         & cur["FTHG"].notna() & cur["FTAG"].notna()]
        if day_games.empty:
            continue

        for _, g in day_games.iterrows():
            home, away = g["HomeTeam"], g["AwayTeam"]
            if home not in model["attack"] or away not in model["attack"]:
                notes.append(f"{league}: {home} vs {away} skipped — no rating "
                             "(promoted/relegated with no prior data yet)")
                continue

            grid = score_matrix(home, away, model)
            mkts = derive_markets(grid)
            hg, ag = int(g["FTHG"]), int(g["FTAG"])

            for market in markets:
                mk_probs = mkts[market]
                options = MARKET_SELECTIONS[market]
                chosen = ([max(options, key=lambda o: mk_probs[o[1]])]
                          if pick_mode == "most_likely" else options)

                for label, key in chosen:
                    rows.append({
                        "league": league, "fixture": f"{home} vs {away}",
                        "market": market, "selection": label,
                        "model_prob": mk_probs[key],
                        "actual_score": f"{hg}-{ag}",
                        "won": grade(market, label, hg, ag),
                        "closing_odds": find_closing_odds(g, market, label),
                    })

    return pd.DataFrame(rows), notes
