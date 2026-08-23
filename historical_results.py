"""
Recomputes what the model would have predicted for a past date — using
only data available BEFORE that date — then grades it against the real
result. No pre-logged picks required.

Mirrors the MLB app's Results page: pick a date, click Load, done.
"""

import pandas as pd
import streamlit as st

from dixon_coles_sketch import fit_league, score_matrix, derive_markets
from league_config import LEAGUES, data_url
from auto_settle import grade, find_closing_odds
from football_data_source import load_results, _attempt, BASE

MARKET_SELECTIONS = {
    "1X2": [("Home", "home"), ("Draw", "draw"), ("Away", "away")],
    "O/U 2.5": [("Over 2.5", "over"), ("Under 2.5", "under")],
    "BTTS": [("Yes", "yes"), ("No", "no")],
}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_season_for(league_name: str, target_date):
    cfg = LEAGUES[league_name]
    return load_results(cfg["data_code"], target_date)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_last_season_fixtures(league_name: str):
    df, _ = _attempt(data_url(league_name))
    if df is None:
        return []
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    return list(df.itertuples(index=False, name=None))


def _fit_as_of(league_name: str, target_date, results_df):
    """Last season in full + this season's completed games strictly before
    target_date, so there's no lookahead into the day being graded."""
    base = _load_last_season_fixtures(league_name)

    prior = results_df[(results_df["_date"] < target_date)
                        & results_df["FTHG"].notna()
                        & results_df["FTAG"].notna()]
    prior_fixtures = list(
        prior[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
        .itertuples(index=False, name=None))

    fixtures = base + prior_fixtures
    teams = sorted({t for fx in fixtures for t in (fx[0], fx[1])})
    if len(teams) < 4 or len(fixtures) < 20:
        return None, "not enough prior data to fit a model yet"
    return fit_league(fixtures, teams), None


def build_historical_results(target_date, leagues,
                              markets=("1X2", "O/U 2.5", "BTTS"),
                              pick_mode="most_likely",
                              uploaded=None):
    """
    pick_mode: 'most_likely' grades only the model's top selection per
    market per fixture. 'all' grades every selection.
    uploaded: {league_name: dataframe} from manual CSV upload, bypassing
    the download entirely.
    Returns (rows_df, notes, diagnostics).
    """
    rows, notes, diagnostics = [], [], {}

    for league in leagues:
        if uploaded and league in uploaded:
            results_df, log = uploaded[league], ["(uploaded manually)"]
        else:
            results_df, log = _load_season_for(league, target_date)
        diagnostics[league] = log

        if results_df is None:
            notes.append(f"{league}: no results file found — see diagnostics below")
            continue

        model, err = _fit_as_of(league, target_date, results_df)
        if model is None:
            notes.append(f"{league}: skipped — {err}")
            continue

        day_games = results_df[(results_df["_date"] == target_date)
                                & results_df["FTHG"].notna()
                                & results_df["FTAG"].notna()]
        if day_games.empty:
            continue

        for _, g in day_games.iterrows():
            home, away = g["HomeTeam"], g["AwayTeam"]
            if home not in model["attack"] or away not in model["attack"]:
                notes.append(f"{league}: {home} vs {away} skipped — no rating "
                             "(promoted/relegated, no prior data)")
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

    return pd.DataFrame(rows), notes, diagnostics
