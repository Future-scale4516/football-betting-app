"""
Shared engine for the football betting app — the single place that
fits models, fetches odds, computes edges, and renders UI components.
Both the Streamlit pages and run_all_leagues.py import from here so
there's one implementation, not two.
"""

import io
from datetime import datetime, date, timezone
import requests
import pandas as pd
import streamlit as st

CSV_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; football-betting-app)"}


def _fetch_csv(url: str) -> pd.DataFrame:
    """Bare pd.read_csv(url) sends no User-Agent, and football-data.co.uk
    sometimes returns an ambiguous HTTP 300 to requests like that instead
    of the file. Fetching with a normal header fixes it."""
    resp = requests.get(url, headers=CSV_HEADERS, timeout=20)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))

from dixon_coles_sketch import fit_league, score_matrix, derive_markets
from league_config import LEAGUES, data_url, normalise_name
from plausibility import check_plausibility
from traffic_light import classify
from promoted_seeding import seed_missing_teams
from env_config import require_api_key

ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/{key}/odds/"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Groups used by the Suggested Bets page
ENGLISH_EXCL_EPL = ["EFL Championship", "League One", "League Two"]
REST_OF_EUROPE = ["Bundesliga", "Serie A", "La Liga", "Ligue 1"]

ACCA_MARKETS = ["1X2", "BTTS", "O/U 2.5"]

TIER_ICON = {"green": "🟢", "amber": "🟡", "verify": "🔵", "red": "⚪"}


def selection_side(selection: str) -> str:
    """Reduces a selection label to a filterable side so 'Over 2.5' and
    'Over 1.5' both filter under 'Over'. Lets you ask for every Over 2.5
    game, or every BTTS Yes, without picking through markets by hand."""
    first = selection.split()[0]
    return first if first in ("Over", "Under", "Home", "Draw", "Away", "Yes", "No") else selection


# ---------------------------------------------------------------- UI helpers

def setup_page(title: str):
    st.set_page_config(page_title=title, layout="wide")
    st.sidebar.title("⚽ Football Model")


def sidebar_date():
    """Date picker + mode toggle. Returns (date, model_only).

    Model-only mode skips the odds API entirely and forecasts from the
    Dixon-Coles model alone. That unlocks every league (including League
    One/Two) and every market (including BTTS), and costs no API credits —
    but it also removes the only external check on the model. Without a
    de-vigged market price to compare against there is no edge, no
    traffic light, and no closing-line value.
    """
    st.sidebar.markdown("### Date")
    sel = st.sidebar.date_input("Fixtures for:", value=date.today())
    st.sidebar.caption(
        "Leagues start at different times — EPL doesn't begin until "
        "mid-August, so an empty result for today is expected rather "
        "than an error."
    )

    st.sidebar.markdown("### Mode")
    model_only = st.sidebar.toggle(
        "Model-only (ignore odds)",
        value=False,
        help="Forecast from the model alone. All 8 leagues, all markets, "
             "no API credits — but no edge, no traffic light, and no way "
             "to sanity-check the model against the market.",
    )
    if model_only:
        st.sidebar.warning(
            "No market comparison. Probabilities are the model's own "
            "estimate, unchecked against bookmaker pricing."
        )

    if st.sidebar.button("🔄 Refresh (re-fit + re-fetch)"):
        st.cache_data.clear()
        st.rerun()
    return sel, model_only


def render_pick_card(icon, headline, subline, metrics, reason=None):
    """Card layout instead of a table — much easier to read on mobile,
    same approach as the MLB app."""
    with st.container(border=True):
        st.markdown(f"**{icon + ' ' if icon else ''}{headline}**")
        st.caption(subline)
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
        if reason:
            st.caption(reason)


def sort_picker(df, options, key):
    labels = [o[0] for o in options]
    choice = st.selectbox("Sort by:", labels, key=key)
    col, ascending = next((o[1], o[2]) for o in options if o[0] == choice)
    if col in df.columns:
        return df.sort_values(col, ascending=ascending)
    return df


# ------------------------------------------------------------ odds / fixtures

def _event_date(event) -> date:
    """Odds API gives commence_time as ISO UTC."""
    ts = event.get("commence_time")
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().date()


def fetch_odds(sport_key: str):
    url = ODDS_BASE_URL.format(key=sport_key)
    # NOTE: "btts" is NOT valid on this bulk endpoint — the Odds API only
    # serves Both Teams To Score via /events/{id}/odds, which costs one call
    # per fixture. So BTTS below is model-only (no market price, no edge).
    params = {"apiKey": require_api_key(), "regions": "uk",
              "markets": "h2h,totals", "oddsFormat": "decimal"}
    try:
        resp = requests.get(url, params=params, timeout=20)
    except Exception as e:
        return [], f"odds request failed: {e}"
    if resp.status_code != 200:
        return [], f"Odds API {resp.status_code}: {resp.text[:120]}"
    return resp.json(), None


@st.cache_data(ttl=900, show_spinner=False)
def _load_fixtures_csv():
    """One download shared by every league. Previously this file was
    fetched once per league — 8 identical downloads per run, which was
    the main reason model-only mode felt slow."""
    return pd.read_csv(FIXTURES_URL)


def fetch_upcoming_fixtures(div_code: str, target: date):
    """For leagues without odds (or every league in model-only mode),
    pull the fixture list so their games can still be forecast."""
    try:
        df = _load_fixtures_csv()
    except Exception as e:
        return [], f"couldn't load fixtures.csv: {e}"

    if "Div" not in df.columns:
        return [], "fixtures.csv missing expected 'Div' column"

    df = df[df["Div"] == div_code].copy()
    if df.empty:
        return [], None

    try:
        df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
        df = df[df["_date"] == target]
    except Exception:
        return [], "couldn't parse dates in fixtures.csv"

    return [(r.HomeTeam, r.AwayTeam) for r in df.itertuples()], None


# ------------------------------------------------------------------- de-vig

def devig(*odds):
    imps = [1 / o for o in odds]
    total = sum(imps)
    return [i / total for i in imps]


def _avg_market(event, market_key, name_map, line=None):
    """Averages odds across every bookmaker quoting this market."""
    buckets = {k: [] for k in name_map.values()}
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != market_key:
                continue
            for o in mkt["outcomes"]:
                if line is not None and o.get("point") != line:
                    continue
                key = name_map.get(o["name"])
                if key:
                    buckets[key].append(o["price"])
    if not all(buckets.values()):
        return None
    return {k: sum(v) / len(v) for k, v in buckets.items()}


# ------------------------------------------------------------------ pipeline

@st.cache_data(ttl=900, show_spinner=False)
def fit_model_for_league(league_name: str):
    df = _fetch_csv(data_url(league_name))
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    fixtures = list(df.itertuples(index=False, name=None))
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    return fit_league(fixtures, teams)


def _evaluate(fixture, league, markets, odds_data, seeded):
    """Turns one fixture's model output + market odds into pick rows."""
    rows = []

    def add(market, selection, model_prob, market_prob=None, odds=None):
        if market_prob is None:
            tier, edge, reason = "forecast", None, "No odds available — model forecast only"
        else:
            edge = model_prob - market_prob
            status, reason = check_plausibility(model_prob, market_prob, edge)
            if seeded:
                tier = "verify"
                reason = f"Provisionally-seeded team — {seeded}"
            else:
                band = "O/U 2.5" if market.startswith("O/U") else market
                tier = classify(band, edge) if status == "ok" else "verify"
        rows.append({
            "league": league, "fixture": fixture, "market": market,
            "selection": selection, "model_prob": model_prob,
            "market_prob": market_prob, "odds": odds, "edge": edge,
            "tier": tier, "reason": reason,
        })

    m = markets["1X2"]
    h2h = odds_data.get("h2h")
    if h2h:
        p = devig(h2h["home"], h2h["draw"], h2h["away"])
        add("1X2", "Home", m["home"], p[0], h2h["home"])
        add("1X2", "Draw", m["draw"], p[1], h2h["draw"])
        add("1X2", "Away", m["away"], p[2], h2h["away"])
    else:
        add("1X2", "Home", m["home"])
        add("1X2", "Draw", m["draw"])
        add("1X2", "Away", m["away"])

    for line in (1.5, 2.5):
        key = f"O/U {line}"
        ou = markets[key]
        odds = odds_data.get(f"totals_{line}")
        if odds:
            p = devig(odds["over"], odds["under"])
            add(key, f"Over {line}", ou["over"], p[0], odds["over"])
            add(key, f"Under {line}", ou["under"], p[1], odds["under"])
        else:
            add(key, f"Over {line}", ou["over"])
            add(key, f"Under {line}", ou["under"])

    # BTTS is model-only — see the note in fetch_odds(). No market price
    # means no edge and no traffic-light tier; these stay "forecast".
    b = markets["BTTS"]
    add("BTTS", "Yes", b["yes"])
    add("BTTS", "No", b["no"])

    return rows


@st.cache_data(ttl=900, show_spinner=False)
def run_all(target_date: date, model_only: bool = False):
    """Returns (rows_df, notes). One row per selection per fixture.

    model_only=True skips the odds API completely and forecasts every
    league from fixtures.csv — no edges, no tiers, just probabilities.
    """
    all_rows, notes = [], []

    for league, cfg in LEAGUES.items():
        try:
            model = fit_model_for_league(league)
        except Exception as e:
            notes.append(f"{league}: couldn't fit model ({e})")
            continue

        if cfg["has_odds"] and not model_only:
            events, err = fetch_odds(cfg["odds_key"])
            if err:
                notes.append(f"{league}: {err}")
                continue

            events = [e for e in events if _event_date(e) == target_date]
            if not events:
                continue

            missing = set()
            for e in events:
                for side in ("home_team", "away_team"):
                    t = normalise_name(league, e[side])
                    if t not in model["attack"]:
                        missing.add(t)
            if missing:
                model = seed_missing_teams(model, league, list(missing))

            seed_methods = model.get("seed_method", {})
            for e in events:
                home = normalise_name(league, e["home_team"])
                away = normalise_name(league, e["away_team"])
                if home not in model["attack"] or away not in model["attack"]:
                    continue
                seeded = seed_methods.get(home) or seed_methods.get(away)
                grid = score_matrix(home, away, model)
                markets = derive_markets(grid)

                odds_data = {
                    "h2h": _avg_market(e, "h2h", {
                        e["home_team"]: "home", "Draw": "draw", e["away_team"]: "away"}),
                }
                for line in (1.5, 2.5):
                    odds_data[f"totals_{line}"] = _avg_market(
                        e, "totals", {"Over": "over", "Under": "under"}, line=line)

                all_rows.extend(_evaluate(
                    f"{home} vs {away}", league, markets, odds_data, seeded))
        else:
            fixtures, err = fetch_upcoming_fixtures(cfg["data_code"], target_date)
            if err:
                notes.append(f"{league}: {err}")
                continue
            if not fixtures:
                continue

            missing = {t for fx in fixtures for t in fx if t not in model["attack"]}
            if missing:
                model = seed_missing_teams(model, league, list(missing))
            seed_methods = model.get("seed_method", {})

            for home, away in fixtures:
                if home not in model["attack"] or away not in model["attack"]:
                    continue
                grid = score_matrix(home, away, model)
                markets = derive_markets(grid)
                all_rows.extend(_evaluate(
                    f"{home} vs {away}", league, markets, {},
                    seed_methods.get(home) or seed_methods.get(away)))

    return pd.DataFrame(all_rows), notes


# ------------------------------------------------------------- acca builder

def build_acca(rows: pd.DataFrame, leagues: list[str], market: str,
                min_legs=5, max_legs=6, exclude_fixtures=None,
                allow_forecast=False):
    """Picks the highest-confidence qualifying legs for one market within
    one group of leagues. Never reuses a fixture within the acca, and
    skips any fixture already used by another acca in the same group."""
    exclude_fixtures = exclude_fixtures or set()

    # BTTS has no market price (see fetch_odds), so its rows are tier
    # "forecast". Allow those for BTTS only, clearly flagged downstream as
    # model-confidence picks rather than edge-backed ones.
    # BTTS never has a market price on the bulk endpoint; in model-only mode
    # nothing does. Either way those rows are tier "forecast" — allow them,
    # flagged downstream as model-confidence picks rather than edge-backed.
    allowed_tiers = ["green", "amber"]
    if allow_forecast or market == "BTTS":
        allowed_tiers.append("forecast")

    pool = rows[
        (rows["league"].isin(leagues))
        & (rows["market"] == market)
        & (rows["tier"].isin(allowed_tiers))
        & (~rows["fixture"].isin(exclude_fixtures))
    ].copy()

    if pool.empty:
        return None

    pool = pool.sort_values("model_prob", ascending=False)

    legs, used = [], set()
    for _, r in pool.iterrows():
        if r["fixture"] in used:
            continue
        legs.append(r)
        used.add(r["fixture"])
        if len(legs) == max_legs:
            break

    if len(legs) < min_legs:
        return {"legs": legs, "short": True, "used": used} if legs else None

    combined_odds = 1.0
    combined_prob = 1.0
    have_odds = True
    for r in legs:
        if r["odds"] and pd.notna(r["odds"]):
            combined_odds *= float(r["odds"])
        else:
            have_odds = False
        combined_prob *= float(r["model_prob"])

    return {"legs": legs, "short": False, "used": used,
            "combined_odds": combined_odds if have_odds else None,
            "combined_prob": combined_prob}
