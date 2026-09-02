"""
Shared engine for the football betting app — the single place that
fits models, fetches odds, computes edges, and renders UI components.
Both the Streamlit pages and run_all_leagues.py import from here so
there's one implementation, not two.
"""

import io
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import streamlit as st

# Odds API timestamps are UTC; fixtures.csv times are already UK-local.
# Streamlit Cloud's server clock runs in UTC regardless of where you are,
# so ANY bare .astimezone()/datetime.now() silently uses server time, not
# UK time — that's what caused kickoff times to show an hour early during
# BST. Everything below converts explicitly to Europe/London instead of
# relying on the server's local timezone.
UK_TZ = ZoneInfo("Europe/London")

CSV_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; football-betting-app)"}


def _fetch_csv(url: str) -> pd.DataFrame:
    """Bare pd.read_csv(url) sends no User-Agent, and football-data.co.uk
    sometimes returns an ambiguous HTTP 300 to requests like that instead
    of the file. Fetching with a normal header fixes that.

    Also handles: an HTML page returned instead of a CSV (usually means
    the file doesn't exist yet), and a stray short/blank row mid-file
    (falls back to skipping just that row)."""
    resp = requests.get(url, headers=CSV_HEADERS, timeout=20)
    resp.raise_for_status()

    text = resp.text
    if text.lstrip()[:1] == "<":
        raise ValueError(
            f"server returned a webpage instead of a CSV — this file "
            f"likely doesn't exist yet ({url})"
        )

    try:
        return pd.read_csv(io.StringIO(text))
    except pd.errors.ParserError:
        return pd.read_csv(io.StringIO(text), engine="python",
                            on_bad_lines="skip")

from dixon_coles_sketch import fit_league, score_matrix, derive_markets
from league_config import LEAGUES, data_url, normalise_name
from plausibility import check_plausibility
from traffic_light import classify
from promoted_seeding import seed_missing_teams
from env_config import require_api_key
from football_data_source import load_results as _load_current_season_results

ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/{key}/odds/"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Groups used by the Suggested Bets page
ENGLISH_EXCL_EPL = ["EFL Championship", "League One", "League Two"]
REST_OF_EUROPE = ["Bundesliga", "Serie A", "La Liga", "Ligue 1"]

ACCA_MARKETS = ["1X2", "BTTS", "O/U 2.5"]

TIER_ICON = {"green": "🟢", "amber": "🟡", "verify": "🔵", "red": "⚪"}


def market_label(market: str, selection: str) -> str:
    """One combined label per market+selection, e.g. 'Match Winner - Home',
    'BTTS - Yes', 'Over 2.5'. Replaces the old two-dropdown Market +
    Selection filter, which was visually noisy and needed both set
    correctly to isolate one thing."""
    if market == "1X2":
        return f"Match Winner - {selection}"
    if market == "BTTS":
        return f"BTTS - {selection}"
    if market.startswith("O/U"):
        return selection          # already reads "Over 2.5" / "Under 1.5"
    return f"{market} - {selection}"


# Sensible default so the page doesn't open with everything at once.
DEFAULT_MARKET_LABELS = ["Match Winner - Home", "Match Winner - Draw",
                          "Match Winner - Away"]


def market_label_options(rows) -> list[str]:
    """All labels present, ordered sensibly rather than alphabetically."""
    present = set(rows["market_label"].unique())
    preferred = ["Match Winner - Home", "Match Winner - Draw", "Match Winner - Away",
                 "BTTS - Yes", "BTTS - No",
                 "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5"]
    ordered = [m for m in preferred if m in present]
    return ordered + sorted(present - set(ordered))


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


def render_pick_card(icon, headline, subline, metrics, reason=None,
                      kickoff=None, started=False):
    """Card layout instead of a table — much easier to read on mobile,
    same approach as the MLB app."""
    with st.container(border=True):
        st.markdown(f"**{icon + ' ' if icon else ''}{headline}**")
        sub = subline
        if kickoff:
            sub = f"{sub} · KO {kickoff}"
        st.caption(sub)
        if started:
            st.warning(
                "⚠️ This match has already kicked off — this prediction was "
                "made before or after kickoff without knowing the live score "
                "or events. Treat it as stale, not a real-time read.",
                icon="⏱️",
            )
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
    """Odds API gives commence_time as ISO UTC — converted to UK time
    explicitly (see UK_TZ note above) rather than server-local."""
    ts = event.get("commence_time")
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UK_TZ).date()


def _event_kickoff(event):
    """Returns (kickoff_local_str, started_bool) from an Odds API event,
    in UK time regardless of what timezone the server itself runs in."""
    ts = event.get("commence_time")
    if not ts:
        return None, False
    dt_uk = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UK_TZ)
    started = dt_uk <= datetime.now(UK_TZ)
    return dt_uk.strftime("%H:%M"), started


def _fixture_kickoff(target_date: date, time_str):
    """Returns (kickoff_str, started_bool) for a fixtures.csv row. The
    time itself is already UK-local as published, so it's shown as-is —
    only the 'has this started' comparison needs an explicit UK 'now'."""
    today_uk = datetime.now(UK_TZ).date()
    if target_date < today_uk:
        return time_str, True
    if target_date > today_uk:
        return time_str, False
    if not time_str or not isinstance(time_str, str):
        return time_str, False
    try:
        kt = datetime.strptime(time_str.strip(), "%H:%M").time()
    except ValueError:
        return time_str, False
    return time_str, datetime.now(UK_TZ).time() >= kt


def kickoff_sort_key(kickoff):
    """Sorts 'HH:MM' strings chronologically; missing/unparseable kickoff
    times sort last rather than crashing or sorting as if past midnight."""
    if not kickoff or not isinstance(kickoff, str):
        return (1, 0)
    try:
        h, m = kickoff.split(":")
        return (0, int(h) * 60 + int(m))
    except (ValueError, AttributeError):
        return (1, 0)


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
    the main reason model-only mode felt slow.

    Doesn't assume the header is line 1 — football-data.co.uk's combined
    fixtures file has occasionally had stray content before the real
    header row, which pandas would otherwise happily parse as column
    names, silently producing a dataframe with no 'Div' column and no
    error. Searching for the real header line fixes that regardless of
    what (if anything) comes before it.
    """
    resp = requests.get(FIXTURES_URL, headers=CSV_HEADERS, timeout=20)
    resp.raise_for_status()

    lines = resp.text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("Div,Date")), None)
    if header_idx is None:
        raise ValueError(
            "couldn't find the expected 'Div,Date,...' header row anywhere "
            "in fixtures.csv — the file's format may have changed"
        )

    clean_text = "\n".join(lines[header_idx:])
    try:
        return pd.read_csv(io.StringIO(clean_text))
    except pd.errors.ParserError:
        return pd.read_csv(io.StringIO(clean_text), engine="python",
                            on_bad_lines="skip")


def fetch_upcoming_fixtures(div_code: str, target: date):
    """For leagues without odds (or every league in model-only mode),
    pull the fixture list so their games can still be forecast."""
    try:
        df = _load_fixtures_csv()
    except Exception as e:
        return [], f"couldn't load fixtures.csv: {e}"

    if "Div" not in df.columns:
        return [], (f"fixtures.csv missing expected 'Div' column — "
                     f"found instead: {list(df.columns)[:8]}")

    df = df[df["Div"] == div_code].copy()
    if df.empty:
        return [], None

    try:
        df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
        df = df[df["_date"] == target]
    except Exception:
        return [], "couldn't parse dates in fixtures.csv"

    has_time = "Time" in df.columns
    out = []
    for r in df.itertuples():
        time_str = getattr(r, "Time", None) if has_time else None
        kickoff, started = _fixture_kickoff(target, time_str)
        out.append((r.HomeTeam, r.AwayTeam, kickoff, started))
    return out, None


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
def fit_model_for_league(league_name: str, as_of_date: date = None):
    """Fits on last season in full, plus this season's completed games
    strictly before as_of_date.

    Added after results tracking showed goals running well above what a
    last-season-only fit expected (BTTS/Over picks underconfident, Under
    picks overconfident across most leagues) — this was previously fitting
    on last season ONLY, with no way to pick up this season's actual form
    as it happens, even though the Results page's backtest already did
    this blending. Live picks were silently stuck a full season behind.
    """
    df = _fetch_csv(data_url(league_name))
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    fixtures = list(df.itertuples(index=False, name=None))

    if as_of_date is not None:
        cfg = LEAGUES[league_name]
        current, _ = _load_current_season_results(cfg["data_code"], as_of_date)
        if current is not None:
            prior = current[(current["_date"] < as_of_date)
                             & current["FTHG"].notna() & current["FTAG"].notna()]
            fixtures += list(
                prior[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
                .itertuples(index=False, name=None))

    teams = sorted({t for fx in fixtures for t in (fx[0], fx[1])})
    return fit_league(fixtures, teams)


def _evaluate(fixture, league, markets, odds_data, seeded, kickoff=None, started=False):
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
            "kickoff": kickoff, "started": started,
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
            model = fit_model_for_league(league, target_date)
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
                kickoff, started = _event_kickoff(e)
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
                    f"{home} vs {away}", league, markets, odds_data, seeded,
                    kickoff=kickoff, started=started))
        else:
            fixtures, err = fetch_upcoming_fixtures(cfg["data_code"], target_date)
            if err:
                notes.append(f"{league}: {err}")
                continue
            if not fixtures:
                continue

            missing = {t for fx in fixtures for t in (fx[0], fx[1])
                       if t not in model["attack"]}
            if missing:
                model = seed_missing_teams(model, league, list(missing))
            seed_methods = model.get("seed_method", {})

            for home, away, kickoff, started in fixtures:
                if home not in model["attack"] or away not in model["attack"]:
                    continue
                grid = score_matrix(home, away, model)
                markets = derive_markets(grid)
                all_rows.extend(_evaluate(
                    f"{home} vs {away}", league, markets, {},
                    seed_methods.get(home) or seed_methods.get(away),
                    kickoff=kickoff, started=started))

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
