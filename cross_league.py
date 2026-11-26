"""
A model read on UEFA competitions without needing ClubElo.

THE PROBLEM
Europa League and Conference League games pair clubs from different leagues.
Dixon-Coles ratings are fitted one league at a time, so "attack = +0.30" in the
Eredivisie and "+0.30" in Ligue 1 are not the same thing — there are no games
linking the two, so the scales float independently. That is why these
competitions had no model: ClubElo was the only thing putting every club on one
scale, and its server has been down since 5 Nov 2026.

WHAT IS COMPARABLE ACROSS LEAGUES
Not ratings — goal RATES. "This team scores 1.9 goals a game against an average
opponent in its own league" is a number in goals, and it means the same thing
everywhere. So is "this team concedes 25% more than its league average".

THE SPLIT
  * WHO WINS comes from the bookmakers' de-vigged 1X2 price. This is exactly the
    part that depends on cross-league strength (is Rennes better than NEC?), and
    exactly the part the market prices most sharply. Guessing it ourselves is
    where a home-made cross-league model would go wrong.
  * HOW MANY GOALS comes from the two clubs' own scoring and conceding rates.
    This is the model's actual contribution, and it is what BTTS and the
    Over/Under lines are made of.

Then one scoreline grid is solved that has the model's total goals AND the
market's home:away ratio, and BTTS / Over / Under are read off that grid.

WHY THIS IS NOT THE SAME AS THE EXISTING "MARKET-IMPLIED" BTTS
That one (club_ratings.implied_grid) needs an Over 2.5 PRICE and simply restates
what the bookmakers already said — it carries no opinion, so it can never
disagree with them. This one takes the total from the clubs' own scoring, so it
CAN disagree, which is the only way a BTTS or Over/Under number is worth having.

HONEST LIMITS (the app shows every one of these games as "verify")
  * Not backtested on UEFA games — we have no UEFA results history, so there is
    no way to score it on the competition it is used for. What HAS been tested
    is the anchoring method itself, on domestic games: see validate_anchor.py.
  * Coverage is only as good as the reference leagues in league_config.py. A
    club from a country we hold no data for still falls back to market-only.
  * Two leagues' defences are not equally good, and the opponent adjustment
    below only corrects for a club being better or worse than ITS OWN league.
    Cross-league defensive quality is a second-order error left in the totals.
  * European nights are not league games: rotation, travel and two-legged ties
    all change how teams play, and none of that is in here.
"""
from datetime import date, timedelta
from math import factorial

import numpy as np
import pandas as pd

from dixon_coles_sketch import fit_league, apply_level_correction
from league_config import LEAGUES, REFERENCE_LEAGUES, SEASON
from shot_goals import build_fit_rows

MAX_GOALS = 8
DEFAULT_RHO = -0.08          # same low-score correction as the domestic model
MIN_GAMES = 40               # a league needs this many results to be usable
HOME_ADV_NEUTRAL = 0.22      # only used when there is no market price to anchor on


# --------------------------------------------------------------- data loading
#
# Two file layouts live in data/:
#   "standard"  — the familiar football-data.co.uk league file (HomeTeam,
#                 AwayTeam, FTHG, FTAG), one file per season.
#   "extra"     — football-data.co.uk/new/<CODE>.csv, one file per country
#                 covering many seasons, with Home/Away/HG/AG column names.
# Both are read into the same shape so the fitter doesn't care which it got.

_STD = ("HomeTeam", "AwayTeam", "FTHG", "FTAG")
_EXTRA = ("Home", "Away", "HG", "AG")


def _normalise(df):
    """-> DataFrame with HomeTeam/AwayTeam/FTHG/FTAG/_date, or None."""
    if df is None or df.empty:
        return None
    cols = set(df.columns)
    if set(_STD) <= cols:
        home, away, hg, ag = _STD
    elif set(_EXTRA) <= cols:
        home, away, hg, ag = _EXTRA
    else:
        return None
    if "Date" not in cols:
        return None

    out = pd.DataFrame({
        "HomeTeam": df[home].astype(str).str.strip(),
        "AwayTeam": df[away].astype(str).str.strip(),
        "FTHG": pd.to_numeric(df[hg], errors="coerce"),
        "FTAG": pd.to_numeric(df[ag], errors="coerce"),
        "_date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date,
    })
    # Shot counts, where the file has them, let build_fit_rows use the
    # shot-implied-goals blend. The extra-format files have none.
    for src, dst in (("HS", "HS"), ("AS", "AS"), ("HST", "HST"), ("AST", "AST")):
        if src in cols:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
    return out.dropna(subset=["_date", "HomeTeam", "AwayTeam"])


def _load_reference(data_dir, code, fmt, target_date, lookback_days=560):
    """Results for one reference league, from the committed snapshot only.

    Reference leagues are never fetched live: Streamlit Cloud is blocked by
    football-data.co.uk, so these files arrive through refresh_data.py.
    """
    frames = []
    if fmt == "extra":
        names = [f"X_{code}.csv"]
    else:
        names = [f"{SEASON}_{code}.csv", f"{_season_code(target_date)}_{code}.csv"]
    for name in names:
        path = data_dir / name
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            try:
                raw = pd.read_csv(path, encoding="utf-8-sig",
                                  engine="python", on_bad_lines="skip")
            except Exception:
                continue
        got = _normalise(raw)
        if got is not None:
            frames.append(got)
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    cutoff = target_date - timedelta(days=lookback_days)
    return df[(df._date >= cutoff) & (df._date < target_date)]


def _season_code(d):
    start = d.year if d.month >= 7 else d.year - 1
    return f"{str(start)[-2:]}{str(start + 1)[-2:]}"


# ------------------------------------------------------------------ profiles

def _profile_from_model(model, league_name):
    """Per-team absolute goal rates, in goals per game against an average
    opponent in that team's own league, averaged over home and away."""
    attack, defence = model["attack"], model["defence"]
    scale = model.get("goal_scale", 1.0)
    ha = model.get("home_adv", 0.0)
    # log-mean-exp, NOT the mean of the ratings: goal rates are exp(rating),
    # and exp(mean) < mean(exp), so averaging the ratings first shrinks every
    # rate toward the middle. (Measured: it cost 2 percentage points of
    # expected goals and made the anchored totals read low.)
    a_avg = float(np.log(np.mean(np.exp(list(attack.values())))))
    d_avg = float(np.log(np.mean(np.exp(list(defence.values())))))
    # A team plays half its games at home, so average the two cases rather
    # than pretending every game is neutral.
    swing = (1.0 + np.exp(ha)) / 2.0

    out = {}
    for t in attack:
        out[t] = {
            "att": float(scale * np.exp(attack[t] + d_avg) * swing),
            "dfn": float(scale * np.exp(a_avg + defence[t]) * swing),
            "league": league_name,
        }
    lg_avg = float(np.mean([v["att"] for v in out.values()])) if out else None
    for v in out.values():
        v["league_avg"] = lg_avg
    return out


def build_profiles(target_date, data_dir, fit_modelled=None):
    """{team name: profile} across every modelled and reference league.

    fit_modelled(league_name) -> fitted model, so the app reuses its own
    cached fits for the nine leagues it already models instead of refitting.
    Returns (profiles, notes).
    """
    profiles, notes = {}, []
    seen_league = {}

    if fit_modelled is not None:
        for name, cfg in LEAGUES.items():
            if not cfg.get("data_code"):
                continue
            try:
                model = fit_modelled(name)
            except Exception as e:
                notes.append(f"{name}: no profiles ({type(e).__name__})")
                continue
            _merge(profiles, _profile_from_model(model, name), seen_league, name)

    for name, cfg in REFERENCE_LEAGUES.items():
        df = _load_reference(data_dir, cfg["code"], cfg.get("format", "standard"),
                             target_date)
        if df is None:
            notes.append(f"{name}: no data file (run refresh_data.py and push)")
            continue
        played = df.dropna(subset=["FTHG", "FTAG"])
        if len(played) < MIN_GAMES:
            notes.append(f"{name}: only {len(played)} results — skipped")
            continue
        try:
            rows = build_fit_rows([played], target_date)
            teams = sorted({t for r in rows for t in (r[0], r[1])})
            model = fit_league(rows, teams)
            this_season = played[played._date >= date(target_date.year
                                                      if target_date.month >= 7
                                                      else target_date.year - 1, 7, 1)]
            if len(this_season):
                model = apply_level_correction(model, list(zip(
                    this_season.HomeTeam, this_season.AwayTeam,
                    this_season.FTHG.astype(int), this_season.FTAG.astype(int))))
        except Exception as e:
            notes.append(f"{name}: couldn't fit ({type(e).__name__})")
            continue
        _merge(profiles, _profile_from_model(model, name), seen_league, name)

    return profiles, notes


def _merge(profiles, new, seen_league, league_name):
    """A club name that appears in two leagues (promotion, or two countries
    spelling a name the same way) is ambiguous — drop it rather than guess."""
    for team, prof in new.items():
        if team in profiles and seen_league.get(team) != league_name:
            profiles[team] = None          # tombstone: known-ambiguous
        elif profiles.get(team) is None and team in profiles:
            continue
        else:
            profiles[team] = prof
            seen_league[team] = league_name


# ------------------------------------------------------------- the goal model

def expected_goals(home_prof, away_prof, home_advantage=0.0):
    """(lambda_home, lambda_away) before any market anchoring.

    Each side's rate is what it scores against an average opponent in its own
    league, adjusted for how leaky THIS opponent is relative to its league.
    That ratio is dimensionless, so it carries across leagues; the absolute
    rate is in goals, so it does too.
    """
    def leak(p):
        if not p.get("league_avg"):
            return 1.0
        return p["dfn"] / p["league_avg"]

    lam_h = home_prof["att"] * leak(away_prof)
    lam_a = away_prof["att"] * leak(home_prof)
    swing = np.exp(home_advantage / 2.0)
    return float(lam_h * swing), float(lam_a / swing)


def _grid(lam_h, lam_a, rho=DEFAULT_RHO, max_goals=MAX_GOALS):
    k = np.arange(max_goals + 1)
    fact = np.array([factorial(int(i)) for i in k], dtype=float)
    ph = np.exp(-lam_h) * lam_h ** k / fact
    pa = np.exp(-lam_a) * lam_a ** k / fact
    g = np.outer(ph, pa)
    # Dixon-Coles low-score correction
    g[0, 0] *= 1 - lam_h * lam_a * rho
    g[0, 1] *= 1 + lam_h * rho
    g[1, 0] *= 1 + lam_a * rho
    g[1, 1] *= 1 - rho
    return g / g.sum()


def anchored_grid(home_prof, away_prof, p_home, p_away, rho=DEFAULT_RHO):
    """Scoreline grid with the MODEL's total goals and the MARKET's home:away
    ratio. p_home/p_away are de-vigged 1X2 probabilities.

    Solves for the goal supremacy s that reproduces the market's split, holding
    the total fixed — so BTTS and the Over/Under lines come out of the model's
    scoring view, not the bookmakers' totals price.
    """
    lam_h, lam_a = expected_goals(home_prof, away_prof)
    total = lam_h + lam_a
    if not np.isfinite(total) or total <= 0.2:
        return None
    target = p_home / (p_home + p_away)

    def ratio_at(s):
        g = _grid((total + s) / 2, (total - s) / 2, rho)
        h = np.tril(g, -1).sum()
        a = np.triu(g, 1).sum()
        return h / (h + a)

    lo, hi = -(total - 0.15), total - 0.15
    for _ in range(50):                      # ratio_at rises with s
        mid = (lo + hi) / 2
        if ratio_at(mid) < target:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return _grid((total + s) / 2, (total - s) / 2, rho)


def model_only_grid(home_prof, away_prof, rho=DEFAULT_RHO):
    """No market price available — the clubs' rates alone, with a typical home
    advantage. Weaker on the result than the anchored version, by exactly the
    amount the cross-league scale is unreliable."""
    lam_h, lam_a = expected_goals(home_prof, away_prof, HOME_ADV_NEUTRAL)
    if not np.isfinite(lam_h + lam_a) or lam_h + lam_a <= 0.2:
        return None
    return _grid(lam_h, lam_a, rho)


# ------------------------------------------------------------- name matching

def find_profile(name, profiles, alias_lookup=None):
    """(profile, matched_name). Exact first, then the shared alias/fuzzy
    matcher. Returns (None, None) when the club isn't covered or its name is
    ambiguous across two leagues."""
    if not name:
        return None, None
    if name in profiles:
        return (profiles[name], name) if profiles[name] else (None, None)
    if alias_lookup:
        hit = alias_lookup(name, profiles)
        if hit and profiles.get(hit):
            return profiles[hit], hit
    return None, None
