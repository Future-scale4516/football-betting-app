"""
Cross-league club ratings for competitions the Dixon-Coles model can't
cover (UEFA club competitions, the FA Cup).

WHY A SEPARATE MODEL
--------------------
Dixon-Coles ratings are relative to each league's own average, so a "+0.3
attack" in the Premier League and in League Two mean very different things.
Competitions that mix leagues need one shared scale. ClubElo (clubelo.com)
publishes exactly that: one Elo rating per club across Europe, built from
every cross-league result (European ties, cups). refresh_data.py saves a
snapshot to data/clubelo.csv; this module turns two ratings into a scoreline
grid, so 1X2, Over/Under and BTTS all come off one set of numbers, as in
the main model.

HOW A RATING GAP BECOMES GOALS
------------------------------
1. Elo expected score for the home side:
       W = 1 / (1 + 10 ** (-(elo_home - elo_away + HOME_ADV) / 400))
   (W = P(home win) + half of P(draw)).
2. Set the match's total expected goals (a per-competition average,
   scaled up for lopsided games) and solve for the goal split whose Poisson grid reproduces W.
3. Dixon-Coles low-score correction with a fixed rho.

THIS IS NOT BACKTESTED. The main model's own history showed that a model
which isn't checked against the market tends to lose to it, and nothing
here has been checked yet: HOME_ADV (60) is set so two equal teams get
~58% of the points at home, a typical football figure; total goals are
rough per-competition averages. Output is therefore always flagged as
unvalidated in the app. Judge it against log_odds.py / clv_report.py data
before trusting any edge.
"""

import csv
import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from team_matching import match_team

DATA_DIR = Path(__file__).parent / "data"
ELO_FILE = "clubelo.csv"

HOME_ADV = 60.0     # Elo points

# Total goals grow with the rating gap. Fitted on 4,963 domestic games
# (9 leagues, 2025/26 + 2026/27) using bookmaker-implied gaps:
#     total goals ~ (1 + 0.294 * gap) where gap = |Elo difference| / 400
# (2.6 goals in even games up to ~3.2 in the biggest mismatches).
# Normalised so avg_goals applies to a typical gap of 0.25.
GAP_GOAL_SLOPE = 0.294
TYPICAL_GAP = 0.25
DEFAULT_RHO = -0.08
MAX_GOALS = 8


# ------------------------------------------------------------------ loading

def load_elo(on_date: date, data_dir: Path = DATA_DIR):
    """{club name: elo} valid on on_date from the committed snapshot, or
    None if there is no snapshot."""
    path = data_dir / ELO_FILE
    if not path.exists():
        return None
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                elo = float(row["Elo"])
                club = row["Club"].strip()
            except (KeyError, ValueError, AttributeError):
                continue
            frm, to = row.get("From"), row.get("To")
            try:
                if frm and date.fromisoformat(frm) > on_date:
                    continue
                if to and date.fromisoformat(to) < on_date:
                    continue
            except ValueError:
                pass
            out[club] = elo
    return out or None


def snapshot_date(data_dir: Path = DATA_DIR):
    path = data_dir / ELO_FILE
    if not path.exists():
        return None
    return date.fromtimestamp(path.stat().st_mtime)


# ---------------------------------------------------------------- matching

def _key(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]", "", s)


# Odds-API spelling -> likely ClubElo spellings (first one that exists wins).
# Not exhaustive: anything unmatched falls back to the bookmakers' own
# prices and is listed in the app's notes so this table can grow.
ALIASES = {
    "paris saint germain": ["Paris SG", "Paris Saint-Germain"],
    "bayern munich": ["Bayern", "Bayern Munich"],
    "inter milan": ["Inter", "Inter Milan"],
    "ac milan": ["Milan", "AC Milan"],
    "atletico madrid": ["Atletico", "Atletico Madrid"],
    "borussia dortmund": ["Dortmund"],
    "bayer leverkusen": ["Leverkusen"],
    "rb leipzig": ["RB Leipzig", "Leipzig"],
    "borussia monchengladbach": ["Gladbach", "Monchengladbach"],
    "eintracht frankfurt": ["Frankfurt", "Eintracht Frankfurt"],
    "manchester city": ["Man City"],
    "manchester united": ["Man United"],
    "tottenham hotspur": ["Tottenham"],
    "newcastle united": ["Newcastle"],
    "wolverhampton wanderers": ["Wolves"],
    "nottingham forest": ["Forest", "Nottingham Forest"],
    "west ham united": ["West Ham"],
    "brighton and hove albion": ["Brighton"],
    "aston villa": ["Aston Villa"],
    "sporting cp": ["Sporting", "Sporting CP"],
    "sporting lisbon": ["Sporting"],
    "sl benfica": ["Benfica"],
    "fc porto": ["Porto"],
    "psv eindhoven": ["PSV"],
    "feyenoord": ["Feyenoord"],
    "ajax": ["Ajax"],
    "red star belgrade": ["Crvena Zvezda", "Red Star", "Red Star Belgrade"],
    "olympiakos piraeus": ["Olympiakos", "Olympiacos"],
    "olympiacos": ["Olympiakos", "Olympiacos"],
    "club brugge": ["Brugge", "Club Brugge"],
    "union saint gilloise": ["St Gillis", "Union SG", "Union Saint-Gilloise"],
    "athletic bilbao": ["Bilbao", "Athletic Club", "Athletic Bilbao"],
    "real sociedad": ["Sociedad", "Real Sociedad"],
    "real betis": ["Betis", "Real Betis"],
    "celta vigo": ["Celta", "Celta Vigo"],
    "sparta prague": ["Sparta Praha", "Sparta Prague"],
    "slavia prague": ["Slavia Praha", "Slavia Prague"],
    "viktoria plzen": ["Plzen", "Viktoria Plzen"],
    "dinamo zagreb": ["Dinamo Zagreb"],
    "shakhtar donetsk": ["Shakhtar"],
    "fk crvena zvezda": ["Crvena Zvezda", "Red Star"],
    "rb salzburg": ["Salzburg", "RB Salzburg"],
    "red bull salzburg": ["Salzburg", "RB Salzburg"],
    "sturm graz": ["Sturm Graz"],
    "young boys": ["Young Boys"],
    "ferencvaros": ["Ferencvaros"],
    "inter club d escaldes": ["Inter d'Escaldes", "Inter Escaldes", "Inter Club d'Escaldes"],
    "rapid vienna": ["Rapid Wien", "Rapid Vienna"],
    "hapoel beer sheva": ["Hapoel Beer Sheva", "Hapoel Be'er Sheva"],
    "maccabi tel aviv": ["Maccabi Tel Aviv"],
    "qarabag": ["Qarabag"],
    "galatasaray": ["Galatasaray"],
    "fenerbahce": ["Fenerbahce"],
    "besiktas": ["Besiktas"],
    "celtic": ["Celtic"],
    "rangers": ["Rangers"],
    "milton keynes dons": ["MK Dons", "Milton Keynes Dons"],
    "afc wimbledon": ["Wimbledon", "AFC Wimbledon"],
    "sheffield wednesday": ["Sheffield Weds", "Sheffield Wednesday"],
    "sheffield united": ["Sheffield United"],
    "leyton orient": ["Leyton Orient", "Orient"],
    "queens park rangers": ["QPR"],
    "west bromwich albion": ["West Brom"],
    "wycombe wanderers": ["Wycombe"],
    "peterborough united": ["Peterborough"],
    "stockport county": ["Stockport"],
    "notts county": ["Notts County"],
    "tranmere rovers": ["Tranmere"],
    "bristol rovers": ["Bristol Rovers"],
    "forest green rovers": ["Forest Green"],
}


def find_elo(name, elo_map):
    """(elo, matched_name) for a club name from another source, or
    (None, None). Prefers returning nothing over a wrong match."""
    keys = {_key(c): c for c in elo_map}
    k = _key(name)
    if k in keys:
        return elo_map[keys[k]], keys[k]
    for cand in ALIASES.get(re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ",
            unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore")
            .decode().lower())).strip(), []):
        if _key(cand) in keys:
            c = keys[_key(cand)]
            return elo_map[c], c
    m = match_team(name, list(elo_map))
    if m:
        return elo_map[m], m
    return None, None


# ------------------------------------------------------------------- model

def _tau(h, a, lh, la, rho):
    if h == 0 and a == 0:
        return 1 - lh * la * rho
    if h == 0 and a == 1:
        return 1 + lh * rho
    if h == 1 and a == 0:
        return 1 + la * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def _grid(lh, la, rho):
    g = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    ph, pa = poisson.pmf(np.arange(MAX_GOALS + 1), lh), poisson.pmf(np.arange(MAX_GOALS + 1), la)
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            g[h, a] = ph[h] * pa[a] * _tau(h, a, lh, la, rho)
    return g / g.sum()


def expected_score(elo_home, elo_away, home_adv=HOME_ADV):
    return 1 / (1 + 10 ** (-(elo_home - elo_away + home_adv) / 400))


def total_goals_for(avg_goals, elo_home, elo_away):
    gap = abs(elo_home - elo_away) / 400
    return avg_goals * (1 + GAP_GOAL_SLOPE * gap) / (1 + GAP_GOAL_SLOPE * TYPICAL_GAP)


def score_grid(elo_home, elo_away, avg_goals, home_adv=HOME_ADV, rho=DEFAULT_RHO):
    """Scoreline probability grid for two clubs on the shared Elo scale.
    avg_goals is the competition's typical total goals per game."""
    total_goals = total_goals_for(avg_goals, elo_home, elo_away)
    w = expected_score(elo_home, elo_away, home_adv)
    lo, hi = -(total_goals - 0.1), total_goals - 0.1   # goal supremacy bounds

    def score_at(s):
        g = _grid((total_goals + s) / 2, (total_goals - s) / 2, rho)
        return np.tril(g, -1).sum() + 0.5 * np.trace(g)

    for _ in range(40):                  # bisection: score_at rises with s
        mid = (lo + hi) / 2
        if score_at(mid) < w:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return _grid((total_goals + s) / 2, (total_goals - s) / 2, rho)


# ------------------------------------------------ market-implied goal grid
#
# When there are no club ratings (or a club isn't in them), the bookmakers'
# own prices still say a lot: 1X2 + Over/Under 2.5 fix both teams' expected
# goals. Solving for the two Poisson rates that reproduce those prices gives
# a scoreline grid, and BTTS (which the bulk odds feed doesn't price) falls
# out of it. This is NOT a model — it simply restates what the market's
# priced markets imply, so it carries no edge by construction.

def implied_grid(p_home, p_draw, p_over25, rho=DEFAULT_RHO, tol=0.02):
    """Scoreline grid reproducing de-vigged 1X2 and Over 2.5 probabilities,
    or None if no pair of goal rates fits them within `tol`."""
    from scipy.optimize import least_squares

    goals = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))

    def probs(x):
        g = _grid(x[0], x[1], rho)
        return np.array([np.tril(g, -1).sum(), np.trace(g), g[goals > 2.5].sum()])

    target = np.array([p_home, p_draw, p_over25])
    best = None
    for start in ((1.4, 1.1), (2.0, 0.7), (0.8, 1.8)):
        r = least_squares(lambda x: probs(x) - target, start, bounds=(0.05, 6.0))
        if best is None or r.cost < best.cost:
            best = r
    if best is None or np.max(np.abs(probs(best.x) - target)) > tol:
        return None
    return _grid(best.x[0], best.x[1], rho)


def implied_btts(p_home, p_draw, p_over25):
    """P(both teams score) implied by the 1X2 and Over 2.5 prices, or None."""
    g = implied_grid(p_home, p_draw, p_over25)
    if g is None:
        return None
    return float(1 - g[0, :].sum() - g[:, 0].sum() + g[0, 0])
