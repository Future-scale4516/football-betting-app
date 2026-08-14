"""
Run the full pipeline across every league in league_config.py.

Odds-enabled leagues: model -> live odds -> de-vig -> edge -> plausibility
ceiling -> traffic light -> logged as a candidate pick.
Forecast-only leagues (League One/Two): model -> ratings printed, no
edge/pick logic, clearly labelled.

At the end: Suggested Bets built from every green/amber pick across
ALL odds-enabled leagues combined.

Usage:
    python3 run_all_leagues.py YOUR_ODDS_API_KEY
"""

import sys
import requests
import pandas as pd

from dixon_coles_sketch import fit_league, score_matrix, derive_markets
from league_config import LEAGUES, data_url, normalise_name
from plausibility import check_plausibility
from traffic_light import classify
from suggested_bets import build_combos, print_combos
from results_tracker import log_pick
from rank_all_markets import collect_selections, rank_all_markets, print_ranking

ODDS_API_KEY = sys.argv[1] if len(sys.argv) > 1 else None
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/{key}/odds/"


def fit_model_for_league(league_name: str):
    df = pd.read_csv(data_url(league_name))
    df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    fixtures = list(df.itertuples(index=False, name=None))
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    return fit_league(fixtures, teams)


def fetch_odds(sport_key: str):
    url = ODDS_BASE_URL.format(key=sport_key)
    params = {"apiKey": ODDS_API_KEY, "regions": "uk",
              "markets": "h2h,totals", "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"  Odds API error {resp.status_code}: {resp.text[:150]}")
        return []
    return resp.json()


def devig_h2h(odds_h, odds_d, odds_a):
    imp_h, imp_d, imp_a = 1 / odds_h, 1 / odds_d, 1 / odds_a
    overround = imp_h + imp_d + imp_a
    return imp_h / overround, imp_d / overround, imp_a / overround


def average_h2h(event):
    home_team, away_team = event["home_team"], event["away_team"]
    home_odds, draw_odds, away_odds = [], [], []
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            for o in mkt["outcomes"]:
                if o["name"] == home_team:
                    home_odds.append(o["price"])
                elif o["name"] == away_team:
                    away_odds.append(o["price"])
                elif o["name"] == "Draw":
                    draw_odds.append(o["price"])
    if not (home_odds and draw_odds and away_odds):
        return None
    return (sum(home_odds) / len(home_odds), sum(draw_odds) / len(draw_odds),
            sum(away_odds) / len(away_odds))


def average_totals_at_line(event, line: float):
    """Averages over/under odds across books that quote exactly this line.
    NOTE: only exact-line matches count — a book quoting 2.0 or 3.0 instead
    of 2.5 is skipped rather than approximated."""
    over_odds, under_odds = [], []
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != "totals":
                continue
            for o in mkt["outcomes"]:
                if o.get("point") != line:
                    continue
                if o["name"] == "Over":
                    over_odds.append(o["price"])
                elif o["name"] == "Under":
                    under_odds.append(o["price"])
    if not (over_odds and under_odds):
        return None
    return sum(over_odds) / len(over_odds), sum(under_odds) / len(under_odds)


def devig_two_way(odds_a, odds_b):
    imp_a, imp_b = 1 / odds_a, 1 / odds_b
    overround = imp_a + imp_b
    return imp_a / overround, imp_b / overround


def process_league(league_name: str, config: dict):
    print(f"\n{'#'*70}\n{league_name}\n{'#'*70}")
    print("Fitting model on last season's results...")
    model = fit_model_for_league(league_name)

    if not config["has_odds"]:
        print("[FORECAST ONLY — no odds coverage for this league]")
        ranked = sorted(model["attack"], key=lambda t: -model["attack"][t])
        print(f"Top 5 attack ratings: {', '.join(ranked[:5])}")
        return [], []

    print("Fetching live odds...")
    events = fetch_odds(config["odds_key"])
    picks = []
    all_selections = []

    for event in events:
        home_raw, away_raw = event["home_team"], event["away_team"]
        home = normalise_name(league_name, home_raw)
        away = normalise_name(league_name, away_raw)
        fixture_label = f"{home} vs {away}"

        if home not in model["attack"] or away not in model["attack"]:
            missing = [t for t in (home, away) if t not in model["attack"]]
            print(f"  SKIPPED {fixture_label} — no rating for {missing} "
                  f"(promoted with no history, or name mismatch — check "
                  f"TEAM_NAME_MAPS['{league_name}'] if this team should exist)")
            continue

        grid = score_matrix(home, away, model)
        markets = derive_markets(grid)
        all_selections.extend(collect_selections(fixture_label, markets))

        # --- 1X2 ---
        h2h = average_h2h(event)
        if h2h:
            odds_h, odds_d, odds_a = h2h
            mkt_h, mkt_d, mkt_a = devig_h2h(odds_h, odds_d, odds_a)
            for selection, model_prob, market_prob, odds in [
                ("Home", markets["1X2"]["home"], mkt_h, odds_h),
                ("Draw", markets["1X2"]["draw"], mkt_d, odds_d),
                ("Away", markets["1X2"]["away"], mkt_a, odds_a),
            ]:
                edge = model_prob - market_prob
                status, reason = check_plausibility(model_prob, market_prob, edge)
                tier = classify("1X2", edge) if status == "ok" else "verify"
                print(f"  {fixture_label:35s} 1X2/{selection:5s} "
                      f"model={model_prob:.1%} mkt={market_prob:.1%} "
                      f"edge={edge:+.1%} -> {tier}"
                      + (f"  [{reason}]" if reason else ""))
                if tier in ("green", "amber"):
                    picks.append({"fixture": fixture_label, "market": "1X2",
                                  "selection": selection, "model_prob": model_prob,
                                  "tier": tier})
                    log_pick(league_name, fixture_label, "1X2", selection,
                              model_prob, market_prob, odds, tier)

        # --- Totals (1.5 and 2.5) ---
        for line, market_key in [(1.5, "O/U 1.5"), (2.5, "O/U 2.5")]:
            totals = average_totals_at_line(event, line)
            if not totals:
                continue
            odds_over, odds_under = totals
            mkt_over, mkt_under = devig_two_way(odds_over, odds_under)
            for selection, model_prob, market_prob, odds in [
                (f"Over {line}", markets[market_key]["over"], mkt_over, odds_over),
                (f"Under {line}", markets[market_key]["under"], mkt_under, odds_under),
            ]:
                edge = model_prob - market_prob
                status, reason = check_plausibility(model_prob, market_prob, edge)
                # traffic_light.py only has explicit bands for O/U 2.5 today —
                # 1.5 reuses those same thresholds via MARKET_BANDS lookup;
                # tune separately later if 1.5 behaves differently once real
                # results come in.
                band_key = "O/U 2.5"
                tier = classify(band_key, edge) if status == "ok" else "verify"
                print(f"  {fixture_label:35s} {market_key}/{selection:10s} "
                      f"model={model_prob:.1%} mkt={market_prob:.1%} "
                      f"edge={edge:+.1%} -> {tier}"
                      + (f"  [{reason}]" if reason else ""))
                if tier in ("green", "amber"):
                    picks.append({"fixture": fixture_label, "market": market_key,
                                  "selection": selection, "model_prob": model_prob,
                                  "tier": tier})
                    log_pick(league_name, fixture_label, market_key, selection,
                              model_prob, market_prob, odds, tier)

    return picks, all_selections


if __name__ == "__main__":
    if not ODDS_API_KEY:
        print("Pass your Odds API key: python3 run_all_leagues.py YOUR_KEY")
        sys.exit(1)

    all_picks = []
    all_selections = []
    for league_name, config in LEAGUES.items():
        picks, selections = process_league(league_name, config)
        all_picks.extend(picks)
        all_selections.extend(selections)

    print(f"\n\nTotal qualifying picks across all leagues: {len(all_picks)}")
    combos = build_combos(all_picks)
    print_combos(combos)

    ranked = rank_all_markets(all_selections, top_n=15)
    print_ranking(ranked)
