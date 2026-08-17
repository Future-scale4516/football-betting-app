"""
Cross-market ranking — same idea as the MLB app's 'rank by Model %'
Suggested Bets view, but across every market type instead of within
one. Takes every evaluated selection (win/draw/away, BTTS yes/no,
over/under at 1.5 and 2.5) from every fixture and ranks them all
together by how confident the model is, regardless of which market
they came from.

Import rank_all_markets() and call it from run_all_leagues.py per
fixture, or accumulate across all fixtures for one big ranked list.
"""


def collect_selections(fixture_label: str, markets: dict,
                        league: str = "Unknown") -> list[dict]:
    """Flattens one fixture's derive_markets() output into a list of
    individually-ranked selections. league is carried through so the
    ranking can be filtered per league in the UI."""
    selections = []

    def add(market, selection, prob):
        selections.append({"league": league, "fixture": fixture_label,
                            "market": market, "selection": selection,
                            "model_prob": prob})

    m = markets["1X2"]
    add("1X2", "Home", m["home"])
    add("1X2", "Draw", m["draw"])
    add("1X2", "Away", m["away"])

    for line_key in ("O/U 1.5", "O/U 2.5"):
        ou = markets[line_key]
        add(line_key, "Over", ou["over"])
        add(line_key, "Under", ou["under"])

    b = markets["BTTS"]
    add("BTTS", "Yes", b["yes"])
    add("BTTS", "No", b["no"])

    return selections


def rank_all_markets(all_selections: list[dict], top_n: int = 15,
                      leagues: list[str] | None = None,
                      markets: list[str] | None = None) -> list[dict]:
    """Sorts every selection across every fixture/market by model
    probability, highest first. This is 'most likely', NOT 'best edge' —
    pair with plausibility.py / traffic_light.py separately if you want
    edge-ranked instead of confidence-ranked.

    leagues: optional list to filter to (None = all leagues).
    markets: optional list to filter to (None = all markets).
    """
    filtered = all_selections
    if leagues:
        filtered = [s for s in filtered if s.get("league") in leagues]
    if markets:
        filtered = [s for s in filtered if s["market"] in markets]

    ranked = sorted(filtered, key=lambda s: -s["model_prob"])
    return ranked[:top_n]


def available_leagues(all_selections: list[dict]) -> list[str]:
    """Leagues actually present in this run — for populating a filter."""
    return sorted({s.get("league", "Unknown") for s in all_selections})


def print_ranking(ranked: list[dict]):
    print(f"\n{'='*70}\nMOST LIKELY ACROSS ALL MARKETS (by model confidence)\n{'='*70}")
    print(f"{'League':18s} {'Fixture':32s} {'Market':10s} {'Pick':8s} {'Model %':>8s}")
    for s in ranked:
        print(f"{s.get('league', 'Unknown'):18s} {s['fixture']:32s} "
              f"{s['market']:10s} {s['selection']:8s} {s['model_prob']:7.1%}")
