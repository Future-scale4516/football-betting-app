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


def collect_selections(fixture_label: str, markets: dict) -> list[dict]:
    """Flattens one fixture's derive_markets() output into a list of
    individually-ranked selections."""
    selections = []

    m = markets["1X2"]
    selections.append({"fixture": fixture_label, "market": "1X2",
                        "selection": "Home", "model_prob": m["home"]})
    selections.append({"fixture": fixture_label, "market": "1X2",
                        "selection": "Draw", "model_prob": m["draw"]})
    selections.append({"fixture": fixture_label, "market": "1X2",
                        "selection": "Away", "model_prob": m["away"]})

    for line_key, label in [("O/U 1.5", "O/U 1.5"), ("O/U 2.5", "O/U 2.5")]:
        ou = markets[line_key]
        selections.append({"fixture": fixture_label, "market": label,
                            "selection": "Over", "model_prob": ou["over"]})
        selections.append({"fixture": fixture_label, "market": label,
                            "selection": "Under", "model_prob": ou["under"]})

    b = markets["BTTS"]
    selections.append({"fixture": fixture_label, "market": "BTTS",
                        "selection": "Yes", "model_prob": b["yes"]})
    selections.append({"fixture": fixture_label, "market": "BTTS",
                        "selection": "No", "model_prob": b["no"]})

    return selections


def rank_all_markets(all_selections: list[dict], top_n: int = 15) -> list[dict]:
    """Sorts every selection across every fixture/market by model
    probability, highest first. This is 'most likely', NOT 'best edge' —
    pair with plausibility.py / traffic_light.py separately if you want
    edge-ranked instead of confidence-ranked."""
    ranked = sorted(all_selections, key=lambda s: -s["model_prob"])
    return ranked[:top_n]


def print_ranking(ranked: list[dict]):
    print(f"\n{'='*70}\nMOST LIKELY ACROSS ALL MARKETS (by model confidence)\n{'='*70}")
    print(f"{'Fixture':35s} {'Market':10s} {'Pick':8s} {'Model %':>8s}")
    for s in ranked:
        print(f"{s['fixture']:35s} {s['market']:10s} {s['selection']:8s} "
              f"{s['model_prob']:7.1%}")
