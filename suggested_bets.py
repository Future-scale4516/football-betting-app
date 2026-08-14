"""
Suggested Bets — combo builder.

Takes a flat list of qualifying picks (green/amber, already passed the
plausibility ceiling) and builds doubles/trebles, same structure as the
MLB app's parlay logic. Stake weight is tier-based for now (green=full,
amber=half) — same "weight by earned trust" idea, using trust_tier as
the input until results_tracker.py can supply a real per-market
trust score.
"""

from itertools import combinations

TIER_STAKE_WEIGHT = {"green": 1.0, "amber": 0.5}


def build_combos(picks: list[dict], combo_sizes=(2, 3)) -> list[dict]:
    """
    picks: each a dict with at minimum:
        fixture, market, selection, model_prob, tier ('green'/'amber')
    Returns a list of combo dicts with combined probability and stake weight.
    """
    qualifying = [p for p in picks if p["tier"] in ("green", "amber")]
    combos = []

    for size in combo_sizes:
        for combo in combinations(qualifying, size):
            # don't combine two legs from the same fixture — correlated outcomes
            fixtures_in_combo = {leg["fixture"] for leg in combo}
            if len(fixtures_in_combo) != size:
                continue

            combined_prob = 1.0
            for leg in combo:
                combined_prob *= leg["model_prob"]

            min_tier_weight = min(TIER_STAKE_WEIGHT[leg["tier"]] for leg in combo)

            combos.append({
                "legs": [f"{leg['fixture']} — {leg['market']}: {leg['selection']}"
                         for leg in combo],
                "size": size,
                "combined_prob": combined_prob,
                "stake_weight": min_tier_weight,
            })

    combos.sort(key=lambda c: -c["combined_prob"])
    return combos


def print_combos(combos: list[dict], top_n=10):
    print(f"\n{'='*70}\nSUGGESTED BETS (top {top_n} by combined probability)\n{'='*70}")
    for c in combos[:top_n]:
        print(f"\n{c['size']}-leg combo (stake weight {c['stake_weight']:.1f}x, "
              f"combined prob {c['combined_prob']:.1%}):")
        for leg in c["legs"]:
            print(f"    - {leg}")
