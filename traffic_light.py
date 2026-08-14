"""
Traffic-light bands per market.

These starting thresholds are reasoned defaults, NOT calibrated —
weaker/harder markets require a bigger edge to earn green. Once
results_tracker.py has real settled picks, get_market_trust() should
be updated to pull actual per-market hit rates instead of the static
defaults below — this is the same evolution the MLB build went
through (colour-coding was the most-touched piece of UI logic there
too).
"""

# edge required (absolute value) to earn each colour, per market.
# Correct score has no market price to de-vig against (Odds API doesn't
# expose it) so it's excluded from auto-green entirely — always "verify".
MARKET_BANDS = {
    "1X2":       {"green": 0.05, "amber": 0.02},
    "O/U 2.5":   {"green": 0.06, "amber": 0.03},
    "BTTS":      {"green": 0.06, "amber": 0.03},
}


def get_market_trust(market: str) -> float:
    """
    Placeholder — returns 1.0 (full trust) for every market until
    results_tracker.py has enough settled picks to compute a real
    per-market hit rate. Wire this up once that data exists.
    """
    return 1.0


def classify(market: str, edge: float) -> str:
    """Returns 'green', 'amber', or 'red'."""
    if market not in MARKET_BANDS:
        return "verify"  # unbanded market (e.g. correct score) — never auto-trust

    bands = MARKET_BANDS[market]
    abs_edge = abs(edge)
    if abs_edge >= bands["green"]:
        return "green"
    elif abs_edge >= bands["amber"]:
        return "amber"
    return "red"
