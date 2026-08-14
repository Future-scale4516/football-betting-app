"""
Plausibility ceiling.

Purpose: catch exactly what happened with Brentford vs Tottenham (+11.4%)
and Newcastle vs Liverpool (+10.4%) in the first live odds run — the
model doesn't know about managerial changes, transfers, or narrative
context, so a suspiciously large edge is more often a blind spot than
real value. This doesn't reject those picks outright — it flags them
for manual review instead of letting a big number auto-qualify as
trustworthy.
"""

MAX_TRUSTED_EDGE = 0.08   # edges above this need manual sanity-check, not auto-green
MIN_PLAUSIBLE_PROB = 0.03  # below this, "the model thinks this is almost impossible" is
MAX_PLAUSIBLE_PROB = 0.92  # itself suspicious for a competitive top-flight fixture


def check_plausibility(model_prob: float, market_prob: float, edge: float):
    """
    Returns (status, reason).
    status is one of: "ok", "verify" (large edge or extreme probability).
    """
    reasons = []

    if abs(edge) > MAX_TRUSTED_EDGE:
        reasons.append(
            f"edge of {edge:+.1%} exceeds the {MAX_TRUSTED_EDGE:.0%} trusted "
            f"ceiling — model may be missing context (injuries, new manager, "
            f"transfers) the market has already priced in"
        )

    if not (MIN_PLAUSIBLE_PROB <= model_prob <= MAX_PLAUSIBLE_PROB):
        reasons.append(
            f"model probability {model_prob:.1%} is outside the plausible "
            f"range for a competitive top-flight fixture — worth checking "
            f"the fixture isn't a data error"
        )

    if reasons:
        return "verify", "; ".join(reasons)
    return "ok", ""
