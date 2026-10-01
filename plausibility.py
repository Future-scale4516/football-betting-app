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

# NOTE (post-calibration): these thresholds were set when the model's raw
# probabilities were badly overconfident — the ceiling was doing the job a
# calibration layer should have been doing. Now that calibration.py pulls
# extremes toward the middle, the model rarely produces probabilities above
# ~85% or below ~15% at all, so the probability bounds below almost never
# fire. They're kept as a backstop against genuine data errors (a fixture
# with a corrupted rating, say) rather than as the primary safety net.
#
# The edge ceiling still matters: calibration fixes systematic
# overconfidence, not the model's blind spots about injuries, managerial
# changes or transfers. A large edge is still more often a blind spot than
# real value.
MAX_TRUSTED_EDGE = 0.08   # edges above this need manual sanity-check, not auto-green
MIN_PLAUSIBLE_PROB = 0.02  # a backstop for data errors, not everyday overconfidence
MAX_PLAUSIBLE_PROB = 0.95


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
