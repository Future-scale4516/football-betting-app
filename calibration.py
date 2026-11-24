"""
Probability calibration — corrects systematic overconfidence.

WHY THIS EXISTS
---------------
Grading all 1,673 selections from September 2026 (239 fixtures, 8 leagues)
showed the raw model is badly overconfident at the extremes:

    model said        actually happened
    0-10%        ->   31%
    10-20%       ->   28%
    40-50%       ->   45%   (fine in the middle)
    70-80%       ->   62%
    90-100%      ->   60%

Binary log loss was 0.6966 against a 0.693 coin-flip baseline — i.e. the
raw probabilities were close to worthless once you grade every selection
rather than only the model's favourite. The middle of the range was fine;
the confident calls were the problem. That also explains the "elite team"
edges spotted back in August (Barcelona at 85.5% etc.) — those weren't a
big-club quirk, they were the same overconfidence showing up where the
model was most sure.

THE FIX
-------
Temperature scaling: divide the log-odds by T, which pulls probabilities
toward 50% without changing their ranking. T was fitted per market by
minimising log loss, and validated with 5-fold cross-validation — every
market improved on held-out data, so this isn't fitting noise:

    market     T      held-out log loss
    1X2        1.75   0.6641 -> 0.6460
    BTTS       3.13   0.7137 -> 0.6858
    O/U 2.5    4.10   0.7284 -> 0.6941

O/U 2.5 needing the largest correction makes sense: total goals are the
noisiest thing here, and the model was claiming the most certainty about
them.

IMPORTANT CAVEATS
-----------------
- Fitted on ONE month. T for O/U 2.5 was the least stable across folds
  (2.5 to 7.2), so treat 4.10 as provisional.
- Ranking is unchanged, so "most likely" ordering stays the same. What
  changes is the probability attached — and therefore every edge
  calculation against bookmaker prices.
- Refit when there's more data: see refit_calibration() below.
"""

import numpy as np

# RESET TO 1.0 (no adjustment) on 28 Oct 2026, when the model moved to
# shot-implied goals (shot_goals.py). The September values (1X2 1.75, BTTS
# 3.13, O/U 4.10) were fitted to the older goals-only model. On the new
# model, cross-validated temperature scaling no longer helps:
#
#     market     fitted T   CV loss with T   uncalibrated
#     1X2        1.19       1.0448           1.0410
#     O/U 2.5    1.28       0.6758           0.6735
#     BTTS       1.26       0.6812           0.6797
#
# October's out-of-sample results had also shown the old values were by then
# flattening the goals-only model too much.
#
# 24 Nov 2026 check (backtest.py, 804 games, 12 Sep - 14 Nov, current model with
# the scoring-level correction, leave-one-week-out):
#     market     fitted T   CV loss T=1   CV loss fitted T
#     1X2        0.95       1.0051        1.0079   (worse)
#     O/U 2.5    1.41       0.6767        0.6751   (-0.0016, within noise)
#     BTTS       1.40       0.6820        0.6803   (-0.0017, within noise)
#     O/U 1.5    1.02       0.5134        0.5159   (worse)
# No market shows a gain worth the risk, so everything stays at 1.0.
# Next review: after ~8 more weeks of games.
CALIBRATION_T = {
    "1X2": 1.0,
    "BTTS": 1.0,
    "O/U 2.5": 1.0,
    "O/U 1.5": 1.0,
}

DEFAULT_T = 1.0

_EPS = 1e-6

CALIBRATION_FITTED_ON = "1.0 for all markets; re-checked 24 Nov 2026 on 804 games, no gain from calibrating"


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calibrate(prob, market: str):
    """Applies the market's temperature correction to a raw model
    probability. Returns a float in (0,1). Ranking within a market is
    preserved — only confidence changes."""
    T = CALIBRATION_T.get(market, DEFAULT_T)
    if T is None or T <= 0:
        return float(prob)
    return float(_sigmoid(_logit(np.asarray(prob, dtype=float)) / T))


def calibrate_markets(markets: dict) -> dict:
    """Applies calibration across a whole derive_markets() output,
    renormalising each market so its outcomes still sum to 1.

    Renormalising matters: temperature scaling is applied per outcome, so
    a 3-way market like 1X2 won't sum to exactly 1 afterwards. Without
    this, de-vigging against bookmaker prices would compare a normalised
    market probability to an unnormalised model one.
    """
    out = {}
    for market, values in markets.items():
        if not isinstance(values, dict):
            out[market] = values          # e.g. the raw scoreline grid
            continue
        cal = {k: calibrate(v, market) for k, v in values.items()}
        total = sum(cal.values())
        out[market] = ({k: v / total for k, v in cal.items()}
                       if total > 0 else values)
    return out


def refit_calibration(results_df, min_samples=200):
    """Refits T per market from graded results.

    results_df needs 'market', 'model_prob' and 'won' columns — exactly
    what the Results page exports. Markets with fewer than min_samples
    graded picks are skipped rather than fitted on noise.

    Returns {market: T}. Review before pasting into CALIBRATION_T above —
    a T that swings wildly from the current value usually means the sample
    is too small, not that the model changed.
    """
    from scipy.optimize import minimize_scalar

    def loss(p, y):
        p = np.clip(p, _EPS, 1 - _EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    fitted = {}
    for market, g in results_df.groupby("market"):
        if len(g) < min_samples:
            continue
        p = g["model_prob"].values
        y = g["won"].astype(float).values
        r = minimize_scalar(lambda T: loss(_sigmoid(_logit(p) / T), y),
                            bounds=(0.5, 8.0), method="bounded")
        fitted[market] = round(float(r.x), 2)
    return fitted
