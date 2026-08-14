"""
Dixon-Coles with time-decay weighting.

The original Dixon-Coles paper weights each match by recency — a match
from last week should influence the ratings more than one from 7 months
ago. Our first sketch fit every match equally, which is the most likely
cause of the shaky calibration in the first backtest.

Weight for a match `t` days before the cutoff date:
    weight = exp(-xi * t)

xi controls how fast old matches get discounted. xi=0.0018 is the
value from Dixon & Coles' original paper (roughly: a match a year old
carries about 40% the weight of today's match). Worth tuning later,
but this is a sane starting point, not a guess.

Requires dixon_coles_sketch.py (uses its tau() function).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from dixon_coles_sketch import tau, score_matrix, derive_markets  # noqa: F401 (re-exported)


def weighted_log_likelihood(params, fixtures_with_days_ago, teams, xi):
    """
    fixtures_with_days_ago: list of (home, away, home_goals, away_goals, days_ago)
    days_ago: how many days before the cutoff date this match was played.
    """
    n = len(teams)
    attack = dict(zip(teams, params[:n]))
    defence = dict(zip(teams, params[n:2*n]))
    home_adv = params[2*n]
    rho = params[2*n + 1]

    ll = 0.0
    for home, away, hg, ag, days_ago in fixtures_with_days_ago:
        lam_h = np.exp(attack[home] + defence[away] + home_adv)
        lam_a = np.exp(attack[away] + defence[home])
        p = (poisson.pmf(hg, lam_h) * poisson.pmf(ag, lam_a)
             * tau(hg, ag, lam_h, lam_a, rho))
        weight = np.exp(-xi * days_ago)
        ll += weight * np.log(max(p, 1e-10))
    return -ll


def fit_league_weighted(fixtures_with_days_ago, teams, xi=0.0018):
    """
    Same as fit_league() but recency-weighted.
    fixtures_with_days_ago: (home, away, home_goals, away_goals, days_ago)
    """
    n = len(teams)
    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [-0.1]])
    result = minimize(weighted_log_likelihood, x0,
                       args=(fixtures_with_days_ago, teams, xi),
                       method="L-BFGS-B")
    params = result.x
    attack = dict(zip(teams, params[:n]))
    defence = dict(zip(teams, params[n:2*n]))
    return {"attack": attack, "defence": defence,
            "home_adv": params[2*n], "rho": params[2*n+1]}
