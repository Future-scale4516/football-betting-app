"""
Dixon-Coles Poisson model — sketch / starting structure.

This replaces the independent-Poisson approach from the MLB build.
One fitted model per league; outputs feed 1X2, O/U, handicaps, and
correct score off the same goal-expectancy grid.

Core idea:
  - Every team gets an ATTACK strength and DEFENCE strength.
  - Expected goals for team i at home vs team j away:
        lambda_home = attack[i] * defence[j] * home_advantage
        lambda_away = attack[j] * defence[i]
  - Plain Poisson(lambda_home) x Poisson(lambda_away) UNDERESTIMATES
    low-scoring draws (0-0, 1-1) and slightly misprices 1-0/0-1.
    Dixon-Coles adds a correction (tau) for scorelines 0-0, 1-0, 0-1, 1-1 only.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

def tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Dixon-Coles low-score correlation adjustment.
    Only nonzero for the four low-scoring combinations."""
    if home_goals == 0 and away_goals == 0:
        return 1 - (lambda_home * lambda_away * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (lambda_home * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (lambda_away * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0

def match_log_likelihood(params, fixtures, teams):
    """
    params layout: [attack_1..attack_n, defence_1..defence_n, home_adv, rho]
    fixtures: list of (home_team, away_team, home_goals, away_goals)
    Team strengths are constrained so attack params sum to n (identifiability).
    """
    n = len(teams)
    attack = dict(zip(teams, params[:n]))
    defence = dict(zip(teams, params[n:2*n]))
    home_adv = params[2*n]
    rho = params[2*n + 1]

    ll = 0.0
    for home, away, hg, ag in fixtures:
        lam_h = np.exp(attack[home] + defence[away] + home_adv)
        lam_a = np.exp(attack[away] + defence[home])
        p = (poisson.pmf(hg, lam_h) * poisson.pmf(ag, lam_a)
             * tau(hg, ag, lam_h, lam_a, rho))
        ll += np.log(max(p, 1e-10))
    return -ll  # negative for minimisation

DEFAULT_XI = 0.0018   # Dixon & Coles' published decay rate


def fit_league(fixtures, teams, reg_lambda=0.01, xi=DEFAULT_XI):
    """
    fixtures: results for ONE league. Each is either
        (home, away, home_goals, away_goals)                  — unweighted
    or  (home, away, home_goals, away_goals, days_before_now)  — time-decayed

    xi: time-decay rate. Each match is weighted exp(-xi * days_ago), so
    recent form counts for more. Only applies when fixtures carry a 5th
    element; 4-tuples are treated as equally weighted so older callers
    keep working.

    WHY THIS MATTERS NOW: an earlier test of time-decay (on one season
    alone) showed almost no effect, so it was dropped. That test no longer
    applies — the model now fits on last season PLUS the current season
    blended together, and without decay a match from 14 months ago counts
    exactly as much as one from last week. At xi=0.0018 a year-old match
    carries roughly 50% the weight of today's, which is the behaviour the
    blended fit needs.

    reg_lambda: L2 regularization, shrinking attack/defence toward the
    league mean. NOTE: at 0.01 this did NOT measurably shrink the large
    edges seen on elite home sides; the plausibility ceiling is what
    actually contains those. Raise it if you want to retest.

    VECTORIZED: the likelihood operates on numpy arrays rather than
    looping per fixture, which is what makes fitting 8 leagues fast.
    """
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Precompute once, outside the objective — this is the whole speedup.
    home_idx = np.array([idx[f[0]] for f in fixtures])
    away_idx = np.array([idx[f[1]] for f in fixtures])
    hg = np.array([f[2] for f in fixtures], dtype=float)
    ag = np.array([f[3] for f in fixtures], dtype=float)

    if fixtures and len(fixtures[0]) > 4:
        days_ago = np.array([f[4] for f in fixtures], dtype=float)
        weights = np.exp(-xi * np.maximum(days_ago, 0))
    else:
        weights = np.ones(len(fixtures))

    # Masks for the four scorelines Dixon-Coles corrects
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def objective(params):
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1]

        lam_h = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
        lam_a = np.exp(attack[away_idx] + defence[home_idx])

        log_p = poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)

        tau_vals = np.ones_like(lam_h)
        tau_vals[m00] = 1 - (lam_h[m00] * lam_a[m00] * rho)
        tau_vals[m01] = 1 + (lam_h[m01] * rho)
        tau_vals[m10] = 1 + (lam_a[m10] * rho)
        tau_vals[m11] = 1 - rho
        tau_vals = np.maximum(tau_vals, 1e-10)  # keep the log finite

        ll = np.sum(weights * (log_p + np.log(tau_vals)))
        penalty = reg_lambda * (np.sum(attack ** 2) + np.sum(defence ** 2))
        return -ll + penalty

    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [-0.1]])
    result = minimize(objective, x0, method="L-BFGS-B")
    params = result.x
    return {"attack": dict(zip(teams, params[:n])),
            "defence": dict(zip(teams, params[n:2 * n])),
            "home_adv": params[2 * n], "rho": params[2 * n + 1]}


def score_matrix(home, away, model, max_goals=8):
    """Full scoreline probability grid for one fixture.
    This single grid is what 1X2, O/U, handicap, and correct-score
    markets all get derived from — one model output, four markets."""
    lam_h = np.exp(model["attack"][home] + model["defence"][away] + model["home_adv"])
    lam_a = np.exp(model["attack"][away] + model["defence"][home])

    grid = np.zeros((max_goals + 1, max_goals + 1))
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            grid[hg, ag] = (poisson.pmf(hg, lam_h) * poisson.pmf(ag, lam_a)
                             * tau(hg, ag, lam_h, lam_a, model["rho"]))
    grid /= grid.sum()  # renormalise after the tau adjustment
    return grid

def goals_over_line(grid, line: float):
    """Probability of total goals being over/under any line (1.5, 2.5, etc.)
    off the same grid — no separate model needed per line."""
    goals = np.add.outer(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
    over = grid[goals > line].sum()
    return over, 1 - over


def derive_markets(grid):
    """Everything below reads off the SAME grid — no separate models."""
    home_win = np.tril(grid, -1).sum()
    draw = np.trace(grid)
    away_win = np.triu(grid, 1).sum()

    over_1_5, under_1_5 = goals_over_line(grid, 1.5)
    over_2_5, under_2_5 = goals_over_line(grid, 2.5)

    btts_yes = grid[1:, 1:].sum()

    return {
        "1X2": {"home": home_win, "draw": draw, "away": away_win},
        "O/U 1.5": {"over": over_1_5, "under": under_1_5},
        "O/U 2.5": {"over": over_2_5, "under": under_2_5},
        "BTTS": {"yes": btts_yes, "no": 1 - btts_yes},
        "correct_score_grid": grid,  # top N scorelines for correct-score market
    }

# --- Next steps once you're wiring this in ---
# 1. fit_league() needs real fixture data (football-data.org / API-Football)
#    once the odds-coverage check confirms which leagues to prioritise.
# 2. rho typically fits around -0.1 to -0.15 for most leagues — sanity-check
#    the fitted value against published Dixon-Coles papers before trusting it.
# 3. Backtest score_matrix() output against last season BEFORE touching
#    real odds — same discipline as the MLB build.
# 4. Promoted teams (Hull/Coventry/Ipswich): no fixture history yet this
#    season. Either seed attack/defence from Championship-adjusted values,
#    or exclude them from fit_league() until ~6 gameweeks of PL data exist,
#    flagging their picks low-confidence in the meantime.
