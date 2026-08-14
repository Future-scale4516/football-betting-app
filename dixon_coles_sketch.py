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

def fit_league(fixtures, teams):
    """
    fixtures: recent results for ONE league (rolling window — consider
    weighting recent gameweeks higher, same instinct as your 'recent form'
    correction in the MLB build).
    Returns fitted attack/defence/home_adv/rho.
    """
    n = len(teams)
    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [-0.1]])
    result = minimize(match_log_likelihood, x0, args=(fixtures, teams),
                       method="L-BFGS-B")
    params = result.x
    attack = dict(zip(teams, params[:n]))
    defence = dict(zip(teams, params[n:2*n]))
    return {"attack": attack, "defence": defence,
            "home_adv": params[2*n], "rho": params[2*n+1]}

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
