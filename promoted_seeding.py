"""
Promoted/relegated team seeding.

Two methods, clearly distinguished:

1. FEEDER-LEAGUE SEED (English pyramid only — Championship/League One/
   League Two data is available via football-data.co.uk):
   fits the team's real rating from the division below, then shrinks it
   toward the new division's average to account for stepping up in
   quality. A genuine, data-backed estimate.

2. GENERIC FALLBACK SEED (every other league — no feeder-league source
   is wired up yet): the team is seeded as a bottom-quartile side in
   its new division. This is an honest placeholder, not a form-based
   estimate, and is flagged as such everywhere it's used.

Both are provisional. Replace with real fitted ratings once ~6+
gameweeks of actual results exist for that team.
"""

import numpy as np
import pandas as pd
from dixon_coles_sketch import fit_league

FEEDER_LEAGUE_CODE = {
    "Premier League": "E1",       # promoted from EFL Championship
    "EFL Championship": "E2",     # promoted from League One
    "League One": "E3",           # promoted from League Two
}

SEASON = "2526"
SHRINKAGE = 0.6  # how much of the feeder-league form carries over — the
                  # rest regresses toward the new division's mean, since
                  # dominating a lower division doesn't fully transfer


def _fit_feeder(league_name: str):
    code = FEEDER_LEAGUE_CODE.get(league_name)
    if not code:
        return None
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{code}.csv"
        df = pd.read_csv(url)
        df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
        fixtures = list(df.itertuples(index=False, name=None))
        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        return fit_league(fixtures, teams)
    except Exception as e:
        print(f"  [seeding] couldn't fetch feeder data for {league_name}: {e}")
        return None


def seed_missing_teams(model: dict, league_name: str, missing_teams: list[str]) -> dict:
    """
    Returns a NEW model dict with provisional ratings added for every
    team in missing_teams. Adds a 'seed_method' dict so callers can
    check which teams are running on a provisional rating and treat
    their picks with appropriate caution (see run_all_leagues.py —
    seeded teams are forced to 'verify' regardless of edge size).
    """
    if not missing_teams:
        return {**model, "seed_method": model.get("seed_method", {})}

    seeded = {
        "attack": dict(model["attack"]), "defence": dict(model["defence"]),
        "home_adv": model["home_adv"], "rho": model["rho"],
        "seed_method": dict(model.get("seed_method", {})),
    }

    league_attack_mean = np.mean(list(model["attack"].values()))
    league_defence_mean = np.mean(list(model["defence"].values()))

    feeder_model = _fit_feeder(league_name)

    for team in missing_teams:
        if feeder_model and team in feeder_model["attack"]:
            feeder_attack_mean = np.mean(list(feeder_model["attack"].values()))
            feeder_defence_mean = np.mean(list(feeder_model["defence"].values()))

            attack_diff = feeder_model["attack"][team] - feeder_attack_mean
            defence_diff = feeder_model["defence"][team] - feeder_defence_mean

            seeded["attack"][team] = league_attack_mean + SHRINKAGE * attack_diff
            seeded["defence"][team] = league_defence_mean + SHRINKAGE * defence_diff
            seeded["seed_method"][team] = "feeder-league (real data, shrunk)"
        else:
            # Bottom-quartile fallback — no real data behind this number.
            attack_sorted = sorted(model["attack"].values())
            defence_sorted = sorted(model["defence"].values(), reverse=True)
            q_idx = max(0, int(len(attack_sorted) * 0.25) - 1)
            seeded["attack"][team] = attack_sorted[q_idx]
            seeded["defence"][team] = defence_sorted[q_idx]
            seeded["seed_method"][team] = "generic fallback (no feeder data — placeholder only)"

    return seeded
