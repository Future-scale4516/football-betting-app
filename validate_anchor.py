"""
Does "market 1X2 + model goals" actually predict BTTS and Over/Under well?

cross_league.py gives UEFA games a model read by taking WHO WINS from the
bookmakers and HOW MANY GOALS from the clubs' domestic scoring profiles. That
cannot be tested on UEFA games — there is no UEFA results history to test on.

But the anchoring machinery itself can be tested, on domestic games, where we
have the model, the prices AND the results. This script does that: it refits
week by week exactly like backtest.py, then scores four ways of producing a
BTTS and an Over/Under 2.5 number for the same games.

    1. model          the full Dixon-Coles grid (what the app shows for leagues)
    2. anchored       model total goals + the market's home:away ratio
                      (what cross_league.py does for UEFA games)
    3. market-implied 1X2 price + Over 2.5 price, solved for goal rates
                      (what the app showed for UEFA before this)
    4. market         the bookmakers' own Over 2.5 price, de-vigged
                      (not available for BTTS — the feed doesn't price it)

If (2) is close to (1), the anchoring costs little and the UEFA read is worth
having. If (2) is much worse, the method is broken and the UEFA numbers should
not be trusted.

    python3 validate_anchor.py
    python3 validate_anchor.py --from 2026-10-01
"""
import argparse
import sys
import warnings
from datetime import date, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from backtest import _read, _played, CUR                      # noqa: E402
from calibration import calibrate_markets                     # noqa: E402
from club_ratings import implied_grid                         # noqa: E402
from cross_league import _grid, anchored_grid                 # noqa: E402
from devig import shin                                        # noqa: E402
from dixon_coles_sketch import (apply_level_correction, derive_markets,  # noqa: E402
                                fit_league, score_matrix)
from league_config import LEAGUES, SEASON                     # noqa: E402
from shot_goals import build_fit_rows                         # noqa: E402


def _profiles(model):
    """Same per-team goal rates cross_league.py builds, for one league."""
    attack, defence = model["attack"], model["defence"]
    scale = model.get("goal_scale", 1.0)
    ha = model.get("home_adv", 0.0)
    a_avg = float(np.log(np.mean(np.exp(list(attack.values())))))
    d_avg = float(np.log(np.mean(np.exp(list(defence.values())))))
    swing = (1.0 + np.exp(ha)) / 2.0
    out = {t: {"att": float(scale * np.exp(attack[t] + d_avg) * swing),
               "dfn": float(scale * np.exp(a_avg + defence[t]) * swing)}
           for t in attack}
    lg = float(np.mean([v["att"] for v in out.values()]))
    for v in out.values():
        v["league_avg"] = lg
    return out


def job(args):
    league, code, cut, data_dir = args
    last, cur = _read(data_dir, SEASON, code), _read(data_dir, CUR, code)
    if cur is None:
        return []
    cur = _played(cur)
    train = cur[cur._date < cut]
    test = cur[(cur._date >= cut) & (cur._date < cut + timedelta(days=7))]
    if test.empty:
        return []
    rows = build_fit_rows([last, train], cut)
    teams = sorted({t for r in rows for t in (r[0], r[1])})
    if len(teams) < 4 or len(rows) < 20:
        return []
    model = fit_league(rows, teams)
    if len(train):
        model = apply_level_correction(model, list(zip(
            train.HomeTeam, train.AwayTeam,
            train.FTHG.astype(int), train.FTAG.astype(int))))
    prof = _profiles(model)

    out = []
    for _, g in test.iterrows():
        h, a = g.HomeTeam, g.AwayTeam
        if h not in model["attack"] or a not in model["attack"]:
            continue
        try:
            o = [float(g[c]) for c in ("AvgH", "AvgD", "AvgA")]
            ou = [float(g[c]) for c in ("Avg>2.5", "Avg<2.5")]
        except (TypeError, ValueError, KeyError):
            continue
        if any(not np.isfinite(x) or x <= 1 for x in o + ou):
            continue
        ph, pd_, pa = shin(*o)
        p_over = shin(*ou)[0]

        m_model = calibrate_markets(derive_markets(score_matrix(h, a, model)))

        g_anch = anchored_grid(prof[h], prof[a], ph, pa)
        if g_anch is None:
            continue
        m_anch = derive_markets(g_anch)

        g_imp = implied_grid(ph, pd_, p_over)
        m_imp = derive_markets(g_imp) if g_imp is not None else None

        hg, ag = int(g.FTHG), int(g.FTAG)
        out.append({
            "league": league, "date": g._date, "fixture": f"{h} vs {a}",
            "btts_won": float(hg > 0 and ag > 0),
            "over_won": float(hg + ag > 2.5),
            "home_won": float(hg > ag), "away_won": float(ag > hg),
            "btts_model": m_model["BTTS"]["yes"],
            "btts_anchored": m_anch["BTTS"]["yes"],
            "btts_implied": m_imp["BTTS"]["yes"] if m_imp else np.nan,
            "over_model": m_model["O/U 2.5"]["over"],
            "over_anchored": m_anch["O/U 2.5"]["over"],
            "over_implied": m_imp["O/U 2.5"]["over"] if m_imp else np.nan,
            "over_market": p_over,
            "home_mkt": ph, "away_mkt": pa,
            "home_anchored": m_anch["1X2"]["home"],
            "away_anchored": m_anch["1X2"]["away"],
        })
    return out


def llb(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2026-09-12")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--workers", type=int, default=max(1, min(4, cpu_count() - 1)))
    a = ap.parse_args()

    start = date.fromisoformat(a.start)
    leagues = {n: c["data_code"] for n, c in LEAGUES.items() if c.get("data_code")}
    latest = max((_played(f)._date.max() for f in
                  (_read(a.data, CUR, c) for c in leagues.values()) if f is not None),
                 default=None)
    if latest is None:
        raise SystemExit(f"No {CUR}_*.csv in {a.data}. Run refresh_data.py first.")

    cuts, c = [], start
    while c <= latest:
        cuts.append(c)
        c += timedelta(days=7)
    jobs = [(lg, code, cut, a.data) for lg, code in leagues.items() for cut in cuts]
    print(f"Refitting {len(jobs)} league-weeks ({start} -> {latest})…")
    with Pool(a.workers) as p:
        df = pd.DataFrame([r for rr in p.map(job, jobs) for r in rr])
    if df.empty:
        raise SystemExit("No games with both prices and results.")

    print(f"\n{len(df)} games with prices, results and a model fit.\n")
    print("=" * 74)
    print("BTTS — log loss (lower is better)")
    print("=" * 74)
    rows = []
    for name, col in (("model (full Dixon-Coles)", "btts_model"),
                      ("anchored (model goals + market 1X2)", "btts_anchored"),
                      ("market-implied (1X2 + O2.5 price)", "btts_implied")):
        d = df.dropna(subset=[col])
        rows.append({"method": name, "n": len(d), "log loss": llb(d.btts_won, d[col]),
                     "says": d[col].mean(), "happened": d.btts_won.mean()})
    base = df.btts_won.mean()
    rows.append({"method": "base rate (always the average)", "n": len(df),
                 "log loss": llb(df.btts_won, np.full(len(df), base)),
                 "says": base, "happened": base})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n" + "=" * 74)
    print("OVER 2.5 — log loss")
    print("=" * 74)
    rows = []
    for name, col in (("model (full Dixon-Coles)", "over_model"),
                      ("anchored (model goals + market 1X2)", "over_anchored"),
                      ("market-implied (1X2 + O2.5 price)", "over_implied"),
                      ("market (the Over 2.5 price itself)", "over_market")):
        d = df.dropna(subset=[col])
        rows.append({"method": name, "n": len(d), "log loss": llb(d.over_won, d[col]),
                     "says": d[col].mean(), "happened": d.over_won.mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n" + "=" * 74)
    print("Does the anchor hold? (it should reproduce the market's 1X2 split)")
    print("=" * 74)
    err = (df.home_anchored / (df.home_anchored + df.away_anchored)
           - df.home_mkt / (df.home_mkt + df.away_mkt)).abs()
    print(f"home:away share, mean error {err.mean():.5f}, worst {err.max():.5f}")
    print("(near zero = the solver is doing what it claims)")

    print("\n" + "=" * 74)
    print("READ THIS")
    print("=" * 74)
    print("These are DOMESTIC games, where both clubs sit on one rating scale.")
    print("UEFA games are harder: the two clubs come from different leagues and")
    print("only their goal RATES are comparable. Treat the anchored numbers above")
    print("as a best case for what cross_league.py can do, not as proof it works.")
    out = HERE / "backtest_output"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "anchor_validation.csv", index=False)
    print(f"\nSaved: {out / 'anchor_validation.csv'}")


if __name__ == "__main__":
    main()
