"""
Rolling out-of-sample backtest of the live model, using only the CSVs in data/.

For every week it refits each league using ONLY games played before that week
(last season + this season so far, with the same time-decay and shot-implied
goals as the app), predicts that week's games, and scores them against what
happened and against the bookmakers' prices.

Run from the repo folder (venv on):

    python3 backtest.py                         # 12 Sep -> latest result, all leagues
    python3 backtest.py --from 2026-10-17       # just the recent weeks
    python3 backtest.py --league "Premier League"
    python3 backtest.py --goal-weight 1.0       # goals only (no shot blend) to compare
    python3 backtest.py --xi 0.0030             # faster time decay

Prints, for each market: model vs market log loss, a per-league table, a
calibration table, a betting simulation by edge size, and how much of the model
(if any) is worth blending with the market price. Everything is also saved to
backtest_output/ so a later run can be compared.

HONEST LIMITS (also printed at the end of every run):
  * The "market" here is football-data's AVERAGE opening/closing price. It is not
    Pinnacle, and it is not the price you could necessarily have bet at.
  * ROI on a few hundred bets is mostly luck. Judge probability quality (log loss,
    calibration) first, betting results second, and trust CLV most.
  * Only 1X2 and Over/Under 2.5 have prices in these files. BTTS and O/U 1.5 can
    be scored for accuracy only.
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

from dixon_coles_sketch import fit_league, score_matrix, derive_markets, apply_level_correction   # noqa: E402
from league_config import LEAGUES, SEASON                                  # noqa: E402
from shot_goals import build_fit_rows                                      # noqa: E402
from calibration import calibrate_markets                                  # noqa: E402
from devig import shin                                                     # noqa: E402

SPLIT_DATE = date(2026, 10, 17)     # model choices were tuned on games up to here
CUR = "2627"


# ----------------------------------------------------------------- loading
def _read(data_dir, season, code):
    p = Path(data_dir) / f"{season}_{code}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date
    return df.dropna(subset=["_date", "HomeTeam", "AwayTeam"])


def _played(df):
    return df[df["FTHG"].notna() & df["FTAG"].notna()]


# ------------------------------------------------------------------ one job
def job(args):
    league, code, cut, data_dir, xi, goal_weight, level = args
    last, cur = _read(data_dir, SEASON, code), _read(data_dir, CUR, code)
    if cur is None:
        return [], f"{league}: no {CUR} file"
    cur = _played(cur)
    train_cur = cur[cur["_date"] < cut]
    test = cur[(cur["_date"] >= cut) & (cur["_date"] < cut + timedelta(days=7))]
    if test.empty:
        return [], None
    fixtures = build_fit_rows([last, train_cur], cut)
    teams = sorted({t for fx in fixtures for t in (fx[0], fx[1])})
    if len(teams) < 4 or len(fixtures) < 20:
        return [], f"{league} {cut}: not enough prior data"
    model = fit_league(fixtures, teams, xi=xi, goal_weight=goal_weight)
    if level:
        model = apply_level_correction(model, list(zip(
            train_cur.HomeTeam, train_cur.AwayTeam, train_cur.FTHG.astype(int), train_cur.FTAG.astype(int))))

    out, skipped = [], 0
    for _, g in test.iterrows():
        h, a = g["HomeTeam"], g["AwayTeam"]
        if h not in model["attack"] or a not in model["attack"]:
            skipped += 1
            continue
        mk = calibrate_markets(derive_markets(score_matrix(h, a, model)))
        hg, ag = int(g["FTHG"]), int(g["FTAG"])
        out.append({
            "league": league, "date": g["_date"], "cut": cut, "home": h, "away": a,
            "hg": hg, "ag": ag,
            "m_home": mk["1X2"]["home"], "m_draw": mk["1X2"]["draw"], "m_away": mk["1X2"]["away"],
            "m_over25": mk["O/U 2.5"]["over"], "m_over15": mk["O/U 1.5"]["over"],
            "m_btts": mk["BTTS"]["yes"],
            **{c: g.get(c, np.nan) for c in (
                "AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA",
                "MaxH", "MaxD", "MaxA", "B365H", "B365D", "B365A",
                "Avg>2.5", "Avg<2.5", "AvgC>2.5", "AvgC<2.5",
                "Max>2.5", "Max<2.5", "B365>2.5", "B365<2.5")},
        })
    return out, (f"{league} {cut}: {skipped} games skipped (team with no rating)"
                 if skipped else None)


# ------------------------------------------------------------------ helpers
def _fair(cols, row):
    """Shin de-vigged probabilities from a row of odds, or None."""
    try:
        o = [float(row[c]) for c in cols]
    except (TypeError, ValueError):
        return None
    if any(np.isnan(x) or x <= 1.0 for x in o):
        return None
    return shin(*o)


def build_frame(raw):
    """Wide per-game frame -> long per-selection frame with model prob, fair
    market probs (open and close), prices and the outcome."""
    rows = []
    for rec in raw.to_dict("records"):      # (itertuples mangles names like "Avg>2.5")
        r = pd.Series(rec)
        res = "H" if r.hg > r.ag else ("A" if r.ag > r.hg else "D")
        tot = r.hg + r.ag
        base = {"league": r.league, "date": r.date, "fixture": f"{r.home} vs {r.away}"}
        fo = _fair(["AvgH", "AvgD", "AvgA"], r)
        fc = _fair(["AvgCH", "AvgCD", "AvgCA"], r)
        for i, (sel, key, won) in enumerate((("Home", "H", res == "H"),
                                             ("Draw", "D", res == "D"),
                                             ("Away", "A", res == "A"))):
            rows.append({**base, "market": "1X2", "selection": sel,
                         "model": r[f"m_{sel.lower()}"], "won": float(won),
                         "mkt_open": fo[i] if fo else np.nan,
                         "mkt_close": fc[i] if fc else np.nan,
                         "odds_avg": r[f"Avg{key}"], "odds_max": r[f"Max{key}"],
                         "odds_b365": r[f"B365{key}"], "group": r.date})
        fo = _fair(["Avg>2.5", "Avg<2.5"], r)
        fc = _fair(["AvgC>2.5", "AvgC<2.5"], r)
        for i, (sel, p, won, oc) in enumerate((
                ("Over 2.5", r.m_over25, tot > 2.5, ">"),
                ("Under 2.5", 1 - r.m_over25, tot < 2.5, "<"))):
            rows.append({**base, "market": "O/U 2.5", "selection": sel,
                         "model": p, "won": float(won),
                         "mkt_open": fo[i] if fo else np.nan,
                         "mkt_close": fc[i] if fc else np.nan,
                         "odds_avg": r[f"Avg{oc}2.5"], "odds_max": r[f"Max{oc}2.5"],
                         "odds_b365": r[f"B365{oc}2.5"], "group": r.date})
        for sel, p, won in (("Yes", r.m_btts, r.hg > 0 and r.ag > 0),
                            ("No", 1 - r.m_btts, not (r.hg > 0 and r.ag > 0))):
            rows.append({**base, "market": "BTTS", "selection": sel, "model": p,
                         "won": float(won), "mkt_open": np.nan, "mkt_close": np.nan,
                         "odds_avg": np.nan, "odds_max": np.nan, "odds_b365": np.nan,
                         "group": r.date})
        for sel, p, won in (("Over 1.5", r.m_over15, tot > 1.5),
                            ("Under 1.5", 1 - r.m_over15, tot < 1.5)):
            rows.append({**base, "market": "O/U 1.5", "selection": sel, "model": p,
                         "won": float(won), "mkt_open": np.nan, "mkt_close": np.nan,
                         "odds_avg": np.nan, "odds_max": np.nan, "odds_b365": np.nan,
                         "group": r.date})
    return pd.DataFrame(rows)


def event_logloss(df, col):
    """Mean log loss per EVENT (one game) from the selection that happened."""
    hit = df[df.won == 1]
    return -np.log(np.clip(hit[col], 1e-9, 1)).mean()


def brier(df, col):
    return ((df[col] - df.won) ** 2).sum() / df.fixture.nunique()


def pool(model, mkt, beta):
    """Log-linear pool: p ∝ mkt^(1-beta) * model^beta, normalised per game.
    beta=0 is the market, beta=1 is the model."""
    z = np.exp((1 - beta) * np.log(np.clip(mkt, 1e-9, 1)) + beta * np.log(np.clip(model, 1e-9, 1)))
    return z


def pooled_logloss(df, beta, mkt_col):
    """Pooled prediction scored by the log loss of what happened. Normalises
    within each fixture so the probabilities sum to 1."""
    d = df.copy()
    d["z"] = pool(d.model.values, d[mkt_col].values, beta)
    d["p"] = d.z / d.groupby("fixture").z.transform("sum")
    hit = d[d.won == 1]
    return -np.log(np.clip(hit.p, 1e-9, 1)).mean()


# ------------------------------------------------------------------- reports
def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def report_logloss(df, label):
    section(f"1. PROBABILITY QUALITY — {label}")
    print("Log loss per game (LOWER is better). 'market' = bookmakers' average price,\n"
          "margin removed. Gap > 0 means the model is WORSE than the market.\n")
    out = []
    for mk in ("1X2", "O/U 2.5"):
        d = df[(df.market == mk)].dropna(subset=["mkt_open", "mkt_close"])
        if d.empty:
            continue
        n = d.fixture.nunique()
        out.append({"market": mk, "games": n,
                    "model": event_logloss(d, "model"),
                    "mkt_open": event_logloss(d, "mkt_open"),
                    "mkt_close": event_logloss(d, "mkt_close"),
                    "gap_vs_close": event_logloss(d, "model") - event_logloss(d, "mkt_close")})
    for mk in ("BTTS", "O/U 1.5"):
        d = df[df.market == mk]
        if d.empty:
            continue
        p = d[d.selection.isin(["Yes", "Over 1.5"])]
        base = p.won.mean()
        ll_m = -np.mean(np.where(p.won == 1, np.log(np.clip(p.model, 1e-9, 1)),
                                 np.log(np.clip(1 - p.model, 1e-9, 1))))
        ll_b = -np.mean(np.where(p.won == 1, np.log(base), np.log(1 - base)))
        out.append({"market": mk + " (no price)", "games": len(p), "model": ll_m,
                    "mkt_open": ll_b, "mkt_close": np.nan, "gap_vs_close": ll_m - ll_b})
    t = pd.DataFrame(out)
    if t.empty:
        print("No games with prices in this window.")
        return
    t = t.rename(columns={"mkt_open": "market_open", "mkt_close": "market_close",
                          "gap_vs_close": "gap"})
    print(t.round(4).to_string(index=False))
    print("\n(BTTS / O/U 1.5 have no price in the data, so their 'market' column is just\n"
          " the base rate: 'always predict the average'. Model < base rate = it has skill.)")


def report_by_league(df):
    section("2. WHICH LEAGUES — model minus market log loss (negative = model better)")
    rows = []
    for (lg, mk), d in df[df.market.isin(["1X2", "O/U 2.5"])].groupby(["league", "market"]):
        d = d.dropna(subset=["mkt_close"])
        if d.fixture.nunique() < 15:
            continue
        rows.append({"league": lg, "market": mk, "games": d.fixture.nunique(),
                     "gap": event_logloss(d, "model") - event_logloss(d, "mkt_close")})
    if not rows:
        print("Not enough games per league.")
        return
    t = pd.DataFrame(rows).pivot(index="league", columns="market", values=["gap", "games"])
    print(t.round(4).to_string())
    print("\nSmall leagues have ~15–60 games a market: differences under ~0.03 are noise.")


def report_calibration(df):
    section("3. CALIBRATION — when the model says X%, how often does it happen?")
    for mk, sel in (("1X2", None), ("O/U 2.5", "Over 2.5"), ("BTTS", "Yes")):
        d = df[df.market == mk]
        if sel:
            d = d[d.selection == sel]
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0]
        d = d.assign(b=pd.cut(d.model, bins))
        g = d.groupby("b", observed=True).agg(n=("won", "size"), said=("model", "mean"),
                                              happened=("won", "mean"))
        g = g[g.n >= 10]
        print(f"\n{mk}{' — ' + sel if sel else ' — all selections'}")
        print(g.round(3).to_string())
    print("\nBuckets with n<10 are hidden. 'said' >> 'happened' means over-confident.")


def report_betting(df):
    section("4. IF YOU HAD BET — flat 1 unit, whenever the model beat the market by >= X")
    print("Edge = model prob − market prob (market = average OPENING price, margin\n"
          "removed, Shin). Returns use the BEST price any book offered (an upper bound\n"
          "for what you could get) and Bet365's price (what one real book offered).\n")
    d0 = df[df.market.isin(["1X2", "O/U 2.5"])].dropna(subset=["mkt_open", "odds_max"]).copy()
    d0["edge"] = d0.model - d0.mkt_open
    rows = []
    for thr in (0.0, 0.03, 0.05, 0.08):
        for mk in ("ALL", "1X2", "O/U 2.5"):
            d = d0 if mk == "ALL" else d0[d0.market == mk]
            d = d[d.edge >= thr]
            if len(d) < 10:
                continue
            for lab, col in (("best price", "odds_max"), ("Bet365", "odds_b365")):
                x = d.dropna(subset=[col])
                r = x.won * x[col] - 1
                rows.append({"edge>=": f"{thr:.0%}", "market": mk, "prices": lab,
                             "bets": len(x), "ROI": r.mean(),
                             "±1.96se": 1.96 * r.std() / np.sqrt(len(x)),
                             "hit%": x.won.mean()})
    t = pd.DataFrame(rows)
    if t.empty:
        print("Not enough qualifying bets.")
        return
    for c in ("ROI", "±1.96se", "hit%"):
        t[c] = t[c].map("{:+.1%}".format if c != "hit%" else "{:.1%}".format)
    print(t.to_string(index=False))

    print("\nBy selection, edge >= 3%, best price:")
    d = d0[d0.edge >= 0.03]
    r = d.assign(ret=d.won * d.odds_max - 1).groupby(["market", "selection"]).agg(
        bets=("ret", "size"), ROI=("ret", "mean"), hit=("won", "mean"))
    r = r[r.bets >= 8]
    r["ROI"] = r.ROI.map("{:+.1%}".format)
    r["hit"] = r.hit.map("{:.1%}".format)
    print(r.to_string() if not r.empty else "  (too few bets per selection)")

    base = d0.assign(ret=d0.won * d0.odds_avg - 1)
    print(f"\nBaseline — betting EVERY selection at the average price: "
          f"{base.ret.mean():+.1%} over {len(base)} bets (this is just the bookmaker margin).")
    print("A ± wider than the ROI itself means: no conclusion can be drawn.")


def report_blend(df):
    section("5. SHOULD THE MODEL BE BLENDED WITH THE MARKET?")
    print("p = market^(1-b) × model^b, rescaled. b=0 is the pure market, b=1 the pure model.\n"
          "'tuned early' picks b on games up to 17 Oct and tests it, unchanged, on later games.\n")
    grid = [0, .1, .2, .3, .5, .75, 1.0]
    rows = []
    for mk in ("1X2", "O/U 2.5"):
        d = df[df.market == mk].dropna(subset=["mkt_open"])
        if d.empty:
            continue
        early = d[d.date <= SPLIT_DATE]
        late = d[d.date > SPLIT_DATE]
        row = {"market": mk, "games": d.fixture.nunique()}
        for b in grid:
            row[f"b={b}"] = pooled_logloss(d, b, "mkt_open")
        best = min(grid, key=lambda b: row[f"b={b}"])
        row["best b"] = best
        if early.fixture.nunique() >= 40 and late.fixture.nunique() >= 40:
            be = min(grid, key=lambda b: pooled_logloss(early, b, "mkt_open"))
            row["tuned early b"] = be
            row["late: market"] = pooled_logloss(late, 0, "mkt_open")
            row["late: at early b"] = pooled_logloss(late, be, "mkt_open")
        rows.append(row)
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    print("\nIf the best b is 0 the model adds nothing to the price. If the 'late' blend\n"
          "beats the 'late' market, the blend is worth building into the app.")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="2026-09-12")
    ap.add_argument("--to", dest="end", default=None, help="default: latest result in the data")
    ap.add_argument("--league", default="all")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--goal-weight", type=float, default=0.5,
                    help="1.0 = goals only, 0.5 = the app's shot blend")
    ap.add_argument("--no-level", action="store_true",
                    help="switch OFF the this-season scoring-level correction (for comparison)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, cpu_count() - 1)))
    ap.add_argument("--out", default=str(HERE / "backtest_output"))
    a = ap.parse_args()

    start = date.fromisoformat(a.start)
    leagues = {n: c["data_code"] for n, c in LEAGUES.items() if c.get("data_code")
               and (a.league == "all" or n == a.league)}
    if not leagues:
        raise SystemExit(f"No such league: {a.league}")

    latest = max((_played(f)["_date"].max() for f in
                  (_read(a.data, CUR, c) for c in leagues.values()) if f is not None),
                 default=None)
    if latest is None:
        raise SystemExit(f"No {CUR}_*.csv files in {a.data}. Run refresh_data.py first.")
    end = date.fromisoformat(a.end) if a.end else latest
    cuts, c = [], start
    while c <= end:
        cuts.append(c)
        c += timedelta(days=7)

    jobs = [(lg, code, cut, a.data, a.xi, a.goal_weight, not a.no_level)
            for lg, code in leagues.items() for cut in cuts]
    print(f"Backtest {start} → {end}: {len(leagues)} leagues × {len(cuts)} weekly refits "
          f"= {len(jobs)} fits (xi={a.xi}, goal_weight={a.goal_weight}, level_correction={not a.no_level}). Takes a few minutes…")
    with Pool(a.workers) as p:
        res = p.map(job, jobs)
    raw = pd.DataFrame([r for rr, _ in res for r in rr])
    notes = sorted({n for _, n in res if n})
    if raw.empty:
        raise SystemExit("No games were predicted. " + "; ".join(notes[:3]))
    raw = raw[raw.date <= end]

    df = build_frame(raw)
    Path(a.out).mkdir(exist_ok=True)
    tag = f"xi{a.xi}_gw{a.goal_weight}" + ("_nolevel" if a.no_level else "")
    df.to_csv(Path(a.out) / f"predictions_{tag}.csv", index=False)

    n_games = raw.shape[0]
    print(f"\nPredicted {n_games} games across {raw.league.nunique()} leagues "
          f"({raw.date.min()} → {raw.date.max()}).")
    for n in notes[:8]:
        print("  note:", n)

    report_logloss(df, f"all {n_games} games")
    recent = df[df.date > SPLIT_DATE]
    if recent.fixture.nunique() >= 30:
        report_logloss(recent, f"only games AFTER {SPLIT_DATE:%d %b} (out-of-sample for earlier tuning)")
    report_by_league(df)
    report_calibration(df)
    report_betting(df)
    report_blend(df)

    section("LIMITS")
    print("Market = football-data AVERAGE price (no Pinnacle in these files). ROI on a few hundred\n"
          "bets can't separate skill from luck: use log loss + calibration, then live CLV.\n"
          f"Saved: {Path(a.out) / f'predictions_{tag}.csv'}")


if __name__ == "__main__":
    main()
