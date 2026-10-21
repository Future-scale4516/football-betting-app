"""
Closing-line value report. Joins the odds you logged earlier (data/odds_log.csv)
with the closing odds in a Results-page export, then shows whether the prices
the app displayed beat the close.

    python3 clv_report.py results_2026-10-24_All.csv [more.csv ...]

CLV = logged_odds / closing_odds - 1 (positive = you'd have got a better price
than the close). Uses the EARLIEST logged price per selection. Consistently
positive CLV on model-flagged picks is the real evidence of an edge; ROI on a
few hundred bets is mostly luck.
"""
import sys
import pandas as pd

KEY = ["fixture", "market", "selection"]


def build(log, res, edge_min=0.03):
    log = log.sort_values("logged_at").drop_duplicates(KEY)
    res = res.drop_duplicates(KEY)[KEY + ["closing_odds", "won"]]
    df = log.merge(res, on=KEY).dropna(subset=["closing_odds"])
    df["clv"] = df.logged_odds / df.closing_odds - 1
    df["edge"] = df.model_prob - df.market_prob
    df["flagged"] = df.edge >= edge_min
    return df


def summarise(df):
    out = []
    for flagged, g in df.groupby("flagged"):
        out.append({"group": "flagged (edge>=3%)" if flagged else "not flagged",
                    "n": len(g), "avg_clv": g.clv.mean(),
                    "beat_close": (g.clv > 0).mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    log = pd.read_csv("data/odds_log.csv")
    res = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)
    df = build(log, res)
    if df.empty:
        raise SystemExit("No overlap between the odds log and these results.")
    s = summarise(df)
    s["avg_clv"] = s.avg_clv.map("{:+.1%}".format)
    s["beat_close"] = s.beat_close.map("{:.0%}".format)
    print(s.to_string(index=False))
