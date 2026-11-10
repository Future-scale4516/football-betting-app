"""
Closing-line value report. Joins the odds you logged earlier (data/odds_log.csv)
with the closing odds in a Results-page export, then shows whether the prices
the app displayed beat the close.

    python3 clv_report.py results_2026-10-24_All.csv [more.csv ...]

Two numbers per bet (positive = you got a better price than the close):
  clv       = logged_odds * closing_fair_prob - 1   (vs the margin-free close;
              the headline figure — it is the expected return at the closing price)
  clv_raw   = logged_odds / closing_odds - 1        (vs the raw close, margin included;
              understates CLV by roughly the closing margin)
The close is Pinnacle's where football-data publishes it (the source is shown),
otherwise the average close. Uses the EARLIEST logged price per selection. Consistently
positive CLV on model-flagged picks is the real evidence of an edge; ROI on a
few hundred bets is mostly luck.
"""
import sys
import pandas as pd

KEY = ["fixture", "market", "selection"]


def build(log, res, edge_min=0.03):
    log = log.sort_values("logged_at").drop_duplicates(KEY)
    cols = KEY + ["closing_odds", "won"]
    for extra in ("closing_fair_prob", "closing_source"):
        if extra in res.columns:
            cols.append(extra)
    res = res.drop_duplicates(KEY)[cols]
    df = log.merge(res, on=KEY).dropna(subset=["closing_odds"])
    df["clv_raw"] = df.logged_odds / df.closing_odds - 1
    if "closing_fair_prob" in df.columns:
        df["clv"] = (df.logged_odds * df.closing_fair_prob - 1).fillna(df.clv_raw)
    else:
        df["clv"] = df.clv_raw
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


def by_edge_bucket(df):
    """Average CLV by how big the model's claimed edge was. If CLV does not rise
    with the edge, the edge size is noise and thresholds mean nothing."""
    bins = [-1, 0, 0.02, 0.04, 0.06, 1]
    names = ["<0%", "0-2%", "2-4%", "4-6%", "6%+"]
    b = pd.cut(df.edge, bins=bins, labels=names)
    g = df.groupby(b, observed=True).agg(n=("clv", "size"), avg_clv=("clv", "mean"),
                                         beat_close=("clv", lambda x: (x > 0).mean()))
    return g.reset_index().rename(columns={"edge": "claimed_edge"})


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
    e = by_edge_bucket(df)
    e["avg_clv"] = e.avg_clv.map("{:+.1%}".format)
    e["beat_close"] = e.beat_close.map("{:.0%}".format)
    print("\nBy claimed edge:\n" + e.to_string(index=False))
    if "closing_source" in df.columns:
        print("\nClosing price source:", df.closing_source.value_counts().to_dict())
