import streamlit as st
import pandas as pd
from datetime import date
from football_core import setup_page, selection_side
from results_tracker import _load as load_results, settle_by_index, delete_by_index

setup_page("Football Model — Results")

st.sidebar.markdown("### Date")
view_mode = st.sidebar.radio("Show picks for:", ["A single day", "All dates"])
sel_date = None
if view_mode == "A single day":
    sel_date = st.sidebar.date_input("Date:", value=date.today())

st.title("📈 Results & CLV")
st.caption(
    "Picks logged from the Today's Picks page, scored against real outcomes "
    "and the closing line. Closing-line value is the real long-run test of "
    "whether the model beats the market — hit rate alone isn't."
)

st.info(
    "On Streamlit Cloud this log resets whenever the app redeploys, because "
    "the filesystem is ephemeral. Local runs keep their own history. A "
    "persistent store (Google Sheets or similar) is a later job."
)

df = load_results()

if df.empty:
    st.warning(
        "No picks logged yet. Go to **Today's Picks**, filter to what you want "
        "to track, and use the *Log these picks* button at the bottom."
    )
    st.stop()

df["fixture_date"] = df["fixture_date"].astype(str)

if sel_date is not None:
    day = df[df["fixture_date"] == str(sel_date)]
    if day.empty:
        st.info(
            f"Nothing logged for {sel_date:%a %d %b}. "
            f"Dates with picks: {', '.join(sorted(df['fixture_date'].unique())[-8:])}"
        )
        st.stop()
    df = day

# ------------------------------------------------------------------ filters
df["side"] = df["selection"].astype(str).map(selection_side)
df["status"] = df["actual_outcome"].fillna("open")

c1, c2, c3 = st.columns(3)
with c1:
    leagues = sorted(df["league"].dropna().unique())
    league_filter = st.multiselect("League", leagues, default=leagues)
with c2:
    markets = sorted(df["market"].dropna().unique())
    market_filter = st.multiselect("Market", markets, default=markets)
with c3:
    statuses = sorted(df["status"].unique())
    status_filter = st.multiselect("Status", statuses, default=statuses)

c4, c5 = st.columns(2)
with c4:
    sides = sorted(df["side"].dropna().unique())
    side_filter = st.multiselect("Selection", sides, default=sides)
with c5:
    tiers = sorted(df["tier"].dropna().unique())
    tier_filter = st.multiselect("Tier", tiers, default=tiers)

view = df[
    df["league"].isin(league_filter) & df["market"].isin(market_filter)
    & df["status"].isin(status_filter) & df["side"].isin(side_filter)
    & df["tier"].isin(tier_filter)
]

if view.empty:
    st.info("Nothing matches those filters.")
    st.stop()

# ------------------------------------------------------------------ summary
settled = view[view["actual_outcome"].notna()]
priced = settled[settled["pnl"].notna()]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Picks", len(view))
m2.metric("Settled", len(settled))
if not settled.empty:
    hit_rate = (settled["actual_outcome"] == "won").mean()
    m3.metric("Hit rate", f"{hit_rate:.1%}")
if not priced.empty:
    pnl = priced["pnl"].sum()
    staked = priced["stake"].sum()
    m4.metric("P/L (units)", f"{pnl:+.2f}",
              f"ROI {pnl/staked:+.1%}" if staked else None)

clv_known = settled[settled["beat_clv"].notna()]
if not clv_known.empty:
    st.metric("Beat closing line", f"{clv_known['beat_clv'].mean():.1%}",
              help="Above 50% consistently is the sign a model genuinely "
                   "beats the market. Needs a few hundred picks to mean much.")

# -------------------------------------------------------------------- table
st.subheader("Picks")

table = view[["fixture_date", "league", "fixture", "market", "selection",
               "model_prob", "odds_at_pick", "tier", "status", "pnl"]].copy()
table["model_prob"] = (table["model_prob"].astype(float) * 100).round(1)
table = table.rename(columns={
    "fixture_date": "Date", "league": "League", "fixture": "Fixture",
    "market": "Market", "selection": "Pick", "model_prob": "Model %",
    "odds_at_pick": "Odds", "tier": "Tier", "status": "Status", "pnl": "P/L",
})
st.dataframe(table, use_container_width=True, hide_index=True)

if not settled.empty:
    st.subheader("By market")
    by_market = settled.groupby("market").agg(
        picks=("actual_outcome", "count"),
        won=("actual_outcome", lambda s: (s == "won").sum()),
        pnl=("pnl", "sum"),
    ).reset_index()
    by_market["hit_rate"] = (by_market["won"] / by_market["picks"] * 100).round(1)
    st.dataframe(by_market, use_container_width=True, hide_index=True)

# --------------------------------------------------------------- settlement
st.divider()
st.subheader("Settle a pick")
st.caption(
    "Mark a result once the match has finished. Closing odds are optional but "
    "worth adding where you have them — without them there's no CLV, which is "
    "the only real measure of whether the model beats the market."
)

open_picks = view[view["actual_outcome"].isna()]
if open_picks.empty:
    st.write("Nothing open to settle in the current filter.")
else:
    labels = {
        i: f"{r['fixture']} — {r['market']}: {r['selection']} "
           f"({r['league']}, {r['fixture_date']})"
        for i, r in open_picks.iterrows()
    }
    chosen = st.selectbox("Pick:", list(labels.keys()),
                           format_func=lambda i: labels[i])

    s1, s2, s3 = st.columns([1, 1, 2])
    with s1:
        won = st.radio("Result", ["Won", "Lost"], horizontal=True)
    with s2:
        odds_at_pick = open_picks.loc[chosen, "odds_at_pick"]
        closing = st.number_input(
            "Closing odds", min_value=0.0, value=0.0, step=0.05,
            help="Leave at 0 to skip. Not available for model-only picks.",
            disabled=pd.isna(odds_at_pick))
    with s3:
        st.write("")
        if st.button("Settle this pick", type="primary"):
            settle_by_index(chosen, won == "Won", closing if closing > 0 else None)
            st.success("Settled.")
            st.rerun()

    with st.expander("Remove a logged pick"):
        st.caption("For picks logged by mistake — this deletes the row entirely.")
        if st.button("Delete selected pick"):
            delete_by_index(chosen)
            st.success("Deleted.")
            st.rerun()
