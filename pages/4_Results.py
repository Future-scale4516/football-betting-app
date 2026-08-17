import streamlit as st
import pandas as pd
from football_core import setup_page
from results_tracker import _load as load_results

setup_page("Football Model — Results")

st.title("📈 Results & CLV")
st.caption(
    "Picks logged from previous runs, settled against real outcomes and the "
    "closing line. Closing-line value is the real long-run test of whether the "
    "model beats the market — hit rate alone isn't."
)

st.info(
    "Note: on Streamlit Cloud the log resets whenever the app redeploys, because "
    "the filesystem is ephemeral. Local runs keep their own history. A persistent "
    "store (Google Sheets or similar) is a later job."
)

df = load_results()

if df.empty:
    st.write("No picks logged yet.")
    st.stop()

settled = df[df["actual_outcome"].notna()]
open_picks = df[df["actual_outcome"].isna()]

c1, c2, c3 = st.columns(3)
c1.metric("Logged", len(df))
c2.metric("Settled", len(settled))
c3.metric("Open", len(open_picks))

if not settled.empty:
    pnl = settled["pnl"].sum()
    staked = settled["stake"].sum()
    roi = pnl / staked if staked else 0
    clv = settled["beat_clv"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("P/L (units)", f"{pnl:+.2f}")
    c2.metric("ROI", f"{roi:+.1%}")
    c3.metric("Beat closing line", f"{clv:.1%}")

    st.subheader("By market")
    st.dataframe(
        settled.groupby("market").agg(
            picks=("pnl", "count"), pnl=("pnl", "sum"),
            clv_rate=("beat_clv", "mean")).reset_index(),
        use_container_width=True, hide_index=True)

if not open_picks.empty:
    st.subheader("Awaiting settlement")
    for _, r in open_picks.iterrows():
        with st.container(border=True):
            st.markdown(f"**{r['selection']} — {r['market']}**")
            st.caption(f"{r['league']} · {r['fixture']}")
    st.caption(
        "Settling isn't wired into the UI yet — call "
        "results_tracker.settle_pick() directly for now."
    )
