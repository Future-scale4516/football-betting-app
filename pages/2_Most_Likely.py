import streamlit as st
import pandas as pd
from football_core import (setup_page, sidebar_date, run_all, render_pick_card,
                            market_label, market_label_options,
                            DEFAULT_MARKET_LABELS)

setup_page("Football Model — Most Likely")
sel_date, model_only = sidebar_date()

st.title("📊 Most Likely")
st.caption(
    "Ranked purely by how confident the model is, ignoring the market entirely. "
    "This answers 'what does the model expect to happen', not 'where is the value'. "
    "Over 1.5 will dominate an unfiltered list because that market is structurally "
    "easy to be right about — use the market filter to see the most likely pick "
    "within a specific market instead."
)

with st.spinner("Fitting models and fetching fixtures..."):
    rows, notes = run_all(sel_date, model_only)

for n in notes:
    st.warning(n)

if rows.empty:
    st.info(f"No fixtures found for {sel_date:%a %d %b}.")
    st.stop()

rows["market_label"] = [market_label(m, sel) for m, sel
                        in zip(rows["market"], rows["selection"])]

leagues = sorted(rows["league"].unique())
label_options = market_label_options(rows)
default_labels = [m for m in DEFAULT_MARKET_LABELS if m in label_options] or label_options

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    league_filter = st.multiselect("League", leagues, default=leagues)
with c2:
    market_filter = st.multiselect(
        "Market", label_options, default=default_labels,
        help="Defaults to Match Winner. Pick exactly the markets you want "
             "ranked rather than loading all of them.",
    )
with c3:
    top_n = st.number_input("Show", min_value=5, max_value=100, value=20, step=5)

view = rows[
    rows["league"].isin(league_filter)
    & rows["market_label"].isin(market_filter)
]

if view.empty:
    st.info("Nothing matches those filters.")
    st.stop()

view = view.sort_values("model_prob", ascending=False).head(int(top_n))

for _, r in view.iterrows():
    metrics = [("Model %", f"{r['model_prob']*100:.1f}%")]
    if r["odds"] and pd.notna(r["odds"]):
        metrics.append(("Odds", f"{r['odds']:.2f}"))
    render_pick_card(
        None,
        r["market_label"],
        f"{r['league']} · {r['fixture']}",
        metrics,
        reason="Forecast only — no odds for this league"
        if r["tier"] == "forecast" else None,
        kickoff=r.get("kickoff"),
        started=bool(r.get("started")),
    )
