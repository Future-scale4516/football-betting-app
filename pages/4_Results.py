import streamlit as st
import pandas as pd
from datetime import date, timedelta
from football_core import setup_page
from historical_results import build_historical_results
from league_config import LEAGUES

setup_page("Football Model — Results")

st.title("📈 Results")
st.caption(
    "How the model's predictions actually turned out for the selected date — "
    "what it projected, what actually happened, and whether it landed. "
    "Refits the model using only data available before that date, so this "
    "is a genuine backtest of that day, not hindsight. Free — no Odds API "
    "credits used."
)

st.sidebar.markdown("### Date")
sel_date = st.sidebar.date_input("Results for:", value=date.today() - timedelta(days=1))

league_options = list(LEAGUES.keys())
league_filter = st.multiselect("Leagues", league_options, default=league_options)

if st.button("Load results", type="primary"):
    with st.spinner("Refitting models as of that date and grading against real scores..."):
        results, notes = build_historical_results(sel_date, league_filter)
    st.session_state["hist_results"] = results
    st.session_state["hist_notes"] = notes
    st.session_state["hist_date"] = sel_date

if "hist_results" not in st.session_state:
    st.info("Pick a date and click **Load results**.")
    st.stop()

results = st.session_state["hist_results"]
notes = st.session_state["hist_notes"]
loaded_date = st.session_state["hist_date"]

for n in notes:
    st.warning(n)

if results.empty:
    st.info(
        f"No completed, gradeable fixtures found for {loaded_date:%a %d %b} in the "
        "selected leagues — either nothing was played, or football-data.co.uk "
        "hasn't published results for it yet."
    )
    st.stop()

if loaded_date != sel_date:
    st.caption(f"Showing results loaded for {loaded_date:%a %d %b} — "
               "change the date and click Load results again to refresh.")

st.caption(f"Scored {len(results)} predictions across "
           f"{results['fixture'].nunique()} completed games.")

# ------------------------------------------------------------- market tabs
markets_present = ["All"] + sorted(results["market"].unique())
tabs = st.tabs(markets_present)

for tab, market in zip(tabs, markets_present):
    with tab:
        view = results if market == "All" else results[results["market"] == market]
        if view.empty:
            st.write("Nothing here.")
            continue

        max_prob = float((view["model_prob"] * 100).max())
        threshold = st.slider(
            "Only show picks the model rated at least this likely (%)",
            0, 100, 0, key=f"thresh_{market}",
            help="Raising this checks calibration on the model's most confident "
                 "picks specifically — where it should be most accurate.",
        )
        filtered = view[view["model_prob"] * 100 >= threshold]

        if filtered.empty:
            st.write("No picks meet that threshold.")
            continue

        landed = int(filtered["won"].sum())
        total = len(filtered)
        actual_rate = landed / total * 100
        predicted_rate = filtered["model_prob"].mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Landed", f"{landed}/{total}")
        c2.metric("Actual hit rate", f"{actual_rate:.1f}%")
        c3.metric("Model predicted", f"{predicted_rate:.1f}%")

        gap = actual_rate - predicted_rate
        if total < 15:
            st.info(f"Only {total} picks — too small a sample to read much into "
                    "the gap either way.")
        elif abs(gap) <= 7:
            st.success(f"Model said {predicted_rate:.1f}%, reality was "
                       f"{actual_rate:.1f}% — closely calibrated on this slate.")
        elif gap < 0:
            st.warning(f"Model said {predicted_rate:.1f}%, reality was only "
                       f"{actual_rate:.1f}% — overconfident on this slate.")
        else:
            st.warning(f"Model said {predicted_rate:.1f}%, reality was "
                       f"{actual_rate:.1f}% — underconfident on this slate.")

        st.caption("One slate is a small sample — a single day swinging either way "
                   "is normal variance, not a verdict on the model. Load several "
                   "dates over time to judge calibration properly.")

        # ---------------------------------------------------------- table
        table = filtered.sort_values("model_prob", ascending=False).copy()
        table["✓"] = table["won"].map({True: "✅", False: "❌"})
        table["Model %"] = (table["model_prob"] * 100).round(1)
        table = table.rename(columns={
            "league": "League", "fixture": "Fixture", "market": "Market",
            "selection": "Pick", "actual_score": "Score",
            "closing_odds": "Closing odds",
        })
        st.dataframe(
            table[["✓", "League", "Fixture", "Market", "Pick", "Model %",
                   "Score", "Closing odds"]],
            use_container_width=True, hide_index=True,
        )

        csv = filtered.to_csv(index=False).encode()
        st.download_button("Download results CSV", csv,
                            file_name=f"results_{loaded_date}_{market}.csv",
                            key=f"dl_{market}")
