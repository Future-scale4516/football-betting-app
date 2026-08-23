import streamlit as st
import pandas as pd
from datetime import date, timedelta
from football_core import setup_page, market_label
from historical_results import build_historical_results
from football_data_source import parse_uploaded, season_candidates
from league_config import LEAGUES

setup_page("Football Model — Results")

st.title("📈 Results")
st.caption(
    "How the model's predictions actually turned out for the selected date — "
    "what it projected, what happened, and whether it landed. The model is "
    "refit using only data available before that date, so this is a genuine "
    "backtest of that day rather than hindsight. Free — no Odds API credits."
)

st.sidebar.markdown("### Date")
sel_date = st.sidebar.date_input("Results for:", value=date.today() - timedelta(days=1))

league_options = list(LEAGUES.keys())
league_filter = st.multiselect("Leagues", league_options, default=league_options)

show_all = st.checkbox(
    "Grade every selection, not just the model's top pick per market",
    value=False,
    help="Off (default) mirrors the MLB app: one pick per market per game — "
         "what the model actually thought would happen. On grades both sides "
         "of every market, which is better for calibration but noisier.")

# ---------------------------------------------------- manual upload fallback
with st.expander("Results not loading? Upload the CSV manually"):
    st.caption(
        "football-data.co.uk is occasionally awkward to fetch from. If the "
        "automatic download fails, grab the file yourself from "
        "football-data.co.uk/englandm.php (or the relevant country page) and "
        "upload it here — same file, no download needed."
    )
    up_league = st.selectbox("Which league is this file for?", league_options)
    up_file = st.file_uploader("Results CSV", type=["csv"])
    if up_file is not None:
        try:
            st.session_state.setdefault("uploaded_results", {})
            st.session_state["uploaded_results"][up_league] = parse_uploaded(up_file)
            st.success(f"Loaded manually for {up_league}.")
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")

if st.button("Load results", type="primary"):
    with st.spinner("Refitting models as of that date and grading against real scores..."):
        results, notes, diagnostics = build_historical_results(
            sel_date, league_filter,
            pick_mode="all" if show_all else "most_likely",
            uploaded=st.session_state.get("uploaded_results"))
    st.session_state.update({"hist_results": results, "hist_notes": notes,
                              "hist_diag": diagnostics, "hist_date": sel_date})

if "hist_results" not in st.session_state:
    st.info("Pick a date and click **Load results**.")
    st.stop()

results = st.session_state["hist_results"]
notes = st.session_state["hist_notes"]
diagnostics = st.session_state.get("hist_diag", {})
loaded_date = st.session_state["hist_date"]

for n in notes:
    st.warning(n)

if diagnostics:
    with st.expander("Data source diagnostics"):
        st.caption(
            f"Season codes tried for {loaded_date}: "
            f"{', '.join(season_candidates(loaded_date))}. Each line shows "
            "what that URL actually returned."
        )
        for lg, log in diagnostics.items():
            st.markdown(f"**{lg}**")
            for line in log:
                st.text(f"  {line}")

if results.empty:
    st.info(
        f"No gradeable fixtures for {loaded_date:%a %d %b} in the selected "
        "leagues. Open the diagnostics above to see whether the files loaded "
        "and simply had no games that day, or failed to download."
    )
    st.stop()

results["market_label"] = [market_label(m, s) for m, s
                            in zip(results["market"], results["selection"])]

st.caption(f"Scored {len(results)} predictions across "
           f"{results['fixture'].nunique()} completed games.")

# ------------------------------------------------------------- market tabs
label_order = ["All"] + [m for m in [
    "Match Winner - Home", "Match Winner - Draw", "Match Winner - Away",
    "BTTS - Yes", "BTTS - No", "Over 2.5", "Under 2.5"]
    if m in set(results["market_label"])]

tabs = st.tabs(label_order)

for tab, label in zip(tabs, label_order):
    with tab:
        view = results if label == "All" else results[results["market_label"] == label]
        if view.empty:
            st.write("Nothing here.")
            continue

        threshold = st.slider(
            "Only show picks the model rated at least this likely (%)",
            0, 100, 0, key=f"thresh_{label}",
            help="Raising this checks calibration on the model's most "
                 "confident picks specifically.")
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
            st.info(f"Only {total} picks — too small a sample to read much "
                    "into the gap either way.")
        elif abs(gap) <= 7:
            st.success(f"Model said {predicted_rate:.1f}%, reality was "
                       f"{actual_rate:.1f}% — closely calibrated on this slate.")
        elif gap < 0:
            st.warning(f"Model said {predicted_rate:.1f}%, reality was only "
                       f"{actual_rate:.1f}% — overconfident on this slate.")
        else:
            st.warning(f"Model said {predicted_rate:.1f}%, reality was "
                       f"{actual_rate:.1f}% — underconfident on this slate.")

        st.caption("One slate is a small sample — a day swinging either way is "
                   "normal variance, not a verdict on the model. Load several "
                   "dates to judge calibration properly.")

        table = filtered.sort_values("model_prob", ascending=False).copy()
        table["✓"] = table["won"].map({True: "✅", False: "❌"})
        table["Model %"] = (table["model_prob"] * 100).round(1)
        table = table.rename(columns={
            "league": "League", "fixture": "Fixture",
            "market_label": "Market", "actual_score": "Score",
            "closing_odds": "Closing odds"})
        st.dataframe(
            table[["✓", "League", "Fixture", "Market", "Model %", "Score",
                   "Closing odds"]],
            use_container_width=True, hide_index=True)

        st.download_button(
            "Download results CSV",
            filtered.to_csv(index=False).encode(),
            file_name=f"results_{loaded_date}_{label.replace(' ', '_')}.csv",
            key=f"dl_{label}")
