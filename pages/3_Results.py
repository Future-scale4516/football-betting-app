import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
from football_core import setup_page, market_label
from historical_results import build_historical_results
from football_data_source import parse_uploaded, season_candidates
from clv_report import build as clv_build, summarise as clv_summarise, by_edge_bucket
from league_config import LEAGUES, modelled_leagues

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

league_options = modelled_leagues()   # UEFA competitions have no model to grade
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

# ------------------------------------------------------- game predictions
view_type = st.radio(
    "View", ["Game predictions", "Market picks"], horizontal=True,
    help="Game predictions shows what the model projected for each match "
         "(expected goals, most likely score, win/BTTS/Over 2.5 chances) "
         "against what happened. Market picks grades individual bets.")
st.caption("Player props aren't part of the football model — it predicts "
           "match outcomes and goals only.")

if view_type == "Game predictions":
    games = results.drop_duplicates("fixture").copy()
    games["call"] = games[["p_home", "p_draw", "p_away"]].idxmax(axis=1).map(
        {"p_home": "Home", "p_draw": "Draw", "p_away": "Away"})
    games["actual"] = np.where(games["home_goals"] > games["away_goals"], "Home",
                               np.where(games["home_goals"] == games["away_goals"],
                                        "Draw", "Away"))
    games["result_ok"] = games["call"] == games["actual"]
    games["btts_ok"] = (games["p_btts"] >= 0.5) == ((games["home_goals"] > 0) & (games["away_goals"] > 0))
    games["o25_ok"] = (games["p_over25"] >= 0.5) == ((games["home_goals"] + games["away_goals"]) > 2.5)
    games["exact_ok"] = games["likely_score"] == games["actual_score"]

    n = len(games)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Result called", f"{int(games.result_ok.sum())}/{n}")
    m2.metric("BTTS called", f"{int(games.btts_ok.sum())}/{n}")
    m3.metric("Over/Under 2.5 called", f"{int(games.o25_ok.sum())}/{n}")
    m4.metric("Exact score", f"{int(games.exact_ok.sum())}/{n}")
    st.caption(
        f"Expected goals per game: model {(games.xg_home + games.xg_away).mean():.2f}, "
        f"actual {(games.home_goals + games.away_goals).mean():.2f}. "
        "Exact scores are rarely hit (even the best model is right ~10% of the "
        "time), and one slate is a small sample.")

    tick = {True: "✅", False: "❌"}
    table = pd.DataFrame({
        "League": games["league"], "Game": games["fixture"],
        "xG": games.xg_home.round(2).astype(str) + " – " + games.xg_away.round(2).astype(str),
        "Likely": games["likely_score"], "Actual": games["actual_score"],
        "Home %": (games.p_home * 100).round(0), "Draw %": (games.p_draw * 100).round(0),
        "Away %": (games.p_away * 100).round(0),
        "Called": games["call"] + " " + games.result_ok.map(tick),
        "BTTS %": (games.p_btts * 100).round(0), "BTTS": games.btts_ok.map(tick),
        "O2.5 %": (games.p_over25 * 100).round(0), "O2.5": games.o25_ok.map(tick),
    })
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download game predictions CSV",
                       table.to_csv(index=False).encode(),
                       file_name=f"game_predictions_{loaded_date}.csv", key="dl_games")

# ------------------------------------------------------------- market tabs
if view_type == "Game predictions":
    market_view = False
else:
    market_view = True

if market_view:
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


# ------------------------------------------------------------------- CLV
st.divider()
st.header("📉 Closing line value (CLV)")
st.caption(
    "Scores the prices the app showed you against the margin-free closing price "
    "(Pinnacle's where available, otherwise the average close, de-vigged with "
    "Shin's method) — the sharpest number the bookmakers publish. Needs prices logged earlier with "
    "`log_odds.py` (free: no Odds API credits are used here). Beating the "
    "close consistently is the real sign of an edge; one slate proves nothing.")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "odds_log.csv")
if not os.path.exists(LOG_PATH):
    st.info("No odds log yet. Run `python3 log_odds.py` before matchday, "
            "commit `data/odds_log.csv`, then load results here afterwards.")
else:
    log = pd.read_csv(LOG_PATH)
    log = log[log["match_date"] == str(loaded_date)]
    if log.empty:
        st.info(f"Nothing was logged for {loaded_date:%a %d %b}.")
    else:
        df = clv_build(log, results)
        if df.empty:
            st.info("Logged prices found, but none matched a graded selection with "
                    "closing odds. Tick **Grade every selection** and reload.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Picks compared", len(df))
            c2.metric("Beat the close", f"{(df.clv > 0).mean():.0%}")
            c3.metric("Average CLV", f"{df.clv.mean():+.1%}")
            summ = clv_summarise(df)
            summ["avg_clv"] = summ["avg_clv"].map("{:+.1%}".format)
            summ["beat_close"] = summ["beat_close"].map("{:.0%}".format)
            st.dataframe(summ.rename(columns={
                "group": "Group", "n": "Picks", "avg_clv": "Avg CLV",
                "beat_close": "Beat close"}), hide_index=True, use_container_width=True)
            if "closing_source" in df.columns and df.closing_source.notna().any():
                st.caption("Closing price used: " + ", ".join(
                    f"{k} ×{v}" for k, v in df.closing_source.value_counts().items()))
            eb = by_edge_bucket(df)
            eb["avg_clv"] = eb["avg_clv"].map("{:+.1%}".format)
            eb["beat_close"] = eb["beat_close"].map("{:.0%}".format)
            st.markdown("**By the edge the model claimed** — if CLV doesn't rise with "
                        "the edge, the edge size is noise.")
            st.dataframe(eb.rename(columns={"claimed_edge": "Claimed edge", "n": "Picks",
                         "avg_clv": "Avg CLV", "beat_close": "Beat close"}),
                         hide_index=True, use_container_width=True)
            if len(df) < 20:
                st.info("Too few picks to read anything into — build this up over many dates.")
            st.dataframe(df[["fixture", "market", "selection", "logged_odds",
                             "closing_odds", "clv", "won"]].sort_values("clv", ascending=False),
                         hide_index=True, use_container_width=True)
