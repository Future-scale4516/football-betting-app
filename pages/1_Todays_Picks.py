import streamlit as st
import pandas as pd
from football_core import (setup_page, sidebar_date, run_all, render_pick_card,
                            TIER_ICON, market_label, kickoff_sort_key)

setup_page("Football Model — Today's Picks")
sel_date, model_only = sidebar_date()

st.title("🎯 Today's Picks")

if model_only:
    st.info(
        "**Model-only mode.** Every league and market is forecast from the "
        "Dixon-Coles model with no bookmaker comparison — so Match Result "
        "shows no edge or traffic light here, only the model's own probability."
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


def _card(r, show_edge=True):
    metrics = [("Model %", f"{r['model_prob']*100:.1f}%")]
    if show_edge and r["market_prob"] is not None and pd.notna(r["market_prob"]):
        metrics.append(("Market %", f"{r['market_prob']*100:.1f}%"))
        metrics.append(("Edge", f"{r['edge']*100:+.1f} pts"))
    if r["odds"] and pd.notna(r["odds"]):
        metrics.append(("Odds", f"{r['odds']:.2f}"))
    render_pick_card(
        TIER_ICON.get(r["tier"], "") if show_edge else None,
        r["market_label"] if show_edge else f"{r['selection']} — {r['market_label']}",
        f"{r['league']} · {r['fixture']}",
        metrics,
        reason=(r["reason"] or None) if show_edge else None,
        kickoff=r.get("kickoff"),
        started=bool(r.get("started")),
    )


tab_match, tab_btts, tab_o25, tab_o15 = st.tabs(
    ["⚽ Match Result", "🥅 BTTS", "📈 Over 2.5", "📊 Over 1.5"])

# ------------------------------------------------------------- Match Result
with tab_match:
    st.caption(
        "Model probability vs de-vigged market probability. 🟢/🟡 cleared the "
        "plausibility ceiling and traffic-light bands. 🔵 needs manual "
        "checking — either a large edge, or a provisionally-seeded promoted "
        "team. ⚪ means no meaningful edge."
    )
    mr = rows[rows["market"] == "1X2"]

    c1, c2 = st.columns(2)
    with c1:
        league_filter = st.multiselect("League", leagues, default=leagues, key="mr_lg")
    with c2:
        tier_filter = st.multiselect(
            "Show", ["green", "amber", "verify", "red", "forecast"],
            default=["green", "amber", "verify", "forecast"], key="mr_tier",
            help="'forecast' = no market price available (League One/Two, "
                 "or model-only mode)")

    view = mr[mr["league"].isin(league_filter) & mr["tier"].isin(tier_filter)]

    if view.empty:
        st.info("Nothing matches those filters.")
    else:
        counts = view["tier"].value_counts()
        st.markdown(" · ".join(f"{TIER_ICON.get(t, '')} {c} {t}"
                                for t, c in counts.items()))

        sort_choice = st.selectbox(
            "Sort by:", ["Kickoff time", "Edge", "Model %", "Odds"], key="mr_sort")
        if sort_choice == "Kickoff time":
            view = view.assign(_k=view["kickoff"].map(kickoff_sort_key))
            view = view.sort_values("_k")
        elif sort_choice == "Edge":
            view = view.assign(_e=view["edge"].abs()).sort_values("_e", ascending=False)
        elif sort_choice == "Model %":
            view = view.sort_values("model_prob", ascending=False)
        else:
            view = view.sort_values("odds", ascending=False)

        for _, r in view.iterrows():
            _card(r, show_edge=True)

# ---------------------------------------------------------------------- BTTS
with tab_btts:
    st.caption(
        "Ranked by how likely the model thinks BTTS is, highest first — "
        "Yes only, since that's what you'd actually stake."
    )
    btts = rows[(rows["market"] == "BTTS") & (rows["selection"] == "Yes")]

    c1, c2 = st.columns([3, 1])
    with c1:
        league_filter = st.multiselect("League", leagues, default=leagues, key="bt_lg")
    with c2:
        top_n = st.number_input("Show", 5, 100, 20, 5, key="bt_n")

    view = btts[btts["league"].isin(league_filter)].sort_values(
        "model_prob", ascending=False).head(int(top_n))

    if view.empty:
        st.info("Nothing matches those filters.")
    else:
        for _, r in view.iterrows():
            _card(r, show_edge=False)

# ----------------------------------------------------------------- Over 2.5
with tab_o25:
    st.caption("Ranked by how likely the model thinks Over 2.5 goals is, highest first.")
    o25 = rows[rows["market_label"] == "Over 2.5"]

    c1, c2 = st.columns([3, 1])
    with c1:
        league_filter = st.multiselect("League", leagues, default=leagues, key="o25_lg")
    with c2:
        top_n = st.number_input("Show", 5, 100, 20, 5, key="o25_n")

    view = o25[o25["league"].isin(league_filter)].sort_values(
        "model_prob", ascending=False).head(int(top_n))

    if view.empty:
        st.info("Nothing matches those filters.")
    else:
        for _, r in view.iterrows():
            _card(r, show_edge=False)

# ----------------------------------------------------------------- Over 1.5
with tab_o15:
    st.caption("Ranked by how likely the model thinks Over 1.5 goals is, highest first.")
    o15 = rows[rows["market_label"] == "Over 1.5"]

    c1, c2 = st.columns([3, 1])
    with c1:
        league_filter = st.multiselect("League", leagues, default=leagues, key="o15_lg")
    with c2:
        top_n = st.number_input("Show", 5, 100, 20, 5, key="o15_n")

    view = o15[o15["league"].isin(league_filter)].sort_values(
        "model_prob", ascending=False).head(int(top_n))

    if view.empty:
        st.info("Nothing matches those filters.")
    else:
        for _, r in view.iterrows():
            _card(r, show_edge=False)
