import streamlit as st
import pandas as pd
from football_core import (setup_page, sidebar_date, run_all, render_pick_card,
                            TIER_ICON, market_label, kickoff_sort_key,
                            add_manual_fixtures, clear_manual_fixtures,
                            manual_rows_for)
from league_config import LEAGUES

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

# Manual entry sits ABOVE the empty-check on purpose: it's the way out when
# no data source has the day's games, so it has to be reachable exactly
# when the automatic list comes back empty.
with st.expander("➕ Add fixtures manually"):
    st.caption(
        "For games the app can't find — a division it doesn't list, a fixture "
        "list that's late, or a blocked source. One per line, optional kickoff "
        "time first, e.g. `12:30 Chesterfield v Tranmere`. The model forecasts "
        "them like any other game."
    )
    man_league = st.selectbox("League", list(LEAGUES), key="man_league")
    man_text = st.text_area("Fixtures", height=150, key="man_text",
                             placeholder="12:30 Chesterfield v Tranmere\n"
                                         "Accrington v Cheltenham")
    b1, b2 = st.columns(2)
    if b1.button("Add these fixtures"):
        if man_text.strip():
            add_manual_fixtures(man_league, man_text, sel_date)
            st.rerun()
    if b2.button("Clear added fixtures"):
        clear_manual_fixtures(sel_date)
        st.rerun()

manual_rows, manual_notes = manual_rows_for(sel_date)
for n in manual_notes:
    st.warning(n)
if not manual_rows.empty:
    rows = pd.concat([rows, manual_rows], ignore_index=True).drop_duplicates(
        subset=["league", "fixture", "market", "selection"])
    st.caption(f"Includes {manual_rows['fixture'].nunique()} fixture(s) added manually.")

if rows.empty:
    st.info(f"No fixtures found for {sel_date:%a %d %b}. If games are on, add "
            "them in the box above.")
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
        f"{r['league']} · {r['fixture']}"
        + (" · ⚠️ provisional rating" if r.get("seeded") else ""),
        metrics,
        reason=(r["reason"] or None) if show_edge else None,
        kickoff=r.get("kickoff"),
        started=bool(r.get("started")),
    )


def _sort_ranked(view, key):
    """Shared 'Sort by' control for the three model-confidence tabs —
    Model % (default), Kickoff time, or Odds where they exist."""
    choice = st.selectbox("Sort by:", ["Model %", "Kickoff time", "Odds"], key=key)
    if choice == "Kickoff time":
        return view.assign(_k=view["kickoff"].map(kickoff_sort_key)).sort_values("_k")
    if choice == "Odds":
        return view.sort_values("odds", ascending=False, na_position="last")
    return view.sort_values("model_prob", ascending=False)


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
            view = view.sort_values("odds", ascending=False, na_position="last")

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

    view = btts[btts["league"].isin(league_filter)]

    if view.empty:
        st.info("Nothing matches those filters.")
    else:
        view = _sort_ranked(view, key="bt_sort").head(int(top_n))
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

    view = o25[o25["league"].isin(league_filter)]

    if view.empty:
        st.info("Nothing matches those filters.")
    else:
        view = _sort_ranked(view, key="o25_sort").head(int(top_n))
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

    view = o15[o15["league"].isin(league_filter)]

    if view.empty:
        st.info("Nothing matches those filters.")
    else:
        view = _sort_ranked(view, key="o15_sort").head(int(top_n))
        for _, r in view.iterrows():
            _card(r, show_edge=False)
