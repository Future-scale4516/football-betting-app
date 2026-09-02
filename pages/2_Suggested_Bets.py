import streamlit as st
import pandas as pd
from football_core import (setup_page, sidebar_date, run_all, build_acca,
                            ENGLISH_EXCL_EPL, REST_OF_EUROPE, ACCA_MARKETS)

setup_page("Football Model — Suggested Bets")
sel_date, _ = sidebar_date()   # this page always runs model-only

st.title("🎟️ Suggested Bets")
st.caption(
    "Six accumulators — WDL, BTTS and Over/Under 2.5, split into English "
    "football excluding the EPL (Championship, League One, League Two) and "
    "the rest of Europe. Legs are the model's most confident picks, and no "
    "fixture is reused anywhere within a group."
)

st.warning(
    "**These are model-only.** This page ignores bookmaker odds entirely, so "
    "there is no edge calculation and no check on whether the model's "
    "probabilities are well-calibrated — backtesting put it only marginally "
    "ahead of guessing. A 5–6 fold also needs every leg to land: your MLB "
    "tracking showed individual legs landing well while the parlay structure "
    "ate the profit. Treat these as the data-collection exercise you asked "
    "for, not as value bets."
)

min_legs = st.slider("Minimum legs per accumulator", 4, 8, 5)
max_legs = st.slider("Maximum legs per accumulator", min_legs, 10, max(6, min_legs))

with st.spinner("Fitting models and loading fixtures..."):
    rows, notes = run_all(sel_date, model_only=True)

for n in notes:
    st.warning(n)

if rows.empty:
    st.info(f"No fixtures found for {sel_date:%a %d %b}.")
    st.stop()

GROUPS = [
    ("🏴 English (excluding EPL)", ENGLISH_EXCL_EPL),
    ("🇪🇺 Rest of Europe", REST_OF_EUROPE),
]

for group_name, group_leagues in GROUPS:
    st.header(group_name)

    available = [l for l in group_leagues if l in set(rows["league"])]
    if not available:
        st.info(f"No fixtures in {', '.join(group_leagues)} on this date.")
        continue

    fixture_count = rows[rows["league"].isin(available)]["fixture"].nunique()
    st.caption(
        f"Drawing from {', '.join(available)} — {fixture_count} fixtures available. "
        f"Three accas of {min_legs}+ legs with no repeats needs at least "
        f"{min_legs * 3} distinct fixtures."
    )

    used_fixtures = set()

    for market in ACCA_MARKETS:
        st.subheader(market)

        acca = build_acca(rows, group_leagues, market,
                           min_legs=min_legs, max_legs=max_legs,
                           exclude_fixtures=used_fixtures,
                           allow_forecast=True)

        if acca is None:
            st.info(
                "No picks left for this market — every available fixture in "
                "this group is already used by an earlier accumulator above."
            )
            continue

        if acca["short"]:
            st.warning(
                f"Only {len(acca['legs'])} fixture(s) left unused in this group — "
                f"below your {min_legs}-leg minimum. Shown for reference, but "
                "this isn't a complete accumulator. Lower the minimum, or pick "
                "a date with more fixtures."
            )

        used_fixtures |= acca["used"]

        for r in acca["legs"]:
            with st.container(border=True):
                st.markdown(f"**{r['selection']}** — {r['fixture']}")
                c1, c2 = st.columns([1, 3])
                c1.metric("Model %", f"{r['model_prob']*100:.1f}%")
                sub = r["league"]
                if r.get("kickoff"):
                    sub = f"{sub} · KO {r['kickoff']}"
                c2.caption(sub)
                if r.get("started"):
                    st.warning("⚠️ Already kicked off — leg picked from a stale "
                               "prediction.", icon="⏱️")

        if not acca["short"]:
            m1, m2 = st.columns(2)
            m1.metric("Legs", len(acca["legs"]))
            m2.metric("Model: chance all legs land",
                      f"{acca['combined_prob']*100:.2f}%")
            st.caption(
                "Combined probability assumes legs are independent, which holds "
                "across different fixtures. No combined odds shown — this page "
                "doesn't fetch prices. Check them at your book."
            )
