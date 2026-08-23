import streamlit as st
import pandas as pd
from football_core import (setup_page, sidebar_date, run_all, render_pick_card,
                            TIER_ICON, market_label, market_label_options,
                            DEFAULT_MARKET_LABELS)

setup_page("Football Model — Today's Picks")
sel_date, model_only = sidebar_date()

st.title("🎯 Today's Picks")
st.caption(
    "Model probability vs de-vigged market probability for every fixture on the "
    "selected date. 🟢/🟡 cleared the plausibility ceiling and traffic-light bands. "
    "🔵 needs manual checking — either a suspiciously large edge, or a "
    "provisionally-seeded promoted team. ⚪ means no meaningful edge."
)

if model_only:
    st.info(
        "**Model-only mode.** Every league and market is forecast from the "
        "Dixon-Coles model with no bookmaker comparison — so there's no edge, "
        "no traffic light, and no check on whether these probabilities are "
        "actually well-calibrated. Backtesting put the model only marginally "
        "ahead of guessing, so treat high percentages as the model's opinion, "
        "not a settled fact."
    )

with st.spinner("Fitting models and fetching fixtures..."):
    rows, notes = run_all(sel_date, model_only)

for n in notes:
    st.warning(n)

if rows.empty:
    st.info(
        f"No fixtures found for {sel_date:%a %d %b}. Different leagues start at "
        "different times of year — try a date when games are actually scheduled."
    )
    st.stop()

rows["market_label"] = [market_label(m, sel) for m, sel
                        in zip(rows["market"], rows["selection"])]

leagues = sorted(rows["league"].unique())
label_options = market_label_options(rows)
default_labels = [m for m in DEFAULT_MARKET_LABELS if m in label_options] or label_options

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    league_filter = st.multiselect("League", leagues, default=leagues)
with c2:
    market_filter = st.multiselect(
        "Market", label_options, default=default_labels,
        help="Defaults to Match Winner. Add or remove markets here rather "
             "than loading everything at once.",
    )
with c3:
    tier_filter = st.multiselect(
        "Show", ["green", "amber", "verify", "red", "forecast"],
        default=["green", "amber", "verify", "forecast"],
        help="'forecast' = no market price available (League One/Two, BTTS, "
             "or anything in model-only mode)",
    )

view = rows[
    rows["league"].isin(league_filter)
    & rows["market_label"].isin(market_filter)
    & rows["tier"].isin(tier_filter)
]

if view.empty:
    st.info("Nothing matches those filters.")
    st.stop()

counts = view["tier"].value_counts()
st.markdown(
    " · ".join(f"{TIER_ICON.get(t, '')} {c} {t}" for t, c in counts.items())
)

view = view.sort_values("model_prob", ascending=False)

for league in league_filter:
    sub = view[view["league"] == league]
    if sub.empty:
        continue
    st.subheader(league)
    for _, r in sub.iterrows():
        metrics = [("Model %", f"{r['model_prob']*100:.1f}%")]
        if r["market_prob"] is not None and pd.notna(r["market_prob"]):
            metrics.append(("Market %", f"{r['market_prob']*100:.1f}%"))
            metrics.append(("Edge", f"{r['edge']*100:+.1f} pts"))
        if r["odds"] and pd.notna(r["odds"]):
            metrics.append(("Odds", f"{r['odds']:.2f}"))
        render_pick_card(
            TIER_ICON.get(r["tier"], ""),
            r["market_label"],
            r["fixture"],
            metrics,
            reason=r["reason"] or None,
            kickoff=r.get("kickoff"),
            started=bool(r.get("started")),
        )
