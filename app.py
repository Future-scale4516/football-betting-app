"""
Football Betting Model — landing page.

Multi-page app: see the pages/ folder for Today's Picks (which now
includes Match Result, BTTS, Over 2.5, and Over 1.5 as tabs), Suggested
Bets, and Results.

Run with:  streamlit run app.py
"""

import streamlit as st
from football_core import setup_page, sidebar_date

setup_page("Football Model — Home")
sel_date, model_only = sidebar_date()

st.title("⚽ Football Betting Model")
st.caption(
    "Dixon-Coles goal model across eight leagues, compared against "
    "de-vigged bookmaker prices to surface genuine edges — not tips."
)

st.markdown("""
### Pages

- **Today's Picks** — Match Result, BTTS, Over 2.5, and Over 1.5 as tabs.
  Match Result compares model vs market with a traffic-light rating; the
  other three rank fixtures purely by model confidence.
- **Suggested Bets** — 5–6 fold accumulators, split into English (excluding EPL)
  and Rest of Europe, one per market.
- **Results** — loads a past date and grades what the model would have
  predicted against the real outcome.
""")

st.info("Pick a date in the sidebar, then head to **Today's Picks**.")
