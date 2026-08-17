"""
Football Betting Model — landing page.

Multi-page app: see the pages/ folder for Today's Picks, Most Likely,
Suggested Bets, and Results.

Run with:  streamlit run app.py
"""

import streamlit as st
from football_core import setup_page, sidebar_date

setup_page("Football Model — Home")
sel_date, model_only = sidebar_date()

st.title("⚽ Football Betting Model")
st.caption(
    "Dixon-Coles goal model across six European leagues, compared against "
    "de-vigged bookmaker prices to surface genuine edges — not tips."
)

st.markdown("""
### Pages

- **Today's Picks** — every fixture and market for the selected date, with model
  probability vs market probability and a traffic-light rating.
- **Most Likely** — ranked purely by model confidence, filterable by league and
  market. This answers *"what does the model expect"*, not *"where's the value"*.
- **Suggested Bets** — 5–6 fold accumulators, split into English (excluding EPL)
  and Rest of Europe, one per market.
- **Results** — logged picks, P/L, and closing-line value once settled.

### Read this before staking anything

Nothing here has been validated against live results yet. The backtest showed
log loss around 1.06 against a 1.099 random-guess baseline — better than
guessing, but only slightly, and the calibration was noisy. Traffic-light
thresholds are reasoned starting points, not numbers earned from real outcomes.

Known issues being tracked:
- The model shows unusually large edges on elite home sides (Barcelona, Bayern).
  Regularisation didn't fix this; the plausibility ceiling contains it, but it's
  not properly calibrated.
- Promoted and relegated teams run on provisional ratings and are always capped
  at 🔵 verify.
- League One and Two have no odds coverage, so they're forecast-only.
""")

st.info("Pick a date in the sidebar, then head to **Today's Picks**.")
