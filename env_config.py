"""
Loads the Odds API key from whichever source is available:

  1. A local .env file (git-ignored) — used when running on your Mac.
  2. Streamlit Cloud's Secrets manager — used when deployed, since
     .env never gets pushed to GitHub.

Local setup (one-time):
    1. pip3 install python-dotenv
    2. Copy .env.example to a file named exactly .env
    3. Put your real key in .env

Streamlit Cloud setup (one-time):
    App -> Settings -> Secrets, then paste:
        ODDS_API_KEY = "your_key_here"
"""

import os
from dotenv import load_dotenv

load_dotenv()

_key = os.getenv("ODDS_API_KEY")

# Fall back to Streamlit secrets when running deployed. Wrapped in
# try/except because streamlit isn't importable in plain-Terminal runs,
# and st.secrets raises if no secrets file exists locally.
if not _key:
    try:
        import streamlit as st
        _key = st.secrets.get("ODDS_API_KEY")
    except Exception:
        _key = None

ODDS_API_KEY = _key


def require_api_key():
    if not ODDS_API_KEY:
        raise SystemExit(
            "No ODDS_API_KEY found. Locally: copy .env.example to .env and "
            "add your key. On Streamlit Cloud: add it under Settings -> Secrets."
        )
    return ODDS_API_KEY
