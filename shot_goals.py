"""
Shot-implied goals ("our own xG").

True xG needs shot-by-shot data (location, type) that no free results file
carries. football-data.co.uk does give shots and shots on target per team
per match, so we fit a simple conversion from those to expected goals:

    expected goals = a * shots_on_target + b * (shots - shots_on_target)

(a, b >= 0, fitted by least squares on the same results the model trains
on — never on games after the date being predicted.) In practice b comes
out at zero: an off-target shot is worth nothing, and a shot on target is
worth about 0.32 goals.

The Dixon-Coles fit then trains on a blend of real goals and shot-implied
goals. Goals are noisy (a dominant team can lose 0-1); shots are a steadier
measure of how well a team really played. Backtest on 408 games from
12 Sep - 24 Oct 2026, 1X2 log loss (lower is better):

    goals only (old model)     1.0723
    50% goals / 50% shots      1.0410   <- used by default
    shots only                 1.0377
    bookmakers                 1.0057

Better in 7 of 8 leagues; paired bootstrap 95% CI for the gain
[-0.057, -0.011]. Still behind the market.

Rows without shot data (e.g. 2025/26 National League) fall back to real
goals, so those leagues behave exactly as before.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

SHOT_COLS = ["HS", "AS", "HST", "AST"]
MIN_ROWS_FOR_FIT = 150


def _conversion(df):
    """Non-negative (on-target, off-target) goal values, or None."""
    d = df.dropna(subset=SHOT_COLS + ["FTHG", "FTAG"])
    if len(d) < MIN_ROWS_FOR_FIT:
        return None
    X = np.vstack([np.c_[d.HST, d.HS - d.HST], np.c_[d.AST, d.AS - d.AST]])
    y = np.r_[d.FTHG, d.FTAG]
    coef, _ = nnls(X.astype(float), y.astype(float))
    return coef if coef.sum() > 0 else None


def build_fit_rows(frames, reference_date):
    """frames: results DataFrames (e.g. last season + this season's games
    before reference_date). Returns 7-tuples for fit_league:
    (home, away, home_goals, away_goals, days_ago, xg_home, xg_away)."""
    parts = []
    for f in frames:
        if f is None or len(f) == 0:
            continue
        f = f.copy()
        for c in SHOT_COLS:
            if c not in f.columns:
                f[c] = np.nan
        if "_date" not in f.columns:
            f["_date"] = (pd.to_datetime(f["Date"], dayfirst=True, errors="coerce").dt.date
                          if "Date" in f.columns else pd.NaT)
        parts.append(f[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "_date"] + SHOT_COLS])
    if not parts:
        return []
    df = pd.concat(parts, ignore_index=True).dropna(
        subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    coef = _conversion(df)
    has_shots = df[SHOT_COLS].notna().all(axis=1).values
    xgh = df.FTHG.astype(float).values.copy()
    xga = df.FTAG.astype(float).values.copy()
    if coef is not None and has_shots.any():
        m = has_shots
        xgh[m] = np.c_[df.HST[m], df.HS[m] - df.HST[m]].astype(float) @ coef
        xga[m] = np.c_[df.AST[m], df.AS[m] - df.AST[m]].astype(float) @ coef

    days = [max((reference_date - d).days, 0) if d is not None and not pd.isna(d) else 365
            for d in df["_date"]]
    return [(h, a, int(g1), int(g2), dd, float(x1), float(x2))
            for h, a, g1, g2, dd, x1, x2 in
            zip(df.HomeTeam, df.AwayTeam, df.FTHG, df.FTAG, days, xgh, xga)]
