"""
Results tracking — closes the loop.

Logs every pick made, then reconciles against the real outcome and the
closing odds, to compute both raw P/L and CLV — the metric that actually
tells you if the model beats the market, not just whether picks landed.

Storage: a simple CSV (results_log.csv), created on first use. Fine for
personal-scale use.

NOTE ON STREAMLIT CLOUD: the filesystem is ephemeral, so this file is
wiped on every redeploy. Local runs keep their own history. A persistent
store (Google Sheets, a hosted DB) is the fix when that starts to matter.
"""

import os
from datetime import datetime, date
import pandas as pd

LOG_PATH = "results_log.csv"
COLUMNS = [
    "date_logged", "fixture_date", "league", "fixture", "market", "selection",
    "model_prob", "market_prob_at_pick", "odds_at_pick", "tier",
    "actual_outcome", "closing_odds", "stake", "pnl", "beat_clv",
]


def _load() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        # Older logs predate fixture_date — backfill so filtering still works.
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        if df["fixture_date"].isna().any():
            df["fixture_date"] = df["fixture_date"].fillna(
                pd.to_datetime(df["date_logged"], errors="coerce").dt.date.astype(str))
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame):
    df.to_csv(LOG_PATH, index=False)


def log_pick(league, fixture, market, selection, model_prob,
             market_prob_at_pick, odds_at_pick, tier, fixture_date=None,
             stake=1.0):
    df = _load()
    df = pd.concat([df, pd.DataFrame([{
        "date_logged": datetime.now().isoformat(timespec="seconds"),
        "fixture_date": str(fixture_date or date.today()),
        "league": league, "fixture": fixture, "market": market,
        "selection": selection, "model_prob": model_prob,
        "market_prob_at_pick": market_prob_at_pick,
        "odds_at_pick": odds_at_pick, "tier": tier, "stake": stake,
        "actual_outcome": None, "closing_odds": None,
        "pnl": None, "beat_clv": None,
    }])], ignore_index=True)
    _save(df)


def log_picks_bulk(rows: pd.DataFrame, fixture_date, stake=1.0) -> int:
    """Logs a batch of picks, skipping any already logged for the same
    fixture/market/selection on that date so re-running a day doesn't
    create duplicates. Returns how many were newly added."""
    df = _load()
    existing = set(zip(df["fixture_date"].astype(str), df["fixture"],
                        df["market"], df["selection"]))

    new_rows = []
    for _, r in rows.iterrows():
        key = (str(fixture_date), r["fixture"], r["market"], r["selection"])
        if key in existing:
            continue
        new_rows.append({
            "date_logged": datetime.now().isoformat(timespec="seconds"),
            "fixture_date": str(fixture_date),
            "league": r.get("league"), "fixture": r["fixture"],
            "market": r["market"], "selection": r["selection"],
            "model_prob": r.get("model_prob"),
            "market_prob_at_pick": r.get("market_prob"),
            "odds_at_pick": r.get("odds"), "tier": r.get("tier"),
            "stake": stake, "actual_outcome": None, "closing_odds": None,
            "pnl": None, "beat_clv": None,
        })
        existing.add(key)

    if new_rows:
        _save(pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True))
    return len(new_rows)


def settle_by_index(idx: int, won: bool, closing_odds=None):
    """Settle one logged pick by its row index. Used by the Results page."""
    df = _load()
    if idx not in df.index:
        return False

    odds = df.loc[idx, "odds_at_pick"]
    stake = df.loc[idx, "stake"] or 1.0

    df.loc[idx, "actual_outcome"] = "won" if won else "lost"
    if pd.notna(odds):
        df.loc[idx, "pnl"] = (stake * (float(odds) - 1)) if won else -stake
    else:
        # Model-only picks have no price, so P/L is undefined — the pick is
        # still worth scoring for hit-rate and calibration purposes.
        df.loc[idx, "pnl"] = None

    if closing_odds and pd.notna(odds):
        df.loc[idx, "closing_odds"] = closing_odds
        # Beating the closing line = we took a bigger price than the market
        # settled at.
        df.loc[idx, "beat_clv"] = float(closing_odds) < float(odds)

    _save(df)
    return True


def delete_by_index(idx: int):
    df = _load()
    if idx in df.index:
        _save(df.drop(index=idx).reset_index(drop=True))
        return True
    return False
