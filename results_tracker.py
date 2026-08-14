"""
Results tracking — closes the loop.

Logs every pick made, then later reconciles against the real outcome
and (crucially) the closing odds, to compute both raw P/L and CLV —
the metric that actually tells you if the model beats the market,
not just whether picks happened to land.

Storage: a simple CSV (results_log.csv), created on first use.
Fine for personal-scale use; swap for a real DB later if this grows.
"""

import os
from datetime import datetime
import pandas as pd

LOG_PATH = "results_log.csv"
COLUMNS = [
    "date_logged", "league", "fixture", "market", "selection",
    "model_prob", "market_prob_at_pick", "odds_at_pick", "tier",
    "actual_outcome", "closing_odds", "stake", "pnl", "beat_clv",
]


def _load() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=COLUMNS)


def log_pick(league, fixture, market, selection, model_prob,
             market_prob_at_pick, odds_at_pick, tier, stake=1.0):
    df = _load()
    new_row = {
        "date_logged": datetime.now().isoformat(timespec="seconds"),
        "league": league, "fixture": fixture, "market": market,
        "selection": selection, "model_prob": model_prob,
        "market_prob_at_pick": market_prob_at_pick,
        "odds_at_pick": odds_at_pick, "tier": tier, "stake": stake,
        "actual_outcome": None, "closing_odds": None, "pnl": None, "beat_clv": None,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


def settle_pick(fixture, market, selection, won: bool, closing_odds: float):
    """Call once the match has been played and the closing line is known."""
    df = _load()
    mask = ((df["fixture"] == fixture) & (df["market"] == market)
             & (df["selection"] == selection) & (df["actual_outcome"].isna()))
    if not mask.any():
        print(f"No open pick found for {fixture} / {market} / {selection}")
        return

    idx = df[mask].index[0]
    odds_at_pick = df.loc[idx, "odds_at_pick"]
    stake = df.loc[idx, "stake"]

    df.loc[idx, "actual_outcome"] = "won" if won else "lost"
    df.loc[idx, "closing_odds"] = closing_odds
    df.loc[idx, "pnl"] = (stake * (odds_at_pick - 1)) if won else -stake
    # CLV: did we get a better price than the closing line? (lower closing
    # odds than our odds = market shortened = we beat the closing line)
    df.loc[idx, "beat_clv"] = closing_odds < odds_at_pick

    df.to_csv(LOG_PATH, index=False)


def summary():
    df = _load()
    settled = df[df["actual_outcome"].notna()]
    if settled.empty:
        print("No settled picks yet.")
        return

    total_pnl = settled["pnl"].sum()
    total_staked = settled["stake"].sum()
    roi = total_pnl / total_staked if total_staked else 0
    clv_rate = settled["beat_clv"].mean()

    print(f"\n{'='*50}\nRESULTS SUMMARY\n{'='*50}")
    print(f"Settled picks: {len(settled)}")
    print(f"Total P/L: {total_pnl:+.2f} units")
    print(f"ROI: {roi:+.1%}")
    print(f"Beat closing line: {clv_rate:.1%} of picks "
          f"(this is the real long-run test — a model beating the market "
          f"should beat closing line noticeably more than 50% of the time)")

    print("\nBy market:")
    for market, group in settled.groupby("market"):
        print(f"  {market:15s} {len(group):3d} picks | "
              f"P/L {group['pnl'].sum():+.2f} | "
              f"CLV {group['beat_clv'].mean():.1%}")
