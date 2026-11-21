from pathlib import Path

from halal_screener import halal_tickers
from halal_screener.api import DATA_DIR


def filter_halal(tickers: list[str], standard: str, data_dir: Path = DATA_DIR) -> list[str]:
    """Keep only tickers that pass the given Shariah standard ("aaoifi" or "wahed").

    Uses the latest snapshot in data/ (refresh monthly: python -m halal_screener refresh).
    shortcut: applies today's compliance to the whole backtest (look-ahead, like the
    current-constituents universe); point-in-time compliance would need dated snapshots.
    """
    allowed = set(halal_tickers(data_dir=data_dir, standard=standard))
    kept = [t for t in tickers if t in allowed]
    print(f"✓ Halal filter ({standard.upper()}): {len(kept)}/{len(tickers)} tickers kept")
    return kept
