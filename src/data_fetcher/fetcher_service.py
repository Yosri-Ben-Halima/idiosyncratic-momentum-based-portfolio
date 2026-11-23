import pandas as pd
import yfinance as yf

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers + benchmark."""
    all_tickers = tickers + ["^GSPC"]
    print(f"Downloading price data for {len(all_tickers)} tickers …")
    # Batches keep only Close as they go: one call for ~500 tickers held ~330 MB
    # of OHLCV data, over the 512 MB a small hosting instance allows.
    size = 50
    batches = [all_tickers[i:i+size] for i in range(0, len(all_tickers), size)]
    raw = pd.concat([yf.download(batch, start=start, end=end, auto_adjust=True,
                                 progress=False, threads=8)["Close"] for batch in batches], axis=1)
    raw = raw[sorted(raw.columns)]  # same column order as a single download
    # yfinance may return a MultiIndex or flat columns depending on version
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    raw.sort_index(inplace=True)
    return raw