import pandas as pd
import yfinance as yf

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers + benchmark."""
    all_tickers = tickers + ["^GSPC"]
    print(f"Downloading price data for {len(all_tickers)} tickers …")
    raw: pd.DataFrame = yf.download(
        all_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )["Close"]
    # yfinance may return a MultiIndex or flat columns depending on version
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    raw.sort_index(inplace=True)
    return raw