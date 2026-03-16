import requests
import pandas as pd
import io


def get_sp500_tickers() -> list[str]:
    """Scrape current SP500 tickers from Wikipedia."""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"✓ Fetched {len(tickers)} SP500 tickers from Wikipedia")
    return tickers
