import yfinance as yf

def _fetch_risk_free_rate() -> float:
    """
    Fetch the current annualised risk-free rate from Yahoo Finance.
    Uses 13-week T-bill yield (^IRX) as the standard short-rate proxy,
    falling back in order to: 5Y yield, 10Y yield, then hardcoded 4.25%.

    Note: yf.download returns a MultiIndex DataFrame in yfinance >= 0.2.x
    so we use yf.Ticker().history() which returns a clean single-level DataFrame.
    """

    def _extract_last_close(ticker: str) -> float:
        """Use Ticker.history() to avoid MultiIndex issues from yf.download."""
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if hist.empty:
            raise ValueError(f"No history returned for {ticker}")
        # history() always returns single-level columns: Open/High/Low/Close/Volume
        val = hist["Close"].dropna().iloc[-1]
        # Ensure scalar — occasionally returns a 1-element Series
        if hasattr(val, "item"):
            val = val.item()
        return float(val)

    sources = [
        ("^IRX", "13-Week T-Bill", 1 / 100),  # quotes in annualised percent
        ("^FVX", "5-Year T-Note", 1 / 100),
        ("^TNX", "10-Year T-Note", 1 / 100),
    ]

    for ticker, label, scale in sources:
        try:
            raw = _extract_last_close(ticker)
            rate = raw * scale
            if 0.0 < rate < 0.25:  # sanity: 0%–25%
                print(
                    f"  Risk-free rate: {rate * 100:.3f}%/yr  "
                    f"(source: {label}  {ticker}  raw={raw:.4f})"
                )
                return rate
            else:
                print(
                    f"  {ticker} returned implausible value {raw:.4f} "
                    f"(scaled={rate * 100:.3f}%) — skipping"
                )
        except Exception as e:
            print(f"  RF rate fetch failed for {ticker}: {e}")

    # Final hardcoded fallback — update this manually if needed
    fallback = 0.0425  # 4.25% as of early 2026
    print(
        f"  All RF rate sources failed — "
        f"using hardcoded fallback {fallback * 100:.2f}%/yr"
    )
    return fallback