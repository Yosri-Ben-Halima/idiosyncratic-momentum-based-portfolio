import pandas as pd

def select_assets(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list[str],
    rebal_date: pd.Timestamp,
    top_k: int,
    min_history: int,
    zscore_threshold: float = 0.0,
) -> list[str]:
    """
    For a single rebalance date, return the list of selected tickers.
      1. Filter by minimum price history
      2. Rank by z-score → take top-K candidates
      3. Drop candidates below zscore_threshold
    Returns an empty list if no stock clears the threshold (→ cash).
    """
    prior_signal = signal.loc[signal.index < rebal_date]
    if prior_signal.empty:
        return []

    last_signal = prior_signal.iloc[-1]

    # Eligibility: enough history
    stock_history = prices[tickers].loc[prices.index < rebal_date].count()
    eligible = stock_history[stock_history >= min_history].index.tolist()
    last_signal = last_signal[eligible].dropna()

    if last_signal.empty:
        return []

    # Top-K candidates by z-score rank
    top_k_candidates = last_signal.nlargest(min(top_k, len(last_signal)))

    # Filter by z-score threshold
    return top_k_candidates[top_k_candidates > zscore_threshold].index.tolist()





