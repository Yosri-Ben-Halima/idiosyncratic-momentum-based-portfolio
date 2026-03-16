import pandas as pd
from .asset_selection import select_assets
from .weights_calculation import compute_weights

def build_portfolio(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list[str],
    top_k: int,
    rebal_freq: str,
    min_history: int,
    zscore_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    At each rebalance date, delegate to:
      - select_assets()   → which tickers to hold
      - compute_weights() → how much of each to hold
    """
    returns = prices[tickers].pct_change()
    rebal_dates = returns.resample(rebal_freq).last().index
    weights = pd.DataFrame(0.0, index=returns.index, columns=tickers)

    print("Building portfolio weights …")
    for i, rd in enumerate(rebal_dates):
        selected = select_assets(
            signal, prices, tickers, rd, top_k, min_history, zscore_threshold
        )

        next_rd = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
        mask = (returns.index > rd) & (returns.index <= next_rd)

        weights.loc[mask] = compute_weights(selected, tickers, returns.index, mask).values

    return weights