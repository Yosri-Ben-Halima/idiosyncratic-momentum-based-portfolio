import numpy as np
import pandas as pd


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
    At each rebalance date, using only data strictly before it:
      1. keep stocks with at least min_history prices
      2. take the top-K by last z-score (ties: first ticker wins)
      3. drop candidates at or below zscore_threshold
    and hold them equal-weight until the next rebalance date (cash if none).
    """
    index = prices.index
    rebal_dates = prices[tickers].resample(rebal_freq).last().index
    sig = signal[tickers].to_numpy(dtype=float)
    history = prices[tickers].notna().to_numpy().cumsum(axis=0)
    weights = np.zeros((len(index), len(tickers)))
    for i, rd in enumerate(rebal_dates):
        row = index.searchsorted(rd, "left") - 1
        sig_row = signal.index.searchsorted(rd, "left") - 1
        if row < 0 or sig_row < 0:
            continue
        last = np.where(history[row] >= min_history, sig[sig_row], np.nan)
        valid = np.flatnonzero(~np.isnan(last))
        top = valid[np.argsort(-last[valid], kind="stable")[:top_k]]
        top = top[last[top] > zscore_threshold]
        end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else index[-1]
        if len(top):
            weights[np.ix_((index > rd) & (index <= end), top)] = 1.0 / len(top)
    return pd.DataFrame(weights, index=index, columns=tickers)
