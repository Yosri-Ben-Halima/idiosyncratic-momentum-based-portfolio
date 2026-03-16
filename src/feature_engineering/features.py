import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_rolling_capm_residuals_vectorized(
    stock_excess: pd.Series,
    mkt_excess: pd.Series,
    window: int,
) -> pd.Series:
    """
    Fully vectorized rolling OLS via closed-form rolling moments.

    For each t, fits OLS on [t-window : t-1] (excludes t), then applies
    estimated alpha/beta to day t to get the true OOS residual.

    No Python loop — uses pandas rolling on shifted series.
    """
    min_periods = int(window * 0.7)

    # Align and extract
    x = mkt_excess.copy().astype(float)
    y = stock_excess.copy().astype(float)

    # NaN-safe products — NaN in either series should blank out the product
    xy = x * y
    xx = x * x

    # ── Rolling moments over [t-window : t-1] ──────────────────────────────
    # shift(1) moves the window back by 1 so day t is excluded from the fit
    def roll(s):
        return s.shift(1).rolling(window=window, min_periods=min_periods)

    roll_x = roll(x)
    roll_y = roll(y)
    roll_xy = roll(xy)
    roll_xx = roll(xx)
    roll_n = roll(x.notna().astype(float))  # effective sample size (non-NaN)

    mean_x = roll_x.mean()
    mean_y = roll_y.mean()
    sum_xy = roll_xy.sum()
    sum_xx = roll_xx.sum()
    n_eff = roll_n.sum()

    # ── Closed-form OLS coefficients ────────────────────────────────────────
    # beta  = (Σxy - n*x̄*ȳ) / (Σx² - n*x̄²)
    # alpha = ȳ - beta * x̄
    denom = sum_xx - n_eff * mean_x**2  # variance of x (scaled)
    beta = (sum_xy - n_eff * mean_x * mean_y) / denom
    alpha = mean_y - beta * mean_x

    # ── OOS residual at t using coefficients from [t-window : t-1] ─────────
    residuals = y - (alpha + beta * x)

    # ── Null out where we had insufficient data ─────────────────────────────
    insufficient = (n_eff < min_periods) | denom.abs().lt(1e-12)
    residuals[insufficient] = np.nan

    return residuals


def build_idiosyncratic_momentum(
    prices: pd.DataFrame,
    tickers: list[str],
    window: int,
    rf_daily: float,
    skip: int,
) -> pd.DataFrame:
    """
    Idiosyncratic momentum at time t = sum of OOS CAPM residuals over
    [t - window - skip : t - skip].

    Vectorized: no per-day OLS loop. Stocks still processed sequentially
    but each stock is now ~500x faster than the loop version.
    """
    returns = prices.pct_change()
    mkt_ret = returns["^GSPC"]
    mkt_excess = mkt_ret - rf_daily

    # Pre-allocate output as float64 numpy array for speed, convert at end
    idx = returns.index
    n_dates = len(idx)
    ticker_map = {t: i for i, t in enumerate(tickers) if t in returns.columns}
    out = np.full((n_dates, len(tickers)), np.nan)

    for tkr, col_i in tqdm(
        ticker_map.items(), desc="Rolling CAPM residuals", unit="stock"
    ):
        stk_excess = returns[tkr] - rf_daily
        resids = compute_rolling_capm_residuals_vectorized(
            stk_excess, mkt_excess, window
        )

        # Momentum = sum of residuals over [t-window-skip : t-skip]
        mom = (
            resids.shift(skip)
            .rolling(window=window, min_periods=int(window * 0.7))
            .sum()
        )
        out[:, col_i] = mom.values

    return pd.DataFrame(out, index=idx, columns=tickers)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SIGNAL CONSTRUCTION  (cross-sectional z-score)
# ─────────────────────────────────────────────────────────────────────────────


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score at each date (row)."""
    return df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
