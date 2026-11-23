import numpy as np
import pandas as pd


def compute_rolling_capm_residuals_vectorized(
    stock_excess: pd.Series | pd.DataFrame,
    mkt_excess: pd.Series,
    window: int,
) -> pd.Series | pd.DataFrame:
    """
    Fully vectorized rolling OLS via closed-form rolling moments.

    For each t, fits OLS on [t-window : t-1] (excludes t), then applies
    estimated alpha/beta to day t to get the true OOS residual.

    stock_excess may be one stock (Series) or many (DataFrame, one column per
    stock); every stock is regressed on the same market series.
    """
    min_periods = int(window * 0.7)

    x = mkt_excess.copy().astype(float)
    y = stock_excess.copy().astype(float)

    # shift(1) moves the window back by 1 so day t is excluded from the fit
    def roll(s):
        return s.shift(1).rolling(window=window, min_periods=min_periods)

    # Market-only moments are shared by every stock
    mean_x = roll(x).mean()
    sum_xx = roll(x * x).sum()
    n_eff = roll(x.notna().astype(float)).sum()  # effective sample size (non-NaN)
    mean_y = roll(y).mean()
    sum_xy = roll(y.mul(x, axis=0)).sum()  # NaN in either series blanks the product

    # beta  = (Σxy - n*x̄*ȳ) / (Σx² - n*x̄²);  alpha = ȳ - beta * x̄
    denom = sum_xx - n_eff * mean_x**2
    beta = (sum_xy - mean_y.mul(n_eff * mean_x, axis=0)).div(denom, axis=0)
    alpha = mean_y - beta.mul(mean_x, axis=0)

    # OOS residual at t using coefficients from [t-window : t-1]
    residuals = y - (alpha + beta.mul(x, axis=0))
    residuals.loc[(n_eff < min_periods) | denom.abs().lt(1e-12)] = np.nan
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
    [t - window - skip : t - skip], computed for all stocks at once.
    """
    returns = prices.pct_change()
    present = [t for t in tickers if t in returns.columns]
    residuals = compute_rolling_capm_residuals_vectorized(
        returns[present] - rf_daily, returns["^GSPC"] - rf_daily, window
    )
    momentum = residuals.shift(skip).rolling(window=window, min_periods=int(window * 0.7)).sum()
    return momentum.reindex(columns=tickers)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SIGNAL CONSTRUCTION  (cross-sectional z-score)
# ─────────────────────────────────────────────────────────────────────────────


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score at each date (row)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
