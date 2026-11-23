"""Fast NumPy / whole-frame code must match the original per-stock and per-date loops exactly."""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (build_idiosyncratic_momentum,
                                     compute_rolling_capm_residuals_vectorized, zscore_normalize)
from src.portfolio_construction import build_portfolio
from src.sensitivity.risk import RISK_FIELDS, RiskDiagnostics, risk_metrics


def reference_portfolio(signal, prices, tickers, top_k, rebal_freq, min_history, thr):
    """The original select_assets / compute_weights loop, kept as the parity oracle."""
    returns = prices[tickers].pct_change(fill_method=None)
    rebal_dates = returns.resample(rebal_freq).last().index
    weights = pd.DataFrame(0.0, index=returns.index, columns=tickers)
    for i, rd in enumerate(rebal_dates):
        selected = []
        prior = signal.loc[signal.index < rd]
        if not prior.empty:
            history = prices[tickers].loc[prices.index < rd].count()
            last = prior.iloc[-1][history[history >= min_history].index.tolist()].dropna()
            if not last.empty:
                top = last.nlargest(min(top_k, len(last)))
                selected = top[top > thr].index.tolist()
        end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
        mask = (returns.index > rd) & (returns.index <= end)
        slice_ = pd.DataFrame(0.0, index=returns.index[mask], columns=tickers)
        if selected:
            slice_[selected] = 1.0 / len(selected)
        weights.loc[mask] = slice_.values
    return weights


def market(seed=0, n_dates=400, n_stocks=25):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    prices = pd.DataFrame(100 * np.exp(rng.normal(0, .01, (n_dates, n_stocks)).cumsum(0)), idx, tickers)
    for j, start in enumerate(rng.integers(0, n_dates, n_stocks // 3)):
        prices.iloc[:start, j] = np.nan          # late listings -> short histories
    prices.iloc[rng.integers(0, n_dates, 30), rng.integers(0, n_stocks, 30)] = np.nan  # gaps
    signal = pd.DataFrame(rng.normal(size=(n_dates, n_stocks)).round(1), idx, tickers)  # rounding -> ties
    signal[signal.abs() > 2.4] = np.nan
    return prices, tickers, signal


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("top_k,min_history,thr,freq", [
    (5, 60, 0.0, "ME"), (5, 60, 1.5, "ME"), (30, 20, -3.0, "ME"),
    (3, 500, 0.0, "ME"), (4, 30, 0.5, "W-FRI"), (1, 1, 2.0, "QE"),
])
def test_build_portfolio_matches_original_loop(seed, top_k, min_history, thr, freq):
    prices, tickers, signal = market(seed)
    expected = reference_portfolio(signal, prices, tickers, top_k, freq, min_history, thr)
    actual = build_portfolio(signal, prices, tickers, top_k, freq, min_history, thr)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def reference_capm_residuals(y, x, window):
    """The original one-stock rolling CAPM, kept as the parity oracle."""
    min_periods = int(window * 0.7)
    roll = lambda s: s.shift(1).rolling(window=window, min_periods=min_periods)
    mean_x, mean_y = roll(x).mean(), roll(y).mean()
    sum_xy, sum_xx, n_eff = roll(x * y).sum(), roll(x * x).sum(), roll(x.notna().astype(float)).sum()
    denom = sum_xx - n_eff * mean_x**2
    beta = (sum_xy - n_eff * mean_x * mean_y) / denom
    residuals = y - ((mean_y - beta * mean_x) + beta * x)
    residuals[(n_eff < min_periods) | denom.abs().lt(1e-12)] = np.nan
    return residuals


def reference_momentum(prices, tickers, window, rf, skip):
    returns = prices.pct_change()
    mkt = returns["^GSPC"] - rf
    out = pd.DataFrame(np.nan, index=returns.index, columns=tickers)
    for t in [t for t in tickers if t in returns.columns]:
        resid = reference_capm_residuals(returns[t] - rf, mkt, window)
        out[t] = resid.shift(skip).rolling(window, min_periods=int(window * 0.7)).sum().values
    return out


@pytest.mark.parametrize("window,skip", [(20, 0), (60, 21), (126, 5)])
def test_momentum_matches_per_stock_loop(window, skip):
    prices, tickers, _ = market(4)
    rng = np.random.default_rng(4)
    prices["^GSPC"] = 100 * np.exp(rng.normal(0, .01, len(prices)).cumsum())
    prices.iloc[50:55, -1] = np.nan                    # benchmark gap
    prices["FLAT"] = 50.0                              # zero-variance stock
    tickers = tickers + ["FLAT", "MISSING"]            # MISSING has no prices column
    expected = reference_momentum(prices, tickers, window, 1e-4, skip)
    pd.testing.assert_frame_equal(build_idiosyncratic_momentum(prices, tickers, window, 1e-4, skip), expected, check_exact=True)


def test_single_stock_residuals_unchanged():
    prices, tickers, _ = market(5)
    y, x = prices[tickers[0]].pct_change(), prices[tickers[1]].pct_change()
    pd.testing.assert_series_equal(compute_rolling_capm_residuals_vectorized(y, x, 30),
                                   reference_capm_residuals(y, x, 30), check_exact=True)


def reference_risk_evaluate(model, weights):
    """The original RiskDiagnostics.evaluate, kept as the parity oracle."""
    months = weights.index.to_period("M")
    monthly = np.r_[True, months[1:] != months[:-1]]
    changes = weights.diff().abs().sum(axis=1).gt(1e-12).to_numpy()
    snapshots = {}
    for date in weights.index[monthly | changes]:
        selected = weights.loc[date][weights.loc[date] > 0]
        metrics = dict.fromkeys(RISK_FIELDS, np.nan)
        if not selected.empty:
            cov = model.covariance(date)
            if selected.index.isin(cov.columns).all():
                metrics = risk_metrics(selected.to_numpy(), cov.loc[selected.index, selected.index].to_numpy())
        snapshots[date] = metrics
    return pd.DataFrame.from_dict(snapshots, orient="index").reindex(weights.index, method="ffill")


@pytest.mark.parametrize("thr", [-3.0, 0.5, 2.3])
def test_risk_evaluate_matches_original_loop(thr):
    prices, tickers, signal = market(6)
    weights = build_portfolio(signal, prices, tickers, 6, "ME", 40, thr).iloc[150:]
    weights.iloc[40:45] = weights.iloc[40:45] * 0                 # cash days mid-month
    model = RiskDiagnostics(prices, window=60, min_observations=30)
    expected = reference_risk_evaluate(RiskDiagnostics(prices, window=60, min_observations=30), weights)
    pd.testing.assert_frame_equal(model.evaluate(weights), expected, check_exact=True)
    pd.testing.assert_frame_equal(model.evaluate(weights.iloc[:0]),
                                  pd.DataFrame(index=weights.index[:0], columns=RISK_FIELDS, dtype=float))


def test_zscore_matches_row_apply():
    _, _, signal = market(3)
    signal.iloc[5] = np.nan       # all-missing row
    signal.iloc[6] = 1.0          # constant row -> std 0
    expected = signal.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
    pd.testing.assert_frame_equal(zscore_normalize(signal), expected, check_exact=True)
