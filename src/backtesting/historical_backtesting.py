import pandas as pd
import numpy as np


def compute_portfolio_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    transaction_cost: float,
) -> pd.Series:
    returns = prices[weights.columns].pct_change()
    port_ret = (weights.shift(1) * returns).sum(axis=1)

    if transaction_cost > 0.0:
        # One-way turnover on each day: sum of absolute weight changes / 2
        # weights.shift(1) is yesterday's target; weights is today's target.
        # Cost hits on the day the new weights first apply (rebalance day).
        daily_turnover = weights.diff().abs().sum(axis=1) / 2
        port_ret -= daily_turnover * transaction_cost

    return port_ret


def performance_metrics(
    returns: pd.Series, benchmark: pd.Series, rf_daily: float, label: str = "Strategy"
) -> dict:
    aligned = returns.align(benchmark, join="inner")
    r, b = aligned[0].dropna(), aligned[1].dropna()

    ann_factor = 252
    excess = r - rf_daily

    cagr = (1 + r).prod() ** (ann_factor / len(r)) - 1
    vol = r.std() * np.sqrt(ann_factor)
    sharpe = (
        excess.mean() / excess.std() * np.sqrt(ann_factor)
        if excess.std() > 0
        else np.nan
    )

    cum = (1 + r).cumprod()
    total_return = cum.iloc[-1] - 1
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    # Beta / Alpha vs benchmark
    cov_matrix = np.cov(r.values, b.values)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha_daily = r.mean() - (rf_daily + beta * (b.mean() - rf_daily))
    alpha_ann = (1 + alpha_daily) ** ann_factor - 1

    # Information Ratio
    active = r - b
    ir = (
        active.mean() / active.std() * np.sqrt(ann_factor)
        if active.std() > 0
        else np.nan
    )

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Win Rate
    win_rate = (r > 0).mean()

    # Sortino
    downside = excess[excess < 0].std() * np.sqrt(ann_factor)
    sortino = excess.mean() * ann_factor / downside if downside > 0 else np.nan

    return {
        "label": label,
        "Total Return": total_return,
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
        "Calmar": calmar,
        "Beta": beta,
        "Alpha": alpha_ann,
        "IR": ir,
        "Win Rate": win_rate,
    }
