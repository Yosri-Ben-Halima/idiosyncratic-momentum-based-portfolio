import warnings

import numpy as np
import pandas as pd

from backtesting import compute_portfolio_returns, performance_metrics
from data_fetcher import fetch_prices
from environment import load_config
from feature_engineering import build_idiosyncratic_momentum, zscore_normalize
from portfolio_construction import build_portfolio
from utils.tick_helper import get_sp500_tickers
from visualizations import fmt_pct, plot_tearsheet

warnings.filterwarnings("ignore")

CFG = load_config()
RF_DAILY = CFG["rf_daily"]
COST_BPS = CFG["cost_bps"]


def main():
    """Pipeline orchestrating data loading, backtesting, and visualization."""
    # 1. Fetch data
    start_dt = pd.Timestamp(CFG["start_date"])
    warmup_days = (
        CFG["momentum_window"] + CFG["momentum_skip"] + CFG["min_history"] + 30
    )
    fetch_start = (start_dt - pd.offsets.BDay(warmup_days)).strftime("%Y-%m-%d")

    print(f"Strategy window  : {CFG['start_date']} → {CFG['end_date']}")
    print(f"Fetching data from: {fetch_start} (warmup = {warmup_days} business days)")

    # ── Fetch data from fetch_start ───────────────────────────────────────────
    tickers = get_sp500_tickers()
    prices = fetch_prices(tickers, fetch_start, CFG["end_date"])

    bench_prices = prices["^GSPC"]
    stock_prices = prices.drop(columns=["^GSPC"])
    valid_tickers = stock_prices.columns[
        stock_prices.notna().sum() >= CFG["min_history"]
    ].tolist()
    stock_prices = stock_prices[valid_tickers]
    prices_full = pd.concat([stock_prices, bench_prices], axis=1)

    print(
        f"✓ Universe: {len(valid_tickers)} stocks | "
        f"{prices_full.index[0].date()} → {prices_full.index[-1].date()}"
    )
    # 2. Compute rolling idiosyncratic momentum
    idio_mom = build_idiosyncratic_momentum(
        prices_full,
        valid_tickers,
        CFG["momentum_window"],
        CFG["momentum_skip"],
    )
    # 3. Z-score normalize cross-sectionally
    signal_z = zscore_normalize(idio_mom)
    # 4. Build portfolio weights
    weights = build_portfolio(
        signal_z,
        prices_full,
        valid_tickers,
        CFG["top_k"],
        CFG["rebalance_freq"],
        CFG["min_history"],
        CFG["z_threshold"],
    )
    prices_full = prices_full.loc[start_dt:]
    idio_mom = idio_mom.loc[start_dt:]
    signal_z = signal_z.loc[start_dt:]
    weights = weights.loc[start_dt:]
    # 5. Compute returns
    port_ret = compute_portfolio_returns(weights, prices_full)
    bench_ret = bench_prices.pct_change().dropna()

    # Align
    port_ret, bench_ret = port_ret.align(bench_ret, join="inner")
    port_ret = port_ret.dropna()
    bench_ret = bench_ret.loc[port_ret.index]

    # Verify no flat warmup remains
    first_invested = (weights.sum(axis=1) > 0).idxmax()
    print(f"✓ First date with holdings : {first_invested.date()}")
    print(f"✓ Strategy start           : {start_dt.date()}")
    # 6. Performance metrics
    metrics_port = performance_metrics(port_ret, bench_ret, label="Idio Momentum")
    bench_only_ret = bench_ret.copy()
    metrics_bench = performance_metrics(bench_only_ret, bench_only_ret, label="SP500")

    print("\n" + "─" * 60)
    print(f"  {'Metric':<14}  {'Strategy':>12}  {'SP500':>12}  {'Delta':>12}")
    print("─" * 60)
    fmts_print = {
        "Total Return": fmt_pct,
        "CAGR": fmt_pct,
        "Vol": fmt_pct,
        "Sharpe": lambda x: f"{x:.2f}",
        "Sortino": lambda x: f"{x:.2f}",
        "Max DD": fmt_pct,
        "Calmar": lambda x: f"{x:.2f}",
        "Alpha": fmt_pct,
        "Beta": lambda x: f"{x:.3f}",
        "IR": lambda x: f"{x:.2f}",
        "Win Rate": fmt_pct,
    }
    for k, fn in fmts_print.items():
        sv = fn(metrics_port[k]) if pd.notna(metrics_port[k]) else "N/A"
        bv = (
            fn(metrics_bench.get(k, np.nan))
            if pd.notna(metrics_bench.get(k, np.nan))
            else "N/A"
        )
        if k in ["Sharpe", "Sortino", "Calmar", "Beta"]:
            delta = (
                f"{metrics_port[k] / metrics_bench[k] - 1:.2%}"
                if pd.notna(metrics_port[k])
                and pd.notna(metrics_bench.get(k, np.nan))
                and metrics_bench.get(k, np.nan) != 0
                else "N/A"
            )
        else:
            delta = (
                f"{(metrics_port[k] - metrics_bench.get(k, np.nan)):.2%}"
                if pd.notna(metrics_port[k]) and pd.notna(metrics_bench.get(k, np.nan))
                else "N/A"
            )
        print(f"  {k:<14}  {sv:>12}  {bv:>12}  {delta:>12}")
    print("─" * 60)
    plot_tearsheet(
        port_ret,
        bench_ret,
        metrics_port,
        metrics_bench,
        signal_z,
        weights,
        CFG["top_k"],
        RF_DAILY,
        252,
        10000,
        504,
    )
