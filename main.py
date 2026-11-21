import argparse
import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.data_fetcher import fetch_prices
from src.engine import Snapshot, normalize_config, run_backtest
from src.engine.exports import export_result
from src.environment import load_config
from src.sensitivity import resolve_grid
from src.sensitivity.plots import render_sensitivity
from utils.halal_helper import filter_halal
from utils.tick_helper import get_sp500_tickers
from src.visualizations import fmt_pct

warnings.filterwarnings("ignore")

def main():
    """Pipeline orchestrating data loading, backtesting, and visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", help="Exclusive download cutoff (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="image", help="Directory for results and tearsheets")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip the parameter grid and fourth tearsheet")
    args = parser.parse_args()
    CFG = load_config()
    sensitivity_enabled = CFG.get("sensitivity", {}).get("enabled", True) and not args.skip_sensitivity
    if args.end_date:
        CFG["end_date"] = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 1. Fetch data
    start_dt = pd.Timestamp(CFG["start_date"])
    warmup_days = (
        CFG["momentum_window"] + CFG["momentum_skip"] + CFG["min_history"] + 30
    )
    if sensitivity_enabled:
        grid_windows, _ = resolve_grid(CFG)
        warmup_days = max(warmup_days, 2*max(grid_windows) + CFG["momentum_skip"] + 60)
    fetch_start = (start_dt - pd.offsets.BDay(warmup_days)).strftime("%Y-%m-%d")

    print(f"Strategy window  : {CFG['start_date']} → {CFG['end_date']}")
    print(f"Fetching data from: {fetch_start} (warmup = {warmup_days} business days)")

    # ── Fetch data from fetch_start ───────────────────────────────────────────
    tickers = get_sp500_tickers()
    if CFG.get("halal_filter", False):
        tickers = filter_halal(tickers, CFG.get("halal_standard", "aaoifi"))
    prices = fetch_prices(tickers, fetch_start, CFG["end_date"])
    if prices.empty or "^GSPC" not in prices or prices["^GSPC"].dropna().empty:
        raise RuntimeError("No usable benchmark prices downloaded; backtest cannot run.")
    prices.to_csv(output_dir / "prices.csv")

    bench_prices = prices["^GSPC"]
    stock_prices = prices.drop(columns=["^GSPC"])
    valid_tickers = stock_prices.columns[
        stock_prices.notna().sum() >= CFG["min_history"]
    ].tolist()
    stock_prices = stock_prices[valid_tickers]
    prices_full = pd.concat([stock_prices, bench_prices], axis=1)
    sensitivity_prices = prices_full

    print(
        f"✓ Universe: {len(valid_tickers)} stocks | "
        f"{prices_full.index[0].date()} → {prices_full.index[-1].date()}"
    )
    CFG.setdefault("sensitivity", {})["enabled"] = sensitivity_enabled
    metadata = {"universe_requested": tickers, "universe_used": valid_tickers,
                "excluded_tickers": sorted(set(tickers) - set(valid_tickers)),
                "price_sha256": hashlib.sha256((output_dir / "prices.csv").read_bytes()).hexdigest()}
    snapshot = Snapshot(metadata["price_sha256"], sensitivity_prices, valid_tickers, CFG, metadata)
    CFG = normalize_config({}, CFG, snapshot)
    result = run_backtest(snapshot, CFG, progress=lambda message: print(message, flush=True))
    metrics_port, metrics_bench = result.metrics["Strategy"], result.metrics["SP500"]
    export_result(result, output_dir)
    if result.sensitivity is not None:
        render_sensitivity(result.sensitivity, output_dir / "sensitivity")

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



if __name__ == "__main__":
    main()
