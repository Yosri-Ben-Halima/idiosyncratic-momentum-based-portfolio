"""Run or redraw sensitivity from a saved backtest: python -m src.sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from .analysis import SensitivityResult, run_sensitivity
from .plots import plot_risk_sheet, plot_sensitivity_sheet, render_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/latest"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    source = args.source
    output = args.output or source / "sensitivity"
    metadata = json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))
    expected = pd.read_csv(source / "daily_returns.csv", index_col=0, parse_dates=True)
    if args.render_only:
        study = json.loads((output / "study_metadata.json").read_text(encoding="utf-8"))
        rows = pd.read_csv(output / "grid.csv")
        windows, cutoffs = sorted(rows.window.unique().tolist()), sorted(rows.cutoff.unique().tolist())
        if len(rows) != len(windows)*len(cutoffs) or rows.duplicated(["window", "cutoff"]).any():
            raise ValueError("Saved sensitivity grid must contain every combination exactly once")
        # Also supports snapshots produced by the original standalone study.
        split = study.get("split_date", study.get("periods", {}).get("recent", ["2023-01-01"])[0])
        result = SensitivityResult(rows, study["config"], windows, cutoffs,
            str(expected.index[0].date()), str(expected.index[-1].date()), split,
            study.get("baseline_verified", False))
        if (output / "latest_risk_contributions.csv").exists():
            result.latest_risk_contributions = pd.read_csv(output / "latest_risk_contributions.csv", index_col=0)
            result.latest_risk_contributions.attrs["snapshot_date"] = study.get("latest_risk_snapshot_date")
    else:
        prices = pd.read_csv(source / "prices.csv", index_col=0, parse_dates=True)
        tickers = metadata["universe_used"]
        result = run_sensitivity(prices[tickers + ["^GSPC"]], tickers, metadata["config"],
            expected_returns=expected.Strategy, progress=lambda message: print(message, flush=True))
        result.save(output, universe_used=tickers,
            price_sha256=hashlib.sha256((source / "prices.csv").read_bytes()).hexdigest())
    render_sensitivity(result, output)
    plot_sensitivity_sheet(result, source / "tearsheet_sheet4_sensitivity.png")
    if "risk_diversification_ratio" in result.grid:
        plot_risk_sheet(result, source / "tearsheet_sheet5_risk_concentration.png")
    print(f"Sensitivity exports saved: {output}")


if __name__ == "__main__":
    main()
