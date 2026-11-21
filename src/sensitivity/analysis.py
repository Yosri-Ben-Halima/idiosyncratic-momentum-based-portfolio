"""Parameter grids evaluated with the strategy's existing execution rules."""
from dataclasses import dataclass
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting import compute_portfolio_returns, performance_metrics
from src.feature_engineering import build_idiosyncratic_momentum, zscore_normalize
from src.portfolio_construction import build_portfolio
from .risk import RiskDiagnostics

DEFAULT_WINDOWS = (126, 168, 210, 231, 252, 273, 294, 336, 378)
DEFAULT_CUTOFFS = (1.0, 1.25, 1.5, 1.75, 1.9, 2.0, 2.1, 2.25, 2.5, 2.75, 3.0)
METRICS = ("Sharpe", "Calmar", "CAGR", "Max DD", "Vol", "Total Return")


def resolve_grid(config, windows=None, cutoffs=None):
    """Validate configured axes and always include the strategy's current setting."""
    options = config.get("sensitivity", {})
    windows = list(options.get("windows", DEFAULT_WINDOWS) if windows is None else windows)
    cutoffs = list(options.get("cutoffs", DEFAULT_CUTOFFS) if cutoffs is None else cutoffs)
    if not windows or not cutoffs:
        raise ValueError("Sensitivity axes must be nonempty")
    windows.append(config["momentum_window"])
    cutoffs.append(config["z_threshold"])
    if not windows or any(isinstance(w, bool) or not np.isfinite(w) or w < 2 or int(w) != w for w in windows):
        raise ValueError("Sensitivity windows must be nonempty integer session counts >= 2")
    if not cutoffs or any(isinstance(c, bool) or not np.isfinite(c) for c in cutoffs):
        raise ValueError("Sensitivity cutoffs must be nonempty finite numbers")
    return sorted(set(map(int, windows))), sorted(set(map(float, cutoffs)))


@dataclass
class SensitivityResult:
    grid: pd.DataFrame
    config: dict
    windows: list
    cutoffs: list
    start_date: str
    end_date: str
    split_date: str
    baseline_verified: bool = False
    baseline_risk_daily: pd.DataFrame | None = None
    latest_risk_contributions: pd.DataFrame | None = None

    @property
    def baseline(self):
        return self.grid[(self.grid.window == self.config["momentum_window"]) &
                         (self.grid.cutoff == self.config["z_threshold"])].iloc[0]

    def summary(self):
        wi = self.windows.index(self.config["momentum_window"])
        ci = self.cutoffs.index(self.config["z_threshold"])
        local = self.grid[self.grid.window.isin(self.windows[max(0, wi-1):wi+2]) &
                          self.grid.cutoff.isin(self.cutoffs[max(0, ci-1):ci+2])]
        z = self.grid.pivot(index="window", columns="cutoff", values="full_Sharpe").to_numpy()
        peaks = []
        for i in range(1, len(self.windows)-1):
            for j in range(1, len(self.cutoffs)-1):
                neighbors = np.delete(z[i-1:i+2, j-1:j+2].ravel(), 4)
                if np.isfinite(neighbors).all() and z[i, j] > neighbors.max():
                    peaks.append({"window": self.windows[i], "cutoff": self.cutoffs[j],
                                  "sharpe": float(z[i, j]),
                                  "above_neighbor_mean": float(z[i,j]-neighbors.mean()),
                                  "above_best_neighbor": float(z[i,j]-neighbors.max())})
        valid = self.grid.dropna(subset=["full_Sharpe"])
        return {"baseline_matches_previous_run": self.baseline_verified, "settings": len(self.grid),
                "baseline": self.baseline.to_dict(),
                "local_sharpe_range": [local.full_Sharpe.min(), local.full_Sharpe.max()],
                "local_sharpe_std": local.full_Sharpe.std(),
                "early_vs_recent_grid_spearman": self.grid.early_Sharpe.corr(self.grid.recent_Sharpe, method="spearman"),
                "interior_local_maxima": peaks,
                "best_full_period_sharpe": valid.loc[valid.full_Sharpe.idxmax()].to_dict() if not valid.empty else None,
                "warning": "Exploratory grid, not independent out-of-sample validation or a parameter recommendation."}

    def save(self, output, **provenance):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.grid.to_csv(output / "grid.csv", index=False)
        metadata = {"config": self.config, "windows": self.windows, "cutoffs": self.cutoffs,
                    "start_date": self.start_date, "end_date": self.end_date,
                    "split_date": self.split_date, "baseline_verified": self.baseline_verified,
                    **provenance}
        options = self.config.get("sensitivity", {})
        metadata["risk_diagnostics"] = {
            "window": options.get("risk_window", 252),
            "min_observations": options.get("risk_min_observations", 126),
            "sampling": "preceding returns; monthly and on target-weight changes",
            "averaging": "daily mean over invested sessions with valid covariance",
            "concentration": "normalized absolute Euler variance contributions",
        }
        if self.baseline_risk_daily is not None:
            self.baseline_risk_daily.to_csv(output / "baseline_risk_daily.csv")
        if self.latest_risk_contributions is not None:
            self.latest_risk_contributions.to_csv(output / "latest_risk_contributions.csv")
            metadata["latest_risk_snapshot_date"] = self.latest_risk_contributions.attrs.get("snapshot_date")
        write_json(output / "study_metadata.json", metadata)
        write_json(output / "summary.json", self.summary())


def write_json(path, value):
    # JSON null preserves undefined risk ratios (e.g. an all-cash setting).
    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item.item() if isinstance(item, np.generic) else item
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def run_sensitivity(prices, tickers, config, *, windows=None, cutoffs=None,
                    expected_returns=None, split_date=None, progress=None):
    """Evaluate a grid without fetching data or changing the caller's configuration.

    Uses identical evaluation dates, including cash sessions, for every setting.
    Pass baseline daily returns to verify that the grid reproduces the main run.
    Empty/constant-return periods retain undefined ratios as NaN.
    """
    windows, cutoffs = resolve_grid(config, windows, cutoffs)
    if not tickers or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Sensitivity requires tickers and sorted, unique price dates")
    if "^GSPC" not in prices or prices["^GSPC"].dropna().empty:
        raise ValueError("Sensitivity requires usable ^GSPC benchmark prices")
    benchmark = prices["^GSPC"].pct_change().dropna()
    evaluation_prices = prices.loc[config["start_date"]:]
    dates = evaluation_prices.index.intersection(benchmark.index)
    if len(dates) < 2:
        raise ValueError("Sensitivity requires at least two evaluation sessions")
    if expected_returns is not None and not expected_returns.index.equals(dates):
        raise ValueError("Sensitivity baseline dates must match the evaluation dates")
    if split_date is None:
        split_date = config.get("sensitivity", {}).get("split_date")
    if split_date is None:
        midpoint = dates[len(dates)//2]
        split_date = pd.Timestamp(year=midpoint.year, month=1, day=1)
        if split_date <= dates[0]:
            split_date = midpoint
    split = pd.Timestamp(split_date)
    periods = {"full": np.ones(len(dates), dtype=bool), "early": dates < split, "recent": dates >= split}
    records = []
    baseline_verified = False
    baseline_risk_daily = latest_risk_contributions = None
    options = config.get("sensitivity", {})
    risk_model = RiskDiagnostics(prices[tickers], window=options.get("risk_window", 252),
                                min_observations=options.get("risk_min_observations", 126))
    for window in windows:
        if progress:
            progress(f"Computing sensitivity signal: {window} sessions")
        # The same window intentionally controls both estimation and summation.
        signal = zscore_normalize(build_idiosyncratic_momentum(
            prices, tickers, window, config["rf_daily"], config["momentum_skip"]))
        for cutoff in cutoffs:
            with contextlib.redirect_stdout(io.StringIO()):
                weights = build_portfolio(signal, prices, tickers, config["top_k"],
                                          config["rebalance_freq"], config["min_history"], cutoff)
            weights = weights.loc[config["start_date"]:]
            returns = compute_portfolio_returns(weights, evaluation_prices, config["cost_bps"]).loc[dates]
            if returns.isna().any() or (returns <= -1).any():
                raise ValueError(f"Invalid sensitivity returns at window={window}, cutoff={cutoff}")
            if window == config["momentum_window"] and cutoff == config["z_threshold"] and expected_returns is not None:
                if not np.allclose(returns, expected_returns, atol=1e-12, rtol=1e-10):
                    raise ValueError("Sensitivity baseline does not reproduce the main backtest")
                baseline_verified = True
            counts = weights.loc[dates].gt(0).sum(axis=1)
            row = {"window": window, "cutoff": cutoff, "mean_holdings": counts.mean(),
                   "min_holdings": counts.min(), "max_holdings": counts.max(),
                   "cash_days": int(counts.eq(0).sum())}
            row.update(risk_model.summarize(weights.loc[dates]))
            if window == config["momentum_window"] and cutoff == config["z_threshold"]:
                baseline_risk_daily = risk_model.evaluate(weights.loc[dates])
                latest_risk_contributions = risk_model.latest_contributions(weights.loc[dates])
            for label, mask in periods.items():
                metrics = (performance_metrics(returns.loc[mask], benchmark.loc[dates[mask]], config["rf_daily"])
                           if mask.sum() >= 2 else {})
                row.update({f"{label}_{metric}": metrics.get(metric, np.nan) for metric in METRICS})
            records.append(row)
        if progress:
            progress(f"Sensitivity: {len(records)}/{len(windows)*len(cutoffs)} settings")
    return SensitivityResult(pd.DataFrame(records), dict(config), windows, cutoffs,
                             str(dates[0].date()), str(dates[-1].date()), str(split.date()), baseline_verified,
                             baseline_risk_daily, latest_risk_contributions)
