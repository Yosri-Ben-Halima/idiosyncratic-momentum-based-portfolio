# Idiosyncratic Momentum-Based Portfolio Strategy

A quantitative trading system that constructs and backtests a long-only portfolio of S&P 500 stocks based on **idiosyncratic momentum signals**. The strategy ranks stocks by their residual returns (alpha) from a rolling CAPM model and holds the top K momentum performers, rebalancing monthly.

## Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [The Signal: Idiosyncratic Momentum](#the-signal-idiosyncratic-momentum)
- [Code Repository Walkthrough](#code-repository-walkthrough)
- [Setup and Configuration](#setup-and-configuration)
- [Commands & Usage](#commands--usage)
- [Local dashboard](#local-dashboard)

## Local dashboard

Choose the existing PNG workflow or launch **Residual / Lab**, the localhost research dashboard. Both call the same computation engine. The dashboard uses the approved dark navy/teal design, with interactive performance, risk, portfolio, and sensitivity views.

With the project's Python 3.12 environment:

```powershell
uv pip install --python .venv312/Scripts/python.exe -r requirements.txt
task dashboard PYTHON=.venv312/Scripts/python.exe SOURCE=results/2026-10-06 CLI_ARGS=--open
```

Or launch directly:

```powershell
.venv312/Scripts/python.exe -m src.dashboard --source results/2026-10-06 --port 8000 --open
```

Open **http://127.0.0.1:8000**. Omit `--source` / `SOURCE` to discover the newest usable saved snapshot under `results/`. Use another `--port` / `PORT` if 8000 is occupied. The app binds to loopback only. To create an environment from scratch, run `uv venv --python 3.12 .venv312` first; ordinary Python 3.12 environments with `python -m pip install -r requirements.txt` also work.

**PNG mode remains available:**

```powershell
task backtest PYTHON=.venv312/Scripts/python.exe END_DATE=2026-10-08 OUTPUT_DIR=results/latest
```

Dashboard operation:

- Use synchronized sliders and numeric inputs for the momentum window, cutoff, and maximum holdings; date pickers for the period; frequency and simulation presets in dropdowns; and advanced numeric fields for costs, risk estimation, and simulation settings. The annual risk-free-rate field is a percentage; costs are basis points.
- Click **Run backtest** to submit settings. Changing a control marks the displayed result as outdated and does not run on every slider movement. Previous results stay visible while a new computation runs, including after a failed run.
- **Include sensitivity** enables a configurable window × cutoff grid. Explore filled rotatable surfaces for Sharpe, Calmar, total return, CAGR, volatility, drawdown loss, holdings, diversification, top-stock risk share, and effective risk contributors. Period controls apply to performance metrics; holdings/risk means cover the full period.
- **Export PNG** renders the completed run's three tearsheets, or five when sensitivity is enabled. Download individual PNGs or a ZIP with CSVs and provenance. Edits not yet run never change an export. The simulation seed and computed Monte Carlo arrays are shared across the dashboard and PNG output.
- Saved prices and chart assets work offline. **Refresh data for these dates** explicitly downloads a new immutable snapshot and refreshes the risk-free-rate default. It preserves earlier snapshots/results. After refresh, **Reset to snapshot** adopts refreshed defaults; run again to use new data. Download failures retain the previous source.
- End dates are **exclusive**. Runs cannot extend beyond observed snapshot coverage; the next business day after the last price is accepted to accommodate trailing weekends. An exchange holiday may require setting an earlier end date. Requested download cutoffs and actual coverage are recorded separately.

The worker runs one computation at a time, with at most two waiting jobs and eight completed jobs retained in memory. Identical completed runs are cached using all parameters and snapshot identity; explicit refreshes always fetch again after the previous refresh finishes. Generated snapshots and export bundles are retained on disk in ignored `results/dashboard/` for reproducibility; remove old outputs when no longer needed. Job IDs expire after eviction or server restart.

Practical limits: at most 225 sensitivity combinations; momentum/risk windows up to 756 sessions; 100–20,000 Monte Carlo paths, 21–1,260 horizon sessions, and at most 12,600,000 path-session cells. The interface and backend enforce these limits. A broader grid and longer history require more computation. Limited warmup and undefined risk values appear as warnings or N/A.

The benchmark remains the **price-only S&P 500 index**. This feature preserves the existing current-constituent universe, fixed equal target weights, signal, execution and trading-cost assumptions. DD/CAGR is a rough recovery heuristic. Sensitivity is exploratory rather than independent validation; effective risk contributors are not independent factor bets.

Verification uses `task test PYTHON=.venv312/Scripts/python.exe`. Browser QA can additionally use Playwright, installed separately; it is not required to run the dashboard.

---

## Project Overview

This project implements a **research-backed quantitative strategy** that exploits idiosyncratic momentum—the tendency of stocks with high residual returns (after removing market beta) to outperform in subsequent periods.

### Key Features

- **Data-Driven**: Fetches live S&P 500 price data via `yfinance`
- **Scalable**: Vectorized rolling CAPM calculations for efficient computation
- **Realistic**: Includes transaction costs (20 bps round-trip by default)
- **Transparent**: Detailed performance metrics (Sharpe, Sortino, Maximum Drawdown, etc.)
- **Visual**: Generates professional tearsheets with performance, signal analysis, and Monte Carlo simulations

### Strategy Parameters

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| **Universe** | S&P 500 stocks | ~500 constituents |
| **Signal Window** | 252 days | 1 trading year of CAPM estimation |
| **Skip Period** | 21 days | Exclude last month (momentum crash prevention) |
| **Portfolio Size** | Top 30 stocks | Number of holdings at any time |
| **Rebalance Frequency** | Monthly (ME) | Month-end rebalancing |
| **Transaction Cost** | 20 bps | Round-trip trading cost |
| **Backtest Period** | Jan 2020 – Jan 2026 | ~6 years |

---

## Methodology

### 5-Step Pipeline

The strategy follows a disciplined, systematic pipeline:

#### 1. **Data Fetching**

- Retrieve daily adjusted close prices for all S&P 500 constituents
- Include S&P 500 index (`^GSPC`) as market benchmark
- Warmup period: momentum window + skip + min history + buffer (≈540 calendar days)
- Filter: Require minimum 300 trading days of data per stock

#### 2. **Feature Engineering: Rolling CAPM Residuals**

- Compute expanding CAPM over a rolling 252-day window (1 year)
- **Residual (Alpha)** = Stock Excess Return − (Beta × Market Excess Return)
- Use out-of-sample (OOS) residuals: fit on `[t−252:t−1]`, apply to day `t`
- Vectorized closed-form OLS for efficiency

#### 3. **Cross-Sectional Normalization**

- Z-score normalize residual returns across all stocks for each date
- Centers signal at 0 with unit standard deviation
- Prepares signal for portfolio construction

#### 4. **Portfolio Construction**

- **Asset Selection**: Rank stocks by z-score, select top 30
- **Weighting**: Equal-weight allocation to selected stocks
- **Rebalancing**: Monthly (month-end) execution
- **Cash**: Hold cash if no stocks meet z-score threshold (default: 0.0)

#### 5. **Backtesting & Valuation**

- Compute portfolio daily returns using lagged weights (realistic timing)
- Deduct transaction costs based on daily turnover
- Calculate comprehensive performance metrics:
  - Total Return, CAGR, Volatility, Sharpe Ratio, Sortino Ratio
  - Maximum Drawdown, Calmar Ratio, Alpha, Beta, Information Ratio
- Compare against S&P 500 benchmark

---

## The Signal: Idiosyncratic Momentum

### Why Idiosyncratic Momentum?

Traditional momentum strategies suffer from **market-regime dependency**. A stock with high absolute returns might simply be catching a bull market. **Idiosyncratic momentum** isolates the stock-specific component:

$$\text{Residual Return}_t = R_{stock,t} - \alpha - \beta \cdot R_{market,t}$$

Where:

- $\alpha$ = intercept from one-year rolling CAPM fit
- $\beta$ = market beta (sensitivity)
- $R_{market,t}$ = market excess return (S&P 500 − risk-free rate)
- $R_{stock,t}$ = stock excess return (Stock - risk-free rate)

### Intuition

Stocks with **high idiosyncratic momentum** are:

1. **Outperforming on fundamentals** (not just market-driven)
2. **Momentum-profitable** after controlling for beta
3. **Less correlated to broad market moves** (better diversification)

### Practical Implementation

The feature engineering module computes this efficiently:

- **Window**: 252 trading days (1 year of data)
- **Skip**: Exclude last 21 days to avoid momentum crash crashes
- **Vectorized**: Fully vectorized rolling OLS — no Python loops

---

## Code Repository Walkthrough

### Directory Structure

```bash
├── main.py                          # Entry point: orchestrates full pipeline
├── requirements.txt                 # Python dependencies
├── taskfile.yaml                    # Task runner commands
├── config.yaml                      # Strategy parameters
│
├── src/
│   ├── backtesting/
│   │   ├── historical_backtesting.py    # Portfolio returns & performance metrics
│   │
│   ├── data_fetcher/
│   │   ├── fetcher_service.py           # Data download via yfinance
│   │
│   ├── environment/
│   │   ├── config.yaml                  # Strategy configuration (all tunable params)
│   │   ├── loader.py                    # YAML config parser
│   │
│   ├── feature_engineering/
│   │   ├── features.py                  # Rolling CAPM → residuals → z-score normalization
│   │
│   ├── portfolio_construction/
│   │   ├── asset_selection.py           # Select top-K by signal
│   │   ├── weights_calculation.py       # Compute portfolio weights
│   │   ├── portfolio_orchestration.py   # Coordinate rebalancing
│   │
│   └── visualizations/
│       ├── viz_ops.py                   # Tearsheets, charts, tables
│
├── utils/
│   ├── tick_helper.py                   # Fetch S&P 500 ticker list
│   ├── rate_helper.py                   # Risk-free rate helpers
│
└── tests/
```

### Key Modules

#### **main.py** — Pipeline Orchestrator

Coordinates the full workflow:

```python
1. Fetch prices (with warmup for indicator estimation)
2. Compute idiosyncratic momentum (rolling CAPM residuals)
3. Z-score normalize cross-sectionally
4. Build portfolio weights (equal-weight top 30)
5. Compute returns and metrics
6. Generate tearsheets
```

**Output**: Portfolio returns, performance metrics, visualizations

---

#### **src/feature_engineering/features.py** — Signal Computation

**`compute_rolling_capm_residuals_vectorized()`**

- Fully vectorized rolling OLS via closed-form formulas
- No Python loops — uses pandas rolling operations
- **Input**: Stock returns, market returns, window size
- **Output**: OOS residuals for each stock-date pair

**`zscore_normalize()`**

- Cross-sectional z-score normalization
- Centers signal at 0, scales by standard deviation
- Prepares signal for portfolio construction

---

#### **src/portfolio_construction/** — Portfolio Assembly

**`asset_selection.py`**

- Filters stocks by minimum price history (300 days)
- Ranks by z-score signal
- Selects top K candidates (default: 30)
- Optionally filters by z-score threshold

**`weights_calculation.py`**

- Allocates equal weights to selected stocks
- Rebalances on specified frequency (monthly)

**`portfolio_orchestration.py`**

- Orchestrates selection → weighting → rebalancing
- Handles edge cases (e.g., no eligible stocks → full cash)

---

#### **src/backtesting/historical_backtesting.py** — Returns & Metrics

**`compute_portfolio_returns()`**

- Calculates daily portfolio returns:
  - Returns = lagged weights × daily stock returns
  - Deducts transaction costs based on daily turnover
  - Realistic timing (weights shift to prior day)

**`performance_metrics()`**

- Computes annualized metrics:
  - **CAGR**: Compound annual growth rate
  - **Volatility**: Annualized standard deviation
  - **Sharpe Ratio**: (Mean Excess Return) / Volatility
  - **Sortino Ratio**: Similar to Sharpe, but penalizes only downside
  - **Maximum Drawdown**: Largest peak-to-trough decline
  - **Calmar Ratio**: CAGR / Max Drawdown
  - **Alpha & Beta**: vs. benchmark
  - **Information Ratio**: Excess return / Tracking error
  - **Win Rate**: % of days with positive returns

---

#### **src/visualizations/viz_ops.py** — Tearsheets & Charts

Generates comprehensive tearsheets:

1. **Performance Chart**: Cumulative returns (Strategy vs. Benchmark)
2. **Drawdown Analysis**: Running maximum and underwater plots
3. **Signal Distribution**: Histogram of z-score signals over time
4. **Portfolio Composition**: Heatmap of holdings
5. **Risk Metrics**: Dual-axis plot (returns vs. rolling volatility)
6. **Monte Carlo Simulation**: 10,000 runs of 2-year forward scenarios
7. **Performance Table**: Side-by-side metrics (Strategy vs. S&P 500)

---

#### **src/data_fetcher/fetcher_service.py** — Data Retrieval

**`fetch_prices()`**

- Downloads daily OHLCV data via `yfinance`
- Returns pandas DataFrame (dates × tickers)
- Handles missing data and gaps gracefully

---

#### **utils/tick_helper.py** — Universe Definition

**`get_sp500_tickers()`**

- Fetches current S&P 500 constituents
- Returns list of ~500 ticker symbols

---

## Setup and Configuration

config is kept in `src/environment/config.yaml`

All strategy parameters are centralized in YAML:

```yaml
start_date: "2020-01-01"              # Strategy inception
end_date: "2026-01-01"                # Backtest end
momentum_window: 252                  # Trading days for CAPM window (1 year)
momentum_skip: 21                     # Skip last N days
top_k: 30                             # Number of stocks to hold
z_threshold: 2                        # Z-score threshold (optional cutoff)
rebalance_freq: "ME"                  # Month-end rebalancing
min_history: 300                      # Minimum days of data per stock
transaction_cost_bps: 20              # 20 basis points = 0.20%
monte_carlo_runs: 10000               # MC simulation count
monte_carlo_days: 504                 # MC simulation length (2 years)
```

Modify these values in `config.yaml` to tune the strategy.

---

## Commands & Usage

### Installation

```bash
# Install all Python dependencies
task install
```

**Dependencies** (see `requirements.txt`):

- `numpy` — Numerical computing
- `pandas` — Data manipulation
- `scipy` — Statistical functions
- `matplotlib`, `seaborn` — Visualization
- `yfinance` — Market data
- `PyYAML` — Configuration parsing
- `tqdm` — Progress bars
- `requests` — HTTP requests

### Running the Strategy

```bash
# Run the full backtesting pipeline
task run
```

To refresh the backtest without changing the configured historical end date:

```powershell
task backtest END_DATE=2026-10-06 OUTPUT_DIR=results/2026-10-06
# With the isolated Python environment used for this run:
task backtest PYTHON=.venv312/Scripts/python.exe END_DATE=2026-10-06 OUTPUT_DIR=results/2026-10-06
```

`END_DATE` is exclusive: the example requests completed daily sessions through
October 5, 2026. Always check `run_metadata.json` for the actual last available
session. The output directory contains metrics, daily and annual returns, latest
holdings, the configuration and universe snapshot, downloaded prices, daily
weights, and five tearsheets. Raw prices and weights are kept locally and ignored
by Git. Python 3.12 supports the pinned dependencies.

Research limitations: the universe is today's S&P 500 membership, not historical
membership; adjusted stock prices are compared with the price-only `^GSPC`
benchmark; the current risk-free quote is applied to the entire history; and
weights stay equal between monthly selections rather than drifting with prices.
The configured `z_threshold: 2` is applied and can produce fewer than 30 holdings.
Trading costs follow the existing half-L1 target-weight turnover formula and do
not charge for maintaining equal weights between selections. These results are
an exploratory backtest of the existing implementation.

To reproduce the window/cutoff sensitivity study from saved data:

```powershell
task sensitivity PYTHON=.venv312/Scripts/python.exe SOURCE=results/2026-10-06
```

This tests 99 settings: nine windows from 126 to 378 trading days and eleven
z-score cutoffs from 1 to 3, including closer steps around 252 days / cutoff 2.
The window changes both CAPM estimation and residual aggregation, as in the
existing strategy. All runs use the saved universe, prices, risk-free quote,
cost convention, and identical evaluation dates. Outputs under `sensitivity/`
include the grid CSV, full/early/recent metrics, 3D Sharpe, Calmar, total-return and average-holdings surfaces,
heatmaps, and a standalone HTML plot with rotation and metric selection.
The current setting must reproduce the previous daily returns before results
are accepted. Period comparisons are exploratory diagnostics, not independent
out-of-sample validation or an instruction to select the grid's maximum.

Sensitivity is now a reusable module in `src/sensitivity`, and the main pipeline
runs it by default using the same downloaded prices and risk-free quote. Its
four surfaces appear in `tearsheet_sheet4_sensitivity.png`, matching the dark
tearsheet theme. Total return is displayed in percent. The default grid adds
several minutes to a backtest; customize `sensitivity.windows`,
`sensitivity.cutoffs` and `sensitivity.split_date` in `src/environment/config.yaml`.
The current window and cutoff are automatically included for baseline checks.
Fresh downloads include enough warmup for the longest configured window.
The holdings surface shows the mean count across every backtest session,
including cash sessions as zero, rather than the latest holdings count.

```powershell
# Redraw existing grids and the fourth tearsheet without recomputing:
task sensitivity PYTHON=.venv312/Scripts/python.exe SOURCE=results/2026-10-06 CLI_ARGS=--render-only
# Run the three original tearsheets without the sensitivity grid:
task run PYTHON=.venv312/Scripts/python.exe CLI_ARGS=--skip-sensitivity
# Run the regression suite:
task test PYTHON=.venv312/Scripts/python.exe
```

Library usage (no network calls):

```python
from src.sensitivity import run_sensitivity

study = run_sensitivity(prices_with_warmup, tickers, config,
                        windows=[231, 252, 273], cutoffs=[1.9, 2, 2.1],
                        expected_returns=portfolio_returns)
study.save("results/custom/sensitivity")
```

`config` includes the derived `rf_daily` and `cost_bps` values from `load_config`.
All-cash settings retain undefined Sharpe/Calmar values as missing, rather than
displaying them as zero. Saved JSON represents these values as `null`.

The fifth page, `tearsheet_sheet5_risk_concentration.png`, compares correlation-aware
diversification and single-stock risk concentration with realized volatility and
drawdown across the same parameter grid. Covariance uses preceding returns only,
refreshed monthly and when target weights change. Configure `risk_window` (252)
and `risk_min_observations` (126) under `sensitivity`.

- Diversification ratio: weighted average stock volatility divided by portfolio
  volatility. Higher values indicate greater diversification benefit.
- Top-stock risk share: largest normalized **absolute** Euler variance contribution,
  accounting for volatility and covariances. Negative hedge contributions are
  counted by magnitude and separately recorded in `risk_hedge_share`.
- Effective risk contributors: inverse concentration of those absolute shares;
  this is not a count of statistically independent bets.
- Risk means exclude cash sessions and unavailable covariance estimates. The grid's
  `risk_coverage` field reports coverage of invested sessions. This differs from
  average holdings, which includes cash sessions as zero.

Exports also include `risk_surfaces.png`, `risk_impact.png`, baseline daily risk
metrics and latest per-stock risk contributions. The impact chart compares
diversification with realized volatility/drawdown and capital weights with the
latest estimated risk shares. These are descriptive associations, not causal
estimates. Older grids must be recomputed once to add risk columns; subsequent
`--render-only` runs reuse them. Methods and the current comparison are recorded
in `results/2026-10-06/sensitivity/risk_findings.md`.

**Pipeline Execution**:

1. Fetches S&P 500 tickers
2. Downloads price data (with warmup)
3. Computes idiosyncratic momentum signal
4. Normalizes signal cross-sectionally
5. Constructs portfolio weights
6. Backtests with transaction costs
7. Calculates performance metrics
8. Generates visualizations

**Output**:

- `image/tearsheet_*.png`: Multi-panel visualizations and performance metrics.

---

## Key Takeaways

1. **Alpha from Idiosyncratic Momentum**: The strategy exploits residual returns after controlling for market beta
2. **Simplicity + Discipline**: Systematic monthly rebalancing, transparent rules
3. **Low Turnover**: Equal-weight positions reduce transaction costs
4. **Realistic**: Includes realistic trading costs and out-of-sample calculations
5. **Scalable**: Vectorized code runs efficiently on 500+ stocks

---

## References

- **Blitz, D., Hanauer, M. X., Vidojevic, M., & Vliet, P. V.** (2018). "How do Factor Premia Vary Over Time? A Century of Evidence". _Journal of Portfolio Management_
- **Fama, E. F., & French, K. R.** (2015). "A five-factor model of asset prices". _Journal of Financial Economics_
- **Harvey, C. R., Liu, Y., & Zhu, H.** (2016). "...and the Cross-Section of Expected Returns". _Review of Financial Studies_
