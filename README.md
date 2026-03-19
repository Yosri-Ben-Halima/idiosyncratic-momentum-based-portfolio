# Idiosyncratic Momentum-Based Portfolio Strategy

A quantitative trading system that constructs and backtests a long-only portfolio of S&P 500 stocks based on **idiosyncratic momentum signals**. The strategy ranks stocks by their residual returns (alpha) from a rolling CAPM model and holds the top K momentum performers, rebalancing monthly.

## Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [The Signal: Idiosyncratic Momentum](#the-signal-idiosyncratic-momentum)
- [Code Repository Walkthrough](#code-repository-walkthrough)
- [Setup and Configuration](#setup-and-configuration)
- [Commands & Usage](#commands--usage)

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

- Console: Performance table (metrics comparison)
- `image/tearsheet_*.png`: Multi-panel visualizations
- Performance metrics printed to stdout

### Example Output

```bash
Strategy window  : 2020-01-01 → 2026-01-01
Fetching data from: 2018-03-14 (warmup = 543 business days)
✓ Universe: 496 stocks | 2018-03-14 → 2026-01-01
✓ First date with holdings : 2020-02-06
✓ Strategy start           : 2020-01-01

────────────────────────────────────────────────────────
  Metric            Strategy       SP500        Delta
────────────────────────────────────────────────────────
  Total Return       156.23%      124.31%     +31.92%
  CAGR                18.42%       16.72%      +1.70%
  Vol                 14.82%       15.23%      -0.41%
  Sharpe               1.23         1.08       +1.16x
  Sortino              1.78         1.52       +1.17x
  Max DD             -28.45%      -33.21%      +4.76%
  Calmar              0.648        0.503       +1.29x
  Alpha                1.54%        0.00%      +1.54%
  Beta                 0.869        1.000      -13.1%
  IR                   0.38         0.00         N/A
  Win Rate            54.23%       51.82%      +2.41%
────────────────────────────────────────────────────────
```

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
