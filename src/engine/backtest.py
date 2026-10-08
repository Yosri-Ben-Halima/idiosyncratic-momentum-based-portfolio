"""One numerical pipeline shared by file and web interfaces."""
from dataclasses import dataclass, field
from copy import deepcopy
import pandas as pd
from src.backtesting import compute_portfolio_returns, performance_metrics
from src.feature_engineering import build_idiosyncratic_momentum, zscore_normalize
from src.portfolio_construction import build_portfolio
from src.sensitivity import run_sensitivity
from src.sensitivity.risk import RiskDiagnostics


@dataclass
class Snapshot:
    id: str
    prices: pd.DataFrame
    tickers: list
    config: dict
    metadata: dict


@dataclass
class BacktestResult:
    config: dict
    snapshot_id: str
    returns: pd.DataFrame
    metrics: dict
    signal: pd.DataFrame
    weights: pd.DataFrame
    risk_daily: pd.DataFrame
    contributions: pd.DataFrame
    sensitivity: object = None
    warnings: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def run_backtest(snapshot, config, progress=None):
    from .diagnostics import build_diagnostics
    report = progress or (lambda message: None)
    cfg = deepcopy(config)
    prices = snapshot.prices.loc[snapshot.prices.index<pd.Timestamp(cfg['end_date'])]
    tickers = snapshot.tickers
    if '^GSPC' not in prices or prices['^GSPC'].dropna().empty:
        raise ValueError('No usable benchmark prices')
    report('Computing CAPM residuals and momentum')
    signal = zscore_normalize(build_idiosyncratic_momentum(prices,tickers,
        cfg['momentum_window'],cfg['rf_daily'],cfg['momentum_skip']))
    report('Constructing portfolio and applying trading costs')
    weights = build_portfolio(signal,prices,tickers,cfg['top_k'],cfg['rebalance_freq'],
                              cfg['min_history'],cfg['z_threshold']).loc[cfg['start_date']:]
    evaluation = prices.loc[cfg['start_date']:]
    ret = compute_portfolio_returns(weights,evaluation,cfg['cost_bps'])
    benchmark = prices['^GSPC'].pct_change().dropna()
    ret,benchmark = ret.align(benchmark,join='inner')
    ret = ret.dropna()
    benchmark = benchmark.loc[ret.index]
    if len(ret)<2:
        raise ValueError('At least two usable evaluation sessions are required')
    report('Measuring portfolio risk contributions')
    options = cfg.get('sensitivity',{})
    risk = RiskDiagnostics(prices[tickers],options.get('risk_window',252),
                           options.get('risk_min_observations',126))
    weights = weights.loc[ret.index]
    result = BacktestResult(cfg,snapshot.id,pd.DataFrame({'Strategy':ret,'SP500':benchmark}),
        {'Strategy':performance_metrics(ret,benchmark,cfg['rf_daily'],'Idio Momentum'),
         'SP500':performance_metrics(benchmark,benchmark,cfg['rf_daily'],'SP500')},
        signal.loc[ret.index],weights,risk.evaluate(weights),risk.latest_contributions(weights))
    result.provenance = {**snapshot.metadata,'snapshot_id':snapshot.id,'universe_used':tickers}
    warmup = len(prices.loc[prices.index<pd.Timestamp(cfg['start_date'])])
    if warmup<2*cfg['momentum_window']+cfg['momentum_skip']:
        result.warnings.append('Limited preceding price history: early signals may be unavailable.')
    if not (weights.sum(axis=1)>0).any():
        result.warnings.append('No stocks passed these settings: the portfolio stayed in cash.')
    if options.get('enabled',False):
        result.sensitivity = run_sensitivity(prices,tickers,cfg,expected_returns=ret,progress=report)
    report('Preparing diagnostics and Monte Carlo simulation')
    result.diagnostics = build_diagnostics(result)
    report('Complete')
    return result
