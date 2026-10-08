"""Reproducible simulation and chart data, independent of renderer."""
import numpy as np
import pandas as pd


def recovery_years(max_dd,cagr):
    return abs(float(max_dd))/float(cagr) if np.isfinite(cagr) and cagr>0 else None


def monte_carlo(returns,paths,horizon,seed=42):
    values = returns.dropna().to_numpy()
    if not len(values):
        raise ValueError('Monte Carlo requires observed returns')
    block = min(21,len(values))
    rng = np.random.RandomState(seed)
    sims = np.empty((paths,horizon))
    for i in range(paths):
        path = []
        while len(path)<horizon:
            start = rng.randint(0,len(values)-block) if len(values)>block else 0
            path.extend(values[start:start+block])
        sims[i] = np.cumprod(1+np.asarray(path[:horizon]))-1
    percentiles = [1,5,25,50,75,95,99]
    return {'bands':{p:np.percentile(sims,p,axis=0) for p in percentiles},
            'terminal':sims[:,-1].copy(),
            'dates':pd.bdate_range(returns.index[-1]+pd.Timedelta(days=1),periods=horizon)}


def build_diagnostics(result):
    cfg = result.config
    returns = result.returns
    cumulative = (1+returns).cumprod()
    window = cfg['momentum_window']
    port,bench = returns.Strategy,returns.SP500
    beta = port.rolling(window).cov(bench)/bench.rolling(window).var()
    alpha = (port.rolling(window).mean()-cfg['rf_daily']-beta*(bench.rolling(window).mean()-cfg['rf_daily']))*252
    cutoff = result.signal.apply(lambda row: row.dropna().nlargest(cfg['top_k']).min()
                                if row.count()>=cfg['top_k'] else np.nan,axis=1)
    return {'cumulative':cumulative-1,'drawdown':cumulative/cumulative.cummax()-1,
            'rolling_sharpe':returns.rolling(window).mean()/returns.rolling(window).std()*np.sqrt(252),
            'rolling_beta':beta,'rolling_alpha':alpha,
            'annual':(1+returns).resample('YE').prod()-1,
            'monthly':(1+returns).resample('ME').prod()-1,
            'turnover':result.weights.diff().abs().sum(axis=1).resample('ME').sum()/2,
            'holdings':(result.weights>0).sum(axis=1),'entry_z':cutoff,
            'mc':monte_carlo(port,cfg.get('monte_carlo_runs',1000),cfg.get('monte_carlo_days',504),cfg.get('monte_carlo_seed',42))}
