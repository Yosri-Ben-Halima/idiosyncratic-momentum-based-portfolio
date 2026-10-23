"""Read-only, reproducible checks of one completed backtest's frozen inputs."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src.backtesting import compute_portfolio_returns
from src.feature_engineering import build_idiosyncratic_momentum, zscore_normalize
from src.feature_engineering.features import compute_rolling_capm_residuals_vectorized
from src.portfolio_construction import build_portfolio

TOLERANCE = 1e-10
SEED = 42


def paired_capm_residuals(stock, market, window):
    """Isolated comparison estimator: all moments use the same observed pairs."""
    minimum = int(window*.7)
    valid = stock.notna() & market.notna()
    x,y = market.where(valid),stock.where(valid)
    def roll(value):
        return value.shift(1).rolling(window,min_periods=minimum)
    count = valid.astype(float).shift(1).rolling(window,min_periods=1).sum()
    mx,my = roll(x).mean(),roll(y).mean()
    denominator = roll(x*x).sum()-count*mx**2
    beta = (roll(x*y).sum()-count*mx*my)/denominator
    alpha = my-beta*mx
    residual = stock-(alpha+beta*market)
    return residual.mask((count<minimum) | denominator.abs().lt(1e-12))


def _replay(prices,tickers,cfg,paired=False,zero_rf=False):
    rf = 0 if zero_rf else cfg['rf_daily']
    if paired:
        returns = prices.pct_change()
        market = returns['^GSPC']-rf
        window = cfg['momentum_window']
        momentum = {}
        for ticker in tickers:
            residual = paired_capm_residuals(returns[ticker]-rf,market,window)
            momentum[ticker] = residual.shift(cfg['momentum_skip']).rolling(window,min_periods=int(.7*window)).sum()
        signal = zscore_normalize(pd.DataFrame(momentum,index=prices.index))
    else:
        signal = zscore_normalize(build_idiosyncratic_momentum(prices,tickers,cfg['momentum_window'],rf,cfg['momentum_skip']))
    weights = build_portfolio(signal,prices,tickers,cfg['top_k'],cfg['rebalance_freq'],cfg['min_history'],cfg['z_threshold']).loc[cfg['start_date']:]
    ret = compute_portfolio_returns(weights,prices.loc[cfg['start_date']:],cfg['cost_bps'])
    benchmark = prices['^GSPC'].pct_change().dropna()
    ret = ret.reindex(ret.index.intersection(benchmark.index)).dropna()
    return signal.loc[ret.index],weights.loc[ret.index],ret


def _metrics(returns):
    wealth = (1+returns).cumprod()
    growth = float(wealth.iloc[-1])
    return {'total_return':growth-1,'cagr':growth**(252/len(returns))-1 if growth>=0 else None,
            'maxdd':float((wealth/wealth.cummax()-1).min()),'sessions':len(returns)}


def drift_returns(weights,stock_returns,cost,rebalance_freq='ME'):
    """Same targets/execution lag, with shares held between scheduled rebalances."""
    raw = stock_returns.reindex(index=weights.index,columns=weights.columns).fillna(0).to_numpy()
    value = cash = 1.
    positions = np.zeros(weights.shape[1])
    previous_target = np.zeros(weights.shape[1])
    calendar = pd.Series(0.,index=stock_returns.index).resample(rebalance_freq).last().index
    locations = weights.index.searchsorted(calendar,side='right')
    scheduled = set(weights.index[locations[locations<len(weights)]])
    result = []
    for i,(target,day_returns) in enumerate(zip(weights.to_numpy(),raw)):
        before = value
        positions *= 1+day_returns
        value = float(positions.sum()+cash)
        if weights.index[i] in scheduled or not np.array_equal(target,previous_target):
            actual = positions/value if value else np.zeros_like(positions)
            # Production starts invested without charging first-row turnover.
            value *= 1-(cost*float(np.abs(target-actual).sum()/2) if i else 0.)
            positions = value*target
            cash = value*(1-target.sum())
            previous_target = target.copy()
        result.append(value/before-1 if before else 0.)
    return pd.Series(result,index=weights.index)


def _difference(left,right):
    a,b = np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if a.shape!=b.shape:
        return {'max_error':None,'missing_changes':1}
    finite = np.isfinite(a)&np.isfinite(b)
    return {'max_error':float(np.abs(a[finite]-b[finite]).max()) if finite.any() else 0.,
            'missing_changes':int(np.count_nonzero(np.isfinite(a)!=np.isfinite(b)))}


def _compare(reference,alternative,dates):
    differences = {};missing = 0
    for name,a,b in zip(['signal','weights','returns'],reference,alternative):
        diff = _difference(a.reindex(dates),b.reindex(dates))
        differences[name] = diff['max_error']
        missing += diff['missing_changes']
    passed = missing==0 and all(v is not None and v<=TOLERANCE for v in differences.values())
    return differences,missing,passed


def _estimator_check():
    rng = np.random.default_rng(SEED)
    x = pd.Series(rng.normal(0,.01,90));y = .001+1.5*x;y.iloc[:8] = np.nan
    original = compute_rolling_capm_residuals_vectorized(y,x,30)
    paired = paired_capm_residuals(y,x,30)
    valid = original.notna()&paired.notna()
    error = float(original.loc[valid].abs().max())
    return {'status':'issue' if error>TOLERANCE else 'passed',
            'original_max_abs_residual':error,'paired_max_abs_residual':float(paired.loc[valid].abs().max()),
            'expected_residual':0.,'description':'A stock with an exact linear market relationship should have zero residual, including after missing early observations.'}


def _clean(value):
    if isinstance(value,dict):return {str(k):_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_clean(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value


def run_audit(snapshot,result,progress=None):
    """Challenge timing and assumptions; never alter the original strategy/result."""
    if snapshot.id!=result.snapshot_id:
        raise ValueError('Audit snapshot does not match the completed run')
    report = progress or (lambda _:None)
    cfg = result.config
    prices = snapshot.prices.loc[snapshot.prices.index<pd.Timestamp(cfg['end_date'])]
    tickers = snapshot.tickers
    dates = result.returns.index
    report('Audit: reproducing the completed run')
    reference = _replay(prices,tickers,cfg)
    original = (result.signal,result.weights,result.returns.Strategy)
    diff,missing,reproduced = _compare(original,reference,dates)
    checks = [{'id':'reproduction','title':'Reproduce the completed run','status':'passed' if reproduced else 'failed',
               'description':'Replay the frozen prices and settings and compare signals, weights and returns.',
               'differences':diff,'missing_changes':missing}]
    cutoff = dates[len(dates)//2]
    prior = dates[dates<cutoff]
    future_dates = dates[dates>=cutoff]
    active = bool((result.weights.loc[prior].sum(axis=1)>0).any())
    enough = len(prior)>=20 and len(future_dates)>=10 and active
    for name,title in [('shock','Change all future prices'),('truncate','Remove all future prices')]:
        if len(prior)<2:
            checks.append({'id':name,'title':title,'status':'insufficient','description':'Not enough earlier evaluation sessions for a comparison.'})
            continue
        report('Audit: '+title.lower())
        changed = prices.loc[prices.index<cutoff].copy() if name=='truncate' else prices.copy()
        if name=='shock':
            mask = changed.index>=cutoff
            rng = np.random.default_rng(SEED)
            factors = np.exp(np.cumsum(rng.normal(0,.025,(int(mask.sum()),len(changed.columns))),axis=0))
            changed.loc[mask] = changed.loc[mask]*factors
        alternative = _replay(changed,tickers,cfg)
        diff,missing,passed = _compare(reference,alternative,prior)
        checks.append({'id':name,'title':title,'status':('passed' if enough else 'insufficient') if passed else 'failed',
                       'description':f'Change data on/after {cutoff.date()}; compare only decisions and returns before that date.',
                       'differences':diff,'missing_changes':missing})
    failed = any(c['status']=='failed' for c in checks)
    timing = 'failed' if failed else 'passed' if enough and all(c['status']=='passed' for c in checks) else 'insufficient'
    baseline = _metrics(result.returns.Strategy)
    comparisons = [{'id':'baseline','label':'Original backtest','metrics':baseline,'delta_cagr':0.,'description':'The completed run, unchanged.'}]
    report('Audit: allowing holdings to drift between rebalances')
    drift = drift_returns(result.weights,prices[tickers].pct_change(),cfg['cost_bps'],cfg['rebalance_freq'])
    comparisons.append({'id':'drift','label':'Hold shares between rebalances','metrics':_metrics(drift),
                        'description':'Hold shares between scheduled rebalances; restore targets even when selections repeat. Charge drifted turnover, preserving the baseline initial-cost convention and execution lag.'})
    report('Audit: measuring the missing-observation estimator effect')
    paired = _replay(prices,tickers,cfg,paired=True)
    comparisons.append({'id':'paired','label':'CAPM with matched observations','metrics':_metrics(paired[2]),
                        'description':'Fit stock and market moments on the same observed dates. This isolated comparison does not modify the production estimator.'})
    report('Audit: removing the fixed risk-free assumption')
    zero = _replay(prices,tickers,cfg,zero_rf=True)
    comparisons.append({'id':'zero_rf','label':'Set constant risk-free rate to zero','metrics':_metrics(zero[2]),
                        'description':'Measure the effect on holdings and portfolio returns, not on excess-return reporting.',
                        'max_weight_difference':_difference(reference[1],zero[1])['max_error']})
    for row in comparisons:
        value = row['metrics']['cagr']
        row['delta_cagr'] = value-baseline['cagr'] if value is not None and baseline['cagr'] is not None else None
    estimator = _estimator_check()
    checks.append({'id':'estimator','title':'Missing-observation CAPM check','status':estimator['status'],
                   'description':'Test an exact linear stock/market relationship with missing early stock data; the expected residual is zero.'})
    report('Audit: reconciling gain attribution and concentration')
    returns = result.returns.Strategy
    raw = prices[tickers].pct_change().reindex(dates)
    contributions = result.weights.shift(1)*raw
    wealth_before = (1+returns).cumprod().shift(1,fill_value=1)
    allocation = contributions.mul(wealth_before,axis=0).sum().sort_values(ascending=False)
    cost = float((result.weights.diff().abs().sum(axis=1)/2*cfg['cost_bps']*wealth_before).sum())
    leading = allocation.reindex(allocation.abs().nlargest(10).index)
    attribution = [{'label':str(k),'value':float(v)} for k,v in leading.items()]
    attribution += [{'label':'Other stocks','value':float(allocation.sum()-leading.sum())},{'label':'Trading costs','value':-cost}]
    count = result.weights.gt(0).sum(axis=1)
    limits = [
        ('Historical universe','Current membership and delisted names are not reconstructed. Both timing probes hold the universe fixed.'),
        ('Initial sample filtering','Initial stock inclusion can use counts across the downloaded history. Per-rebalance eligibility is historical; the initial sample is not proven point-in-time.'),
        ('Independent validation','Changing parameters after viewing this sample is not an untouched out-of-sample test.'),
        ('Data and execution','Vendor history, corporate actions, missing/delisted returns, spreads, market impact and executable fills are not independently validated.')]
    return _clean({'version':1,'computed_at':datetime.now(timezone.utc).isoformat(),
        'scope':{'snapshot_id':snapshot.id,'config':cfg,'dates':[str(dates[0].date()),str(dates[-1].date())],
                 'stocks':len(tickers),'observations':len(dates),'cutoff':str(cutoff.date()),'seed':SEED,'tolerance':TOLERANCE},
        'timing':{'status':timing,'prior_sessions':len(prior),'future_sessions':len(future_dates),
                  'active_before_cutoff':active,'note':'A passing result is evidence at one fixed cutoff and universe, not proof that the entire backtest is unbiased.' if enough else 'At least 20 earlier sessions, 10 later sessions and active earlier holdings are required for a timing verdict.'},
        'checks':checks,'estimator':estimator,'comparisons':comparisons,'attribution':attribution,
        'attribution_reconciliation_error':float(sum(row['value'] for row in attribution)-baseline['total_return']),
        'concentration':{'mean_holdings':float(count.mean()),'min_holdings':int(count.min()),'max_holdings':int(count.max()),
                         'max_stock_weight':float(result.weights.max().max()),'fraction_5_or_fewer':float((count<=5).mean())},
        'limitations':[{'title':title,'description':description,'status':'not_tested'} for title,description in limits]})
