"""Server-side parameter validation and units conversion."""
from copy import deepcopy
import math
import pandas as pd
from src.sensitivity import resolve_grid

LIMITS = dict(momentum_window=(2,756), momentum_skip=(0,252), top_k=(1,503),
              min_history=(2,2520), z_threshold=(-3,6), transaction_cost_bps=(0,500),
              rf_annual=(-.05,.30), monte_carlo_runs=(100,20000),
              monte_carlo_days=(21,1260), monte_carlo_seed=(0,2**32-1))
INTEGER = {'momentum_window','momentum_skip','top_k','min_history',
           'monte_carlo_runs','monte_carlo_days','monte_carlo_seed'}
SENSITIVITY_KEYS = {'enabled','windows','cutoffs','split_date','risk_window','risk_min_observations'}


def normalize_config(values, defaults, snapshot=None):
    allowed = set(LIMITS) | {'start_date','end_date','rebalance_freq','sensitivity'}
    if set(values)-allowed:
        raise ValueError('Unknown parameters: '+', '.join(sorted(set(values)-allowed)))
    cfg = deepcopy(defaults)
    options = cfg.setdefault('sensitivity', {})
    supplied = values.get('sensitivity', {})
    if not isinstance(supplied, dict) or set(supplied)-SENSITIVITY_KEYS:
        raise ValueError('sensitivity: unsupported settings')
    options.update(supplied)
    cfg.update({k:v for k,v in values.items() if k!='sensitivity'})
    cfg.setdefault('monte_carlo_runs',1000)
    cfg.setdefault('monte_carlo_days',504)
    cfg.setdefault('monte_carlo_seed',42)
    cfg.setdefault('rf_annual',.04)
    cfg.setdefault('transaction_cost_bps',20)
    for field,(low,high) in LIMITS.items():
        number = cfg[field]
        if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number):
            raise ValueError(f'{field}: enter a finite number')
        high = len(snapshot.tickers) if field=='top_k' and snapshot else high
        if not low <= number <= high or (field in INTEGER and int(number)!=number):
            raise ValueError(f'{field}: must be {"an integer " if field in INTEGER else ""}between {low} and {high}')
        if field in INTEGER:
            cfg[field] = int(number)
    if cfg['rebalance_freq'] not in ('W-FRI','ME','QE'):
        raise ValueError('rebalance_freq: choose weekly, month end, or quarter end')
    try:
        start,end = pd.Timestamp(cfg['start_date']),pd.Timestamp(cfg['end_date'])
        if pd.isna(start) or pd.isna(end) or start.tzinfo or end.tzinfo or start>=end:
            raise ValueError()
    except (ValueError,TypeError,KeyError):
        raise ValueError('dates: start must precede the exclusive end date') from None
    cfg['start_date'],cfg['end_date'] = str(start.date()),str(end.date())
    if snapshot:
        # Weekends have no missing trading observations; metadata cannot extend coverage.
        coverage_end = snapshot.prices.index[-1]+pd.offsets.BDay(1)
        if start < snapshot.prices.index[0] or end > coverage_end:
            raise ValueError('dates: outside saved snapshot coverage; refresh data first')
        if len(snapshot.prices.loc[(snapshot.prices.index>=start)&(snapshot.prices.index<end)])<2:
            raise ValueError('dates: at least two evaluation sessions are required')
    options.setdefault('enabled',False)
    if not isinstance(options['enabled'],bool):
        raise ValueError('sensitivity.enabled: must be a boolean')
    for field,default in [('risk_window',252),('risk_min_observations',126)]:
        options.setdefault(field,default)
        v = options[field]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or int(v)!=v or not 2<=v<=756:
            raise ValueError(f'{field}: must be an integer between 2 and 756')
        options[field] = int(v)
    if options['risk_min_observations']>options['risk_window']:
        raise ValueError('risk_min_observations: cannot exceed risk_window')
    if options['enabled']:
        windows,cutoffs = resolve_grid(cfg)
        if len(windows)*len(cutoffs)>225 or max(windows)>756 or min(cutoffs)<-3 or max(cutoffs)>6:
            raise ValueError('sensitivity: at most 225 settings, windows <= 756, cutoffs between -3 and 6')
        options['windows'],options['cutoffs'] = windows,cutoffs
        split = pd.Timestamp(options.get('split_date',start+(end-start)/2))
        if pd.isna(split) or not start<split<end:
            raise ValueError('split_date: must lie inside the evaluation period')
        options['split_date'] = str(split.date())
    if cfg['monte_carlo_runs']*cfg['monte_carlo_days']>12600000:
        raise ValueError('monte_carlo_runs: paths × horizon cannot exceed 12,600,000')
    cfg['rf_daily'] = (1+cfg['rf_annual'])**(1/252)-1
    cfg['cost_bps'] = cfg['transaction_cost_bps']/10000
    return cfg
