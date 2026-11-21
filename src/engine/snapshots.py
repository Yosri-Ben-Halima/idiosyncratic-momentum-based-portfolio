"""Immutable, source-identified saved market data."""
from pathlib import Path
import hashlib
import json
import uuid
from datetime import datetime, timezone
import pandas as pd
from .backtest import Snapshot


def load_snapshot(source):
    source = Path(source)
    meta = json.loads((source/'run_metadata.json').read_text(encoding='utf-8'))
    raw = (source/'prices.csv').read_bytes()
    prices = pd.read_csv(source/'prices.csv',index_col=0,parse_dates=True)
    if len(prices)<2 or '^GSPC' not in prices or prices['^GSPC'].dropna().empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Snapshot requires sorted unique price dates and usable benchmark prices')
    tickers = [t for t in meta['universe_used'] if t in prices and t!='^GSPC']
    if not tickers:
        raise ValueError('Snapshot has no stock prices')
    digest = hashlib.sha256(raw+json.dumps(meta,sort_keys=True).encode()).hexdigest()
    return Snapshot(digest,prices[tickers+['^GSPC']],tickers,meta['config'],
                    {**meta,'source':str(source.resolve()),'price_sha256':hashlib.sha256(raw).hexdigest()})


def discover_snapshot(root,source=None):
    if source:
        return load_snapshot(source)
    candidates = sorted(Path(root).rglob('run_metadata.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for meta in candidates:
        if 'runs' in meta.parts or any('verification' in p for p in meta.parts):
            continue
        try:
            return load_snapshot(meta.parent)
        except (OSError,ValueError,KeyError):
            continue
    return None


def refresh_snapshot(config,root,progress=None):
    from src.data_fetcher import fetch_prices
    from utils.tick_helper import get_sp500_tickers
    from utils.rate_helper import _fetch_risk_free_rate
    from src.sensitivity import resolve_grid
    report = progress or (lambda _:None)
    cfg = dict(config)
    maximum = max(resolve_grid(cfg)[0]) if cfg.get('sensitivity',{}).get('enabled') else cfg['momentum_window']
    warmup = max(cfg['momentum_window']+cfg['momentum_skip']+cfg['min_history']+30,
                 2*maximum+cfg['momentum_skip']+60)
    start = (pd.Timestamp(cfg['start_date'])-pd.offsets.BDay(warmup)).strftime('%Y-%m-%d')
    report('Fetching current S&P 500 constituents')
    tickers = get_sp500_tickers()
    report('Downloading adjusted stock and benchmark prices')
    prices = fetch_prices(tickers,start,cfg['end_date'])
    if prices.empty or '^GSPC' not in prices or prices['^GSPC'].dropna().empty:
        raise ValueError('Download returned no usable benchmark; previous data retained')
    used = [t for t in tickers if t in prices and prices[t].count()>=cfg['min_history']]
    if not used:
        raise ValueError('Download returned no eligible stocks; previous data retained')
    cfg['rf_annual'] = float(_fetch_risk_free_rate())
    cfg['rf_daily'] = (1+cfg['rf_annual'])**(1/252)-1
    report('Saving a new immutable data snapshot')
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    directory = Path(root)/'dashboard'/'snapshots'/name
    directory.mkdir(parents=True)
    prices.to_csv(directory/'prices.csv')
    # Metadata is the discovery marker; write it only after prices succeed.
    requested_end = cfg['end_date']
    cfg['end_date'] = str(min(pd.Timestamp(requested_end),prices.index[-1]+pd.offsets.BDay(1)).date())
    (directory/'run_metadata.json').write_text(json.dumps({'config':cfg,'requested_download_end':requested_end,'universe_used':used,
        'universe_requested':tickers,'excluded_tickers':sorted(set(tickers)-set(used)),
        'exclusion_reason':'missing prices or fewer than min_history observed sessions'},indent=2),encoding='utf-8')
    return load_snapshot(directory)
