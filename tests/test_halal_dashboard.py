"""Halal filter in the shared engine used by the dashboard."""
import pytest

import src.engine.backtest as backtest
from src.engine import Snapshot, normalize_config, run_backtest
from test_engine import fixture  # tests/ is on sys.path under pytest


def snap():
    prices, cfg = fixture()
    return Snapshot('fixture', prices, ['AAA', 'BBB'], cfg, {}), cfg


def test_defaults_off_and_aaoifi_and_idempotent():
    s, cfg = snap()
    n = normalize_config({}, cfg, s)
    assert n['halal_filter'] is False and n['halal_standard'] == 'aaoifi'
    assert normalize_config({}, n, s) == n


@pytest.mark.parametrize('bad', [{'halal_filter': 'yes'}, {'halal_standard': 'msci'}])
def test_rejects_invalid_halal_settings(bad):
    s, cfg = snap()
    with pytest.raises(ValueError):
        normalize_config(bad, cfg, s)


def test_run_uses_only_compliant_stocks(monkeypatch):
    calls = []
    monkeypatch.setattr(backtest, 'filter_halal',
                        lambda tickers, standard: calls.append(standard) or ['AAA'])
    s, cfg = snap()
    result = run_backtest(s, normalize_config({'halal_filter': True, 'halal_standard': 'wahed'}, cfg, s))
    assert calls == ['wahed']
    assert result.provenance['universe_used'] == ['AAA']
    assert result.provenance['halal_standard'] == 'wahed'
    assert 'BBB' not in result.weights.columns or (result.weights['BBB'] == 0).all()
    assert any('look-ahead' in w for w in result.warnings)


def test_filter_off_leaves_universe_alone(monkeypatch):
    monkeypatch.setattr(backtest, 'filter_halal', lambda *a: pytest.fail('filter must not run'))
    s, cfg = snap()
    result = run_backtest(s, normalize_config({}, cfg, s))
    assert result.provenance['universe_used'] == ['AAA', 'BBB']


def test_no_compliant_stock_in_snapshot_raises(monkeypatch):
    monkeypatch.setattr(backtest, 'filter_halal', lambda tickers, standard: [])
    s, cfg = snap()
    with pytest.raises(ValueError, match='No stock in this snapshot passes'):
        run_backtest(s, normalize_config({'halal_filter': True}, cfg, s))
