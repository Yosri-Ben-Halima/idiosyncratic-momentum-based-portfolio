"""Engine parity, validation, and diagnostics regressions."""
import importlib
import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path


def fixture():
    dates = pd.bdate_range('2023-01-02', periods=330)
    t = np.arange(len(dates))
    values = np.column_stack([.001 + .013*np.sin(t*.41),
                              .0003+.009*np.cos(t*.27), .0005+.006*np.sin(t*.19)])
    prices = pd.DataFrame(100*np.cumprod(1+values, axis=0), index=dates,
                          columns=['AAA', 'BBB', '^GSPC'])
    config = dict(start_date='2023-05-01', end_date='2024-04-01', momentum_window=10,
                  momentum_skip=2, top_k=2, z_threshold=.1, rebalance_freq='ME',
                  min_history=12, transaction_cost_bps=20, rf_annual=.04,
                  monte_carlo_runs=100, monte_carlo_days=21,
                  sensitivity={'enabled': False, 'risk_window': 30, 'risk_min_observations': 10})
    return prices, config


class EngineTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('src.engine')
        except ImportError:
            self.fail('Shared backtest engine is missing')

    def test_normalization_derives_once_and_rejects_invalid_inputs(self):
        api = self.api()
        prices, cfg = fixture()
        snap = api.Snapshot('fixture', prices, ['AAA','BBB'], cfg, {})
        normalized = api.normalize_config({}, cfg, snap)
        self.assertAlmostEqual(normalized['cost_bps'], .002)
        self.assertAlmostEqual(normalized['rf_daily'], 1.04**(1/252)-1)
        self.assertEqual(api.normalize_config({}, normalized, snap), normalized)
        for bad in [dict(momentum_window=True), dict(z_threshold=float('nan')),
                    dict(start_date='2030-01-01'), dict(end_date='2025-01-01'),
                    dict(rebalance_freq='nonsense'), dict(unknown=1),
                    dict(sensitivity={'risk_window':10,'risk_min_observations':20})]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                api.normalize_config(bad, cfg, snap)

    def test_backtest_matches_original_computation_and_end_is_exclusive(self):
        api = self.api()
        from src.feature_engineering import build_idiosyncratic_momentum, zscore_normalize
        from src.portfolio_construction import build_portfolio
        from src.backtesting import compute_portfolio_returns
        prices, cfg = fixture()
        snap = api.Snapshot('fixture',prices,['AAA','BBB'],cfg,{})
        config = api.normalize_config({},cfg,snap)
        result = api.run_backtest(snap,config)
        source = prices.loc[prices.index < pd.Timestamp(cfg['end_date'])]
        signal = zscore_normalize(build_idiosyncratic_momentum(source,snap.tickers,10,config['rf_daily'],2))
        weights = build_portfolio(signal,source,snap.tickers,2,'ME',12,.1).loc[cfg['start_date']:]
        expected = compute_portfolio_returns(weights,source.loc[cfg['start_date']:],.002)
        pd.testing.assert_series_equal(result.returns.Strategy,expected,check_names=False)
        self.assertTrue((result.returns.index < pd.Timestamp(cfg['end_date'])).all())
        self.assertFalse(result.weights.empty)
        self.assertEqual(cfg.get('cost_bps'),None)
        free = api.run_backtest(snap,api.normalize_config({'transaction_cost_bps':0},cfg,snap))
        self.assertTrue((free.returns.Strategy >= result.returns.Strategy-1e-12).all())

    def test_future_requested_download_does_not_extend_observed_coverage(self):
        api=self.api()
        prices,cfg=fixture()
        cfg['end_date']='2030-01-01'
        snap=api.Snapshot('incomplete',prices,['AAA','BBB'],cfg,{})
        with self.assertRaisesRegex(ValueError,'coverage'):
            api.normalize_config({},cfg,snap)

    def test_rolling_sharpe_matches_existing_png_formula(self):
        api=self.api()
        prices,cfg=fixture()
        snap=api.Snapshot('fixture',prices,['AAA','BBB'],cfg,{})
        result=api.run_backtest(snap,api.normalize_config({},cfg,snap))
        r=result.returns.Strategy
        expected=r.rolling(10).mean()/r.rolling(10).std()*np.sqrt(252)
        np.testing.assert_allclose(result.diagnostics['rolling_sharpe'].Strategy,expected,equal_nan=True)

    def test_diagnostics_cash_short_history_and_seed_are_honest(self):
        api = self.api()
        from src.engine.diagnostics import monte_carlo, recovery_years
        cash = pd.Series(0.,index=pd.bdate_range('2024-01-01',periods=5))
        simulation = monte_carlo(cash,100,21,42)
        self.assertTrue(np.array_equal(simulation['terminal'],np.zeros(100)))
        varied = cash + np.linspace(-.01,.02,5)
        first = monte_carlo(varied,100,21,42)
        second = monte_carlo(varied,100,21,42)
        np.testing.assert_array_equal(first['terminal'],second['terminal'])
        self.assertIsNone(recovery_years(-.4,0))
        self.assertAlmostEqual(recovery_years(-.4,.2),2)

    def test_cash_short_history_exports_png_and_exact_returns(self):
        api = self.api()
        from src.engine.exports import export_result
        prices,cfg = fixture()
        cfg.update(z_threshold=6,end_date='2023-06-01')
        snap = api.Snapshot('cash',prices,['AAA','BBB'],cfg,{})
        result = api.run_backtest(snap,api.normalize_config({},cfg,snap))
        with tempfile.TemporaryDirectory() as folder:
            files = export_result(result,Path(folder))
            self.assertEqual(len([x for x in files if x.endswith('.png')]),3)
            actual = pd.read_csv(files['daily_returns.csv'],index_col=0,parse_dates=True)
            pd.testing.assert_frame_equal(actual,result.returns,check_freq=False)
            from PIL import Image
            for name,path in files.items():
                if name.endswith('.png'):
                    with Image.open(path) as picture:
                        self.assertGreater(picture.width,1000)


if __name__ == '__main__':
    unittest.main()
