import unittest
import json
from test_engine import fixture


class ChartTests(unittest.TestCase):
    def test_interactive_charts_keep_returns_and_surface_units(self):
        try:
            from src.dashboard.charts import build_dashboard_payload
        except ImportError:
            self.fail('Interactive dashboard figures are missing')
        from src.engine import Snapshot, normalize_config, run_backtest
        prices,cfg = fixture()
        cfg['sensitivity'].update(enabled=True,windows=[8,10],cutoffs=[0,.1],split_date='2023-10-01')
        snap = Snapshot('fixture',prices,['AAA','BBB'],cfg,{})
        result = run_backtest(snap,normalize_config({},cfg,snap))
        payload = build_dashboard_payload(result)
        self.assertAlmostEqual(payload['metrics']['Strategy']['Total Return'],(1+result.returns.Strategy).prod()-1)
        self.assertEqual(len(payload['surfaces']['windows']),2)
        self.assertAlmostEqual(payload['surfaces']['metrics']['full_Total Return']['z'][1][1],
                               result.sensitivity.baseline['full_Total Return']*100)
        self.assertEqual(payload['surfaces']['metrics']['risk_top_share']['unit'],'%')
        self.assertGreaterEqual(len(payload['figures']['risk']),6)
        json.dumps(payload,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
