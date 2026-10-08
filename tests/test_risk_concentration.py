"""Hand-derived risk decomposition and no-future-data checks."""
import importlib
import unittest

import numpy as np
import pandas as pd


class RiskConcentrationTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module("src.sensitivity.risk")
        except ImportError:
            self.fail("Risk concentration diagnostics are missing")

    def test_equal_holdings_can_have_different_correlation_diversification(self):
        api = self.api()
        independent = api.risk_metrics(np.array([.5, .5]), np.diag([.01, .01]))
        correlated = api.risk_metrics(np.array([.5, .5]), np.full((2, 2), .01))
        self.assertAlmostEqual(independent["risk_diversification_ratio"], 2**.5)
        self.assertAlmostEqual(correlated["risk_diversification_ratio"], 1)
        self.assertAlmostEqual(independent["risk_top_share"], .5)
        self.assertAlmostEqual(correlated["risk_effective_contributors"], 2)

    def test_equal_capital_weights_can_put_ninety_percent_of_risk_in_one_stock(self):
        api = self.api()
        result = api.risk_metrics(np.array([.5, .5]), np.diag([.01, .09]))
        self.assertAlmostEqual(result["risk_top_share"], .9)
        self.assertAlmostEqual(result["risk_effective_contributors"], 1/.82)

    def test_cash_and_missing_covariance_are_not_treated_as_diversified(self):
        api = self.api()
        for weights, covariance in [(np.zeros(2), np.eye(2)),
                                     (np.ones(2)/2, np.full((2, 2), np.nan))]:
            result = api.risk_metrics(weights, covariance)
            self.assertTrue(np.isnan(result["risk_diversification_ratio"]))
            self.assertTrue(np.isnan(result["risk_top_share"]))

    def test_future_price_shocks_do_not_change_earlier_risk_estimates(self):
        api = self.api()
        index = pd.bdate_range("2024-01-01", periods=100)
        t = np.arange(100)
        prices = pd.DataFrame({"AAA":100*np.cumprod(1+.01*np.sin(t*.4)),
                               "BBB":100*np.cumprod(1+.02*np.cos(t*.3))}, index=index)
        weights = pd.DataFrame(.5,index=index[40:],columns=prices.columns)
        altered = prices.copy()
        altered.loc[index[70]:,"AAA"] *= 5
        original = api.RiskDiagnostics(prices, window=30, min_observations=20).evaluate(weights)
        changed = api.RiskDiagnostics(altered, window=30, min_observations=20).evaluate(weights)
        pd.testing.assert_frame_equal(original.loc[:index[69]], changed.loc[:index[69]])
        self.assertTrue(original.risk_diversification_ratio.notna().all())


if __name__ == "__main__":
    unittest.main()
