"""Checks for comparable grids, arbitrary settings and tearsheet delivery."""
import importlib
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd


class SensitivityTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module("src.sensitivity")
        except ImportError:
            self.fail("The reusable sensitivity module is missing")

    def fixture(self):
        index = pd.bdate_range("2024-09-02", periods=150)
        t = np.arange(len(index))
        returns = np.column_stack([
            0.001 + 0.012 * np.sin(t * 0.41),
            0.0003 + 0.009 * np.cos(t * 0.27),
            0.0005 + 0.006 * np.sin(t * 0.19),
        ])
        prices = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0),
                              index=index, columns=["AAA", "BBB", "^GSPC"])
        cfg = {"start_date": "2024-11-01", "momentum_window": 10,
               "momentum_skip": 2, "top_k": 2, "min_history": 12,
               "rebalance_freq": "ME", "z_threshold": 0.1,
               "rf_daily": 0.0, "cost_bps": 0.002}
        return prices, cfg

    def test_custom_grid_includes_current_setting_and_rejects_wrong_baseline(self):
        # Catches hardcoded 252/2 baselines and skipped baseline verification.
        api = self.api()
        prices, cfg = self.fixture()
        result = api.run_sensitivity(prices, ["AAA", "BBB"], cfg,
                                     windows=[8], cutoffs=[0.0], split_date="2025-01-01")
        self.assertEqual(set(map(tuple, result.grid[["window", "cutoff"]].to_numpy())),
                         {(8, 0.0), (8, 0.1), (10, 0.0), (10, 0.1)})
        self.assertEqual(result.start_date, "2024-11-01")
        self.assertEqual(result.end_date, "2025-03-28")
        self.assertFalse(result.baseline_verified)
        wrong = pd.Series(0.5, index=prices.loc[cfg["start_date"]:].index)
        with self.assertRaisesRegex(ValueError, "baseline"):
            api.run_sensitivity(prices, ["AAA", "BBB"], cfg,
                                windows=[8], cutoffs=[0.0], expected_returns=wrong)

    def test_cash_grid_keeps_dates_and_reports_undefined_risk_ratios(self):
        # Catches dropping cash dates or presenting undefined ratios as zero.
        api = self.api()
        prices, cfg = self.fixture()
        result = api.run_sensitivity(prices, ["AAA", "BBB"], cfg,
                                     windows=[10], cutoffs=[5.0], split_date="2025-01-01")
        cash = result.grid[result.grid.cutoff == 5.0].iloc[0]
        self.assertEqual(cash.cash_days, len(prices.loc[cfg["start_date"]:]))
        self.assertEqual(cash["full_Total Return"], 0.0)
        self.assertTrue(np.isnan(cash.full_Sharpe))
        self.assertTrue(np.isnan(cash.full_Calmar))

    def test_invalid_windows_fail_before_running(self):
        api = self.api()
        prices, cfg = self.fixture()
        for windows in [[], [0], [2.5]]:
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                api.run_sensitivity(prices, ["AAA", "BBB"], cfg, windows=windows, cutoffs=[0])

    def test_tearsheet_exports_sensitivity_page_for_supplied_result(self):
        # Catches failure to connect the new module to the public tearsheet flow.
        api = self.api()
        prices, cfg = self.fixture()
        result = api.run_sensitivity(prices, ["AAA", "BBB"], cfg,
                                     windows=[8], cutoffs=[0.0], split_date="2025-01-01")
        from src.visualizations import plot_tearsheets
        from unittest.mock import patch
        index = prices.loc[cfg["start_date"]:].index
        returns = prices.AAA.pct_change().loc[index]
        with tempfile.TemporaryDirectory() as folder:
            prefix = str(Path(folder) / "tearsheet")
            # The legacy MC sheets require long history and are unrelated here.
            with patch("src.visualizations.viz_ops.plot_sheet1"), \
                 patch("src.visualizations.viz_ops.plot_sheet2"), \
                 patch("src.visualizations.viz_ops.plot_sheet3"):
                plot_tearsheets(returns, returns, {}, {}, prices, prices,
                               top_k=2, rf_daily=0, save_prefix=prefix,
                               sensitivity_result=result)
            from PIL import Image
            with Image.open(prefix + "_sheet4_sensitivity.png") as image:
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 500)
            self.assertTrue(Path(prefix + "_sheet5_risk_concentration.png").is_file())

    def test_total_return_surface_uses_percent_units(self):
        # Catches omitting the third surface or confusing 1.25 with 1.25%.
        api = self.api()
        from src.sensitivity.plots import _surface_figure
        import matplotlib.pyplot as plt
        grid = pd.DataFrame({"window": [8, 8, 10, 10], "cutoff": [0, .1, 0, .1],
                             "full_Sharpe": [.5, .7, .8, .9],
                             "full_Calmar": [.4, .5, .6, .7],
                             "full_Total Return": [1, 1.1, 1.2, 1.25],
                             "mean_holdings": [30, 20, 15, 13.25]})
        result = api.SensitivityResult(grid, {"momentum_window": 10, "z_threshold": .1},
                                      [8, 10], [0, .1], "2024-01-01", "2025-01-01", "2024-07-01")
        with plt.rc_context():
            figure = _surface_figure(result)
            try:
                axes = [ax for ax in figure.axes if hasattr(ax, "get_zlabel")]
                self.assertEqual(len(axes), 4)
                total = next(ax for ax in axes if ax.get_zlabel() == "Total return (%)")
                self.assertIn("125.0%", total.get_legend().get_texts()[0].get_text())
                holdings = next(ax for ax in axes if ax.get_zlabel() == "Average holdings")
                self.assertIn("13.25", holdings.get_legend().get_texts()[0].get_text())
            finally:
                plt.close(figure)

    def test_all_cash_grid_can_render_without_inventing_risk_ratios(self):
        api = self.api()
        from src.sensitivity.plots import render_sensitivity
        prices, cfg = self.fixture()
        cfg["z_threshold"] = 5
        result = api.run_sensitivity(prices, ["AAA", "BBB"], cfg,
                                     windows=[8], cutoffs=[5], split_date="2025-01-01")
        with tempfile.TemporaryDirectory() as folder:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                render_sensitivity(result, folder)
            self.assertTrue((Path(folder) / "surfaces.png").is_file())
            result.save(folder)
            import json
            summary = json.loads((Path(folder) / "summary.json").read_text())
            self.assertIsNone(summary["baseline"]["full_Sharpe"])


if __name__ == "__main__":
    unittest.main()
