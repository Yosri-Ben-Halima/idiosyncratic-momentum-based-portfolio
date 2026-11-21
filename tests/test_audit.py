"""Current-run audit evidence, isolated comparisons and point-in-time probes."""
import importlib
import json
import unittest
import numpy as np
import pandas as pd
from test_engine import fixture
from src.engine import Snapshot, normalize_config, run_backtest


class AuditTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('src.engine.audit')
        except ImportError:
            self.fail('Current-run audit engine is missing')

    def result(self, cash=False, short=False):
        p,cfg=fixture()
        if cash:cfg['z_threshold']=6
        if short:cfg.update(start_date='2023-05-01',end_date='2023-05-04')
        snap=Snapshot('audit-fixture',p,['AAA','BBB'],cfg,{})
        cfg=normalize_config({},cfg,snap)
        return snap,run_backtest(snap,cfg)

    def test_paired_estimator_matches_least_squares_with_missing_observations(self):
        api=self.api()
        rng=np.random.default_rng(4)
        x=pd.Series(rng.normal(0,.01,100));y=.002+1.4*x
        y.iloc[:15]=np.nan;x.iloc[38:42]=np.nan
        actual=api.paired_capm_residuals(y,x,30)
        self.assertTrue(np.isnan(actual.iloc[40]))
        for t in [45,50,80]:
            prior=pd.DataFrame({'x':x,'y':y}).iloc[t-30:t].dropna()
            fit=np.linalg.lstsq(np.column_stack([np.ones(len(prior)),prior.x]),prior.y,rcond=None)[0]
            expected=y.iloc[t]-np.array([1,x.iloc[t]])@fit
            self.assertAlmostEqual(actual.iloc[t],expected,places=12)

    def test_real_audit_is_scoped_reconciled_and_does_not_mutate_run(self):
        api=self.api();snap,result=self.result()
        original_prices=snap.prices.copy();original_weights=result.weights.copy()
        report=api.run_audit(snap,result)
        self.assertEqual(report['scope']['snapshot_id'],'audit-fixture')
        self.assertEqual(report['scope']['config'],result.config)
        self.assertEqual(report['timing']['status'],'passed')
        for check in report['checks'][:3]:self.assertEqual(check['status'],'passed',check)
        for check in report['checks'][1:3]:
            self.assertLessEqual(check['differences']['weights'],1e-10)
            self.assertLessEqual(check['differences']['returns'],1e-10)
        self.assertEqual(len(report['comparisons']),4)
        self.assertAlmostEqual(report['comparisons'][0]['metrics']['cagr'],result.metrics['Strategy']['CAGR'])
        self.assertAlmostEqual(sum(row['value'] for row in report['attribution']),result.metrics['Strategy']['Total Return'])
        self.assertEqual(report['estimator']['status'],'issue')
        self.assertTrue(all(item['status']=='not_tested' for item in report['limitations']))
        self.assertNotIn('NaN',json.dumps(report,allow_nan=False))
        pd.testing.assert_frame_equal(snap.prices,original_prices)
        pd.testing.assert_frame_equal(result.weights,original_weights)
        self.assertEqual(api.run_audit(snap,result)['comparisons'],report['comparisons'])

    def test_cash_and_short_runs_do_not_claim_decision_timing_pass(self):
        api=self.api()
        for cash,short in [(True,False),(False,True)]:
            with self.subTest(cash=cash,short=short):
                snap,result=self.result(cash=cash,short=short)
                report=api.run_audit(snap,result)
                self.assertEqual(report['timing']['status'],'insufficient')
                self.assertNotIn('NaN',json.dumps(report,allow_nan=False))
                if cash:
                    self.assertEqual(report['concentration']['mean_holdings'],0)
                    self.assertEqual(report['comparisons'][0]['metrics']['total_return'],0)

    def test_snapshot_mismatch_is_rejected(self):
        api=self.api();snap,result=self.result()
        snap.id='different'
        with self.assertRaisesRegex(ValueError,'snapshot'):
            api.run_audit(snap,result)

    def test_drift_counterfactual_holds_shares_between_target_changes(self):
        api=self.api()
        dates=pd.bdate_range('2024-01-01',periods=3)
        raw=pd.DataFrame({'AAA':[0.,1.,0.],'BBB':[0.,0.,1.]},index=dates)
        w=pd.DataFrame(.5,index=dates,columns=raw.columns)
        ret=api.drift_returns(w,raw,0)
        np.testing.assert_allclose(ret,[0,.5,1/3])

    def test_drift_resets_at_scheduled_rebalance_even_when_targets_are_identical(self):
        api=self.api()
        dates=pd.bdate_range('2024-01-30',periods=4)
        raw=pd.DataFrame({'AAA':[0.,1.,0.,0.],'BBB':[0.,0.,0.,1.]},index=dates)
        weights=pd.DataFrame(.5,index=dates,columns=raw.columns)
        actual=api.drift_returns(weights,raw,0)
        # Jan 31 selection becomes the Feb 1 target, applied to Feb 2 returns.
        np.testing.assert_allclose(actual,[0,.5,0,.5])

    def test_drift_preserves_baseline_initialization_cost_convention(self):
        api=self.api()
        dates=pd.bdate_range('2024-01-01',periods=4)
        raw=pd.DataFrame({'AAA':0.},index=dates)
        weights=pd.DataFrame({'AAA':1.},index=dates)
        actual=api.drift_returns(weights,raw,.002)
        np.testing.assert_allclose(actual,np.zeros(4))


if __name__=='__main__':unittest.main()
