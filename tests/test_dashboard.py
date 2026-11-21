"""Real computation through the localhost API and snapshot boundary."""
import importlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from threading import Event
from test_engine import fixture


class DashboardTests(unittest.TestCase):
    def modules(self):
        try:
            from src.dashboard.app import create_app
            from src.engine.snapshots import discover_snapshot
            return create_app,discover_snapshot
        except ImportError:
            self.fail('Local dashboard and saved snapshot service are missing')

    def snapshot(self,folder):
        prices,cfg = fixture()
        source = Path(folder)/'snapshot'
        source.mkdir()
        prices.to_csv(source/'prices.csv')
        (source/'run_metadata.json').write_text(json.dumps({'config':cfg,'universe_used':['AAA','BBB']}))
        return source

    def test_snapshot_identity_and_no_network_loading(self):
        _,discover = self.modules()
        with tempfile.TemporaryDirectory() as folder:
            source = self.snapshot(folder)
            snap = discover(Path(folder))
            self.assertEqual(snap.tickers,['AAA','BBB'])
            previous = snap.id
            prices,cfg = fixture()
            prices.iloc[-1,0] += 1
            prices.to_csv(source/'prices.csv')
            self.assertNotEqual(previous,discover(Path(folder)).id)
            self.assertEqual(snap.prices.iloc[-1,0]+1,prices.iloc[-1,0])

    def test_real_run_results_validation_and_origin_protection(self):
        create_app,_ = self.modules()
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as folder:
            source = self.snapshot(folder)
            app = create_app(source,Path(folder))
            with TestClient(app,base_url='http://127.0.0.1:8000') as client:
                bootstrap = client.get('/api/bootstrap').json()
                self.assertEqual(bootstrap['snapshot']['stocks'],2)
                self.assertEqual(client.post('/api/runs',json={'momentum_window':True}).status_code,422)
                self.assertEqual(client.post('/api/runs',json={},headers={'Origin':'https://evil.example'}).status_code,403)
                self.assertEqual(client.get('/api/bootstrap',headers={'Host':'evil.example'}).status_code,403)
                job = client.post('/api/runs',json={}).json()
                for _ in range(100):
                    state = client.get('/api/jobs/'+job['id']).json()
                    if state['state'] in ('succeeded','failed'):
                        break
                    time.sleep(.05)
                self.assertEqual(state['state'],'succeeded',state)
                result = client.get('/api/runs/'+job['id']+'/results')
                self.assertEqual(result.status_code,200)
                data = result.json()
                self.assertEqual(data['config']['momentum_window'],10)
                self.assertIn('performance',data['figures'])
                self.assertNotIn('NaN',result.text)
                cached = client.post('/api/runs',json={}).json()
                self.assertEqual(cached['id'],job['id'])
                changed = client.post('/api/runs',json={'transaction_cost_bps':0}).json()
                self.assertNotEqual(changed['id'],job['id'])
                self.assertEqual(client.get('/api/runs/'+job['id']+'/files/private.txt').status_code,404)

    def test_worker_bounds_queue_and_reports_failure_without_losing_previous_jobs(self):
        self.modules()
        from src.dashboard.jobs import JobManager
        entered,release=Event(),Event()
        def unavailable(*args):
            entered.set()
            release.wait(3)
            raise ValueError('Controlled computation failure')
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(Path(folder))
            try:
                with patch('src.dashboard.jobs.run_backtest',side_effect=unavailable):
                    first=manager.submit('run',{'cost':1})
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(manager.submit('run',{'cost':1})['id'],first['id'])
                    second=manager.submit('run',{'cost':2})
                    third=manager.submit('run',{'cost':3})
                    with self.assertRaisesRegex(ValueError,'queue'):
                        manager.submit('run',{'cost':4})
                    release.set()
                    for job in [first,second,third]:
                        for _ in range(100):
                            status=manager.status(job['id'])
                            if status['state']=='failed':break
                            time.sleep(.01)
                        self.assertEqual(status['state'],'failed')
                        self.assertEqual(status['error'],'Controlled computation failure')
            finally:
                release.set()
                manager.close()

    def test_explicit_refresh_does_not_reuse_completed_refresh(self):
        self.modules()
        from src.dashboard.jobs import JobManager
        from src.engine import Snapshot
        prices,cfg=fixture()
        fresh=Snapshot('new',prices,['AAA','BBB'],cfg,{})
        with tempfile.TemporaryDirectory() as folder:
            seen=[]
            manager=JobManager(Path(folder),lambda snap:seen.append(snap.id))
            try:
                with patch('src.dashboard.jobs.refresh_snapshot',return_value=fresh):
                    first=manager.submit('refresh',cfg)
                    for _ in range(100):
                        if manager.status(first['id'])['state']=='succeeded':break
                        time.sleep(.01)
                    second=manager.submit('refresh',cfg)
                    self.assertNotEqual(first['id'],second['id'])
                    for _ in range(100):
                        if manager.status(second['id'])['state']=='succeeded':break
                        time.sleep(.01)
                self.assertEqual(seen,['new','new'])
            finally:manager.close()


if __name__ == '__main__':
    unittest.main()
