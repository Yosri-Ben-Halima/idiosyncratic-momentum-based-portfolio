"""Run backtest downloads newer prices first when the end date is past the saved data."""
import json
import pandas as pd
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.dashboard.app import create_app
from src.engine import Snapshot
from test_engine import fixture


class AutoRefreshTests(unittest.TestCase):
    def finish(self, client, job):
        for _ in range(500):
            state = client.get('/api/jobs/'+job['id']).json()
            if state['state'] in ('succeeded', 'failed'):
                return state
            time.sleep(.02)
        self.fail('job did not finish')

    def app(self, folder):
        prices, cfg = fixture()
        source = Path(folder)/'snapshot'
        source.mkdir()
        prices.loc[:'2024-02-29'].to_csv(source/'prices.csv')          # stale: ends Feb 2024
        (source/'run_metadata.json').write_text(json.dumps(
            {'config': {**cfg, 'end_date': '2024-02-01'}, 'universe_used': ['AAA', 'BBB']}))
        fresh = Snapshot('fresh', prices, ['AAA', 'BBB'], cfg, {'source': str(source)})
        return create_app(source, Path(folder)), fresh

    def test_end_past_saved_data_refreshes_then_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            app, fresh = self.app(folder)
            with TestClient(app, base_url='http://127.0.0.1:8000') as client, \
                    patch('src.dashboard.jobs.refresh_snapshot', return_value=fresh) as refresh:
                job = client.post('/api/runs', json={'end_date': '2024-04-01'})
                self.assertEqual(job.status_code, 202, job.text)
                self.assertEqual(self.finish(client, job.json())['state'], 'succeeded')
                refresh.assert_called_once()
                self.assertEqual(refresh.call_args.args[0]['end_date'], '2024-04-01')
                self.assertEqual(client.get('/api/bootstrap').json()['snapshot']['id'], 'fresh')
                results = client.get('/api/runs/'+job.json()['id']+'/results')
                self.assertEqual(results.status_code, 200)
                run = app.state.jobs.jobs[job.json()['id']]
                self.assertEqual(run['snapshot'].id, 'fresh')
                self.assertGreater(run['result'].returns.index[-1], fresh.prices.loc[:'2024-02-29'].index[-1])

    def test_end_inside_saved_data_does_not_download(self):
        with tempfile.TemporaryDirectory() as folder:
            app, fresh = self.app(folder)
            with TestClient(app, base_url='http://127.0.0.1:8000') as client, \
                    patch('src.dashboard.jobs.refresh_snapshot', return_value=fresh) as refresh:
                job = client.post('/api/runs', json={'end_date': '2024-02-01'}).json()
                self.assertEqual(self.finish(client, job)['state'], 'succeeded')
                refresh.assert_not_called()

    def test_end_past_even_the_new_data_fails_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            app, fresh = self.app(folder)
            with TestClient(app, base_url='http://127.0.0.1:8000') as client, \
                    patch('src.dashboard.jobs.refresh_snapshot', return_value=fresh):
                job = client.post('/api/runs', json={'end_date': '2030-01-01'}).json()
                state = self.finish(client, job)
                self.assertEqual(state['state'], 'failed')
                self.assertIn('coverage', state['error'])

    def test_end_today_runs_through_latest_published_session(self):
        # After a holiday (or before any new close) the fresh download stops short of today.
        with tempfile.TemporaryDirectory() as folder:
            app, fresh = self.app(folder)
            last = fresh.prices.index[-1]
            fresh.config['end_date'] = str((last+pd.offsets.BDay(1)).date())
            today = str(pd.Timestamp.today().date())
            with TestClient(app, base_url='http://127.0.0.1:8000') as client, \
                    patch('src.dashboard.jobs.refresh_snapshot', return_value=fresh):
                job = client.post('/api/runs', json={'end_date': today}).json()
                self.assertEqual(self.finish(client, job)['state'], 'succeeded', job)
                run = app.state.jobs.jobs[job['id']]
                self.assertEqual(run['config']['end_date'], fresh.config['end_date'])
                self.assertEqual(run['result'].returns.index[-1], last)

    def test_invalid_settings_still_rejected_before_any_download(self):
        with tempfile.TemporaryDirectory() as folder:
            app, fresh = self.app(folder)
            with TestClient(app, base_url='http://127.0.0.1:8000') as client, \
                    patch('src.dashboard.jobs.refresh_snapshot', return_value=fresh) as refresh:
                bad = client.post('/api/runs', json={'end_date': '2030-01-01', 'top_k': 0})
                self.assertEqual(bad.status_code, 422)
                refresh.assert_not_called()
